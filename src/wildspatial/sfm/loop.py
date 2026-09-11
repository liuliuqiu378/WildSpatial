"""回环检测（Loop Closure Detection）

=========================== 为什么必须有它 ===========================
VO 逐帧 PnP，每帧的小误差**沿轨迹累积** → 漂移。
BA 能摊平局部重投影误差，但在**没有回环约束的开阔轨迹**上，
它消不掉整体的尺度/旋转偏差 —— 本项目实测：BA 把重投影误差 1.54px → 0.20px，
**ATE 却 0.506 → 0.506 m 纹丝不动**。

回环检测的核心洞察：
    如果相机**回到了以前到过的地方**，那么"当前帧 i" 与 "很久以前的帧 j"
    看到的是同一个场景。这两帧之间的位姿关系可以**一次直接测量**得到 ——
    它只含一份测量误差，**不会累积中间 N 帧的误差**。
    把这个"准确的长程约束"丢进位姿图（PGO）做优化，累积漂移就被**分摊回整条轨迹**。

这正是 SLAM（ORB-SLAM）精度碾压纯 VO 的根本原因，也是我们 ATE
从 0.5m 量级往 0.02~0.05m 走的必经之路。

=========================== 本模块的检测流程 ===========================
1. **关键帧抽样**：每隔 kf_stride 帧取一个关键帧（不必每帧都比，省时间）
2. **外观匹配**：当前关键帧 i ↔ 历史关键帧 j（要求 i - j >= min_gap，排除相邻帧）
   用描述子比值检验（同 M1 前端的 match_ratio_test）
3. **几何验证**：对匹配点做 Essential 矩阵 RANSAC，要求内点率足够高
   （这一步能滤掉绝大多数"看起来像但几何上不成立"的误检 —— 感知歧义的护栏）
4. **求回环位姿**：用 j 帧可见的**地图点**（3D）+ i 帧的 2D 观测做 PnP，
   直接解出 i 帧的绝对位姿 T_i^loop。
   ⭐ 关键：这个位姿是"直接用地图解出来的"，绕过了 j→i 之间逐帧累积的漂移，
   所以它比 VO 链式推出来的 T_i^vo 更准 —— 二者之差就是回环要修正的漂移量。

⚠️ 本实现是**教学版**（暴力两两匹配 + SIFT），生产系统用
词袋模型（DBoW2/DBoW3）+ 反向索引做到毫秒级检索。但原理完全一致。
"""

import numpy as np

from .matching import match_ratio_test
from .ransac import ransac_essential
from ..geometry.pnp import pnp_with_refine

__all__ = ["camera_center", "detect_loops", "estimate_loop_pose", "build_loop_priors"]


def camera_center(T_cw):
    """相机在世界系下的位置：C = −R^T t"""
    return -T_cw[:3, :3].T @ T_cw[:3, 3]


def detect_loops(vo, kf_stride=5, min_gap=3, ratio=0.8, min_matches=60,
                 min_inlier_ratio=0.5, max_per_frame=1, ransac_thresh_px=2.0,
                 verbose=False):
    """在已处理的帧里检测回环。

    参数
        vo               : MonocularVO 实例（需已完成 process）
        kf_stride        : 每隔多少帧取一个关键帧参与检测
        min_gap          : 关键帧索引间隔下限（i 与 j 至少隔开这么多个关键帧）
        ratio            : 描述子比值检验阈值
        min_matches      : 候选所需的最少匹配数
        min_inlier_ratio : 几何验证（E 矩阵 RANSAC）所需的最小内点率
        max_per_frame    : 每个关键帧最多保留几个最优回环（避免重复约束）
    返回
        list of dict: {i, j, n_matches, inlier_ratio}  （i 为较新的帧）
    """
    frames = vo.frames
    K = vo.K
    if len(frames) < 2 * kf_stride:
        return []

    kf = list(range(0, len(frames), kf_stride))
    cands = []
    for a in range(len(kf)):
        i = kf[a]
        best_for_i = []
        for b in range(a):
            if a - b < min_gap:
                continue
            j = kf[b]
            dj, di = frames[j].descriptors, frames[i].descriptors
            if dj is None or di is None or len(dj) < 2 or len(di) < 2:
                continue
            m = match_ratio_test(dj, di, ratio=ratio)
            if len(m) < min_matches:
                continue
            p_j = frames[j].keypoints[m[:, 0]]
            p_i = frames[i].keypoints[m[:, 1]]
            E, mask, stats = ransac_essential(
                p_j, p_i, K, thresh_px=ransac_thresh_px,
                max_iters=500, seed=j, return_stats=True)
            ir = float(stats.get("inlier_ratio", 0.0))
            if E is None or ir < min_inlier_ratio:
                continue
            best_for_i.append({"i": int(i), "j": int(j),
                               "n_matches": int(len(m)), "inlier_ratio": ir})
        # 每个关键帧只保留最好的几个（匹配数最多 = 最可信）
        best_for_i.sort(key=lambda d: -d["n_matches"])
        cands.extend(best_for_i[:max_per_frame])

    cands.sort(key=lambda d: -d["n_matches"])
    if verbose:
        print(f"[回环] 检测到 {len(cands)} 个候选：")
        for c in cands:
            print(f"       i={c['i']:4d} ← j={c['j']:4d}  "
                  f"matches={c['n_matches']:4d}  inlier={c['inlier_ratio']:.1%}")
    return cands


def estimate_loop_pose(vo, i, j, window=3, ratio=0.8, min_points=25,
                       max_drift_m=3.0, max_rot_deg=60.0):
    """求回环帧 i 的**绝对**位姿（绕过 j→i 的漂移链）。

    做法：把 **j 邻域窗口内各帧**上可见的地图点（3D，世界系）与 i 帧的特征（2D）
    通过描述子匹配关联起来，池化成 (3D ↔ 2D) 对应，再 PnP 直接解出 T_i。

    ⭐ 为什么要用"窗口"而不只是 j 这一帧？
    本项目的地图里，每个点只在**它被三角化的那一帧**记录观测（`_add_points` 只写创建帧），
    单帧上带地图点的关键点只占约 5%，凑不够 PnP 需要的点数（实测单帧仅约 8 个）。
    取 j 前后各 window 帧一起匹配池化，才够解 PnP —— 这就是 ORB-SLAM
    "在**共视关键帧**中搜索地图点"的同款思路。

    这样解出的 T_i 是"由地图直接定位"得到的，不经过 j→i 之间逐帧累积；
    而 VO 的 T_i^vo 是链式推过来的。二者之差 ≈ 这一段累积的漂移。

    返回 T_cw (4,4)；失败（点不够 / PnP 异常 / 结果离谱）返回 None。
    """
    frames = vo.frames
    X_all, uv_all = [], []
    seen_kp = set()          # 同一个 i 帧关键点若匹配到多个地图点，只取第一个（避免歧义）
    for f in range(max(0, j - window), min(len(frames), j + window + 1)):
        df, di = frames[f].descriptors, frames[i].descriptors
        if df is None or di is None or len(df) < 2 or len(di) < 2:
            continue
        m = match_ratio_test(df, di, ratio=ratio)
        for a, b in m:
            b = int(b)
            if b in seen_kp:
                continue
            pid = vo.obs.get((int(f), int(a)))
            if pid is None or pid >= len(vo.points_3d):
                continue
            seen_kp.add(b)
            X_all.append(vo.points_3d[pid])
            uv_all.append(frames[i].keypoints[b])

    if len(X_all) < min_points:
        return None
    X = np.asarray(X_all, dtype=float)
    uv_i = np.asarray(uv_all, dtype=float)

    try:
        T_loop = pnp_with_refine(X, uv_i, vo.K, iters=30)
    except Exception:
        return None
    if not np.all(np.isfinite(T_loop)):
        return None

    # ---- 护栏：回环修正不该是"跳变"，与 VO 位姿差太远就判为误检 ----
    T_vo = frames[i].T_cw
    if T_vo is not None:
        dpos = float(np.linalg.norm(camera_center(T_loop) - camera_center(T_vo)))
        Rrel = T_loop[:3, :3] @ T_vo[:3, :3].T
        c = float(np.clip((np.trace(Rrel) - 1.0) / 2.0, -1.0, 1.0))
        dang = float(np.degrees(np.arccos(c)))
        if dpos > max_drift_m or dang > max_rot_deg:
            return None
    return T_loop


def build_loop_priors(vo, cands, window=3, ratio=0.8, min_points=25,
                      max_drift_m=3.0, verbose=False):
    """把回环候选转成 PGO 可用的绝对位姿先验 [(i, T_abs), ...]"""
    priors = []
    for c in cands:
        T = estimate_loop_pose(vo, c["i"], c["j"], window=window, ratio=ratio,
                               min_points=min_points, max_drift_m=max_drift_m)
        if T is None:
            continue
        priors.append((c["i"], T))
        if verbose:
            d = np.linalg.norm(camera_center(T) - camera_center(vo.frames[c["i"]].T_cw))
            print(f"[回环] i={c['i']} ← j={c['j']}: 修正量 {d:.3f} m")
    return priors
