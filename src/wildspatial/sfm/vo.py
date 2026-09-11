"""单目视觉里程计（VO）—— 把 M0 的几何 + M1 的特征串成完整管线

管线的三个阶段
--------------
1. **初始化**（两帧）
   匹配 → RANSAC 估 E → recover_pose 得 (R, t_unit) → 三角化建初始地图。
   ⚠️ t 只有方向没有长度（单目尺度歧义），需要人为定一个尺度。

2. **跟踪**（每新帧）
   与上一帧匹配 → 找出"上一帧特征点中已有 3D 地图点"的那些 → **PnP 定位**。
   用 PnP 而不是重新解 E，是因为 PnP 能**继承已有地图的尺度**，避免尺度逐帧漂移。

3. **扩展**
   对尚未三角化的匹配点做三角化，补充地图。

为什么要亲手实现它？
    只有亲手搭一遍，才知道 VGGT（M4）"一次前向取代整条管线"到底取代了什么：
    特征检测 → 描述子 → 匹配 → RANSAC → 位姿分解 → 三角化 → PnP → BA。
    而且只有亲手搭，才知道它在**哪里会崩**（M3/M5 的主题）。

输出诊断信息（M3 失效归因用）
----------------------------
    每帧记录：特征数 / 匹配数 / 内点数 / 内点率 / PnP 点数 / 地图点数 / 跟踪状态。
    这些"过程量"比最终的 ATE 更能定位失效根因。
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geometry import lie, camera, epipolar, triangulation, pnp
from .features import extract_features
from .matching import match_ratio_test
from .ransac import ransac_essential
from .ba import local_ba
from .loop import detect_loops, build_loop_priors
from .pgo import pose_graph_optimize

__all__ = ["VOFrame", "MonocularVO", "VOConfig"]


@dataclass
class VOConfig:
    """VO 参数（调参即工程，M5 会系统扫这些）"""
    feature: str = "sift"
    max_features: int = 3000
    ratio: float = 0.8                 # Lowe 比值检验阈值
    ransac_thresh_px: float = 1.5      # RANSAC 内点阈值（像素）
    ransac_max_iters: int = 1000
    min_inliers: int = 20              # 低于此值认为跟踪失败
    min_pnp_points: int = 8            # PnP 最少点数
    min_parallax_deg: float = 1.0      # 常规三角化的最小视差角
    max_depth: float = 20.0            # 剔除过远的三角化点
    init_median_depth: float = 3.0     # 初始化时把中值深度归一化到此值（定尺度）
    init_min_parallax_deg: float = 3.0 # ⭐ 初始化所需的最小**中位**视差角
    init_max_wait_frames: int = 60     # 最多等多少帧来累积视差，超过则放弃
    # ---- 运动连续性护栏（防止位姿爆炸）----
    max_step_m: float = 0.5            # 单帧最大位移（米），超出视为异常跳变
    max_step_ratio: float = 6.0        # 相对历史中位位移的最大倍数
    step_history: int = 10             # 用最近多少帧的位移中位数作参考
    verbose: bool = False


@dataclass
class VOFrame:
    idx: int
    keypoints: np.ndarray                    # (N,2)
    descriptors: np.ndarray
    T_cw: np.ndarray = None                  # 世界 → 相机
    image: np.ndarray = None
    depth: np.ndarray = None


class MonocularVO:
    """单目视觉里程计

    用法：
        vo = MonocularVO(K)
        for img in images:
            info = vo.process(img)
        traj = vo.trajectory()          # (N,3) 相机位置（世界系）
    """

    def __init__(self, K, config: VOConfig = None):
        self.K = np.asarray(K, dtype=float).reshape(3, 3)
        self.cfg = config or VOConfig()
        self.frames = []
        self.points_3d = np.zeros((0, 3))          # 世界坐标系下的地图点
        self.obs = {}                              # (frame_idx, kp_idx) -> point_id
        self.initialized = False
        self.history = []                          # 每帧诊断信息
        self.scale = 1.0                           # 当前地图尺度（单目下是相对尺度）
        self.init_ref_idx = 0                      # 初始化参考帧（累积视差期间固定）
        self.init_frame_idx = None                 # 成功初始化发生在第几帧（评估起点）

    # ------------------------------------------------------------------ 主入口
    def process(self, image, depth=None):
        """处理一帧。返回 dict（含位姿与诊断）"""
        idx = len(self.frames)
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        fs = extract_features(gray, self.cfg.feature,
                              max_features=self.cfg.max_features)
        frame = VOFrame(idx=idx, keypoints=fs.keypoints,
                        descriptors=fs.descriptors, image=gray, depth=depth)

        info = {"idx": idx, "n_features": len(fs), "status": "init"}

        if idx == 0:
            frame.T_cw = np.eye(4)
            self.frames.append(frame)
            info.update({"n_matches": 0, "n_inliers": 0, "inlier_ratio": 0.0,
                         "n_pnp": 0, "n_map": 0, "status": "first"})
            self.history.append(info)
            return info

        # 未初始化时固定用**参考帧**做匹配（累积视差），已初始化后跟踪上一帧
        ref = self.frames[self.init_ref_idx] if not self.initialized else self.frames[-1]
        prev = ref
        matches = match_ratio_test(prev.descriptors, fs.descriptors,
                                   ratio=self.cfg.ratio)
        info["n_matches"] = len(matches)

        if len(matches) < self.cfg.min_inliers:
            # 匹配太少 —— 跟踪失败（弱纹理/快速运动/光照剧变的典型表现）
            frame.T_cw = prev.T_cw.copy()
            self.frames.append(frame)
            info.update({"n_inliers": 0, "inlier_ratio": 0.0, "n_pnp": 0,
                         "n_map": len(self.points_3d), "status": "lost_no_matches"})
            self.history.append(info)
            return info

        p_prev = prev.keypoints[matches[:, 0]]
        p_cur = fs.keypoints[matches[:, 1]]

        if not self.initialized:
            ok = self._initialize(prev, frame, p_prev, p_cur, matches, info, depth)
        else:
            ok = self._track(prev, frame, p_prev, p_cur, matches, info, depth)

        if not ok:
            frame.T_cw = prev.T_cw.copy()      # 失败时沿用上一帧位姿（VO 的常规做法）
            info["status"] = "lost"

        self.frames.append(frame)
        info["n_map"] = len(self.points_3d)
        self.history.append(info)
        return info

    # ------------------------------------------------------------------ 初始化
    def _initialize(self, prev, cur, p_prev, p_cur, matches, info, depth=None):
        E, mask, stats = ransac_essential(
            p_prev, p_cur, self.K,
            thresh_px=self.cfg.ransac_thresh_px,
            max_iters=self.cfg.ransac_max_iters, seed=cur.idx,
            return_stats=True)
        info.update({"n_inliers": int(stats.get("n_inliers", 0)),
                     "inlier_ratio": stats.get("inlier_ratio", 0.0)})

        if E is None or stats.get("n_inliers", 0) < self.cfg.min_inliers:
            return False

        R, t_unit, cmask, n_pos = epipolar.recover_pose(E, p_prev, p_cur, self.K)
        if n_pos < self.cfg.min_inliers:
            return False

        # T_cur_prev: 把点从 prev 相机系转到 cur 相机系
        T_cu_pr = np.eye(4)
        T_cu_pr[:3, :3] = R
        T_cu_pr[:3, 3] = t_unit

        # 三角化（在 prev 相机系下）
        T_pr_w = prev.T_cw
        T_cu_w = T_cu_pr @ T_pr_w
        X_prev, valid, _ = triangulation.triangulate_points(
            self.K, T_pr_w, T_cu_w, p_prev[mask], p_cur[mask])

        if valid.sum() < self.cfg.min_inliers:
            return False

        X_prev = X_prev[valid]

        # ⭐ 视差角检查（**与尺度无关**，是初始化能否成功的关键判据）
        # 实测：TUM fr1/desk 相邻帧视差中位数仅 0.41° → 深度中位数被放大到 100m，完全不可用。
        # 标准做法（同 ORB-SLAM）：固定参考帧，等运动累积到视差足够再初始化。
        sample = X_prev[: min(60, len(X_prev))]
        angs = np.array([triangulation.parallax_angle(
            self.K, T_pr_w, T_cu_w, x.reshape(1, 3)) for x in sample])
        med_ang = float(np.median(angs))
        info["init_parallax_deg"] = med_ang
        if med_ang < self.cfg.init_min_parallax_deg:
            if cur.idx - self.init_ref_idx > self.cfg.init_max_wait_frames:
                info["status"] = "init_failed_timeout"
            else:
                info["status"] = "wait_parallax"     # 继续累积，参考帧不变
            return False

        # 定尺度：让中值深度 = init_median_depth
        d = X_prev[:, 2]
        s = self.cfg.init_median_depth / (np.median(d) + 1e-9)

        # 若有真实深度图，用真实尺度（RGB-D 模式，用于对比实验）
        if depth is not None:
            s_gt = self._scale_from_depth(X_prev, T_pr_w, depth,
                                          p_prev[mask][valid])
            if s_gt is not None:
                s = s_gt
                info["scale_source"] = "gt_depth"
            else:
                info["scale_source"] = "median_depth"
        else:
            info["scale_source"] = "median_depth"

        # ⚠️ 深度合理性检查必须在**尺度归一化之后**做。
        # （曾经踩坑：用单位平移下的深度（可达 100m）去比 20m 阈值，几乎全部被误杀。）
        X_scaled = X_prev * s
        dz = X_scaled[:, 2]
        good = (dz > 0.1) & (dz < self.cfg.max_depth)
        if good.sum() < self.cfg.min_inliers:
            info["status"] = "init_few_valid_depth"
            return False

        X_prev = X_prev[good]

        # 应用尺度：平移 × s
        T_cu_pr[:3, 3] = t_unit * s
        T_cu_w = T_cu_pr @ T_pr_w
        cur.T_cw = T_cu_w
        self.scale = s

        # 地图点转到世界坐标系
        T_w_pr = np.linalg.inv(T_pr_w)
        X_w = (T_w_pr[:3, :3] @ (X_prev * s).T).T + T_w_pr[:3, 3]

        # 关键：点要关联到**当前帧**的特征索引，下一帧才能用 (frame_idx, kp_idx) 查到 3D 点
        self._add_points(X_w, cur.idx, matches[mask][valid][good])
        self.initialized = True
        self.init_frame_idx = cur.idx
        info["status"] = "initialized"
        info["init_inliers"] = int(good.sum())
        info["init_scale"] = float(s)
        return True

    # ------------------------------------------------------------------ 护栏
    def _recent_step_median(self):
        """最近若干帧的位移中位数 —— 运动连续性的参考基准"""
        ts = []
        frames = self.frames[-(self.cfg.step_history + 1):]
        for i in range(1, len(frames)):
            a, b = frames[i - 1], frames[i]
            if a.T_cw is not None and b.T_cw is not None:
                ts.append(np.linalg.norm(b.T_cw[:3, 3] - a.T_cw[:3, 3]))
        return float(np.median(ts)) if ts else 0.05

    def _is_plausible_step(self, T_new, T_prev):
        """运动连续性检查：新位姿相对上一帧的位移是否合理

        🔴 这是 VO 最重要的护栏之一。没有它，一次异常跳变会被逐帧放大，
        最终轨迹指数爆炸（实测 RPE 达到 49940 m —— 就是这么来的）。
        """
        if T_prev is None or not np.all(np.isfinite(T_new)):
            return False
        delta = np.linalg.norm(T_new[:3, 3] - T_prev[:3, 3])
        med = self._recent_step_median()
        limit = min(self.cfg.max_step_m,
                    max(med * self.cfg.max_step_ratio, self.cfg.max_step_m * 0.5))
        return delta <= limit

    def _scale_from_depth(self, X_prev, T_pr_w, depth, px):
        """用真实深度图求尺度因子（RGB-D 模式）"""
        if depth is None:
            return None
        pts = np.asarray(px)
        H, W = depth.shape
        zs = []
        for (u, v) in pts:
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < W and 0 <= vi < H:
                z = depth[vi, ui]
                if 0.1 < z < 20:
                    zs.append(z)
        if len(zs) < 8:
            return None
        # X_prev 的 z 就是 prev 相机系下的深度
        ratio = np.array(zs) / (X_prev[:len(zs), 2] + 1e-9)
        return float(np.median(ratio))

    # ------------------------------------------------------------------ 跟踪
    def _track(self, prev, cur, p_prev, p_cur, matches, info, depth=None):
        """用地图点 + PnP 跟踪（保持尺度）"""
        # 1) 找出上一帧中"已有 3D 地图点"的匹配
        pt_ids, kp_prev_idx, kp_cur_idx = [], [], []
        for m_i, (i_prev, i_cur) in enumerate(matches):
            pid = self.obs.get((prev.idx, int(i_prev)))
            if pid is not None:
                pt_ids.append(pid)
                kp_prev_idx.append(m_i)
                kp_cur_idx.append(m_i)

        info["n_pnp"] = len(pt_ids)

        # 2) 先用 RANSAC 剔除外点（在 2D-2D 层面），提升 PnP 鲁棒性
        E, emask, stats = ransac_essential(
            p_prev, p_cur, self.K,
            thresh_px=self.cfg.ransac_thresh_px,
            max_iters=self.cfg.ransac_max_iters, seed=cur.idx,
            return_stats=True)
        info["n_inliers"] = int(stats.get("n_inliers", 0))
        info["inlier_ratio"] = stats.get("inlier_ratio", 0.0)

        if E is not None and emask.sum() >= self.cfg.min_inliers:
            p_prev, p_cur, matches = p_prev[emask], p_cur[emask], matches[emask]
            # 重新筛选有地图点的匹配
            pt_ids, kp_cur_idx = [], []
            for m_i, (i_prev, i_cur) in enumerate(matches):
                pid = self.obs.get((prev.idx, int(i_prev)))
                if pid is not None:
                    pt_ids.append(pid)
                    kp_cur_idx.append(m_i)

        if len(pt_ids) >= self.cfg.min_pnp_points:
            X = self.points_3d[np.array(pt_ids)]
            uv = p_cur[np.array(kp_cur_idx)]
            try:
                T_cu_w = pnp.pnp_with_refine(X, uv, self.K, iters=20)
                if not self._is_plausible_step(T_cu_w, prev.T_cw):
                    # PnP 给了异常跳变 → 拒绝，交给更保守的回退路径
                    info["status"] = "rejected_jump"
                    return self._fallback_essential(prev, cur, p_prev, p_cur, info)
                cur.T_cw = T_cu_w
                info["status"] = "tracked_pnp"
                info["pnp_points"] = len(pt_ids)
                used = set(np.array(kp_cur_idx))
            except Exception:
                return self._fallback_essential(prev, cur, p_prev, p_cur, info)
        else:
            # 地图点不足，退回两帧几何（会引入尺度误差）
            if not self._fallback_essential(prev, cur, p_prev, p_cur, info):
                return False
            used = set()

        # 3) 三角化新点补充地图（只三角化尚未有地图点的匹配）
        self._triangulate_new(prev, cur, p_prev, p_cur, matches, used)
        return True

    def _fallback_essential(self, prev, cur, p_prev, p_cur, info):
        """退化路径：无地图点时用 E + 上一帧尺度估计位姿

        ⚠️ 这条路径会累积尺度误差 —— 地图点不足时 VO 会明显漂移。
        """
        E, mask, stats = ransac_essential(
            p_prev, p_cur, self.K, thresh_px=self.cfg.ransac_thresh_px,
            max_iters=self.cfg.ransac_max_iters, seed=cur.idx,
            return_stats=True)
        if E is None or stats.get("n_inliers", 0) < self.cfg.min_inliers:
            return False
        R, t_unit, cmask, n_pos = epipolar.recover_pose(E, p_prev, p_cur, self.K)
        if n_pos < self.cfg.min_inliers:
            return False

        # 用**最近若干帧位移的中位数**近似当前尺度（比只用上一帧稳健得多），
        # 并强制 clip 到 [1e-3, max_step_m]，避免异常值被逐帧放大。
        step = self._recent_step_median()
        s = float(np.clip(step, 1e-3, self.cfg.max_step_m))

        T_cu_pr = np.eye(4)
        T_cu_pr[:3, :3] = R
        T_cu_pr[:3, 3] = t_unit * s
        T_new = T_cu_pr @ prev.T_cw
        if not self._is_plausible_step(T_new, prev.T_cw):
            info["status"] = "lost_implausible"
            return False
        cur.T_cw = T_new
        info["status"] = "tracked_essential"
        info["pnp_points"] = 0
        return True

    # ------------------------------------------------------------------ 建图
    def _triangulate_new(self, prev, cur, p_prev, p_cur, matches, used_idx):
        """三角化尚未有地图点的匹配点"""
        if cur.T_cw is None:
            return
        sel = [i for i in range(len(matches)) if i not in used_idx]
        if len(sel) < 8:
            return
        sel = np.array(sel)

        X_prev, valid, _ = triangulation.triangulate_points(
            self.K, prev.T_cw, cur.T_cw, p_prev[sel], p_cur[sel])
        if valid.sum() == 0:
            return

        X_prev = X_prev[valid]
        sel = sel[valid]
        d = X_prev[:, 2]
        good = (d > 0.2) & (d < self.cfg.max_depth)
        if good.sum() == 0:
            return
        X_prev, sel = X_prev[good], sel[good]

        # 视差角筛选：太小的视差 → 深度不可信
        angles = []
        for X in X_prev:
            angles.append(triangulation.parallax_angle(
                self.K, prev.T_cw, cur.T_cw, X.reshape(1, 3)))
        angles = np.array(angles)
        keep = angles > self.cfg.min_parallax_deg
        if keep.sum() == 0:
            return
        X_prev, sel = X_prev[keep], sel[keep]

        T_w_pr = np.linalg.inv(prev.T_cw)
        X_w = (T_w_pr[:3, :3] @ X_prev.T).T + T_w_pr[:3, 3]
        self._add_points(X_w, cur.idx, matches[sel])

    def _add_points(self, X_w, frame_idx, match_pairs):
        """把三角化出的点加入地图，并记录观测

        match_pairs: (M,2) 每行 [kp_idx_in_prev, kp_idx_in_cur]
        这里把点关联到 **当前帧（frame_idx）** 的特征索引上，
        这样下一帧做 PnP 时能用 (frame_idx, kp_idx) 查到 3D 点。
        """
        n = len(X_w)
        start = len(self.points_3d)
        self.points_3d = np.vstack([self.points_3d, X_w]) if len(self.points_3d) \
            else X_w.copy()
        # match_pairs 的第 2 列是 frame_idx 帧中的特征索引
        cur_kp_idx = np.asarray(match_pairs)[:n, 1].astype(int)
        for j, kp_i in enumerate(cur_kp_idx):
            self.obs[(frame_idx, int(kp_i))] = start + j

    # ------------------------------------------------------------------ 输出
    def trajectory(self):
        """相机位置轨迹 (N,3)（世界坐标系下的光心位置）"""
        pts = []
        for f in self.frames:
            if f.T_cw is None:
                continue
            T_wc = np.linalg.inv(f.T_cw)
            pts.append(T_wc[:3, 3])
        return np.array(pts)

    def poses_cw(self):
        return np.array([f.T_cw for f in self.frames if f.T_cw is not None])

    def diagnostics(self):
        """把每帧诊断汇总成 dict of arrays（M3 失效归因的数据源）"""
        keys = ["idx", "n_features", "n_matches", "n_inliers", "inlier_ratio",
                "n_pnp", "n_map", "status"]
        out = {k: [] for k in keys}
        for h in self.history:
            for k in keys:
                out[k].append(h.get(k, None))
        for k in keys:
            if k != "status":
                out[k] = np.array([0 if v is None else v for v in out[k]], dtype=float)
        return out

    # ------------------------------------------------------------------ BA 精修
    def refine(self, max_iter=80, max_obs=8000, max_points=5000, verbose=False):
        """用局部 BA 联合优化所有位姿与（部分）地图点，消除累积漂移。

        构建观测 obs = [(frame_idx, point_id, uv)]，调用 sfm.ba.local_ba。
        BA 固定第 0 帧（6 DOF）+ 尺度锚（1 DOF），详见 ba.py 顶部 docstring。

        ⚡ **规模控制（教学关键点）**：全局 BA 在万级地图点上是"又慢又吃内存"的——
        这正是 ORB-SLAM 之类必须用**局部 / 增量 BA（只在关键帧窗口内做）**的根本原因。
        本项目为演示可复现，用**确定性子采样**把问题规模钉在预算内：
          · 地图点超过 max_points → 均匀采样保留子集（其余点不参与优化但保留在地图里）
          · 观测超过 max_obs → 子采样，仅减少参与优化的约束条数
        所有位姿仍全部参与优化，优化目标（最小化重投影误差）不变，只是用子集近似。
        返回 info dict（含 cost_before/after、rmse、n_obs_used、n_points_used 等）；
        未初始化或观测过少则返回 skip。
        """
        if not self.initialized or len(self.points_3d) == 0:
            return {"status": "skipped_not_ready"}
        # 由 self.obs 反查每条观测的像素坐标（存在对应帧的 keypoints 里）
        obs = []
        for (frame_idx, kp_idx), pid in self.obs.items():
            fi = int(frame_idx)
            kpi = int(kp_idx)
            if 0 <= fi < len(self.frames):
                f = self.frames[fi]
                if f.keypoints is not None and 0 <= kpi < len(f.keypoints):
                    obs.append((fi, int(pid), np.asarray(f.keypoints[kpi], dtype=float)))
        if len(obs) < 20:
            return {"status": "skipped_few_obs", "n_obs": len(obs)}

        # 1) 地图点过多 → 均匀采样保留一个子集（其余点不动，仅不参与本轮 BA）
        points = self.points_3d.copy()
        if len(points) > max_points:
            rng = np.random.default_rng(1)
            keep = np.sort(rng.choice(len(points), size=max_points, replace=False))
            points = points[keep]
            remap = {int(old): int(new) for new, old in enumerate(keep)}
            obs = [(i, remap[j], uv) for (i, j, uv) in obs if j in remap]
            info_points_used = int(len(points))
        else:
            keep = np.arange(len(points))
            info_points_used = int(len(points))

        # 2) 观测过多 → 子采样，仅减少参与优化的约束条数
        if len(obs) > max_obs:
            rng = np.random.default_rng(0)
            sel = rng.choice(len(obs), size=max_obs, replace=False)
            obs = [obs[i] for i in sel]

        poses = self.poses_cw()                 # (N,4,4) T_cw，顺序与帧一一对应
        poses_opt, points_opt, info = local_ba(
            self.K, poses, points, obs,
            n_fixed_poses=1, max_iter=max_iter, verbose=verbose)

        # 写回：位姿全部更新；地图点只更新被优化过的子集，其余原样保留
        for i in range(min(len(poses_opt), len(self.frames))):
            self.frames[i].T_cw = poses_opt[i]
        new_points = self.points_3d.copy()
        new_points[keep] = points_opt
        self.points_3d = new_points
        info["n_obs_used"] = len(obs)
        info["n_points_used"] = info_points_used
        info["n_points_total"] = len(self.points_3d)
        return info

    # ------------------------------------------------------------------ 回环 + PGO
    def close_loops(self, kf_stride=5, min_gap=3, min_matches=60,
                    min_inlier_ratio=0.5, window=3, min_points=25,
                    max_drift_m=3.0, loop_weight=1.0, max_iter=50, verbose=False):
        """回环检测 + 位姿图优化（PGO），把累积漂移分摊回整条轨迹。

        流程：
          1. `detect_loops`    ：关键帧两两匹配 + E 矩阵几何验证，找出"回到旧地"的帧对
          2. `build_loop_priors`：用旧帧可见的地图点 + 新帧 2D 做 PnP，解出新帧的**绝对**位姿
                                 （这条约束不累积 j→i 之间的逐帧误差）
          3. 里程计边：相邻帧由 VO 给出的相对位姿 M = T_{i+1}·T_i^{-1}
          4. `pose_graph_optimize`：联合优化所有位姿，让两类边都尽量满足
          5. 写回位姿（第 0 帧固定，保证世界系不变）

        返回 info dict（含 cost/rmse 前后、检测到的回环数等）。
        """
        if not self.initialized or len(self.frames) < 4:
            return {"status": "skipped_not_ready"}

        cands = detect_loops(self, kf_stride=kf_stride, min_gap=min_gap,
                             min_matches=min_matches,
                             min_inlier_ratio=min_inlier_ratio,
                             verbose=verbose)
        if not cands:
            return {"status": "no_loop_detected", "n_candidates": 0}

        priors = build_loop_priors(self, cands, window=window,
                                   min_points=min_points,
                                   max_drift_m=max_drift_m, verbose=verbose)
        if not priors:
            return {"status": "no_loop_pose", "n_candidates": len(cands),
                    "n_priors": 0}

        poses0 = self.poses_cw()                       # (N,4,4)
        if len(poses0) < 2:
            return {"status": "skipped_few_poses"}

        # 里程计边：相邻帧相对位姿（来自 VO 链式结果，含累积误差）
        odo = []
        for i in range(len(poses0) - 1):
            M = poses0[i + 1] @ np.linalg.inv(poses0[i])
            odo.append((i, i + 1, M))

        poses_opt, info = pose_graph_optimize(
            poses0, odo, priors, fix_first=True,
            loop_weight=loop_weight, max_iter=max_iter, verbose=verbose)

        # 写回（第 0 帧固定，后续帧用优化后的位姿）
        for i in range(min(len(poses_opt), len(self.frames))):
            self.frames[i].T_cw = poses_opt[i]

        info["n_candidates"] = len(cands)
        info["n_priors"] = len(priors)
        info["loop_pairs"] = [[c["i"], c["j"]] for c in cands]
        return info
