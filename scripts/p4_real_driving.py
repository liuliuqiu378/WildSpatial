"""p4_real_driving.py — 真实驾驶动态建图 demo（真实 RGB + 真实 LiDAR）
==========================================================================

把「真实多传感器数据」当实时输入流，构建机器人视角的动态场景，并用
**俯视全景（BEV）+ 第一视角（egocentric）双画面**同步回放「车开完全程」。

数据（已在磁盘，无需下载）：KITTI depth_completion · val_selection_cropped
  - image/        : 真实车载相机 RGB（image_02，左目）
  - velodyne_raw/ : 真实稀疏激光雷达深度（16-bit PNG，depth = png/256 m，已投影到 image_02 坐标系）
  - intrinsics/   : 相机内参 P2（3×3）

流程（真实传感器 → 场景构建 → 双视角回放）：
  1. 取同一 drive 的连续片段，作为"实时"输入流（真实驾驶时序，带真实噪声）。
  2. 项目方法动物园 **VGGT**：用真实 RGB 做视觉定位，一次前向出相机轨迹 T_cw（up to scale）。
  3. 真实 LiDAR 深度给 VGGT 轨迹定标（median 比例）→ 度量级轨迹。
 4. 每帧真实 LiDAR 点云按 VGGT 位姿反投影到世界 → 增量累积成俯视（BEV）高度/占据图。
 5. **移动目标检测**：把上一帧相机系 LiDAR 点用相邻帧相对位姿反投影到当前相机系，在图像
    平面重建"静态预期深度"；当前帧中无法被该预期解释的离地点 → 移动车辆/行人，双视角标红。
 6. 双视角视频：左 = 第一视角 RGB + 真实 LiDAR 扫描（移动目标标红）；右 = 俯视全景建图 +
    轨迹 + 车体朝向 + 移动目标（红点）。

诚实边界：
  - 轨迹来自视觉模型（非 GPS/IMU 真值）；尺度由真实 LiDAR 锚定。
  - 移动目标检测为启发式（自运动补偿帧间差分），可能误标"新揭示的静态障碍"，已用离地约束抑制路面误检；属于演示级而非检测器 benchmark。
  - 该基准无逐帧真值位姿，故不报 ATE；只报轨迹长度、尺度因子、点云量等可量化指标。
  - velodyne_raw 为真实稀疏激光（非仿真），正是真实雷达形态。

运行（wildspatial 环境，需 GPU 跑 VGGT）：
  PYTHONPATH=src python scripts/p4_real_driving.py --drive 2011_09_26_drive_0023_sync --max-frames 30
"""
import os
import re
import json
import argparse

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(
    ROOT, "data/raw/ms/OmniData__KITTI_depth_completion/raw/KITTI_depth_completion",
    "depth_selection", "val_selection_cropped")
IMAGE_DIR = os.path.join(BASE, "image")
VELO_DIR = os.path.join(BASE, "velodyne_raw")
INTR_DIR = os.path.join(BASE, "intrinsics")
OUT = os.path.join(ROOT, "experiments", "P4_real_driving")
FIGS = os.path.join(OUT, "figs")
os.makedirs(FIGS, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. 数据加载（真实传感器，归一成帧列表）
# --------------------------------------------------------------------------- #
def decode_kitti_depth(p):
    """KITTI 深度图为 16-bit PNG：depth_m = png / 256.0，0 为无效。"""
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    d = img.astype(np.float32) / 256.0
    d[d <= 0] = np.nan
    return d


def load_intr(txt):
    nums = [float(x) for x in open(txt).read().split()]
    return np.array(nums[:9]).reshape(3, 3)


def load_drive(drive, max_frames, stride=1):
    """取同一 drive 的连续真实驾驶片段。"""
    pat = re.compile(re.escape(drive) + r"_image_(\d+)_image_02\.png$")
    files = []
    for nm in sorted(os.listdir(IMAGE_DIR)):
        m = pat.match(nm)
        if m:
            files.append((int(m.group(1)), nm))
    files.sort()
    if stride > 1:
        files = [files[i] for i in range(0, len(files), stride)]
    if max_frames:
        files = files[:max_frames]

    frames = []
    for fid, nm in files:
        ip = os.path.join(IMAGE_DIR, nm)
        vp = os.path.join(VELO_DIR, f"{drive}_velodyne_raw_{fid:010d}_image_02.png")
        itp = os.path.join(INTR_DIR, f"{drive}_image_{fid:010d}_image_02.txt")
        if not (os.path.exists(vp) and os.path.exists(itp)):
            continue
        rgb = cv2.imread(ip)
        if rgb is None:
            continue
        ld = decode_kitti_depth(vp)
        if ld is None:
            continue
        frames.append({"rgb": rgb, "lidar": ld, "frame": fid,
                       "K": load_intr(itp), "name": nm})
    return frames


# --------------------------------------------------------------------------- #
# 2. 视觉定位（项目方法动物园 VGGT，真实 RGB → 相机轨迹 T_cw）
# --------------------------------------------------------------------------- #
def run_vggt(frames):
    from wildspatial.methods import get_method
    m = get_method("vggt")
    if m is None or not m.available():
        print("[!] VGGT 不可用（需 GPU + vggt 包），仅用真实 LiDAR 建图、轨迹缺失。")
        return None
    rgbs = [f["rgb"] for f in frames]
    K = frames[0]["K"]
    print(f"[*] 运行 VGGT 视觉定位（{len(rgbs)} 帧真实 RGB）...")
    res = m.run(rgbs, K, return_dense=True)
    if not res.ok:
        print(f"[!] VGGT 失败：{res.note}")
        return None
    return res


# --------------------------------------------------------------------------- #
# 3. 尺度定标（真实 LiDAR 深度 anchor VGGT up-to-scale 轨迹）
# --------------------------------------------------------------------------- #
def compute_scale(res, frames):
    vd = res.extra.get("depth") if res else None
    if vd is None:
        return 1.0
    Hs, Ws = vd.shape[1], vd.shape[2]
    scales = []
    for i, f in enumerate(frames):
        ld = f["lidar"]
        ld_r = cv2.resize(ld, (Ws, Hs), interpolation=cv2.INTER_NEAREST)
        a, b = vd[i], ld_r
        m = np.isfinite(a) & (a > 0) & np.isfinite(b) & (b > 0)
        if m.sum() > 50:
            scales.append(np.median(b[m]) / np.median(a[m]))
    return float(np.median(scales)) if scales else 1.0


# --------------------------------------------------------------------------- #
# 4. 真实 LiDAR 反投影 → 世界点云（按 VGGT 位姿累积）
# --------------------------------------------------------------------------- #
def build_world_points(frames, res, scale):
    if res is None:
        return [np.zeros((0, 3), float) for _ in frames]
    Tcw = res.extra["T_cw"]
    per = []
    for i, f in enumerate(frames):
        ld, K = f["lidar"], f["K"]
        H, W = ld.shape
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        Z = ld
        X = (uu - cx) * Z / fx
        Y = (vv - cy) * Z / fy
        pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        valid = np.isfinite(Z.reshape(-1))
        pts = pts[valid]
        M = Tcw[i].copy()
        M[:3, 3] = M[:3, 3] * scale          # 平移按真实 LiDAR 定标到度量级
        pw = (M[:3, :3] @ pts.T + M[:3, 3:4]).T
        per.append(pw)
    return per


# --------------------------------------------------------------------------- #
# 4b. 相机系 LiDAR 点（用于帧间差分，保留图像像素坐标）
# --------------------------------------------------------------------------- #
def build_cam_points(frames):
    """每帧 LiDAR 在相机系 (X,Y,Z) 及对应 image_02 像素 (u,v)。"""
    cam = []
    for f in frames:
        ld, K = f["lidar"], f["K"]
        H, W = ld.shape
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        Z = ld.reshape(-1)
        X = (uu.reshape(-1) - cx) * Z / fx
        Y = (vv.reshape(-1) - cy) * Z / fy
        mask = np.isfinite(Z)
        pts = np.stack([X[mask], Y[mask], Z[mask],
                        uu.reshape(-1)[mask].astype(float),
                        vv.reshape(-1)[mask].astype(float)], axis=-1)
        cam.append(pts)
    return cam


# --------------------------------------------------------------------------- #
# 4c. 移动目标检测：自运动补偿 + 累积静态地图差分
#     把前若干帧的真实 LiDAR 世界点（同一套 VGGT 位姿，故与当前帧自洽、抗漂移）
#     反投影到当前相机系，在图像平面重建“稠密静态预期深度”；当前帧中无法被该
#     静态预期解释、且离地点 → 移动车辆/行人，双视角标红。
#     返回 dyn_world[t] (N,3 世界系, 已定标) 与 dyn_uv[t] (N,2 image_02 像素)。
# --------------------------------------------------------------------------- #
def detect_dynamic(frames, cam, per, res, scale):
    empty3 = lambda: np.zeros((0, 3), float)
    empty2 = lambda: np.zeros((0, 2), float)
    if res is None:
        return [empty3() for _ in frames], [empty2() for _ in frames]
    Tcw = res.extra["T_cw"]
    cam_h = 1.65                           # KITTI image_02 相机离地高度（米）
    r = 1.2                                # 世界系邻域基础半径（米），容忍 VGGT 漂移
    dyn_world, dyn_uv = [], []
    accum = np.zeros((0, 3), float)        # 累积静态地图（世界系，已定标）
    from scipy.spatial import cKDTree
    for t in range(len(frames)):
        cur_w = per[t]                     # 当前帧世界点（已定标）
        cur_c = cam[t]                     # 当前帧相机系点（含像素 u,v 与 Y）
        if len(accum) == 0 or len(cur_w) == 0:   # 首帧无前序地图可比
            dyn_world.append(empty3()); dyn_uv.append(empty2())
            if len(cur_w):
                accum = cur_w.copy()
            continue
        # 3D 邻近匹配：当前世界点若能在累积静态地图中找到近邻 → 静态（被解释）
        # 匹配半径随深度增大（位姿角误差在远处放大为位置误差）
        d2, _ = cKDTree(accum).query(cur_w, k=1)
        rad = r + 0.03 * cur_c[:, 2]
        explained = d2 <= rad
        # 离地判定：相机系 Y(下正) 明显小于相机高度 → 位于地面之上（障碍/车辆/行人）
        offground = cur_c[:, 1] < (cam_h - 0.5)
        dmask = (~explained) & offground
        duv = cur_c[dmask, 3:5].astype(float)
        dyn_world.append(cur_w[dmask]); dyn_uv.append(duv)
        accum = np.vstack([accum, cur_w])
    return dyn_world, dyn_uv


# --------------------------------------------------------------------------- #
# 5. 双视角回放（第一视角 RGB+LiDAR  |  俯视 BEV 建图+轨迹+车体）
# --------------------------------------------------------------------------- #
def _height_color(y, ymin, ymax):
    t = np.clip((y - ymin) / (ymax - ymin + 1e-6), 0, 1)
    # 低处=深蓝（地面），高处=暖色（车/树/建筑）
    r = int(255 * t)
    g = int(200 * (1 - abs(t - 0.5) * 2) + 30)
    b = int(255 * (1 - t))
    return (b, g, r)


def render(frames, per, res, scale, out_mp4, dyn_world=None, dyn_uv=None, fps=3.0):
    has_pose = res is not None
    if has_pose:
        positions = res.positions * scale          # (S,3) 度量级相机光心
    # 全局 BEV 范围
    allp = np.concatenate([p for p in per if len(p) > 0], axis=0)
    xmin, xmax = allp[:, 0].min(), allp[:, 0].max()
    zmin, zmax = allp[:, 2].min(), allp[:, 2].max()
    ymin, ymax = allp[:, 1].min(), allp[:, 1].max()
    res_ = 0.4                                     # BEV 栅格分辨率（米）
    PX = 5                                         # 每栅格像素
    gw = int((xmax - xmin) / res_) + 1
    gh = int((zmax - zmin) / res_) + 1

    # 持久 BEV 画布（按高度着色），逐帧增量填充
    bev = np.zeros((gh, gw, 3), np.uint8)

    def world_to_grid(x, z):
        return int((x - xmin) / res_), int((z - zmin) / res_)

    def paint_frame(pw):
        for (x, y, z) in pw:
            gx, gz = world_to_grid(x, z)
            if 0 <= gx < gw and 0 <= gz < gh:
                bev[gz, gx] = _height_color(y, ymin, ymax)

    Hl, Wl = frames[0]["rgb"].shape[:2]
    LH = BH = 400                       # 两个面板的显示高度（像素）
    BW = 560                            # BEV 面板显示宽度
    left_disp = (int(Wl * LH / Hl) // 2 * 2, LH)

    # 车体朝向（相机前向 = OpenCV +Z）
    def heading(i):
        if not has_pose:
            return np.array([0.0, 0.0, 1.0])
        R = res.extra["T_cw"][i][:3, :3]
        return R @ np.array([0.0, 0.0, 1.0])

    def to_disp(x, z):
        """世界 (x, z) → BEV 显示像素（forward 朝上）"""
        px = int((x - xmin) / (xmax - xmin + 1e-9) * (BW - 1))
        py = int(BH - 1 - (z - zmin) / (zmax - zmin + 1e-9) * (BH - 1))
        return px, py

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_w = (left_disp[0] + 6 + BW) // 2 * 2
    out_h = LH // 2 * 2
    vw = cv2.VideoWriter(out_mp4, fourcc, fps, (out_w, out_h))
    if not vw.isOpened():
        print("[!] 视频写入器打不开，仅输出关键帧 PNG。")
        vw = None

    # 关键帧（示例帧）：保存间隔更小 → 展示更多例子（约每 3 帧一张）
    keyframes = max(1, len(frames) // 10)
    saved = []

    for t, f in enumerate(frames):
        paint_frame(per[t])

        # ---- 右：BEV 显示（forward 朝上）----
        bev_disp = cv2.resize(bev, (BW, BH), interpolation=cv2.INTER_NEAREST)
        bev_disp = cv2.flip(bev_disp, 0)
        if has_pose:
            # 轨迹（已走过部分，画在最终分辨率上保证可见）
            traj_px = [to_disp(positions[k, 0], positions[k, 2])
                       for k in range(t + 1)]
            for k in range(1, len(traj_px)):
                cv2.line(bev_disp, traj_px[k - 1], traj_px[k],
                         (0, 255, 120), 3, cv2.LINE_AA)
            # 车体三角（沿heading方向）
            cx_px, cz_px = traj_px[-1]
            h = heading(t)
            ang = np.arctan2(-h[2], h[0])       # 显示系 y 向下 → z 取负
            tri = np.array([
                [cx_px + 14 * np.cos(ang), cz_px + 14 * np.sin(ang)],
                [cx_px + 9 * np.cos(ang + 2.6), cz_px + 9 * np.sin(ang + 2.6)],
                [cx_px + 9 * np.cos(ang - 2.6), cz_px + 9 * np.sin(ang - 2.6)],
            ], np.int32)
            cv2.fillPoly(bev_disp, [tri], (0, 0, 255))
            cv2.circle(bev_disp, (cx_px, cz_px), 4, (255, 255, 255), -1)
            # 移动目标（红色，俯视图）
            if dyn_world is not None and t < len(dyn_world) and len(dyn_world[t]):
                for (x, y, z) in dyn_world[t]:
                    dx, dz = to_disp(x, z)
                    cv2.circle(bev_disp, (int(dx), int(dz)), 3, (0, 0, 255), -1)

        # ---- 左：第一视角 RGB + 真实 LiDAR 扫描（距离着色：近=暖 远=蓝）----
        left = cv2.resize(f["rgb"], left_disp)
        ld = f["lidar"]
        H, W = ld.shape
        sx = left_disp[0] / W
        sy = left_disp[1] / H
        ys, xs = np.where(np.isfinite(ld) & (ld <= 80))
        if len(ys):
            px = np.clip((xs * sx).astype(int), 0, left_disp[0] - 2)
            py = np.clip((ys * sy).astype(int), 0, LH - 2)
            col = np.clip(255.0 * ld[ys, xs] / 60.0, 0, 255).astype(np.uint8)
            c = np.stack([col, np.zeros_like(col), 255 - col], axis=1)
            left[py, px] = c                    # 2×2 点，保证可见
            left[py, px + 1] = c
            left[py + 1, px] = c
            left[py + 1, px + 1] = c
        # 移动目标（红色，第一视角）
        ndyn = 0
        if dyn_uv is not None and t < len(dyn_uv) and len(dyn_uv[t]):
            du = np.clip((dyn_uv[t][:, 0] * sx).astype(int), 0, left_disp[0] - 2)
            dv = np.clip((dyn_uv[t][:, 1] * sy).astype(int), 0, LH - 2)
            ndyn = len(du)
            for (px, py) in zip(du, dv):
                cv2.circle(left, (px, py), 2, (0, 0, 255), -1)

        # ---- 合成（固定尺寸，无需二次缩放）----
        sep = np.full((LH, 6, 3), 200, np.uint8)
        combo = np.hstack([left, sep, bev_disp])
        cv2.putText(combo, f"frame {t+1}/{len(frames)}  real RGB + real LiDAR (KITTI)",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
        cv2.putText(combo, f"moving objects (red): {ndyn}",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2)
        cv2.putText(combo, "EGOCENTRIC (camera)", (10, LH - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)
        cv2.putText(combo, "BEV MAP (built from real LiDAR)",
                    (left_disp[0] + 16, LH - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)

        if vw is not None:
            vw.write(combo)
        if t % keyframes == 0 or t == len(frames) - 1:
            kp = os.path.join(FIGS, f"frame_{t:03d}.png")
            cv2.imwrite(kp, combo)
            saved.append(kp)
        print(f"    frame {t+1}/{len(frames)} | 真实LiDAR点={len(per[t])}", end="\r")

    if vw is not None:
        vw.release()
    print()
    print(f"[✓] 双视角视频 → {out_mp4}")
    print(f"[✓] 关键帧 {len(saved)} 张 → {FIGS}")
    return saved


def render_final_map(frames, per, res, scale, dyn_world=None):
    """高质量静态终图：完整 BEV（高度着色）+ 完整轨迹 + 起终点。"""
    allp = np.concatenate([p for p in per if len(p) > 0], axis=0)
    xmin, xmax = allp[:, 0].min(), allp[:, 0].max()
    zmin, zmax = allp[:, 2].min(), allp[:, 2].max()
    ymin, ymax = allp[:, 1].min(), allp[:, 1].max()
    res_ = 0.3
    gw = int((xmax - xmin) / res_) + 1
    gh = int((zmax - zmin) / res_) + 1
    bev = np.zeros((gh, gw, 3), np.uint8)
    for p in per:
        for (x, y, z) in p:
            gx, gz = int((x - xmin) / res_), int((z - zmin) / res_)
            if 0 <= gx < gw and 0 <= gz < gh:
                bev[gz, gx] = _height_color(y, ymin, ymax)
    bev = cv2.resize(bev, (gw * 6, gh * 6), interpolation=cv2.INTER_NEAREST)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(bev, origin="lower", extent=[xmin, xmax, zmin, zmax])
    if res is not None:
        pos = res.positions * scale
        ax.plot(pos[:, 0], pos[:, 2], "-", color="lime", lw=2,
                label="estimated trajectory (VGGT)")
        ax.plot(pos[0, 0], pos[0, 2], "go", ms=10, label="start")
        ax.plot(pos[-1, 0], pos[-1, 2], "ro", ms=10, label="end")
        if dyn_world is not None:
            dpts = np.concatenate([p for p in dyn_world if len(p) > 0], axis=0)
            if len(dpts):
                ax.scatter(dpts[:, 0], dpts[:, 2], s=10, c="red",
                           label="moving objects (LiDAR frame-diff)")
    ax.set_xlabel("X right (m)"); ax.set_ylabel("Z forward (m)")
    ax.set_title("Real-driving BEV map built from KITTI real LiDAR\n"
                 "(height-colored: blue=ground, warm=obstacles/structures)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = os.path.join(FIGS, "bev_final_map.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[✓] 终图 → {out}")
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="2011_09_26_drive_0023_sync")
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=float, default=3.0,
                    help="双视角视频帧率（默认 3.0 = 半速播放，对比原 6 fps）")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    figs = os.path.join(args.out, "figs")
    os.makedirs(figs, exist_ok=True)
    global FIGS
    FIGS = figs                       # 关键帧/终图跟随 --out，支持多场景分目录

    frames = load_drive(args.drive, args.max_frames, args.stride)
    if not frames:
        raise RuntimeError(f"未找到 drive={args.drive} 的连续片段（检查 BASE 路径）")
    print(f"[*] 载入真实驾驶片段：{len(frames)} 帧（drive={args.drive}）")

    res = run_vggt(frames)
    scale = compute_scale(res, frames) if res else 1.0
    print(f"[*] 尺度定标因子（真实LiDAR / VGGT）= {scale:.3f}")
    per = build_world_points(frames, res, scale)
    cam = build_cam_points(frames)
    dyn_world, dyn_uv = detect_dynamic(frames, cam, per, res, scale)

    n_points = [int(len(p)) for p in per]
    total_points = int(sum(n_points))
    traj_len = 0.0
    if res is not None:
        pos = res.positions * scale
        traj_len = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))

    mp4 = os.path.join(args.out, "driving_dualview.mp4")
    saved = render(frames, per, res, scale, mp4, dyn_world=dyn_world, dyn_uv=dyn_uv,
                   fps=args.fps)
    final_map = render_final_map(frames, per, res, scale, dyn_world=dyn_world)

    dyn_counts = [int(len(d)) for d in dyn_uv]
    total_dynamic = int(sum(dyn_counts))
    metrics = {
        "dataset": "KITTI depth_completion (val_selection_cropped, image_02)",
        "drive": args.drive,
        "n_frames": len(frames),
        "scale_factor_lidar_over_vggt": round(scale, 4),
        "trajectory_length_m": round(traj_len, 2),
        "total_real_lidar_points": total_points,
        "real_lidar_points_per_frame": n_points,
        "moving_object_detection": {
            "method": "LiDAR frame-diff with ego-motion compensation (VGGT relative pose)",
            "total_detected_points": total_dynamic,
            "detected_points_per_frame": dyn_counts,
        },
        "video": os.path.relpath(mp4, ROOT),
        "keyframes": [os.path.relpath(s, ROOT) for s in saved],
        "final_map": os.path.relpath(final_map, ROOT),
        "note": ("真实 RGB + 真实 LiDAR 输入；VGGT 用 RGB 做视觉定位（up-to-scale），"
                 "真实 LiDAR 深度定标到度量级；BEV 由真实 LiDAR 反投影累积。"
                 "基准无逐帧真值位姿，故不报 ATE。"),
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    print(f"[✓] metrics → {os.path.join(args.out, 'metrics.json')}")
    print(f"    轨迹长度={metrics['trajectory_length_m']} m | 真实LiDAR点={total_points} | 移动目标点={total_dynamic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
