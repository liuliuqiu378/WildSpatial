"""SUN RGB-D 深度真值 demo（接入新下载的多模态数据）

============================ 目的 ============================
项目原只有 TUM fr1/desk（单一室内序列）。新下 **SUN RGB-D**（6853 场景，含 RGB +
深度 + 内参）后，可做**「一般室内场景」的深度真值**认知：
    · 真实深度长什么样（不是合成退化）
    · 从深度反投影出的 3D 点云长什么样
    · 内参 K 怎么从 intrinsics.txt 读

============================ 方法原理 ============================
SUN RGB-D 深度 PNG 为 uint16，数值单位 = 1/8 mm（即 value * 0.125 mm = 距离）。
fullres 内参（3x3 K）存于 `fullres/intrinsics.txt`。
反投影：像素 (u,v,Z) → 相机系 (X,Y,Z)，X=(u-cx)Z/fx, Y=(v-cy)Z/fy。

============================ 用法 ============================
    conda activate wildspatial
    PYTHONPATH=src python scripts/sunrgbd_depth_demo.py            # 默认 NYU1235
    PYTHONPATH=src python scripts/sunrgbd_depth_demo.py --scene kv2/kinect2data/000002

产出 experiments/SUNRGBD_depth/figs/{rgb,depth,pointcloud}.png + metrics.json
"""
import os
import sys
import json
import zipfile
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from wildspatial.viz import setup_plot_style  # noqa: E402

setup_plot_style()

ZIP = os.path.join(ROOT, "data", "raw", "ms", "OmniData__SUN_RGB-D", "raw", "SUNRGBD.zip")
PREFIX = "SUNRGBD/"


def read_intrinsics(z, scene_path):
    """intrinsics.txt 是一行 9 个数（3x3 K）"""
    cand = [f"{PREFIX}{scene_path}/fullres/intrinsics.txt",
            f"{PREFIX}{scene_path}/intrinsics.txt",
            f"{PREFIX}{scene_path}/depth_bfx/intrinsics.txt"]
    for c in cand:
        try:
            txt = z.read(c).decode()
            vals = [float(x) for x in txt.replace("\n", " ").split()][:9]
            if len(vals) == 9:
                return np.array(vals).reshape(3, 3)
        except KeyError:
            continue
    return None


def load_scene(z, scene_path):
    import cv2
    def _img(name):
        try:
            data = z.read(f"{PREFIX}{scene_path}/{name}")
        except KeyError:
            return None
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)

    rgb = _img("fullres") or None
    # fullres 是目录，逐个尝试
    rgb = None
    for nm in z.namelist():
        if nm.startswith(f"{PREFIX}{scene_path}/fullres/") and nm.lower().endswith((".jpg", ".png")):
            rgb = cv2.imdecode(np.frombuffer(z.read(nm), np.uint8), cv2.IMREAD_UNCHANGED)
            break
    depth = None
    for sub in ("depth_bfx", "depth"):
        for nm in z.namelist():
            if nm.startswith(f"{PREFIX}{scene_path}/{sub}/") and nm.lower().endswith(".png"):
                depth = cv2.imdecode(np.frombuffer(z.read(nm), np.uint8), cv2.IMREAD_UNCHANGED)
                break
        if depth is not None:
            break
    K = read_intrinsics(z, scene_path)
    return rgb, depth, K


def list_scenes(z, limit=20):
    scenes = set()
    for nm in z.namelist():
        parts = nm.split("/")
        if len(parts) >= 4 and parts[0] == "SUNRGBD" and parts[3]:
            scenes.add("/".join(parts[1:4]))
        if len(scenes) >= limit:
            break
    return sorted(scenes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="kv1/NYUdata/NYU1235")
    ap.add_argument("--max-depth", type=float, default=8.0)
    args = ap.parse_args()

    if not os.path.exists(ZIP):
        print(f"[✗] 未找到 {ZIP}（请先 python scripts/fetch_datasets.py --tier B）")
        return 1

    out_dir = os.path.join(ROOT, "experiments", "SUNRGBD_depth")
    figs = os.path.join(out_dir, "figs")
    os.makedirs(figs, exist_ok=True)

    with zipfile.ZipFile(ZIP) as z:
        rgb, depth, K = load_scene(z, args.scene)
        if rgb is None or depth is None or K is None:
            print(f"[✗] 场景 {args.scene} 数据不全 (rgb={rgb is not None}, depth={depth is not None}, K={K is not None})")
            print("     可用场景示例:", list_scenes(z, 6))
            return 1

    # 深度单位：SUN RGB-D = 1/8 mm  → 米
    depth_m = depth.astype(np.float32) / 8.0 / 1000.0
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # SUN RGB-D 的 depth_bfx 尺寸(561x427) 与 fullres RGB(640x480) 不一致 →
    # 把深度上采样到 RGB 尺寸，使反投影点云能取到正确颜色。
    rgb_shape = None
    if rgb is not None:
        rgb_shape = rgb.shape[:2]
        if depth_m.shape != rgb_shape:
            import cv2
            depth_m = cv2.resize(depth_m, (rgb_shape[1], rgb_shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

    valid = (depth_m > 0.3) & (depth_m < args.max_depth)

    # 反投影
    H, W = depth_m.shape
    vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    Z = depth_m[valid]
    X = (uu[valid] - cx) * Z / fx
    Y = (vv[valid] - cy) * Z / fy
    P = np.stack([X, Y, Z], axis=1)
    cols = None
    if rgb is not None and rgb.shape[:2] == depth_m.shape:
        rgb_f = rgb.astype(np.float32)
        if rgb_f.max() > 1.5:                          # uint8/uint16 → [0,1]
            rgb_f = rgb_f / (255.0 if rgb_f.max() <= 255 else float(rgb_f.max()))
        rgb_f = np.clip(rgb_f, 0.0, 1.0)
        if rgb_f.ndim == 3 and rgb_f.shape[2] == 3:
            cols = rgb_f[valid][:, ::-1]               # BGR->RGB
        else:                                          # 灰度
            g = rgb_f[valid]
            cols = np.stack([g, g, g], axis=1)

    # ---- 可视化 ----
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f"SUN RGB-D 深度真值 · {args.scene}（室内，真实采集）", fontsize=13)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(rgb[:, :, ::-1] if rgb.ndim == 3 else rgb)
    ax1.axis("off"); ax1.set_title("① RGB（fullres）", fontsize=10)

    ax2 = fig.add_subplot(1, 3, 2)
    dm = np.ma.masked_where(~valid, depth_m)
    im2 = ax2.imshow(dm, cmap="viridis", vmin=0.3, vmax=args.max_depth)
    ax2.axis("off"); ax2.set_title("② 深度真值（米，viridis）", fontsize=10)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02, label="Z (m)")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    idx = np.random.RandomState(0).choice(len(P), min(20000, len(P)), replace=False)
    sc = ax3.scatter(P[idx, 0], P[idx, 2], -P[idx, 1],
                     c=(cols[idx] if cols is not None else P[idx, 2]),
                     s=0.6, marker=".")
    ax3.set_xlabel("X 右 (m)"); ax3.set_ylabel("Z 前 (m)"); ax3.set_zlabel("Y 上 (m)")
    ax3.set_title("③ 深度反投影彩色点云", fontsize=10)
    ax3.view_init(elev=20, azim=-70)

    fig.tight_layout()
    out_png = os.path.join(figs, "sunrgbd_scene.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "scene": args.scene,
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth.shape),
        "K": K.tolist(),
        "valid_pixel_ratio": round(float(valid.mean()), 4),
        "depth_min_m": round(float(depth_m[valid].min()), 3),
        "depth_max_m": round(float(depth_m[valid].max()), 3),
        "points": int(len(P)),
        "note": "深度单位 1/8 mm 已转米；SUN RGB-D 真实采集（非合成退化）",
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[✓] {args.scene} 演示完成 → {out_dir}")
    print(f"    RGB {rgb.shape} | 深度 {depth.shape} | K=({fx:.1f},{fy:.1f},{cx:.1f},{cy:.1f})")
    print(f"    有效像素 {valid.mean()*100:.1f}% | 深度 {depth_m[valid].min():.2f}~{depth_m[valid].max():.2f} m")
    print(f"    点云 {len(P)} 点")
    return 0


if __name__ == "__main__":
    sys.exit(main())
