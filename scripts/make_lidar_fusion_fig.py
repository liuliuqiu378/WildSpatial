"""生成「视觉 + 深度/LiDAR 融合几何」示意图（正面回应教程里视觉+激光雷达融合缺口）。

用 TUM fr1/desk 的**真实深度**反投影成 3D 点云、用 RGB 上色，演示：
  - 相机图像提供「有什么 / 长什么样」（语义、纹理、颜色）
  - 深度 / 激光雷达提供「在什么位置 / 离多远」（公制 3D 几何）
  - 二者几何对齐（同一相机内参 K）后 → 带颜色的点云 = 视觉语义 × 几何距离 的融合

注：TUM 是 RGB-D（主动深度），与 LiDAR 在几何本质上等价（都给公制 3D），
区别在 LiDAR 更稀疏、可 360°、不受光照影响。本图用 RGB-D 深度作为
「点云/LiDAR 几何源」的代理来演示融合几何，已在图注标注。

用法：
    PYTHONPATH=src python scripts/make_lidar_fusion_fig.py
产出：experiments/projects/figs/vision_lidar_fusion.png
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from wildspatial.projects.loaders import get_sequence


def main():
    out_dir = os.path.join(ROOT, "experiments", "projects", "figs")
    os.makedirs(out_dir, exist_ok=True)

    seq = get_sequence("tum", seq="fr1/desk", max_frames=1, stride=1)
    import cv2
    rgb = cv2.cvtColor(seq.rgbs[0], cv2.COLOR_BGR2RGB)
    depth = seq.depths[0]
    H, W = depth.shape
    fx, fy = seq.K[0, 0], seq.K[1, 1]
    cx, cy = seq.K[0, 2], seq.K[1, 2]

    # 反投影：像素 (u,v,depth) -> 相机系 3D，再按颜色采样得到彩色点云
    vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    z = depth
    valid = (z > 0.1) & (z < 5.0)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    pts = np.stack([x[valid], y[valid], z[valid]], axis=1)
    cols = rgb[valid] / 255.0
    # 下采样以加快 3D 散点渲染
    if len(pts) > 20000:
        idx = np.random.RandomState(0).choice(len(pts), 20000, replace=False)
        pts, cols = pts[idx], cols[idx]

    fig = plt.figure(figsize=(13, 4.2))
    fig.suptitle("Camera + Depth/LiDAR Fusion: appearance x metric geometry", fontsize=12, y=1.02)

    # (a) RGB
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(rgb); ax1.set_title("Camera: 'what / how it looks'\n(semantics, texture, color)", fontsize=9)
    ax1.axis("off")

    # (b) Depth colormap
    m = np.ma.masked_where(~valid, depth)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(m, cmap="viridis", vmin=0, vmax=5)
    ax2.set_title("Depth / LiDAR: 'where / how far'\n(metric 3D geometry)", fontsize=9)
    ax2.axis("off")

    # (c) Fused colored point cloud (camera frame, X right, Y down, Z forward)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c=cols, s=0.6, marker=".")
    ax3.set_xlabel("X (right, m)"); ax3.set_ylabel("Z (forward, m)"); ax3.set_zlabel("Y (up, m)")
    ax3.set_title("Fused: colored point cloud\n(appearance x geometry)", fontsize=9)
    ax3.view_init(elev=20, azim=-60)

    fig.tight_layout()
    out = os.path.join(out_dir, "vision_lidar_fusion.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] 视觉+深度/LiDAR 融合图 → {out}")


if __name__ == "__main__":
    main()
