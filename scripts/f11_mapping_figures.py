"""F11 建图实操全链路：生成示意/模拟图（纯 numpy+matplotlib，零下载）。

产出 experiments/F11_mapping/figs/：
  f11_pipeline.png       建图全链路 pipeline（传感器→位姿→融合→地图→用）
  f11_representations.png 5 种场景表示示意（点云/网格/TSDF/占据栅格/BEV）
  f11_storage.png         5 种表示体积量级对比（对数轴，经验估算）
  f11_frames.png          坐标帧示意（sensor / robot / map）

图内标签用英文（matplotlib 无 CJK 字形）；中文说明在 docs/F11_mapping_in_practice.md。
所有几何为合成示意，非真实采集数据。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.transforms as mtransforms

OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "F11_mapping", "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 130})


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    boxes = [
        (0.02, 0.55, 0.16, 0.35, "Sensors\nRGB-D / LiDAR\n+ IMU", "#cfe8ff"),
        (0.24, 0.55, 0.16, 0.35, "Pose\nVO / VGGT\n/ LIO", "#d6f5d6"),
        (0.46, 0.55, 0.16, 0.35, "Geometry\nFusion\nTSDF / cloud", "#fff0c2"),
        (0.68, 0.55, 0.16, 0.35, "Map\n5 formats\n(.ply/.pgm..)", "#ffd6d6"),
        (0.90, 0.55, 0.08, 0.35, "use", "#e6d6ff"),
    ]
    # lower row: map users
    users = [
        (0.68, 0.12, 0.16, 0.30, "Localization\n(re-localize)", "#ffd6d6"),
        (0.46, 0.12, 0.16, 0.30, "Planning\nA* + inflate", "#d6f5d6"),
        (0.24, 0.12, 0.16, 0.30, "Control\nv / omega", "#cfe8ff"),
    ]
    for x, y, w, h, t, c in list(boxes) + list(users):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=c, edgecolor="0.3", lw=1.2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9)
    # arrows top row
    for (x1, y1, w1, h1, _, _), (x2, y2, w2, h2, _, _) in zip(boxes[:-1], boxes[1:]):
        ax.add_patch(FancyArrowPatch((x1 + w1, y1 + h1 / 2), (x2, y2 + h2 / 2),
                                     arrowstyle="-|>", mutation_scale=12, color="0.3", lw=1.5))
    # map down to users
    mx, my, mw, mh, _, _ = boxes[3]
    for x, y, w, h, _, _ in users:
        ax.add_patch(FancyArrowPatch((mx + mw / 2, my), (x + w / 2, y + h),
                                     arrowstyle="-|>", mutation_scale=10, color="0.5", lw=1.2, linestyle="--"))
    # users chain
    for (x1, y1, w1, h1, _, _), (x2, y2, w2, h2, _, _) in zip(users[::-1][1:], users[::-1][:-1]):
        ax.add_patch(FancyArrowPatch((x1 + w1, y1 + h1 / 2), (x2, y2 + h2 / 2),
                                     arrowstyle="-|>", mutation_scale=10, color="0.3", lw=1.5))
    ax.text(0.5, 0.97, "Mapping pipeline: sensor -> pose -> fusion -> map -> use",
            ha="center", va="top", fontsize=11, weight="bold")
    fig.tight_layout()
    p = os.path.join(OUT, "f11_pipeline.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_representations():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.6))
    fig.suptitle("Five scene representations (schematic, synthetic)", fontsize=12, weight="bold")

    # (a) point cloud: synthetic room corner
    ax = axes[0, 0]
    n = 1200
    pts = np.concatenate([
        rng.uniform([0, 0, 0], [3, 3, 0.05], (n // 3, 3)),
        rng.uniform([0, 0, 0], [0.05, 3, 2.5], (n // 3, 3)),
        rng.uniform([0, 0, 0], [3, 0.05, 2.5], (n // 3, 3)),
    ])
    ax.scatter(pts[:, 0], pts[:, 2], s=3, c=pts[:, 1], cmap="viridis")
    ax.set_title("(a) Point cloud\n(x,y,z[,intensity])")
    ax.set_xlabel("x"); ax.set_ylabel("z")

    # (b) mesh: sin surface triangulation look (3D)
    ax = fig.add_subplot(2, 3, 2, projection="3d")
    axes[0, 1] = ax
    X, Y = np.meshgrid(np.linspace(-3, 3, 40), np.linspace(-3, 3, 40))
    Z = np.sin(X) * np.cos(Y)
    ax.plot_wireframe(X, Y, Z, rstride=3, cstride=3, color="0.35", lw=0.5)
    ax.set_title("(b) Mesh\n(vertices + faces)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    # (c) TSDF slice: distance field heatmap with a wall
    ax = axes[0, 2]
    g = np.linspace(-2, 2, 80)
    xx, yy = np.meshgrid(g, g)
    tsdf = np.clip(np.abs(xx - 0.6) - 0.15, -1, 1)  # wall at x=0.6
    im = ax.imshow(tsdf, extent=[-2, 2, -2, 2], origin="lower", cmap="coolwarm")
    ax.contour(xx, yy, tsdf, levels=[0], colors="k", lw=1.5)
    ax.set_title("(c) TSDF slice\n(signed distance field)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (d) occupancy grid
    ax = axes[1, 0]
    grid = np.zeros((40, 40))
    grid[:] = 0  # unknown
    grid[5:35, 8:32] = 1  # free
    grid[20:26, 18:22] = 2  # obstacle
    grid[10:14, 28:31] = 2
    cmap = matplotlib.colors.ListedColormap(["#bbbbbb", "#ffffff", "#444444"])
    ax.imshow(grid, cmap=cmap, origin="lower", vmin=0, vmax=2)
    # a path
    path = np.array([[10, 12], [14, 14], [18, 16], [22, 20], [26, 24], [30, 26]])
    ax.plot(path[:, 1], path[:, 0], "g-", lw=2, label="A* path")
    ax.set_title("(d) Occupancy grid\n(0 unknown /1 free /2 obst)")
    ax.set_xlabel("x cell"); ax.set_ylabel("y cell"); ax.legend(loc="lower right", fontsize=7)

    # (e) BEV
    ax = axes[1, 1]
    bg = np.zeros((40, 40))
    bg[5:35, 8:32] = 0.3
    bg[20:26, 18:22] = 1.0
    bg[10:14, 28:31] = 1.0
    ax.imshow(bg, cmap="gray", origin="lower")
    # detections
    ax.scatter([20, 29], [23, 12], c="r", s=40, marker="o", label="detect")
    ax.set_title("(e) BEV\n(top-down for detect/plan)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(loc="lower right", fontsize=7)

    # (f) 3DGS note panel
    ax = axes[1, 2]
    ax.axis("off")
    ax.text(0.5, 0.6, "3D Gaussian\nSplatting", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(0.5, 0.35, "many glowing\nellipsoids\n(neural render)", ha="center", va="center", fontsize=9)
    ax.set_title("(f) 3DGS (bonus)")

    fig.tight_layout()
    p = os.path.join(OUT, "f11_representations.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_storage():
    labels = ["BEV\n(.png)", "Occupancy\ngrid @5cm", "Point cloud\n(1M pts)", "Mesh\n(848k vert)",
              "TSDF\n@2cm room"]
    sizes_mb = [0.2, 4.0, 16.0, 20.0, 75.0]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.bar(labels, sizes_mb, color="#6fa8dc", edgecolor="0.3")
    ax.set_yscale("log")
    ax.set_ylabel("Size (MB, log scale)")
    ax.set_title("Storage per representation (illustrative estimates, NOT measured)")
    for b, s in zip(bars, sizes_mb):
        ax.text(b.get_x() + b.get_width() / 2, s * 1.1, f"{s:g} MB", ha="center", va="bottom", fontsize=8)
    ax.text(0.5, -0.28, "Rule: occupancy grid cheapest (planning-only); TSDF/dense cloud grow with resolution^3 -> tile/decimate for large sites.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="0.4")
    fig.tight_layout()
    p = os.path.join(OUT, "f11_storage.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_frames():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.set_aspect("equal")
    # map frame (origin)
    ax.quiver(0, 0, 1, 0, color="k", scale=3, width=0.01, headwidth=4, label="map frame")
    ax.quiver(0, 0, 0, 1, color="k", scale=3, width=0.01, headwidth=4)
    # robot frame
    rx, ry = 3, 2
    ax.quiver(rx, ry, 1, 0, color="b", scale=3, width=0.01, headwidth=4, label="robot/base frame")
    ax.quiver(rx, ry, 0, 1, color="b", scale=3, width=0.01, headwidth=4)
    # sensor frame (on robot, tilted)
    sx, sy = rx + 0.8, ry + 0.5
    ax.quiver(sx, sy, 0.9, 0.3, color="r", scale=3, width=0.01, headwidth=4, label="sensor frame")
    ax.quiver(sx, sy, -0.3, 0.9, color="r", scale=3, width=0.01, headwidth=4)
    # transform arrows
    ax.annotate("", xy=(rx, ry), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="b", lw=1.2))
    ax.text(rx / 2, ry / 2 + 0.2, "T_wc\n(pose)", color="b", fontsize=8, ha="center")
    ax.annotate("", xy=(sx, sy), xytext=(rx, ry), arrowprops=dict(arrowstyle="-|>", color="r", lw=1.2))
    ax.text((rx + sx) / 2 + 0.1, (ry + sy) / 2, "T_cl\n(extrinsic)", color="r", fontsize=8, ha="center")
    ax.text(0, -0.6, "map frame", fontsize=8)
    ax.text(rx, ry - 0.5, "robot frame", fontsize=8, color="b")
    ax.text(sx, sy - 0.5, "sensor frame", fontsize=8, color="r")
    ax.set_xlim(-1, 6); ax.set_ylim(-1, 5)
    ax.set_title("Coordinate frames: all maps must unify to MAP frame")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    p = os.path.join(OUT, "f11_frames.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    for f in (fig_pipeline, fig_representations, fig_storage, fig_frames):
        print("wrote", f())
    print("DONE")
