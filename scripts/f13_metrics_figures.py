"""F13 指标速查：生成"你想评什么 -> 看哪个指标"决策图（纯 numpy+matplotlib，零下载）。

产出 experiments/F13_metrics/figs/f13_metric_tree.png
图内英文标签（matplotlib 无 CJK 字形）。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "F13_metrics", "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})

fig, ax = plt.subplots(figsize=(8.5, 5.6))
ax.axis("off")

# nodes: (x, y, w, h, text, color)
nodes = [
    (0.30, 0.90, 0.40, 0.08, "What to evaluate?", "#dddddd"),
    # left branch: motion/geometry
    (0.02, 0.74, 0.34, 0.08, "Trajectory / Geometry", "#cfe8ff"),
    (0.02, 0.60, 0.34, 0.07, "ATE (+Sim3 align)", "#cfe8ff"),
    (0.02, 0.50, 0.34, 0.07, "RPE (drift/frame)", "#cfe8ff"),
    (0.02, 0.40, 0.34, 0.07, "Depth: AbsRel/RMSE/d1", "#cfe8ff"),
    # middle: perception
    (0.33, 0.74, 0.34, 0.08, "Perception", "#d6f5d6"),
    (0.33, 0.60, 0.34, 0.07, "Detection: mAP/recall", "#d6f5d6"),
    (0.33, 0.50, 0.34, 0.07, "Segmentation: mIoU", "#d6f5d6"),
    (0.33, 0.40, 0.34, 0.07, "Online health: inlier%", "#d6f5d6"),
    # right: system/edge
    (0.64, 0.74, 0.34, 0.08, "Closed loop / Edge", "#ffd6d6"),
    (0.64, 0.60, 0.34, 0.07, "Path len / avoid steps", "#ffd6d6"),
    (0.64, 0.50, 0.34, 0.07, "Latency / Hz (edge)", "#ffd6d6"),
    (0.64, 0.40, 0.34, 0.07, "Calibration residual px", "#ffd6d6"),
]
for x, y, w, h, t, c in nodes:
    ax.add_patch(Rectangle((x, y), w, h, facecolor=c, edgecolor="0.3", lw=1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.5)

# arrows from root to 3 branches
rx, ry, rw, rh = nodes[0][:4]
for (bx, by, bw, bh, _, _) in nodes[1:4]:
    ax.add_patch(FancyArrowPatch((rx + rw / 2, ry), (bx + bw / 2, by + bh),
                                 arrowstyle="-|>", mutation_scale=12, color="0.4", lw=1.2))
# branch boxes connected vertically (root->branch label already)
for (bx, by, bw, bh, _, _), children in [
    (nodes[1], nodes[2:5]),
    (nodes[5], nodes[6:9]),
    (nodes[9], nodes[10:13]),
]:
    for (cx, cy, cw, ch, _, _) in children:
        ax.add_patch(FancyArrowPatch((bx + bw / 2, by), (cx + cw / 2, cy + ch),
                                     arrowstyle="-|>", mutation_scale=10, color="0.5", lw=1))

ax.text(0.5, 0.02, "One metric lies (ATE non-monotonic when degraded) -> use a COMBINATION",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color="0.4")
fig.tight_layout()
p = os.path.join(OUT, "f13_metric_tree.png")
fig.savefig(p, bbox_inches="tight")
plt.close(fig)
print("wrote", p)
