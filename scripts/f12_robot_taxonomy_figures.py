"""F12 机器人分类：生成地形图（纯 numpy+matplotlib，零下载）。

产出 experiments/F12_robot_taxonomy/figs/f12_landscape.png
横轴=环境先验(右=known)，纵轴=自主难度；气泡大小=速度/风险。
图内英文标签（matplotlib 无 CJK 字形）。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "F12_robot_taxonomy", "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})

# (name, prior 0..1 right=known, difficulty 0..1, size)
robots = [
    ("sweeper/AMR", 0.95, 0.12, 120),
    ("warehouse", 0.92, 0.18, 130),
    ("campus", 0.88, 0.22, 130),
    ("hotel delivery", 0.85, 0.25, 110),
    ("AGV port", 0.7, 0.35, 140),
    ("farm robot", 0.65, 0.4, 120),
    ("AV (HD-map)", 0.6, 0.6, 220),
    ("AV (mapless)", 0.35, 0.75, 220),
    ("USV", 0.4, 0.7, 180),
    ("wild drone", 0.3, 0.8, 170),
    ("inspection", 0.8, 0.3, 110),
    ("rescue", 0.25, 0.82, 160),
    ("underwater AUV", 0.1, 0.9, 150),
    ("tunnel/mining", 0.15, 0.88, 150),
    ("lunar/ruins", 0.05, 0.98, 140),
]

fig, ax = plt.subplots(figsize=(8.5, 5.2))
rng = np.random.default_rng(1)
for name, prior, diff, sz in robots:
    x = prior + rng.uniform(-0.02, 0.02)
    y = diff + rng.uniform(-0.02, 0.02)
    color = plt.cm.viridis(1 - diff)  # hard=red-ish
    ax.scatter(x, y, s=sz, color=color, alpha=0.75, edgecolor="0.3", lw=0.8)
    ax.text(x, y + 0.03, name, fontsize=7.5, ha="center", va="bottom")

ax.set_xlabel("Environment prior  ->  (right = known / can pre-map)")
ax.set_ylabel("Autonomy difficulty  ->")
ax.set_xlim(-0.05, 1.1)
ax.set_ylim(-0.05, 1.1)
ax.set_title("Robot landscape: where does your robot sit?")
ax.grid(alpha=0.25)
# region shading
ax.axvspan(0.55, 1.1, alpha=0.06, color="green")
ax.axvspan(-0.05, 0.45, alpha=0.06, color="red")
ax.text(0.82, 0.02, "known / easy", color="green", fontsize=8, ha="center")
ax.text(0.2, 0.02, "unknown / hard", color="red", fontsize=8, ha="center")
fig.tight_layout()
p = os.path.join(OUT, "f12_landscape.png")
fig.savefig(p, bbox_inches="tight")
plt.close(fig)
print("wrote", p)
