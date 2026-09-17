"""
P6 · USV 感知→控制输入 模拟演练脚本（纯合成数据，[模拟] 非真实采集）

目的：把 docs/P6_new_scenario_perception.md §6 的「假设设计」变成一张**可看的数据图**，
加深认知：① 占据栅格在水域长什么样（岸/浅滩/障碍/反光区）；② A* 怎么绕行；
③ 强反光区如何触发 M6 定位兜底（ATE 曲线）。

全部为合成假设，无真实 USV 数据。输出：
  experiments/P6_usv_sim/figs/usv_occupancy_plan.png   （俯视航道 + 栅格 + A* 路径 + 动船）
  experiments/P6_usv_sim/figs/usv_glare_ate.png        （ATE vs 反光强度 + M6 兜底触发）
  experiments/P6_usv_sim/metrics.json
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "experiments/P6_usv_sim"
FIG = os.path.join(OUT, "figs")
os.makedirs(FIG, exist_ok=True)

# ---------- 场景参数（与 P6 §6 一致，[模拟]） ----------
W, H = 600, 400          # 栅格尺寸 (cell)
RES = 0.4                # 分辨率 m/cell -> 世界 240m x 160m 内河航道
ROBOT_R = 0.5            # 船半径 m
MARGIN = 0.3             # 安全余量 m
INFL = int(np.ceil((ROBOT_R + MARGIN) / RES))   # 膨胀半径(格) = 2

# 栅格语义类别
FREE, SHORE, SHOAL, OBST, GLARE = 0, 1, 2, 3, 4

grid = np.full((H, W), FREE, dtype=np.uint8)
def rect(r0, r1, c0, c1, val):
    grid[r0:r1, c0:c1] = val

# 两岸（沿航道上下边界）
rect(0, 40, 0, W, SHORE)
rect(H - 40, H, 0, W, SHORE)
# 浅滩（航道中部的圆形浅水区）
yy, xx = np.ogrid[:H, :W]
shoal = (xx - 300) ** 2 / 30 ** 2 + (yy - 200) ** 2 / 18 ** 2 <= 1
grid[shoal] = SHOAL
# 桥墩（中部矩形障碍）
rect(120, 200, 450, 470, OBST)
# 浮标（两个小点障碍）
rect(149, 151, 200, 202, OBST)
rect(249, 251, 520, 522, OBST)
# 强反光区（晴天正午镜面反射，定位退化，可通行但高代价）→ 只覆盖当前水域，不覆盖浅滩/障碍
glare_mask = np.zeros((H, W), dtype=bool)
glare_mask[150:250, 260:360] = True
grid[glare_mask & (grid == FREE)] = GLARE

# 移动船只（[模拟] 当前位置 + 速度 m/s），放在 A* 主航道(r≈250)上以触发避障
moving_boats = [
    {"name": "boat_A", "cy": 250, "cx": 350, "vy": 0.0, "vx": 0.3},
    {"name": "boat_B", "cy": 250, "cx": 250, "vy": 0.0, "vx": -0.2},
]

start = (20, 200)      # (cx, cy) 左端航道口（中部水域）
goal = (580, 200)      # (cx, cy) 右端航道口

# ---------- A* (8 邻接, 欧氏启发, 反光区高代价) ----------
def astar(grid, start, goal, glare_cost=5.0):
    H_, W_ = grid.shape
    cost = np.ones((H_, W_), dtype=float)
    cost[grid == OBST] = np.inf
    cost[grid == SHORE] = np.inf
    cost[grid == SHOAL] = np.inf
    cost[grid == GLARE] = glare_cost
    sr, sc = start[1], start[0]
    gr, gc = goal[1], goal[0]
    import heapq
    open_ = [(0.0, sr, sc)]
    came = {}
    gsc = {sr * W_ + sc: 0.0}
    while open_:
        f, r, c = heapq.heappop(open_)
        if (r, c) == (gr, gc):
            break
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H_ and 0 <= nc < W_):
                    continue
                if not np.isfinite(cost[nr, nc]):
                    continue
                step = np.hypot(dr, dc)
                ng = gsc[r * W_ + c] + step * cost[nr, nc]
                if ng < gsc.get(nr * W_ + nc, np.inf):
                    gsc[nr * W_ + nc] = ng
                    came[nr * W_ + nc] = (r, c)
                    heapq.heappush(open_, (ng + np.hypot(nr - gr, nc - gc), nr, nc))
    if (gr * W_ + gc) not in came and (gr, gc) != (sr, sc):
        return None
    path = []
    cur = (gr, gc)
    while cur != (sr, sc):
        path.append(cur)
        cur = came[cur[0] * W_ + cur[1]]
    path.append((sr, sc))
    return path[::-1]

path = astar(grid, start, goal)
assert path is not None, "A* 未找到路径"

# ---------- 膨胀（障碍 + 浅滩 + 岸，反光区不膨胀） ----------
inflate_mask = np.isin(grid, [OBST, SHORE, SHOAL])
def dilate(mask, r):
    out = mask.copy()
    ys, xs = np.where(mask)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = ys + dy, xs + dx
            ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            out[ny[ok], nx[ok]] = True
    return out
inflated = dilate(inflate_mask, INFL)

# ---------- 避障步数（路径靠近移动船当前位置 <1.5m） ----------
def avoidance_steps(path, boats, radius_m=1.5):
    rad = int(radius_m / RES)
    cnt = 0
    for (r, c) in path:
        for b in boats:
            if abs(c - b["cx"]) <= rad and abs(r - b["cy"]) <= rad:
                cnt += 1
                break
    return cnt

n_avoid = avoidance_steps(path, moving_boats)
path_len_m = sum(np.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                 for i in range(len(path) - 1)) * RES

# ---------- 图1：俯视航道 + 栅格 + A* 路径 + 动船 ----------
cmap = matplotlib.colors.ListedColormap(
    ["#2b6cb0", "#c2a36b", "#8c6b3f", "#c0392b", "#f6e05e"])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
fig, ax = plt.subplots(figsize=(11, 7.5))
ax.imshow(grid, cmap=cmap, norm=norm, origin="upper")
pr = [p[0] for p in path]; pc = [p[1] for p in path]
ax.plot(pc, pr, "-", color="#22c55e", lw=2.2, label="A* path")
ax.plot(start[0], start[1], "o", color="white", ms=9, mec="k", label="start")
ax.plot(goal[0], goal[1], "*", color="gold", ms=14, mec="k", label="goal")
for b in moving_boats:
    ax.plot(b["cx"], b["cy"], "s", color="#ea580c", ms=10, mec="k")
    ax.arrow(b["cx"], b["cy"], b["vx"] * 12, b["vy"] * 12,
             color="#ea580c", head_width=6, head_length=8, lw=1.5)
ax.text(start[0] + 6, start[1], "Pier A", color="white", fontsize=9)
ax.text(goal[0] - 42, goal[1], "Pier B", color="gold", fontsize=9)
ax.text(300, 175, "shoal", color="white", fontsize=8, ha="center")
ax.text(460, 128, "pier", color="white", fontsize=8, ha="center")
ax.text(310, 152, "glare zone (M6)", color="black", fontsize=8, ha="center")
ax.set_title("USV inland waterway: occupancy grid + A* plan (synthetic, [SIM])", fontsize=12)
ax.set_xlabel("along-channel (cell -> 240 m)"); ax.set_ylabel("cross-channel (cell -> 160 m)")
from matplotlib.patches import Patch
leg = [Patch(facecolor="#2b6cb0", label="navigable water"),
       Patch(facecolor="#c2a36b", label="shore"),
       Patch(facecolor="#8c6b3f", label="shoal"),
       Patch(facecolor="#c0392b", label="obstacle (pier/buoy)"),
       Patch(facecolor="#f6e05e", label="glare zone (loc degraded)")]
leg += [plt.Line2D([0],[0], color="#22c55e", lw=2.2, label="A* path"),
        plt.Line2D([0],[0], marker="s", color="#ea580c", linestyle="none", label="moving boat")]
ax.legend(handles=leg, loc="upper right", fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "usv_occupancy_plan.png"), dpi=130); plt.close(fig)

# ---------- 图2：ATE vs 反光强度 + M6 兜底触发 ----------
glare = np.linspace(0, 1, 100)
ate = 0.18 + 0.67 * np.clip((glare - 0.40) / 0.30, 0, 1)   # 干净0.18m -> 强反光0.85m
fallback_thr = 0.5
trigger = glare[ate >= fallback_thr][0] if np.any(ate >= fallback_thr) else None
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(glare, ate, "-", color="#2b6cb0", lw=2.2, label="VGGT ATE (sim)")
ax.axhline(fallback_thr, color="#c0392b", ls="--", lw=1.5, label="M6 fallback thr 0.5 m")
ax.axvline(trigger, color="#ea580c", ls=":", lw=1.5,
           label=f"fallback at glare~{trigger:.2f}" if trigger else "no trigger")
ax.scatter([0.0], [0.18], color="green", zorder=5)
ax.text(0.02, 0.20, "clean 0.18 m (TUM-scale)", color="green", fontsize=8)
ax.set_xlabel("water glare intensity (0=none, 1=specular)"); ax.set_ylabel("localization ATE (m)")
ax.set_title("Glare degrades VGGT loc -> M6 diagnostic switch (sim, [SIM])", fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "usv_glare_ate.png"), dpi=130); plt.close(fig)

# ---------- metrics.json（镜像 P5 格式 + [模拟] 标注） ----------
metrics = {
    "project": "P6_usv_sim",
    "title": "USV 内河航道 感知→控制输入 模拟演练（合成数据，[模拟]非真实采集）",
    "scenario": "内河港口巡检，码头A→B 240m x 160m 航道",
    "grid": {"shape": [H, W], "res_m": RES, "world_m": [H * RES, W * RES]},
    "cells": {"shore": int((grid == SHORE).sum()), "shoal": int((grid == SHOAL).sum()),
              "obstacle": int((grid == OBST).sum()), "glare": int((grid == GLARE).sum())},
    "inflation": {"robot_radius_m": ROBOT_R, "margin_m": MARGIN, "radius_cells": INFL,
                  "inflated_cells": int(inflated.sum())},
    "moving_boats": moving_boats,
    "closed_loop": {"start_cell": list(start)[::-1], "goal_cell": list(goal)[::-1],
                    "path_steps": len(path), "path_length_m": round(path_len_m, 2),
                    "n_cmds": len(path) - 1, "avoidance_steps": n_avoid,
                    "glare_cells_passed": int(sum(1 for (r, c) in path if grid[r, c] == GLARE))},
    "localization_sim": {"clean_ate_m": 0.18, "glare_ate_m": 0.85,
                         "fallback_threshold_m": fallback_thr,
                         "fallback_trigger_glare": round(float(trigger), 2) if trigger else None,
                         "note": "与 §6.4 假设一致；clean 点量级参考 TUM fr1/desk VGGT 0.022m"},
    "figs": {"usv_occupancy_plan": os.path.join(FIG, "usv_occupancy_plan.png"),
             "usv_glare_ate": os.path.join(FIG, "usv_glare_ate.png")},
    "honest_bounds": ["全部为合成模拟，非真实 USV 采集", "控制为运动学仿真",
                      "真实需出海+RTK真值载荷(见 §8)", "ATE 曲线为假设模型非实测"],
}
with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print("=== P6 USV 模拟演练完成（[模拟]）===")
print(f"栅格 {W}x{H}@{RES}m | 岸{metrics['cells']['shore']} 浅滩{metrics['cells']['shoal']} "
      f"障碍{metrics['cells']['obstacle']} 反光{metrics['cells']['glare']}")
print(f"A* 步数 {len(path)} | 路径长 {path_len_m:.1f}m | 避障步数 {n_avoid} | 经过反光格 {metrics['closed_loop']['glare_cells_passed']}")
print(f"膨胀格 {metrics['inflation']['inflated_cells']} | M6 兜底触发反光强度≈{metrics['localization_sim']['fallback_trigger_glare']}")
print(f"产物: {FIG}/usv_occupancy_plan.png , {FIG}/usv_glare_ate.png , {OUT}/metrics.json")
