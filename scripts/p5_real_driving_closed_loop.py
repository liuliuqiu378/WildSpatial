#!/usr/bin/env python
"""p5_real_driving_closed_loop.py — 真实驾驶数据上的「感知→建图→规划→控制」端到端闭环
======================================================================================

================== 为什么做这个（目的）==================
P4 回答了"真实传感器数据怎么变成一张随车生长的地图"（**自动驾驶栈的前半段**：
感知 → 定位 → 建图 → 可视化），但它**不含规划/控制**——产物是地图和回放视频，不是控制指令，
而且那张 BEV 地图**不能直接拿去导航**（无 inflation 膨胀、无机器人几何约束）。

P3 回答了"定位崩了怎么兜底 + 闭环控制"，但它的输入是 **TUM 手持桌面序列**（室内、非真实驾驶）。

**P5 = 把 P4 的"真实驾驶"输入，接上 P3 的"规划→控制"闭环**，补上中间缺的那一环，
得到本仓库唯一一条**在同一份真实驾驶数据上跑通的「感知→建图→规划→控制」全链路**：

    P4 的前半段（真实 RGB + 真实 LiDAR → VGGT 定位 → 定标 → 累积 BEV 占据栅格）
                            │
                            ▼  ← P5 从这里接上（把"地图"变成"决策"的关键一跃）
        ① 2.5D 占据栅格（地面 free / 障碍 occupied）
                            │
                            ▼
        ② inflation 膨胀（机器人半径 + 安全余量）→ 规划用代价地图
                            │
                            ▼
        ③ A* 全局规划（起点=自车，目标=沿轨迹前方可达点）
                            │
                            ▼
        ④ 差速轮控制 (v, ω)（含加速度/角速度限幅）
                            │
                            ▼
        ⑤ 速度障碍法（VO）动态避障减速（对向车/行人）
                            │
                            ▼
           控制指令序列 + 闭环可视化（俯视）

产出（`experiments/P5_real_driving_closed_loop/`）：
    figs/bev_occupancy.png      累积 BEV 占据栅格（真实 LiDAR 建图）
    figs/costmap_inflated.png   膨胀后的规划代价地图（对比未膨胀）
    figs/map_plan.png           代价地图 + A* 全局路径 + 起终点
    figs/control_cmd.png        差速轮控制指令序列 (v, ω)
    figs/dynamic_avoid.png      动态障碍（真实检测的移动目标）→ 避让减速
    figs/closed_loop.gif        闭环俯视回放（车沿轨迹走 + 规划路径 + 避障）
    metrics.json                全部量化
    README.md                   四段式 + 三视角讲解

⚠️ 诚实声明（必读）：
  1. 本闭环的**自车运动学的"执行"是运动学仿真**（用 P4 的估计轨迹驱动，不是真车）；
     真车执行需要底盘 + 电机驱动 + 实时控制（见 F7 §8）。
  2. **规划/控制的"地图"来自视觉+激光估计**（无真值位姿）→ 地图有 VGGT 漂移；
     这不是"带真值的闭环评测"，是"真实数据上的系统贯通"。
  3. 动态障碍用的是 P4 检出的**真实移动点**（KD-tree 帧间差分，启发式，非检测器 benchmark）；
     避障逻辑是**反应式减速（VO 简化版）**，不是完整 DWA/MPC。
  4. 真实驾驶场景下"自车"本就在往前开 → 这里的"规划"演示的是**沿已知走廊到前方目标点**的
     路径规划与运动学可行性，不是"随意想到哪就到哪"。

运行（wildspatial 环境，需 GPU 跑 VGGT）：
  PYTHONPATH=src python scripts/p5_real_driving_closed_loop.py \
      --drive 2011_09_26_drive_0023_sync --max-frames 30
"""
import os
import sys
import json
import heapq
import argparse

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# ---- 复用 P4 的感知/建图前端（真实 RGB + 真实 LiDAR → 世界系点 + 移动目标）----
import importlib.util
_p4_path = os.path.join(ROOT, "scripts", "p4_real_driving.py")
_spec = importlib.util.spec_from_file_location("p4_real_driving", _p4_path)
p4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p4)

from wildspatial.viz import setup_plot_style
setup_plot_style()

OUT = os.path.join(ROOT, "experiments", "P5_real_driving_closed_loop")


# --------------------------------------------------------------------------- #
# ① 世界点（真实 LiDAR，含移动目标标记）→ 2.5D 占据栅格
#    与 P3 的 build_occupancy 同思路，但**数据源是真实驾驶 LiDAR**（非 TUM 深度），
#    且**显式剔除移动目标点**（动态物体不应进静态地图 —— P4 §⑧ 提到的"地图幽灵"）。
# --------------------------------------------------------------------------- #
def build_occupancy_from_lidar(per, dyn_world, res=0.4, margin=4.0,
                               h_min=0.30, h_max=3.0):
    """真实 LiDAR 世界点 → 2.5D 占据栅格 (0=未知, 1=可通, 2=障碍)。

    地面点 (Y < h_min) → 可通；障碍点 (h_min ≤ Y ≤ h_max) → 占据；
    移动目标点（P4 检出的动态点）→ **不进静态地图**（否则地图会留下移动物体的幽灵）。
    """
    allp = np.concatenate([p for p in per if len(p) > 0], axis=0)
    xs, zs = allp[:, 0], allp[:, 2]
    xmin, xmax = xs.min() - margin, xs.max() + margin
    zmin, zmax = zs.min() - margin, zs.max() + margin
    nx = int(np.ceil((xmax - xmin) / res))
    nz = int(np.ceil((zmax - zmin) / res))
    occ = np.zeros((nz, nx), np.uint8)

    # 动态点集合（用于剔除）——按栅格坐标做集合查询，避免逐点 O(N·M)
    dyn_set = set()
    if dyn_world is not None:
        for dw in dyn_world:
            if len(dw) == 0:
                continue
            gx = np.floor((dw[:, 0] - xmin) / res).astype(int)
            gz = np.floor((dw[:, 2] - zmin) / res).astype(int)
            for a, b in zip(gx, gz):
                dyn_set.add((int(a), int(b)))

    n_dyn_cells = 0
    for p in per:
        if len(p) == 0:
            continue
        gx = np.floor((p[:, 0] - xmin) / res).astype(int)
        gz = np.floor((p[:, 2] - zmin) / res).astype(int)
        ok = (gx >= 0) & (gx < nx) & (gz >= 0) & (gz < nz)
        Y = p[:, 1]
        free = ok & (Y < h_min)
        ob = ok & (Y >= h_min) & (Y <= h_max)
        # 逐点写入（真实 LiDAR 点数适中，且要剔除动态栅格）
        for a, b, f, o in zip(gx, gz, free, ob):
            if not (f or o):
                continue
            if (int(a), int(b)) in dyn_set:
                n_dyn_cells += 1
                continue
            if f:
                if occ[b, a] == 0:
                    occ[b, a] = 1
            else:
                occ[b, a] = 2                      # 障碍优先覆盖
    bounds = (xmin, xmax, zmin, zmax)
    return occ, bounds, res, len(allp), n_dyn_cells


# --------------------------------------------------------------------------- #
# ② inflation 膨胀：把障碍按机器人半径 + 安全余量"长胖"
#    这是 P4 地图**不能直接导航**的关键原因，也是本篇要补的第一课。
# --------------------------------------------------------------------------- #
def inflate_obstacles(occ, res, robot_radius=0.5, safety_margin=0.3):
    """障碍膨胀：占据格向四周扩张 (robot_radius+margin) 距离。返回代价地图。"""
    radius_cells = int(np.ceil((robot_radius + safety_margin) / res))
    if radius_cells <= 0:
        return occ.copy(), 0
    obst = (occ == 2).astype(np.uint8)
    # 用形态学膨胀实现（方形核，近似圆形足够）
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * radius_cells + 1, 2 * radius_cells + 1))
    inflated = cv2.dilate(obst, k, iterations=1)
    cost = occ.copy()
    new_obst = (inflated == 1) & (occ != 2)
    cost[new_obst] = 2                    # 膨胀区也视为不可通行（保守）
    n_inflated = int(new_obst.sum())
    return cost, n_inflated


# --------------------------------------------------------------------------- #
# ③ A* 全局规划（与 P3 同实现；障碍格 occ==2 不可通行）
# --------------------------------------------------------------------------- #
def astar(cost, start, goal):
    nz, nx = cost.shape

    def neigh(r, c):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < nz and 0 <= nc < nx and cost[nr, nc] != 2:
                yield nr, nc, (1.414 if dr and dc else 1.0)

    openh = [(0.0, 0, start)]
    came = {start: None}
    gcost = {start: 0.0}
    while openh:
        _, _, node = heapq.heappop(openh)
        if node == goal:
            break
        for nr, nc, c in neigh(*node):
            ng = gcost[node] + c
            if (nr, nc) not in gcost or ng < gcost[(nr, nc)]:
                gcost[(nr, nc)] = ng
                h = ((nr - goal[0]) ** 2 + (nc - goal[1]) ** 2) ** 0.5
                heapq.heappush(openh, (ng + h, len(came), (nr, nc)))
                came[(nr, nc)] = node
    if goal not in came:
        return None
    path, n = [], goal
    while n is not None:
        path.append(n)
        n = came[n]
    return path[::-1]


def world_to_cell(x, z, bounds, res):
    xmin, xmax, zmin, zmax = bounds
    return (int((z - zmin) / res), int((x - xmin) / res))


def cell_to_world(r, c, bounds, res):
    xmin, xmax, zmin, zmax = bounds
    return ((c + 0.5) * res + xmin, (r + 0.5) * res + zmin)


def pick_goal_forward(occ, start_rc, bounds, res, min_cells=15):
    """目标点：沿自车前进方向（Z 增大）找最远的可通行格。

    真实驾驶中车是往前开的，所以"目标"取前方可达点，而不是地图里任意最远点。
    """
    cost = (occ != 2)
    nz, nx = occ.shape
    r0, c0 = start_rc
    # 在前方（Z 更大）的窗口里找可通行格，优先最远
    for dz in range(min_cells, nz):
        r = r0 + dz
        if r >= nz:
            break
    best = None
    for r in range(nz - 1, r0 + min_cells - 1, -1):
        row_free = np.where(occ[r] != 2)[0]
        if len(row_free) == 0:
            continue
        # 选离当前列最近的自由格
        c = row_free[np.argmin(np.abs(row_free - c0))]
        best = (r, int(c))
        break
    return best


# --------------------------------------------------------------------------- #
# ④ 差速轮控制（与 P3 同实现）
# --------------------------------------------------------------------------- #
def diff_drive_control(path_rc, bounds, res, dt=0.5, v_max=0.8,
                       a_max=1.0, omega_max=1.5, turn_slow=0.6):
    pts = np.array([cell_to_world(r, c, bounds, res) for r, c in path_rc])
    if len(pts) < 2:
        return [], pts
    dxy = np.diff(pts, axis=0)
    dist = np.linalg.norm(dxy, axis=1)
    theta = np.arctan2(dxy[:, 1], dxy[:, 0])
    dtheta = np.diff(theta)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    cmds, v_prev = [], 0.0
    for i in range(len(dist)):
        ang = abs(dtheta[i]) if i < len(dtheta) else 0.0
        v_des = v_max * (turn_slow if ang > 0.4 else 1.0)
        dv = np.clip(v_des - v_prev, -a_max * dt, a_max * dt)
        v = v_prev + dv
        omega = float(np.clip(dtheta[i] / dt if i < len(dtheta) else 0.0,
                              -omega_max, omega_max))
        cmds.append({"step": i, "v": round(float(v), 3),
                     "omega": round(omega, 3), "seg_len": round(float(dist[i]), 3)})
        v_prev = v
    return cmds, pts


# --------------------------------------------------------------------------- #
# ⑤ 动态避障（VO 简化版）：用 P4 的**真实移动目标点**做障碍，沿路径减速
# --------------------------------------------------------------------------- #
def dynamic_avoid_from_detections(cmds, path_world, dyn_world, bounds, res,
                                  robot_radius=0.5, obs_pad=0.5, lookahead=3,
                                  slow_factor=0.30):
    if path_world is None or len(path_world) == 0:
        return cmds, 0, np.zeros((0, 2))
    # 收集所有真实移动目标点的 (X,Z)
    if dyn_world is not None:
        obs = np.concatenate([d[:, [0, 2]] for d in dyn_world if len(d) > 0], axis=0) \
            if any(len(d) > 0 for d in dyn_world) else np.zeros((0, 2))
    else:
        obs = np.zeros((0, 2))
    dmin = robot_radius + obs_pad
    n_slow = 0
    for c in cmds:
        idx = min(c["step"], len(path_world) - 1)
        hit = False
        for la in range(0, lookahead + 1):
            ti = idx + la
            if ti >= len(path_world):
                break
            px, pz = path_world[ti]
            if len(obs):
                d = np.hypot(obs[:, 0] - px, obs[:, 1] - pz)
                if d.min() < dmin:
                    hit = True
                    break
        if hit:
            c["v"] = round(float(c["v"] * slow_factor), 3)
            c["avoid"] = True
            n_slow += 1
    return cmds, n_slow, obs


# --------------------------------------------------------------------------- #
# 可视化
# --------------------------------------------------------------------------- #
def render_bev_occupancy(occ, bounds, res, out_png):
    xmin, xmax, zmin, zmax = bounds
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = matplotlib.colors.ListedColormap(["#dddddd", "#ffffff", "#333333"])
    ax.imshow(occ, origin="lower", extent=[xmin, xmax, zmin, zmax],
              cmap=cmap, vmin=0, vmax=2)
    ax.set_xlabel("X right (m)"); ax.set_ylabel("Z forward (m)")
    ax.set_title("① 真实 LiDAR 累积的 2.5D 占据栅格\n"
                 "(灰=未知, 白=可通, 黑=障碍; 移动目标已剔除)")
    ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    return out_png


def render_costmap_compare(occ, cost, bounds, res, out_png):
    xmin, xmax, zmin, zmax = bounds
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6.5))
    cmap = matplotlib.colors.ListedColormap(["#dddddd", "#ffffff", "#333333"])
    axL.imshow(occ, origin="lower", extent=[xmin, xmax, zmin, zmax],
               cmap=cmap, vmin=0, vmax=2)
    axL.set_title("②a 原始占据栅格（不可直接导航）")
    axL.set_xlabel("X (m)"); axL.set_ylabel("Z (m)"); axL.set_aspect("equal")
    cmap2 = matplotlib.colors.ListedColormap(["#dddddd", "#ffffff", "#d62728"])
    axR.imshow(cost, origin="lower", extent=[xmin, xmax, zmin, zmax],
               cmap=cmap2, vmin=0, vmax=2)
    axR.set_title("②b 膨胀后代价地图（机器人半径+安全余量 → 可规划）")
    axR.set_xlabel("X (m)"); axR.set_ylabel("Z (m)"); axR.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    return out_png


def render_map_plan(cost, bounds, res, path_rc, start, goal, traj_world, out_png):
    xmin, xmax, zmin, zmax = bounds
    fig, ax = plt.subplots(figsize=(8.5, 7))
    cmap = matplotlib.colors.ListedColormap(["#dddddd", "#ffffff", "#d62728"])
    ax.imshow(cost, origin="lower", extent=[xmin, xmax, zmin, zmax],
              cmap=cmap, vmin=0, vmax=2)
    if traj_world is not None and len(traj_world) > 1:
        ax.plot(traj_world[:, 0], traj_world[:, 2], "-", color="lime", lw=2,
                label="自车真实轨迹（VGGT）")
    if path_rc:
        px = [(c + 0.5) * res + xmin for r, c in path_rc]
        pz = [(r + 0.5) * res + zmin for r, c in path_rc]
        ax.plot(px, pz, "-", color="#1f77b4", lw=2.5, label="A* 规划路径")
    if start:
        ax.scatter([(start[1] + 0.5) * res + xmin], [(start[0] + 0.5) * res + zmin],
                   marker="s", s=140, color="lime", edgecolor="black", zorder=5, label="起点(自车)")
    if goal:
        ax.scatter([(goal[1] + 0.5) * res + xmin], [(goal[0] + 0.5) * res + zmin],
                   marker="X", s=180, color="#ff7f0e", edgecolor="black", zorder=5, label="目标点(前方可达)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.set_title("③ 代价地图 + A* 全局规划（真实驾驶数据上）")
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    return out_png


def render_control(cmds, out_png):
    fig, ax = plt.subplots(figsize=(9, 4))
    steps = [c["step"] for c in cmds]
    ax.plot(steps, [c["v"] for c in cmds], "-o", ms=3, color="#1f77b4",
            label="线速度 v (m/s)")
    ax.plot(steps, [c["omega"] for c in cmds], "-s", ms=3, color="#d62728",
            label="角速度 ω (rad/s)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("控制步（dt=0.5s）"); ax.set_ylabel("指令量")
    ax.set_title("④ 差速轮控制指令序列（含 a_max/ω_max 限幅）")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    return out_png


def render_dynamic_avoid(cost, bounds, res, path_rc, cmds, obs, out_png):
    xmin, xmax, zmin, zmax = bounds
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))
    cmap = matplotlib.colors.ListedColormap(["#dddddd", "#ffffff", "#d62728"])
    axL.imshow(cost, origin="lower", extent=[xmin, xmax, zmin, zmax],
               cmap=cmap, vmin=0, vmax=2)
    if path_rc:
        px = [(c + 0.5) * res + xmin for r, c in path_rc]
        pz = [(r + 0.5) * res + zmin for r, c in path_rc]
        axL.plot(px, pz, "-", color="#1f77b4", lw=2, label="A* 路径")
    if obs is not None and len(obs):
        axL.scatter(obs[:, 0], obs[:, 1], s=8, c="orange", alpha=0.6,
                    label="真实移动目标（P4 检出）")
    axL.set_title("⑤a 路径 + 真实动态目标（对向车/行人）")
    axL.set_xlabel("X (m)"); axL.set_ylabel("Z (m)"); axL.set_aspect("equal")
    axL.legend(fontsize=8, loc="upper right")
    if cmds:
        steps = [c["step"] for c in cmds]
        axR.plot(steps, [c["v"] for c in cmds], "-o", ms=3, color="#1f77b4",
                 label="v (m/s)")
        for s in [c["step"] for c in cmds if c.get("avoid")]:
            axR.axvline(s, color="#ff7f0e", alpha=0.30, lw=1.2)
        if any(c.get("avoid") for c in cmds):
            axR.plot([], [], color="#ff7f0e", lw=1.2, label="触发避让减速")
        axR.axhline(0, color="gray", lw=0.8)
        axR.set_xlabel("控制步"); axR.set_ylabel("v (m/s)")
        axR.set_title("⑤b 避让减速（VO 简化版）")
        axR.legend(fontsize=8); axR.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    return out_png


def render_closed_loop_gif(cost, bounds, res, path_rc, traj_world, obs, out_gif):
    """闭环俯视回放：车沿轨迹走，路径与障碍同屏（写多帧 PNG 再合成 GIF）。"""
    xmin, xmax, zmin, zmax = bounds
    H = 480
    W = 480
    tmp = os.path.join(os.path.dirname(out_gif), "_cl_tmp")
    os.makedirs(tmp, exist_ok=True)
    n = max(2, len(traj_world))
    for k in range(n):
        fig, ax = plt.subplots(figsize=(5, 5))
        cmap = matplotlib.colors.ListedColormap(["#dddddd", "#ffffff", "#d62728"])
        ax.imshow(cost, origin="lower", extent=[xmin, xmax, zmin, zmax],
                  cmap=cmap, vmin=0, vmax=2)
        if path_rc:
            px = [(c + 0.5) * res + xmin for r, c in path_rc]
            pz = [(r + 0.5) * res + zmin for r, c in path_rc]
            ax.plot(px, pz, "--", color="#1f77b4", lw=1.5, alpha=0.8, label="A* 路径")
        if obs is not None and len(obs):
            ax.scatter(obs[:, 0], obs[:, 1], s=6, c="orange", alpha=0.5)
        ax.plot(traj_world[:k + 1, 0], traj_world[:k + 1, 2], "-",
                color="lime", lw=2.5, label="自车轨迹")
        ax.scatter([traj_world[k, 0]], [traj_world[k, 2]], marker="o", s=90,
                   color="red", edgecolor="black", zorder=6)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
        ax.set_title(f"闭环回放 {k+1}/{n}（感知→建图→规划→控制）")
        ax.set_aspect("equal"); ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(tmp, f"f_{k:03d}.png"), dpi=70)
        plt.close(fig)
    # 合成 GIF
    import glob
    files = sorted(glob.glob(os.path.join(tmp, "f_*.png")))
    frames = [cv2.imread(f) for f in files]
    if frames:
        h, w = frames[0].shape[:2]
        frames = [cv2.resize(f, (w, h)) for f in frames]
        # 用 ffmpeg 合成更稳；退化为 cv2 逐帧写
        gif_ok = _make_gif_ffmpeg(tmp, out_gif)
        if not gif_ok:
            cv2.imwrite(out_gif.replace(".gif", "_last.png"), frames[-1])
    for f in files:
        try:
            os.remove(f)
        except OSError:
            pass
    try:
        os.rmdir(tmp)
    except OSError:
        pass
    return out_gif


def _make_gif_ffmpeg(tmpdir, out_gif):
    import subprocess
    try:
        cmd = ["ffmpeg", "-y", "-framerate", "2", "-i",
               os.path.join(tmpdir, "f_%03d.png"),
               "-vf", "split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse",
               out_gif]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return os.path.exists(out_gif)
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="2011_09_26_drive_0023_sync")
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--res", type=float, default=0.4, help="占据栅格分辨率（m）")
    ap.add_argument("--robot-radius", type=float, default=0.5, help="机器人半径（m）")
    ap.add_argument("--safety-margin", type=float, default=0.3, help="安全余量（m）")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    out_dir = args.out
    figs = os.path.join(out_dir, "figs")
    os.makedirs(figs, exist_ok=True)

    print("[P5] 真实驾驶数据上的「感知→建图→规划→控制」端到端闭环")
    print(f"[·] 复用 P4 前端：drive={args.drive}, max_frames={args.max_frames}")

    # ---------- (0) P4 前端：真实 RGB+LiDAR → 定位 → 定标 → 世界点 + 移动目标 ----------
    frames = p4.load_drive(args.drive, args.max_frames, args.stride)
    if not frames:
        raise RuntimeError(f"未找到 drive={args.drive} 的连续片段（检查 BASE 路径）")
    print(f"[·] 载入真实驾驶片段：{len(frames)} 帧")
    res_vggt = p4.run_vggt(frames)
    scale = p4.compute_scale(res_vggt, frames) if res_vggt else 1.0
    print(f"[·] 尺度定标因子 = {scale:.3f}")
    per = p4.build_world_points(frames, res_vggt, scale)
    cam = p4.build_cam_points(frames)
    dyn_world, dyn_uv = p4.detect_dynamic(frames, cam, per, res_vggt, scale)

    traj_world = None
    if res_vggt is not None:
        pos = res_vggt.positions * scale
        traj_world = np.asarray(pos)

    # ---------- (1) 2.5D 占据栅格（真实 LiDAR，剔除移动目标）----------
    occ, bounds, res, n_points, n_dyn_removed = build_occupancy_from_lidar(
        per, dyn_world, res=args.res)
    print(f"[·] 占据栅格 {occ.shape} @ {res}m | 真实点 {n_points} | "
          f"剔除移动栅格 {n_dyn_removed} | 障碍格 {int((occ==2).sum())}")
    bev_png = render_bev_occupancy(occ, bounds, res, os.path.join(figs, "bev_occupancy.png"))

    # ---------- (2) inflation 膨胀 ----------
    cost, n_inflated = inflate_obstacles(occ, res,
                                         robot_radius=args.robot_radius,
                                         safety_margin=args.safety_margin)
    print(f"[·] 膨胀：新增不可通行格 {n_inflated}（半径 {args.robot_radius}m "
          f"+ 余量 {args.safety_margin}m）")
    cmap_png = render_costmap_compare(occ, cost, bounds, res,
                                     os.path.join(figs, "costmap_inflated.png"))

    # ---------- (3) A* 规划：起点=自车首帧，目标=前方最远可达 ----------
    closed = {}
    plan_png = ctrl_png = avoid_png = gif_path = None
    if traj_world is not None and len(traj_world) > 1:
        s0 = traj_world[0]
        start = world_to_cell(s0[0], s0[2], bounds, res)
        # 若起点落在障碍/膨胀区，就近找可通行格
        if not (0 <= start[0] < cost.shape[0] and 0 <= start[1] < cost.shape[1]) \
                or cost[start] == 2:
            free = np.argwhere(cost != 2)
            if len(free):
                d = np.linalg.norm(free - np.array(start), axis=1)
                start = tuple(free[d.argmin()])
        goal = pick_goal_forward(occ, start, bounds, res)
        print(f"[·] 起点={start} 目标={goal}")
        path_rc = astar(cost, start, goal) if goal else None
        cmds, pts = ([], None)
        if path_rc:
            cmds, pts = diff_drive_control(path_rc, bounds, res)
        cmds, n_avoid, obs = dynamic_avoid_from_detections(
            cmds, pts, dyn_world, bounds, res, robot_radius=args.robot_radius)
        plan_png = render_map_plan(cost, bounds, res, path_rc, start, goal,
                                   traj_world, os.path.join(figs, "map_plan.png"))
        ctrl_png = render_control(cmds, os.path.join(figs, "control_cmd.png"))
        avoid_png = render_dynamic_avoid(cost, bounds, res, path_rc, cmds, obs,
                                         os.path.join(figs, "dynamic_avoid.png"))
        try:
            gif_path = render_closed_loop_gif(cost, bounds, res, path_rc,
                                              traj_world, obs,
                                              os.path.join(figs, "closed_loop.gif"))
        except Exception as e:
            print(f"[!] GIF 生成失败：{e}")
        closed = {
            "start_cell": [int(start[0]), int(start[1])],
            "goal_cell": [int(goal[0]), int(goal[1])] if goal else None,
            "path_steps": int(len(path_rc)) if path_rc else 0,
            "path_length_m": round(float(sum(c["seg_len"] for c in cmds)), 2),
            "n_cmds": len(cmds),
            "max_v": round(max((c["v"] for c in cmds), default=0), 3),
            "max_abs_omega": round(max((abs(c["omega"]) for c in cmds), default=0), 3),
            "n_avoid_steps": n_avoid,
            "n_dynamic_obstacle_points": int(len(obs)),
        }
        print(f"[·] 闭环：A*={closed['path_steps']}步 | 路径={closed['path_length_m']}m | "
              f"控制={closed['n_cmds']}条 | 避让={n_avoid}步")
    else:
        print("[!] 无 VGGT 轨迹 → 无法规划（前端降级为仅建图）")

    # ---------- metrics ----------
    metrics = {
        "project": "real_driving_closed_loop",
        "title": "真实驾驶数据上的感知→建图→规划→控制端到端闭环",
        "dataset": "KITTI depth_completion (val_selection_cropped, image_02)",
        "drive": args.drive,
        "n_frames": len(frames),
        "scale_factor_lidar_over_vggt": round(scale, 4),
        "perception_frontend": {
            "source": "P4 (scripts/p4_real_driving.py, 复用)",
            "total_real_lidar_points": int(sum(len(p) for p in per)),
            "moving_object_detection": {
                "method": "LiDAR frame-diff with ego-motion compensation (P4)",
                "total_detected_points": int(sum(len(d) for d in dyn_uv)),
            },
        },
        "mapping": {
            "occupancy_shape": list(occ.shape),
            "res_m": res,
            "occupied_cells": int((occ == 2).sum()),
            "free_cells": int((occ == 1).sum()),
            "moving_cells_excluded": int(n_dyn_removed),
        },
        "inflation": {
            "robot_radius_m": args.robot_radius,
            "safety_margin_m": args.safety_margin,
            "inflated_cells": int(n_inflated),
        },
        "closed_loop": closed,
        "figs": {
            "bev_occupancy": os.path.relpath(bev_png, ROOT),
            "costmap_inflated": os.path.relpath(cmap_png, ROOT),
            "map_plan": os.path.relpath(plan_png, ROOT) if plan_png else None,
            "control_cmd": os.path.relpath(ctrl_png, ROOT) if ctrl_png else None,
            "dynamic_avoid": os.path.relpath(avoid_png, ROOT) if avoid_png else None,
            "closed_loop_gif": os.path.relpath(gif_path, ROOT) if gif_path else None,
        },
        "note": ("真实 RGB + 真实 LiDAR 输入（P4 前端）；BEV 占据栅格由真实 LiDAR 累积并剔除移动目标；"
                 "inflation 后 A* 规划 + 差速轮控制 + VO 简化避障。"
                 "自车运动的『执行』为运动学仿真；地图来自视觉估计（无真值位姿），非带真值闭环评测。"),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False, default=str)
    print(f"[✓] metrics → {os.path.join(out_dir, 'metrics.json')}")

    write_readme(out_dir, metrics)
    print(f"[✓] 完成 → {out_dir}")
    return 0


def write_readme(out_dir, m):
    cl = m["closed_loop"]
    lines = []
    lines.append("# P5 · 真实驾驶端到端闭环：感知 → 建图 → 规划 → 控制\n")
    lines.append("> **定位**：把 `P4`（真实驾驶 感知→建图，**前半段**）接上 `P3`/`F6`（规划→控制，**后半段**），")
    lines.append("> 在**同一份真实驾驶数据**上跑通全链路——补上 P4 明确缺的那一环。\n")
    lines.append(f"> 数据：{m['dataset']}（drive `{m['drive']}`，{m['n_frames']} 帧真实驾驶，零下载）。\n")
    lines.append("\n## ① 目的（场景与需求）\n")
    lines.append("- **场景**：真实城市街道驾驶（KITTI），自车需要沿走廊前进并避开对向车/行人。\n")
    lines.append("- **真实问题**：P4 能建出地图，但**地图 ≠ 决策**——原始占据栅格没有膨胀、没有机器人几何约束，")
    lines.append("  不能直接规划；中间缺「inflation → 规划 → 控制」这条链。\n")
    lines.append("- **需求**：① 真实 LiDAR 建图 ② 剔除移动目标（否则地图留幽灵）③ 膨胀成规划用代价地图 ")
    lines.append(" ④ A* 规划 ⑤ 差速轮控制 ⑥ 用真实检出的动态目标做避让。\n")
    lines.append("\n## ② 方法原理（专业）\n")
    lines.append("- **感知前端（复用 P4）**：真实 RGB → VGGT 位姿；真实 LiDAR 深度定标 → 世界点云；KD-tree 帧间差分检出移动目标。\n")
    lines.append("- **建图（本模块）**：世界点云 → 2.5D 占据栅格（地面=可通/障碍=占据），**移动目标格直接剔除**。\n")
    lines.append("- **inflation**：按 `机器人半径 + 安全余量` 形态学膨胀障碍 → 规划用代价地图（**这是「地图能不能导航」的分水岭**）。\n")
    lines.append("- **规划**：A* 在代价地图上求 8 邻域最短路径（起点=自车，目标=前方最远可达点）。\n")
    lines.append("- **控制**：路径 → 差速轮 `(v, ω)` 序列，含 `a_max`/`ω_max` 限幅与转弯降速。\n")
    lines.append("- **避障**：用 P4 检出的真实移动目标点，沿路径前瞻检测 → 触发 VO 式减速（反应式）。\n")
    lines.append("\n## ③ 直白讲解\n")
    lines.append("P4 给了你一张「车走出来的俯视图」，但那只是「照片」——车不能照着照片开。\n")
    lines.append("P5 做三件事：**先把障碍物按车宽「撑胖」一圈**（不然算出来的路会贴着墙，真车过不去）；\n")
    lines.append("**再用 A* 找一条从当前位置到前方的可行路线**；**最后把它翻译成左右轮转速**。\n")
    lines.append("路上遇到 P4 标红的那辆对向车，就减速让行。这就从「看得见」走到了「能行动」。\n")
    lines.append("\n## ④ 真实数据验证 + 效果\n")
    lines.append(f"- 真实 LiDAR 点：**{m['perception_frontend']['total_real_lidar_points']}**；")
    lines.append(f"移动目标点：**{m['perception_frontend']['moving_object_detection']['total_detected_points']}**。\n")
    lines.append(f"- 占据栅格 **{m['mapping']['occupancy_shape']} @ {m['mapping']['res_m']}m**：")
    lines.append(f"障碍格 **{m['mapping']['occupied_cells']}**、可通格 **{m['mapping']['free_cells']}**、")
    lines.append(f"剔除移动格 **{m['mapping']['moving_cells_excluded']}**。\n")
    lines.append(f"- 膨胀（半径 **{m['inflation']['robot_radius_m']}m** + 余量 **{m['inflation']['safety_margin_m']}m**）：")
    lines.append(f"新增不可通行格 **{m['inflation']['inflated_cells']}**。\n")
    if cl:
        lines.append(f"- 闭环：A* **{cl['path_steps']}** 步 / 路径长 **{cl['path_length_m']}m**；")
        lines.append(f"控制 **{cl['n_cmds']}** 条（v_max {cl['max_v']} m/s，ω_max {cl['max_abs_omega']} rad/s）；")
        lines.append(f"避让 **{cl['n_avoid_steps']}** 步；动态目标点 **{cl['n_dynamic_obstacle_points']}**。\n")
    lines.append("\n![占据栅格](figs/bev_occupancy.png)\n")
    lines.append("![膨胀对比](figs/costmap_inflated.png)\n")
    lines.append("![规划](figs/map_plan.png)\n")
    lines.append("![控制指令](figs/control_cmd.png)\n")
    lines.append("![动态避障](figs/dynamic_avoid.png)\n")
    lines.append("\n## ⑤ 与 P4 的分工\n")
    lines.append("| | P4 | P5（本篇） |\n|---|---|---|\n")
    lines.append("| 覆盖 | 感知→定位→建图→可视化（前半段） | 建图→inflation→规划→控制（后半段） |\n")
    lines.append("| 产物 | 地图 + 双视角回放视频 | 代价地图 + A* 路径 + 控制指令 |\n")
    lines.append("| 输入 | 真实 KITTI RGB + LiDAR | **复用 P4 的建图结果** |\n")
    lines.append("\n## ⑥ 工程叙事：讲给三种人听\n")
    lines.append("- **面试官/同行**：强调「**地图 ≠ 决策**」——原始占据栅格必须经过 inflation（机器人几何约束）才能规划；")
    lines.append("  强调「**动态物体不能进静态地图**」（否则留下幽灵障碍）；强调「感知→控制的接口是代价地图 + 运动学约束」。\n")
    lines.append("- **客户**：「真实驾驶数据进去，出来的是一条可执行的路线和轮速指令，还会给对向车让行。」\n")
    lines.append("- **初学者常漏的环节**：① inflation 半径与机器人几何 ② 膨胀后可能「堵死」窄通道 ③ 控制限幅")
    lines.append("  （加速度/角速度超限真车会打滑/翻）④ 地图有漂移但规划局部仍可用 ⑤ 位姿估计误差 ≠ 地图误差。\n")
    lines.append("\n## 如何复现\n")
    lines.append("```bash")
    lines.append("PYTHONPATH=src python scripts/p5_real_driving_closed_loop.py \\")
    lines.append(f"    --drive {m['drive']} --max-frames 30")
    lines.append("```\n")
    lines.append(f"> ⚠️ 诚实边界：{m['note']}\n")
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
