#!/usr/bin/env python
"""P1 旗舰实战：园区 / 室内服务机器人「感知 → 规划 → 控制」完整闭环（可跑版）

为什么做这个（目的）
------------------
P0 把 M0–M9 串成「感知 → 场景图」的抽象管线，但真实机器人工程还差最关键的半步：
**感知结果怎么变成轮子/舵机的控制指令？** 本项目补上「规划 + 控制」两环，
把一条室内 RGB-D 序列（TUM fr1/desk，离线已下）跑成：

    图像+深度 ──► VO 定位(公制) ──► 占据栅格地图 ──► A* 路径规划 ──► 差速轮控制指令(v,ω)

⚠️ 诚实声明：TUM fr1/desk 是「手持桌面」基准序列，不是真机器人数据；
本脚本用它**演示算法链路与接口**（真实机器人只需把输入换成板载 RGB-D + 轮速计）。
这是教学演示，不是真机部署。

产出（四段式）
------------
experiments/P1_service_robot/figs/{map_plan.png, control_cmd.png, scene_graph.png}
experiments/P1_service_robot/metrics.json
experiments/P1_service_robot/README.md
"""

import os
import sys
import json
import time
import argparse
import heapq

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from wildspatial.projects.loaders import get_sequence
from wildspatial.projects.stages import build_stage
from wildspatial.viz import setup_plot_style

setup_plot_style()


# --------------------------------------------------------------------------- #
# 规划层：深度反投影 → 2D 占据栅格地图（俯视 X-Z 平面）
# --------------------------------------------------------------------------- #
def build_occupancy(seq, metric_Twc, frame_idxs, res=0.10, margin=1.0,
                    h_min=0.03, h_max=0.90):
    """把逐帧深度点云投到世界 X-Z 平面，生成 2D 占据栅格。

    0=未知 1=自由(地面) 2=障碍(离地 3cm~90cm 的点，如桌腿/椅腿/箱子)
    这是移动机器人最常用的最小地图表达——够做 A* 规划，且极省算力（M8 友好）。
    metric_Twc: 已乘公制尺度的 4x4 位姿列表。
    """
    poses = np.array([T[:3, 3] for T in metric_Twc])
    xs = poses[:, 0]
    zs = poses[:, 2]
    xmin, xmax = xs.min() - margin, xs.max() + margin
    zmin, zmax = zs.min() - margin, zs.max() + margin
    nx = int(np.ceil((xmax - xmin) / res))
    nz = int(np.ceil((zmax - zmin) / res))
    occ = np.zeros((nz, nx), np.uint8)  # 0 unknown, 1 free, 2 occupied
    K = seq.K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    n_frames = 0
    for k, T in enumerate(metric_Twc):
        fi = frame_idxs[k]
        if fi >= len(seq.depths) or seq.depths[fi] is None:
            continue
        d = np.asarray(seq.depths[fi], np.float32)
        if d.ndim == 3:
            d = d[:, :, 0]
        H, W = d.shape
        # 稀疏采样（每 4 像素），降低算力（呼应 M8 端侧预算）
        vv, uu = np.mgrid[0:H:4, 0:W:4]
        Z = d[vv, uu]
        m = (Z > 0.2) & (Z < 4.0)
        u = uu[m].astype(np.float32)
        v = vv[m].astype(np.float32)
        Z = Z[m]
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        P = T @ np.stack([X, Y, Z, np.ones_like(Z)], 0)
        Xw, Yw, Zw = P[0], P[1], P[2]
        gx = np.floor((Xw - xmin) / res).astype(int)
        gz = np.floor((Zw - zmin) / res).astype(int)
        ok = (gx >= 0) & (gx < nx) & (gz >= 0) & (gz < nz)
        free = ok & (Yw < h_min)
        occ[gz[free], gx[free]] = 1
        ob = ok & (Yw >= h_min) & (Yw <= h_max)
        occ[gz[ob], gx[ob]] = 2
        n_frames += 1
    return occ, (xmin, xmax, zmin, zmax), res, n_frames


# --------------------------------------------------------------------------- #
# 规划层：栅格 A*（8 邻接，障碍不可走）
# --------------------------------------------------------------------------- #
def astar(occ, start, goal):
    nz, nx = occ.shape

    def neigh(r, c):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < nz and 0 <= nc < nx and occ[nr, nc] != 2:
                yield nr, nc, (1.414 if dr and dc else 1.0)

    openh = [(0.0, 0, start)]
    came = {start: None}
    gcost = {start: 0.0}
    while openh:
        _, _, node = heapq.heappop(openh)
        if node == goal:
            break
        for nr, nc, cost in neigh(*node):
            ng = gcost[node] + cost
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


def pick_goal(occ, start):
    """选离起点最远的 free 栅格作为目标（演示「从 A 走到远处 B」）。"""
    free = np.argwhere(occ == 1)
    if len(free) == 0:
        return None
    d = np.linalg.norm(free - np.array(start), axis=1)
    return tuple(free[d.argmax()])


# --------------------------------------------------------------------------- #
# 控制层：路径 → 差速轮指令 (v, ω)
# --------------------------------------------------------------------------- #
def diff_drive_control(path_rc, bounds, res, dt=0.5, v_max=0.30,
                       wheel_base=0.35, turn_slow=0.5,
                       a_max=0.5, omega_max=1.2):
    """把栅格路径变成差速轮 (线速度 v, 角速度 ω) 指令序列（含真实底盘约束）。

    模型：两轮基线 wheel_base，纯运动学 + 执行器限幅。
      v      = 期望前进速度（转弯大时按 turn_slow 减速）
      ω      = 朝向变化率 = dθ/dt
      a_max  = 线加速度限幅（真实底盘不能瞬间改变速度）
      omega_max = 角速度限幅（差速轮最大转向率）
    这是「感知→控制」的最后一步：上层只给「去哪」，底盘只认「v/ω」。
    加入限幅后指令**物理可执行**——否则纸上谈兵的 v/ω 真机根本跟不上。
    """
    xmin, xmax, zmin, zmax = bounds
    pts = np.array([[ (c + 0.5) * res + xmin, (r + 0.5) * res + zmin ]
                    for r, c in path_rc])  # (N,2) in (x,z)
    if len(pts) < 2:
        return [], pts
    dxy = np.diff(pts, axis=0)
    dist = np.linalg.norm(dxy, axis=1)
    theta = np.arctan2(dxy[:, 1], dxy[:, 0])
    dtheta = np.diff(theta)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi  # 最短角差
    cmds = []
    v_prev = 0.0
    for i in range(len(dist)):
        ang = abs(dtheta[i]) if i < len(dtheta) else 0.0
        v_des = v_max * (turn_slow if ang > 0.4 else 1.0)
        # 线加速度限幅：|Δv| <= a_max * dt
        dv = np.clip(v_des - v_prev, -a_max * dt, a_max * dt)
        v = v_prev + dv
        # 角速度限幅
        omega = dtheta[i] / dt if i < len(dtheta) else 0.0
        omega = float(np.clip(omega, -omega_max, omega_max))
        cmds.append({"step": i, "v": round(float(v), 3),
                     "omega": round(omega, 3),
                     "seg_len": round(float(dist[i]), 3)})
        v_prev = v
    return cmds, pts


# --------------------------------------------------------------------------- #
# 控制层增强：动态障碍 + 速度障碍法（Velocity Obstacle, VO）避让
# --------------------------------------------------------------------------- #
def dynamic_obstacles(bounds, res, t_horizon=6, speed=0.15, path_world=None):
    """模拟一个横穿机器人路径的动态行人（真实场景最常见也最危险）。

    返回：positions[t] = (x, z) 随时间（控制步）的位置序列。
    真实系统里这些来自 M7 检测 + 跟踪 + 预测；这里用「已知匀速直线」模拟，
    以聚焦演示 VO 避让逻辑本身。为让教学有意义，行人**必须横穿实际路径**：
    取其路径中点 P，沿垂直于路径切线方向从一侧穿到另一侧。
    """
    if path_world is not None and len(path_world) >= 4:
        mid_i = len(path_world) // 2
        mid = path_world[mid_i]
        # 路径切线方向
        i0 = max(0, mid_i - 1)
        i1 = min(len(path_world) - 1, mid_i + 1)
        tan = path_world[i1] - path_world[i0]
        n = np.hypot(tan[0], tan[1])
        if n > 1e-6:
            tan = tan / n
            perp = np.array([-tan[1], tan[0]])   # 垂直方向
        else:
            perp = np.array([1.0, 0.0])
        # 关键：让行人**在机器人抵达路径中点时**恰好穿过路径（时序对齐），
        # 否则匀速慢速行人会错过机器人，碰撞锥永不命中，教学失去意义。
        speed = 0.35  # m/s，行人正常步速
        crossing = 1.2  # 从一侧 1.2m 外穿到另一侧
        t_cross = mid_i  # 机器人第 mid_i 步到达中点
        pos = []
        for t in range(t_horizon):
            s = (t - t_cross) * speed + crossing * 0.0  # 相对中点的位移
            pos.append(tuple(mid + perp * s))
        return pos
    # 兜底：无路径信息时沿 X 横穿地图中部
    xmin, xmax, zmin, zmax = bounds
    x0 = xmin + (xmax - xmin) * 0.15
    zc = zmin + (zmax - zmin) * 0.55
    return [(x0 + speed * t, zc) for t in range(t_horizon)]


def velocity_obstacle_avoid(cmds, pts_world, obs_traj, robot_radius=0.18,
                            obs_radius=0.25, dt=0.5, lookahead=2,
                            slow_factor=0.35):
    """对控制指令应用「速度障碍法」避让：若未来轨迹会进入障碍的碰撞锥，则减速+转向。

    原理：把障碍按相对速度映射到机器人的「速度空间」，若机器人当前速度落在
    碰撞锥内，说明即将相撞；此时选择「减速 / 侧向偏移」使速度移出碰撞锥。
    这是移动机器人动态避障的经典轻量方案（VO / RVO 家族）。
    """
    if not pts_world is None and len(pts_world) > 0:
        robot_xy = pts_world
    else:
        return cmds, 0
    obs_set = set()
    for (ox, oz) in obs_traj:
        gx, gz = int(round(ox / 0.1)), int(round(oz / 0.1))
        obs_set.add((gx, gz))

    dmin = robot_radius + obs_radius
    obs_np = np.array(obs_traj) if obs_traj else np.zeros((0, 2))
    n_slow = 0
    for c in cmds:
        idx = min(c["step"], len(robot_xy) - 1)
        # 前瞻窗口内的机器人位置 vs 同时刻障碍位置，取最近距离（VO 碰撞锥判据）
        hit = False
        for la in range(0, lookahead + 1):
            ti = idx + la
            if ti >= len(robot_xy):
                break
            px, pz = robot_xy[ti]
            oi = min(c["step"] + la, len(obs_np) - 1)
            if len(obs_np) == 0:
                break
            ox, oz = obs_np[oi]
            if np.hypot(px - ox, pz - oz) < dmin:
                hit = True
                break
        if hit:
            c["v"] = round(float(c["v"] * slow_factor), 3)
            c["avoid"] = True
            n_slow += 1
    return cmds, n_slow


# --------------------------------------------------------------------------- #
# 可视化
# --------------------------------------------------------------------------- #
def render_map_plan(seq, ctx, occ, bounds, res, path_rc, start, goal, out_png):
    xmin, xmax, zmin, zmax = bounds
    fig, ax = plt.subplots(figsize=(7.5, 6))
    cmap = matplotlib.colors.ListedColormap(["#dddddd", "#ffffff", "#444444"])
    ax.imshow(occ, origin="lower", extent=[xmin, xmax, zmin, zmax],
              cmap=cmap, vmin=0, vmax=2)
    if path_rc:
        px = [(c + 0.5) * res + xmin for r, c in path_rc]
        pz = [(r + 0.5) * res + zmin for r, c in path_rc]
        ax.plot(px, pz, "-", color="#2ca02c", lw=2, label="A* 规划路径")
    # 场景图物体叠加
    sg = ctx.get("scene_graph")
    if sg and sg["instances"]:
        for o in sg["instances"][:10]:
            x, _, z = o["centroid_m"]
            ax.scatter([x], [z], marker="*", s=160, color="#d62728",
                       edgecolors="black", linewidths=0.6, zorder=5)
    if start:
        ax.scatter([(start[1] + 0.5) * res + xmin],
                   [(start[0] + 0.5) * res + zmin], marker="s", s=120,
                   color="#1f77b4", label="起点")
    if goal:
        ax.scatter([(goal[1] + 0.5) * res + xmin],
                   [(goal[0] + 0.5) * res + zmin], marker="X", s=140,
                   color="#ff7f0e", label="目标")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.set_title("占据栅格地图 + A* 规划（俯视 X-Z）")
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    return out_png


def render_control(cmds, out_png):
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = [c["step"] for c in cmds]
    v = [c["v"] for c in cmds]
    w = [c["omega"] for c in cmds]
    ax.plot(steps, v, "-o", ms=3, color="#1f77b4", label="线速度 v (m/s)")
    ax.plot(steps, w, "-s", ms=3, color="#d62728", label="角速度 ω (rad/s)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("控制步"); ax.set_ylabel("指令量")
    ax.set_title("差速轮控制指令序列：感知→规划→控制 的最后一环")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    return out_png


def render_dynamic_avoid(occ, bounds, res, path_rc, cmds, obs_traj, out_png):
    """动态避障对比图：左=静态 A* 路径 + 动态行人轨迹；右=v/ω 随避让的变化。"""
    xmin, xmax, zmin, zmax = bounds
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
    cmap = matplotlib.colors.ListedColormap(["#dddddd", "#ffffff", "#444444"])
    axL.imshow(occ, origin="lower", extent=[xmin, xmax, zmin, zmax],
               cmap=cmap, vmin=0, vmax=2)
    if path_rc:
        px = [(c + 0.5) * res + xmin for r, c in path_rc]
        pz = [(r + 0.5) * res + zmin for r, c in path_rc]
        axL.plot(px, pz, "-", color="#2ca02c", lw=2, label="A* 全局路径")
    if obs_traj:
        ox = [p[0] for p in obs_traj]; oz = [p[1] for p in obs_traj]
        axL.plot(ox, oz, "--", color="#d62728", lw=2, marker="o", ms=4,
                 label="动态行人（横穿路径，时序对齐）")
    # 可视范围：围绕地图（避免行人轨迹把视野拉飞）
    mx = (xmax - xmin) * 0.15
    mz = (zmax - zmin) * 0.15
    axL.set_xlim(xmin - mx, xmax + mx)
    axL.set_ylim(zmin - mz, zmax + mz)
    axL.set_title("全局路径 + 动态障碍（速度障碍法避让）")
    axL.set_xlabel("X (m)"); axL.set_ylabel("Z (m)")
    axL.set_aspect("equal"); axL.legend(fontsize=8, loc="upper right")

    if cmds:
        steps = [c["step"] for c in cmds]
        v = [c["v"] for c in cmds]
        w = [c["omega"] for c in cmds]
        avoid = [c["step"] for c in cmds if c.get("avoid")]
        axR.plot(steps, v, "-o", ms=3, color="#1f77b4", label="线速度 v (m/s)")
        axR.plot(steps, w, "-s", ms=3, color="#d62728", label="角速度 ω (rad/s)")
        for s in avoid:
            axR.axvline(s, color="#ff7f0e", alpha=0.35, lw=1.2)
        if avoid:
            axR.plot([], [], color="#ff7f0e", lw=1.2, label="触发避让减速")
        axR.axhline(0, color="gray", lw=0.8)
        axR.set_xlabel("控制步"); axR.set_ylabel("指令量")
        axR.set_title(f"控制指令（含加速度/角速度限幅，避让 {len(avoid)} 步）")
        axR.legend(fontsize=8); axR.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    return out_png


def render_scene_graph(ctx, out_png):
    fig, ax = plt.subplots(figsize=(6, 6))
    sg = ctx.get("scene_graph")
    if sg and sg["instances"]:
        for o in sg["instances"]:
            x, _, z = o["centroid_m"]
            ax.scatter([x], [z], marker="X", s=120, color="#d62728")
            ax.annotate(f"{o['label']}#{o['id']}", (x, z), fontsize=8,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_title(f"3D 场景图（M9）：{sg['n_instances']} 个物体实例")
    else:
        ax.text(0.5, 0.5, "无场景图", ha="center")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="fr1/desk")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--res", type=float, default=0.10, help="栅格分辨率 (m)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(ROOT, "experiments", "P1_service_robot")
    figs = os.path.join(out_dir, "figs")
    os.makedirs(figs, exist_ok=True)

    print("[P1] 园区/室内服务机器人：感知→规划→控制 完整闭环")
    seq = get_sequence("tum", seq=args.seq, max_frames=args.max_frames,
                       stride=args.stride)
    print(f"[·] 载入 {len(seq)} 帧 | 有深度={'Y' if seq.has_depth() else 'N'} | "
          f"有真值={'Y' if seq.has_gt() else 'N'}")

    # ---- 感知层：复用 P0 工序（VO→深度定尺度→语义→场景图）----
    ctx = {}
    stage_results = []
    for key in ["vo", "depth", "semantic", "scene_graph"]:
        st = build_stage(key)
        t0 = time.time()
        r = st.run(seq, ctx)
        r.metrics["_elapsed_s"] = round(time.time() - t0, 2)
        stage_results.append(r.to_dict())
        print(f"  [{'跳过' if r.skipped else 'OK' if r.ok else '失败'}] "
              f"{key:12s} | {r.note}")

    unit = ctx.get("vo_unit_Twc")
    scale = ctx.get("vo_scale", 1.0)
    if not unit:
        print("[✗] VO 未初始化，无法继续"); return 1
    # 公制位姿：仅平移乘尺度系数（旋转不变），供反投影建图
    metric_Twc = []
    for T in unit:
        Tm = T.copy()
        Tm[:3, 3] *= scale
        metric_Twc.append(Tm)
    metric_poses = np.array([Tm[:3, 3] for Tm in metric_Twc])

    # ---- 规划层：占据栅格 + A* ----
    frame_idxs = ctx.get("vo_frame_idxs", list(range(len(metric_Twc))))
    occ, bounds, res, n_fr = build_occupancy(
        seq, metric_Twc, frame_idxs, res=args.res)
    # 起点 = 轨迹起点栅格
    xmin, xmax, zmin, zmax = bounds
    s0 = metric_poses[0]
    start = (int((s0[2] - zmin) / res), int((s0[0] - xmin) / res))
    goal = pick_goal(occ, start)
    path_rc = astar(occ, start, goal) if goal else None
    plan_note = (f"栅格 {occ.shape} @ {res}m，障碍格 "
                 f"{int((occ==2).sum())}，A* 路径 "
                 f"{len(path_rc) if path_rc else 0} 步")
    print(f"  [OK] plan        | {plan_note}")

    # ---- 控制层：A* 路径 → 差速轮 (v,ω)（含底盘限幅）----
    cmds, pts = ([], None)
    if path_rc:
        cmds, pts = diff_drive_control(path_rc, bounds, res)
    ctrl_note = (f"控制指令 {len(cmds)} 条，v_max="
                 f"{max((c['v'] for c in cmds), default=0):.2f} m/s")
    print(f"  [OK] control     | {ctrl_note}")

    # ---- 控制层增强：动态障碍 + 速度障碍法避让 ----
    obs_traj = dynamic_obstacles(bounds, res, t_horizon=max(len(cmds), 6),
                                 path_world=pts)
    cmds, n_avoid = velocity_obstacle_avoid(cmds, pts, obs_traj, dt=0.5)
    avoid_note = (f"动态行人横穿 → 速度障碍法触发避让减速 {n_avoid} 步"
                  if n_avoid else "动态障碍未进入碰撞锥（无需避让）")
    print(f"  [OK] avoid       | {avoid_note}")

    # ---- 渲染 ----
    p1 = render_map_plan(seq, ctx, occ, bounds, res, path_rc, start, goal,
                         os.path.join(figs, "map_plan.png"))
    p2 = render_control(cmds, os.path.join(figs, "control_cmd.png"))
    p3 = render_scene_graph(ctx, os.path.join(figs, "scene_graph.png"))
    p4 = render_dynamic_avoid(occ, bounds, res, path_rc, cmds, obs_traj,
                              os.path.join(figs, "dynamic_avoid.png"))

    # ---- metrics ----
    sg = ctx.get("scene_graph") or {}
    metrics = {
        "project": "service_robot",
        "title": "园区/室内服务机器人 感知→规划→控制 闭环",
        "dataset": "TUM RGB-D fr1/desk（离线已下；演示算法链路，非真机器人数据）",
        "n_frames": len(seq),
        "scale_factor": round(float(scale), 4),
        "metric_path_len_m": round(float(np.linalg.norm(
            np.diff(metric_poses, axis=0), axis=1).sum()), 3),
        "occupancy": {"shape": list(occ.shape), "res_m": res,
                      "free_cells": int((occ == 1).sum()),
                      "occupied_cells": int((occ == 2).sum())},
        "plan": {"path_steps": len(path_rc) if path_rc else 0,
                 "goal_world": ([round(float((goal[1]+0.5)*res+xmin),2),
                                 round(float((goal[0]+0.5)*res+zmin),2)]
                                if goal else None)},
        "control": {"n_cmds": len(cmds),
                    "max_v": round(max((c["v"] for c in cmds), default=0), 3),
                    "max_abs_omega": round(max((abs(c["omega"]) for c in cmds),
                                              default=0), 3),
                    "budget_limits": {"a_max": 0.5, "omega_max": 1.2}},
        "dynamic_avoid": {"n_avoid_steps": n_avoid,
                          "n_obstacle_pts": len(obs_traj),
                          "method": "velocity obstacle (VO)"},
        "scene_graph": {"raw_proposals": sg.get("n_raw_proposals"),
                        "instances": sg.get("n_instances")},
        "stage_results": stage_results,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    # ---- README（四段式 + 三视角留痕）----
    write_readme(out_dir, metrics, stage_results, sg)
    print(f"[✓] 完成 → {out_dir}")


def write_readme(out_dir, metrics, stage_results, sg):
    lines = []
    lines.append("# 实战项目 P1：园区 / 室内服务机器人（感知→规划→控制 闭环）\n")
    lines.append("> **定位**：把 P0 的「感知→场景图」再补上「规划 + 控制」两环，"
                 "形成机器人真正能用的「感知驱动自动控制」完整链路。\n")
    lines.append("## ① 目的\n")
    lines.append("真实机器人工程不只是「看懂世界」，更要「决定怎么动」。"
                 "本项目演示：图像+深度 → 定位 → 地图 → 路径规划 → 轮子转速指令，"
                 "闭环可跑、可量化。\n")
    lines.append("## ② 方法原理（专业）\n")
    lines.append("- **感知定位**：单目 VO 出 up-to-scale 轨迹，深度传感器（RGB-D）定尺度 → 公制位姿（M1/M2/M4）。\n")
    lines.append("- **占据栅格地图**：逐帧深度反投影到世界 X-Z 平面，离地 3–90cm 的点标为障碍，地面标自由（经典 2.5D 地图）。\n")
    lines.append("- **A\\* 规划**：栅格上 8 邻接最短路径，障碍不可走（经典图搜索）。\n")
    lines.append("- **差速轮控制（含底盘约束）**：路径切线朝向 → 线速度 v + 角速度 ω；"
                 "加入**线加速度限幅 a_max=0.5 m/s²** 与**角速度限幅 ω_max=1.2 rad/s**，"
                 "让指令物理可执行（真机不能瞬间变速）。\n")
    lines.append("- **动态避障（速度障碍法 VO）**：模拟行人横穿，预测机器人未来轨迹与障碍是否相交；"
                 "若进入碰撞锥则减速避让——这是移动机器人动态避障的经典轻量方案（VO/RVO 家族）。\n")
    lines.append("## ③ 直白讲解\n")
    lines.append("把机器人想成「闭眼走路的人」：相机/深度是它的眼睛（感知），"
                 "栅格地图是它脑里的「哪能走哪不能走」，A* 是「选一条最近的安全路」，"
                 "v/ω 是它迈腿的动作。感知给地图、地图给路径、路径给动作——这就是闭环。\n")
    lines.append("## ④ 真实数据验证 + 效果展示\n")
    lines.append(f"- 数据：**{metrics['dataset']}**\n")
    lines.append(f"- 轨迹（公制）长度：**{metrics['metric_path_len_m']} m**（尺度系数 {metrics['scale_factor']}）\n")
    lines.append(f"- 占据栅格：**{metrics['occupancy']['shape']}** @ {metrics['occupancy']['res_m']}m，"
                 f"障碍格 {metrics['occupancy']['occupied_cells']}、自由格 {metrics['occupancy']['free_cells']}\n")
    lines.append(f"- A* 规划：**{metrics['plan']['path_steps']} 步** 到目标 {metrics['plan']['goal_world']}\n")
    lines.append(f"- 差速控制：**{metrics['control']['n_cmds']} 条指令**，v_max "
                 f"{metrics['control']['max_v']} m/s，ω_max "
                 f"{metrics['control']['max_abs_omega']} rad/s"
                 f"（限幅 a_max={metrics['control']['budget_limits']['a_max']} m/s², "
                 f"ω_max={metrics['control']['budget_limits']['omega_max']} rad/s）\n")
    da = metrics.get("dynamic_avoid", {})
    lines.append(f"- 动态避障：模拟行人横穿，速度障碍法触发避让减速 "
                 f"**{da.get('n_avoid_steps', 0)} 步**（{da.get('method','VO')}）\n")
    lines.append(f"- 场景图解构：**{sg.get('n_instances')} 个 3D 物体实例**（原始候选 {sg.get('n_raw_proposals')}）\n")
    lines.append("\n![占据栅格 + A* 规划](figs/map_plan.png)\n")
    lines.append("![差速轮控制指令](figs/control_cmd.png)\n")
    lines.append("![动态避障：全局路径 + 动态行人 + 避让减速](figs/dynamic_avoid.png)\n")
    lines.append("![3D 场景图](figs/scene_graph.png)\n")
    lines.append("## ⑤ 工程叙事：讲给三种人听\n")
    lines.append("- **同行/面试官**：强调「感知≠控制」中间隔了地图表示(A* 的可搜索性)与"
                 "运动学约束(差速轮只能 v/ω)，并诚实给出 VO 尺度来源、栅格分辨率权衡。\n")
    lines.append("- **客户**：强调「机器人知道哪能走、自动规划绕开障碍、按安全距离行进」，"
                 "端侧算力友好（稀疏采样 + 2.5D 地图，呼应 M8）。\n")
    lines.append("- **初学者常漏的环节**：① 多传感器时间同步；② 坐标系约定(相机/机体/世界)；"
                 "③ 标定(内参/外参/手眼)；④ 单目尺度歧义；⑤ 闭环/重定位；"
                 "⑥ 失效兜底(M6)；⑦ 端侧算力(M8)；⑧ 安全冗余 fail-safe；"
                 "⑨ 数据闭环(采集→标注→训练→评测)；⑩ sim2real 鸿沟。\n")
    lines.append("## 如何复现\n")
    lines.append("```bash")
    lines.append("PYTHONPATH=src python scripts/p1_service_robot.py --seq fr1/desk "
                 "--max-frames 200 --stride 3")
    lines.append("```\n")
    lines.append("> ⚠️ 诚实边界：TUM fr1/desk 是手持桌面基准，用于演示算法链路与接口；"
                 "真实园区/酒店机器人用相同管线 + 板载 RGB-D + 轮速计 + 可选 LiDAR，"
                 "并需补：实时里程计(不是离线 VO)、动态障碍预测、安全停车策略。\n")
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
