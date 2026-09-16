#!/usr/bin/env python
"""P2 · 第一视角（FPV）送货演示 —— 机器人去固定地点送货，路上遇到各种情况

为什么做这个（背景）
-------------------
基础版 `p2_dynamic_obstacle_gif.py` 是**俯视上帝视角**，能看清"谁在哪、怎么动"，
但始终像一张俯视图，缺乏"机器人自己在走路、自己看世界"的临场感。
用户要求做一个**机器人第一视角（FPV）的 3D 画面**：机器人去一个**固定地点送货**，
途中遇到**各种真实路况**——这正是真实配送机器人（园区/酒店送物）每天面对的场景。

本脚本的做法（方案 B：纯合成 3D，零 GPU 依赖）
------------------------------------------------
· 不依赖 Gazebo GPU 渲染（本机 `/dev/dri/renderD128` 无权限，headless 相机输出灰屏）。
· **自研轻量软件渲染器**：针孔相机模型 + 透视投影 + 画家算法（远→近排序覆盖），
  在普通 CPU 上把 3D 世界（地面网格 / 墙 / 障碍 / 行人 / 目标建筑）投影成
  **机器人眼里的 2D 画面**。效果与真实相机透视一致（近大远小、汇聚到地平线）。
· 机器人沿**固定路线（航点表）**前往**唯一送达点（固定地点）**；途中按位置触发
  6 类工况并做出符合常理的反应（减速 / 停车让行 / 靠右避让 / 盲区减速）。

与"真实无人配送车"的关系（用户原问）
------------------------------------
原理**同源**：都是「感知（看到障碍/行人）→ 决策（减速/让行）→ 控制（v/ω）」的闭环。
真实车只是**传感器更强（激光雷达+相机+GPU 实时建图）、模型更复杂（学习式预测、
多传感器融合）、规则更细（交规/礼让策略）**。本项目用几何+规则把**第一性原理**
跑通、做可视化，正是理解真实系统的入口。

6 类路况（沿路线依次出现）
-------------------------
① 静态障碍（路边停着的快递推车）→ 小幅绕行
② 横向穿行行人（斑马线式横穿）→ 停车等其通过
③ 对向行人（窄通道里迎面走来）→ 靠右礼让、减速错车
④ 视觉盲区（拐角墙挡住视线）→ 拐角前减速
⑤ 窄通道（两面墙夹着走）→ 居中精准通行
⑥ 送达点（终点建筑，先被遮挡后现身）→ 到达、完成送货

产出
----
experiments/P2_delivery_fpv/
    figs/delivery_fpv.gif    第一视角送货全过程（3D 透视 + HUD + 小地图）
    figs/delivery_fpv.mp4    同内容视频
    figs/keyframes.png       关键帧拼接（6 类工况各一帧）
    metrics.json             逐帧数据（位姿/速度/距离/工况）
"""

import os
import json
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
# 项目中文字体：本机装了 Noto / 文泉驿，避免 HUD/标题中文显示成方框
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei",
                                          "DejaVu Sans"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "P2_delivery_fpv")
FIGS = os.path.join(OUT, "figs")
FRAMES = os.path.join(OUT, "gif_frames")

# ===== 相机内参（像素）=====
W, H = 640, 360
CX, CY = W / 2.0, H / 2.0
HFOV = 75.0  # 水平视场角（度）
F = (W / 2.0) / math.tan(math.radians(HFOV / 2.0))
NEAR = 0.05
CAM_H = 0.55  # 相机离地高度（配送机器人典型相机高度）

# ===== 机器人运动 =====
V_MAX = 1.1        # 巡航速度 (m/s)
DT = 0.03          # 仿真步长 (s)
RECORD_DT = 0.20   # 渲染采样间隔 (s)


# ===================================================================
# 世界几何：盒子（墙 / 障碍 / 建筑）与行人（billboard）
# ===================================================================
class Box:
    """轴对齐立方体；按面投影 + 简单方向光照，得到有立体感的方块。"""

    def __init__(self, center, size, color):
        self.c = np.array(center, float)
        self.s = np.array(size, float)
        self.color = np.array(color, float)  # 0-1 RGB

    def faces(self):
        cx, cy, cz = self.c
        hx, hy, hz = self.s / 2
        c = [  # 8 个角
            (cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
            (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
            (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
            (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz),
        ]
        faces = [
            ([0, 1, 2, 3], (0, 0, -1)),   # 底
            ([4, 5, 6, 7], (0, 0, 1)),    # 顶
            ([0, 3, 7, 4], (-1, 0, 0)),   # -X
            ([1, 2, 6, 5], (1, 0, 0)),    # +X
            ([0, 1, 5, 4], (0, -1, 0)),   # -Y
            ([3, 2, 6, 7], (0, 1, 0)),    # +Y
        ]
        out = []
        for idx, n in faces:
            pts = [c[i] for i in idx]
            out.append((pts, self.color, n))
        return out


class Pedestrian:
    """站在地面的行人，用朝向相机的 billboard（矩形身体 + 头圆）表示。"""

    def __init__(self, x, y, color=(0.95, 0.55, 0.15)):
        self.x = x
        self.y = y
        self.color = np.array(color, float)

    def drawables(self, r):
        """返回 (depth, kind, ...) 列表；r 为相机右向量（billboard 朝向相机）。"""
        px, py = self.x, self.y
        bc = np.array([px, py, 0.0])
        tc = np.array([px, py, 1.55])  # 身体顶
        w = 0.26
        bl = bc - w * r
        br = bc + w * r
        tr = tc + w * r
        tl = tc - w * r
        body = [bl, br, tr, tl]
        head_c = np.array([px, py, 1.66])
        return [("body", body, self.color), ("head", head_c, 0.16)]


# ===================================================================
# 针孔相机投影（世界 → 像素）
# ===================================================================
def camera_basis(yaw):
    """相机在 (rx,ry,CAM_H)，朝向 yaw；返回右/上/前向量。"""
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)  # = (sinθ, -cosθ, 0)
    return right, up, fwd


def project(P, C, right, up, fwd):
    d = np.array(P, float) - C
    xc = float(d @ right)
    yc = float(d @ up)
    zc = float(d @ fwd)
    if zc <= NEAR:
        return None
    u = CX + F * xc / zc
    v = CY + F * yc / zc   # 屏幕下=0（origin lower），高处 v 大 → 向上
    return (u, v, zc)


# ===================================================================
# 背景（天空渐变 + 地面底色），视场固定故只算一次
# ===================================================================
def make_background():
    sky_top = np.array([0.42, 0.62, 0.92])
    horizon = np.array([0.80, 0.88, 0.95])
    ground_far = np.array([0.62, 0.64, 0.55])
    ground_near = np.array([0.46, 0.48, 0.42])
    bg = np.zeros((H, W, 3))
    vh = int(CY)
    for v in range(H):
        if v >= vh:  # 天空（上半，v 大）
            t = (v - vh) / max(1, (H - vh))
            bg[v] = horizon * (1 - t) + sky_top * t
        else:        # 地面（下半，v 小）
            t = (vh - v) / max(1, vh)
            bg[v] = horizon * (1 - t) + ground_near * t
            # 远处(far, t→0)偏 ground_far
            bg[v] = ground_far * (1 - t) + ground_near * t
    return bg


# ===================================================================
# 世界定义
# ===================================================================
def build_world():
    obstacles = [
        # ① 静态障碍：路边停着的快递推车（偏离路线中心，需小幅绕行）
        Box([3.4, 1.15, 0.4], [1.0, 1.0, 0.8], [0.30, 0.55, 0.35]),
        # ⑤ 窄通道：两面长墙（x≈11.2 / 12.8，y 0..10），通道宽约 1.6m
        Box([11.2, 5.0, 1.0], [0.3, 10.0, 2.0], [0.55, 0.55, 0.58]),
        Box([12.8, 5.0, 1.0], [0.3, 10.0, 2.0], [0.55, 0.55, 0.58]),
        # ④ 视觉盲区：拐角内侧挡墙（机器人上坡到顶、转弯前看不到西侧）
        Box([10.9, 10.6, 1.0], [2.2, 0.3, 2.0], [0.50, 0.45, 0.42]),
        # ⑥ 送达点建筑（终点，固定地点）：改小、让机器人停在正面 2m 外
        Box([0.0, 10.0, 1.2], [2.2, 2.2, 2.4], [0.35, 0.45, 0.65]),
        # 装饰：终点旁的小屋
        Box([-2.2, 8.5, 0.9], [1.6, 1.6, 1.8], [0.60, 0.50, 0.45]),
    ]
    # 路线（航点表）：起点(0,0) → 东行 → 北行(窄通道) → 西行 → 送达(0,10)
    # 已把"靠右礼让"的横向偏移直接烤进航点（通道内 x=12.5 贴右墙）。
    route = np.array([
        [0.0, 0.0], [6.0, 0.0], [8.0, 0.0], [11.0, 0.0],
        [12.0, 0.0], [12.0, 2.0], [12.5, 4.0], [12.5, 6.0],
        [12.0, 8.0], [12.0, 10.0], [10.0, 10.0], [6.0, 10.0],
        [2.0, 10.0], [3.0, 10.0],   # 终点停在建筑东立面约 2m 外
    ], float)
    goal = np.array([3.0, 10.0])
    peds = {
        # ② 横向穿行行人：在 x=8 横穿路线（斑马线式）
        "cross": {"x": 8.0, "y": -3.2, "active": False, "done": False},
        # ③ 对向行人：在窄通道里从北向南迎面走来
        "oncoming": {"x": 11.6, "y": 8.6, "active": False, "done": False},
    }
    return obstacles, route, goal, peds


# ===================================================================
# 沿折线参数化前进（保证必达送达点，且航向=切线）
# ===================================================================
def route_geometry(route):
    seg = np.diff(route, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    total = cum[-1]
    return seg, seg_len, cum, total


def pose_at(route, seg, seg_len, cum, total, s):
    s = min(max(s, 0), total)
    i = np.searchsorted(cum, s, side="right") - 1
    i = min(i, len(seg) - 1)
    local = (s - cum[i]) / seg_len[i] if seg_len[i] > 1e-9 else 0.0
    pos = route[i] + seg[i] * local
    yaw = math.atan2(seg[i][1], seg[i][0])
    return pos, yaw


# ===================================================================
# 仿真：推进机器人 + 行人，产出逐帧状态
# ===================================================================
def simulate(obstacles, route, goal, peds):
    seg, seg_len, cum, total = route_geometry(route)
    s = 0.0
    t = 0.0
    frames = []
    last_rec = -1.0
    while s < total - 0.05 and t < 90.0:
        pos, yaw = pose_at(route, seg, seg_len, cum, total, s)
        rx, ry = float(pos[0]), float(pos[1])

        # ---- 工况判定 ----
        scenario = "巡航送货"
        v = V_MAX

        # ① 静态障碍：接近推车(x≈3.4)时轻微减速（已偏离中心，无需停车）
        if 2.4 < rx < 4.6 and abs(ry - 1.15) < 1.6:
            scenario = "① 静态障碍·绕行"
            v = min(v, 0.8)

        # ② 横向穿行行人：机器人接近 x=5 时行人开始横穿；
        #    若行人还在路面上且离机器人很近 → 停车让行
        cr = peds["cross"]
        if rx > 2.5 and not cr["done"]:
            cr["active"] = True
        if cr["active"] and not cr["done"]:
            cr["y"] += 0.55 * DT
            if cr["y"] > 3.4:
                cr["y"] = 3.4
                cr["done"] = True
        ped_near_cross = (abs(rx - 5.0) < 1.5 and abs(cr["y"]) < 1.4)
        if ped_near_cross:
            scenario = "② 横向行人·停车让行"
            v = 0.0

        # ③ 对向行人（窄通道）：机器人进入通道且对向行人活动时减速错车
        oc = peds["oncoming"]
        in_corridor = (11.0 < rx < 13.0 and 0.0 < ry < 10.0)
        if in_corridor and ry > 2.0 and not oc["done"]:
            oc["active"] = True
        if oc["active"] and not oc["done"]:
            oc["y"] -= 0.42 * DT
            if oc["y"] < 2.0:
                oc["y"] = 2.0
                oc["done"] = True
        if in_corridor and oc["active"] and (ry - oc["y"]) < 4.0 and (ry - oc["y"]) > -1.0:
            scenario = "③ 对向行人·靠右礼让"
            v = min(v, 0.45)

        # ④ 视觉盲区：接近拐角(12,10)前减速
        if in_corridor and ry > 8.3:
            scenario = "④ 视觉盲区·拐角减速"
            v = min(v, 0.5)

        # ⑤ 窄通道：通道内居中通行（已在航点烤入偏移），标注工况
        if in_corridor and scenario.startswith("巡航"):
            scenario = "⑤ 窄通道·居中通行"

        # ⑥ 送达点：最后 2m 减速停靠
        dgoal = math.hypot(rx - goal[0], ry - goal[1])
        if dgoal < 2.0:
            scenario = "⑥ 送达点·到达"
            v = min(v, 0.35)

        # ---- 推进 ----
        s += v * DT
        t += DT

        if t - last_rec >= RECORD_DT or not frames:
            last_rec = t
            frames.append({
                "t": round(t, 2), "rx": rx, "ry": ry, "yaw": yaw,
                "v": v, "dgoal": dgoal, "scenario": scenario,
                "peds": {k: (peds[k]["x"], peds[k]["y"]) for k in peds},
            })
    return frames


# ===================================================================
# 渲染一帧 FPV（含 HUD + 小地图）
# ===================================================================
def render_frame(frame, obstacles, route, goal, peds, bg, fidx):
    rx, ry, yaw = frame["rx"], frame["ry"], frame["yaw"]
    C = np.array([rx, ry, CAM_H])
    right, up, fwd = camera_basis(yaw)

    fig = plt.figure(figsize=(9.2, 5.2), dpi=92)
    ax = fig.add_axes([0.0, 0.0, 0.72, 1.0])
    ax.imshow(bg, extent=[0, W, 0, H], origin="lower", aspect="auto")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ---- 收集可绘制体 ----
    draw = []  # (depth, type, data)

    # 地面网格（透视汇聚到地平线）：X 线 / Y 线
    for gx in range(-6, 31):
        pts = [(gx, y, 0.0) for y in np.linspace(-8, 22, 40)]
        draw.append((9999, "line", (pts, (0.32, 0.34, 0.30), 0.6)))
    for gy in range(-8, 23):
        pts = [(x, gy, 0.0) for x in np.linspace(-6, 30, 40)]
        draw.append((9999, "line", (pts, (0.32, 0.34, 0.30), 0.6)))

    # 盒子（墙/障碍/建筑）各面
    LIGHT = np.array([0.45, 0.5, 0.85])
    LIGHT = LIGHT / np.linalg.norm(LIGHT)
    for ob in obstacles:
        for pts, color, n in ob.faces():
            proj = [project(p, C, right, up, fwd) for p in pts]
            if any(p is None for p in proj):
                continue
            zs = [p[2] for p in proj]
            shade = 0.45 + 0.55 * max(0.0, float(np.dot(n, LIGHT)))
            fc = np.clip(color * shade, 0, 1)
            draw.append((np.mean(zs), "poly",
                         ([(p[0], p[1]) for p in proj], fc)))

    # 行人 billboard
    for k, (px, py) in frame["peds"].items():
        ped = Pedestrian(px, py)
        for item in ped.drawables(right):
            if item[0] == "body":
                proj = [project(p, C, right, up, fwd) for p in item[1]]
                if any(p is None for p in proj):
                    continue
                zs = [p[2] for p in proj]
                draw.append((np.mean(zs), "poly",
                             ([(p[0], p[1]) for p in proj], item[2])))
            else:  # head
                p = project(item[1], C, right, up, fwd)
                if p is None:
                    continue
                r_pix = 0.17 * F / p[2]
                draw.append((p[2], "circle", (p[0], p[1], r_pix,
                                              (0.95, 0.80, 0.65))))

    # ---- 画家算法：远→近 ----
    draw.sort(key=lambda d: d[0], reverse=True)
    for d in draw:
        if d[1] == "line":
            pts, col, lw = d[2]
            us, vs = [], []
            for p in pts:
                pr = project(p, C, right, up, fwd)
                if pr is not None:
                    us.append(pr[0]); vs.append(pr[1])
            if len(us) > 1:
                ax.plot(us, vs, color=col, lw=lw, alpha=0.55, zorder=1)
        elif d[1] == "poly":
            xy, fc = d[2]
            ax.add_patch(Polygon(xy, closed=True, facecolor=fc,
                                 edgecolor="0.15", lw=0.6, zorder=2))
        elif d[1] == "circle":
            u, v, r, fc = d[2]
            ax.add_patch(Circle((u, v), r, facecolor=fc,
                                edgecolor="0.15", lw=0.5, zorder=3))

    # ---- HUD：工况横幅 ----
    ax.text(0.012, 0.95, frame["scenario"], transform=ax.transAxes,
            fontsize=14, weight="bold", color="white", va="top",
            bbox=dict(facecolor="#d62728", alpha=0.92, boxstyle="round,pad=0.4"))
    # 底部状态条
    info = (f"速度 v = {frame['v']:.2f} m/s\n"
            f"到送达点 = {frame['dgoal']:.1f} m\n"
            f"t = {frame['t']:.1f} s")
    ax.text(0.012, 0.05, info, transform=ax.transAxes, fontsize=11,
            color="white", va="bottom",
            bbox=dict(facecolor="black", alpha=0.55, boxstyle="round,pad=0.35"))

    # ---- 小地图（右上角 inset）----
    axm = fig.add_axes([0.745, 0.60, 0.235, 0.36])
    rmin = route.min(0); rmax = route.max(0)
    m = 2.0
    axm.plot(route[:, 0], route[:, 1], ":", color="#9467bd", lw=1.3,
             label="送货路线")
    axm.scatter([goal[0]], [goal[1]], marker="*", s=200, color="#9467bd",
                edgecolors="black", zorder=5, label="送达点")
    # 障碍盒俯视
    for ob in obstacles:
        cx, cy, _ = ob.c; sx, sy, _ = ob.s
        axm.add_patch(plt.Rectangle((cx - sx/2, cy - sy/2), sx, sy,
                                    facecolor="0.5", edgecolor="0.2", alpha=0.8))
    for k, (px, py) in frame["peds"].items():
        axm.scatter([px], [py], c="#d62728", s=40, zorder=6)
    axm.scatter([rx], [ry], marker="s", s=70, c="#1f77b4",
                edgecolors="black", zorder=7, label="机器人")
    axm.arrow(rx, ry, 0.5*math.cos(yaw), 0.5*math.sin(yaw),
              color="#1f77b4", head_width=0.25, zorder=8)
    axm.set_xlim(rmin[0]-m, rmax[0]+m)
    axm.set_ylim(rmin[1]-m, rmax[1]+m)
    axm.set_aspect("equal")
    axm.set_title("小地图", fontsize=9)
    axm.tick_params(labelsize=6)
    axm.grid(alpha=0.25)

    # 标题
    ax.text(0.5, 1.02, "P2 · 配送机器人第一视角（FPV）送货演示",
            transform=ax.transAxes, fontsize=12, weight="bold",
            ha="center", va="bottom")

    path = os.path.join(FRAMES, f"f{fidx:04d}.png")
    fig.savefig(path, dpi=92)
    plt.close(fig)


# ===================================================================
# GIF / MP4 / 关键帧拼接
# ===================================================================
def make_gif(out_dir, gif_path, fps=12):
    from PIL import Image
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
    imgs = [Image.open(os.path.join(out_dir, f)).convert("P") for f in files]
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                 duration=int(1000/fps), loop=0, optimize=True)
    return True


def make_mp4(out_dir, mp4_path, fps=12):
    code = os.system(
        f"ffmpeg -y -framerate {fps} -pattern_type glob -i '{out_dir}/f*.png' "
        f"-c:v libx264 -pix_fmt yuv420p -vf scale=1190:-2 {mp4_path}")
    return code == 0 and os.path.exists(mp4_path)


def make_keyframes(frames, obstacles, route, goal, peds, bg, selectors, path):
    targets = []
    for label, score_fn in selectors:
        cands = [f for f in frames if f["scenario"] == label]
        if not cands:
            # 回退：按编号前缀匹配
            cands = [f for f in frames if f["scenario"].startswith(label[:2])]
        if cands:
            best = max(cands, key=score_fn)
            targets.append((label, best))
    if not targets:
        return False
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    for ax, (label, f) in zip(axes.ravel(), targets):
        C = np.array([f["rx"], f["ry"], CAM_H])
        right, up, fwd = camera_basis(f["yaw"])
        ax.imshow(bg, extent=[0, W, 0, H], origin="lower", aspect="auto")
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
        draw = []
        for ob in obstacles:
            for pts, color, n in ob.faces():
                proj = [project(p, C, right, up, fwd) for p in pts]
                if any(p is None for p in proj):
                    continue
                zs = [p[2] for p in proj]
                shade = 0.5 + 0.5*max(0.0, float(np.dot(n, [0.45,0.5,0.85])/1.18))
                fc = np.clip(color*shade, 0, 1)
                draw.append((np.mean(zs), "poly", ([(p[0],p[1]) for p in proj], fc)))
        for k, (px, py) in f["peds"].items():
            ped = Pedestrian(px, py)
            for item in ped.drawables(right):
                if item[0] == "body":
                    proj = [project(p, C, right, up, fwd) for p in item[1]]
                    if any(p is None for p in proj):
                        continue
                    zs = [p[2] for p in proj]
                    draw.append((np.mean(zs), "poly", ([(p[0],p[1]) for p in proj], item[2])))
        draw.sort(key=lambda d: d[0], reverse=True)
        for d in draw:
            if d[1] == "poly":
                xy, fc = d[2]
                ax.add_patch(Polygon(xy, closed=True, facecolor=fc,
                                     edgecolor="0.15", lw=0.6))
        ax.set_title(label, fontsize=11, weight="bold")
    fig.suptitle("P2 · 第一视角送货 · 六类路况关键帧", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=95)
    plt.close(fig)
    return True


def main():
    os.makedirs(FRAMES, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    print("[P2·FPV 送货] 构建世界 + 仿真 + 渲染第一视角...")

    obstacles, route, goal, peds = build_world()
    frames = simulate(obstacles, route, goal, peds)
    print(f"[OK] 仿真 {len(frames)} 帧（RECORD_DT={RECORD_DT}s）")

    bg = make_background()
    for i, f in enumerate(frames):
        render_frame(f, obstacles, route, goal, peds, bg, i)
    print(f"[✓] 渲染 {len(frames)} 张 FPV")

    gif = os.path.join(FIGS, "delivery_fpv.gif")
    make_gif(FRAMES, gif)
    print(f"[OK] GIF → {gif} ({os.path.getsize(gif)/1e6:.1f} MB)")

    mp4 = os.path.join(FIGS, "delivery_fpv.mp4")
    if make_mp4(FRAMES, mp4):
        print(f"[OK] MP4 → {mp4} ({os.path.getsize(mp4)/1e6:.1f} MB)")
    else:
        print("[✗] MP4 生成失败（ffmpeg）")

    kf = os.path.join(FIGS, "keyframes.png")
    selectors = [
        ("① 静态障碍·绕行", lambda f: -math.hypot(f["rx"]-3.4, f["ry"]-1.15)),
        ("② 横向行人·停车让行", lambda f: -abs(f["peds"]["cross"][1])),
        ("③ 对向行人·靠右礼让", lambda f:
            -math.hypot(f["rx"]-f["peds"]["oncoming"][0],
                        f["ry"]-f["peds"]["oncoming"][1])),
        ("④ 视觉盲区·拐角减速", lambda f: -abs(f["ry"]-9.6)),
        ("⑤ 窄通道·居中通行", lambda f: -abs(f["rx"]-12.0)),
        ("⑥ 送达点·到达", lambda f: -abs(f["dgoal"]-1.5)),
    ]
    make_keyframes(frames, obstacles, route, goal, peds, bg, selectors, kf)
    print(f"[OK] 关键帧 → {kf}")

    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(frames, fh, indent=1)
    print(f"[✓] 完成 → {OUT}")


if __name__ == "__main__":
    main()
