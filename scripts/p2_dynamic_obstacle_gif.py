#!/usr/bin/env python
"""P2 · 动态障碍闭环 —— GIF 动图（让"运动"看得见）

为什么做这个
------------
静态图只能展示"某一刻"，看不出**机器人怎么走、行人怎么动、避障怎么响应**。
本脚本在跑闭环的同时**逐帧渲染"世界俯视图"**（机器人+行人+轨迹+状态），
合成 **GIF 动图**，让整个过程"动起来"。

就像看比赛回放：**俯视上帝视角**（谁在哪、怎么动）+ 状态条（前往目标/避让），
一眼看懂"感知→决策→控制"。

机器人在干什么（重要）
----------------------
**航点追逐（waypoint pursuit）**：机器人沿固定航点表（八边形环线）依次前进，
紫星 = **当前正在追逐的目标点**，控制律是 `atan2(goal - robot)`（**直接朝目标转**），
距目标 < ARRIVE(0.28m) 即**切换下一航点**；遇行人则在"朝目标"基础上叠加绕行偏置。
→ 图上可见：蓝色朝向箭头指向紫星；HUD 显示"到目标距离 + 朝向误差"（巡航时误差≈0°）。

> 历史教训：早期版本跟的是"环线解析切线"（时间驱动），**控制律里没用目标点**，
> 导致"机器人不朝紫星走、紫星自己转圈"（用户一眼看穿）。这正是
> **轨迹跟踪 vs 目标点导航**两个范式的混淆 —— 详见 `docs/P2_simulation.md §4.14.3b 修复三`。

产出
----
experiments/P2_dynamic_obstacle/
    figs/  closed_loop.gif    完整闭环动图（俯视 + 状态）
           closed_loop.mp4    同内容视频（若 ffmpeg 可用）
    traj_rich.npy             逐帧数据 [t, rx, ry, dmin, v, w, gx, gy, theta]

诚实声明
--------
· 俯视图由**仿真真值位姿**渲染（非相机画面）；相机画面对比见 §4.14 sensor_view。
· 机器人位置用**脚本运动学积分**（保证演示轨迹清晰），也向 Gazebo 发 cmd_vel；
  行人位置读 Gazebo **真值**。
· GIF 帧率高会体积大，本脚本控制在合理体积（~几 MB）。
"""

import os
import sys
import json
import time
import signal
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from wildspatial.viz import setup_plot_style
setup_plot_style()

CONDA_SH = "/home/hmn-cjy/miniforge3/etc/profile.d/conda.sh"
ENV = "ros2jazzy"
OUT = os.path.join(ROOT, "experiments", "P2_dynamic_obstacle")
FIGS = os.path.join(OUT, "figs")
FRAMES = os.path.join(OUT, "gif_frames")
WURDF = os.path.join(ROOT, "data", "gz_models", "urdf", "gz_waffle.sdf.xacro")
WORLD = os.path.join(ROOT, "data", "gz_models", "worlds", "open_plaza_dynamic.sdf.xacro")
PROCS = []
# 巡逻航点（半径 1.5m 的八边形环线）——渲染与采集共用同一份定义
WAYPOINTS = [[1.5, 0.0], [1.06, 1.06], [0.0, 1.5], [-1.06, 1.06],
             [-1.5, 0.0], [-1.06, -1.06], [0.0, -1.5], [1.06, -1.06]]
ARRIVE = 0.28
# ===== 分层安全区（比例避障）=====
# 半径关系：WARN > DANGER > STOP > 碰撞距离(0.44m)
V_MAX = 0.28            # 巡航速度 (m/s)
WARN = 1.2              # 预警圈：进入即开始比例减速 + 转向
DANGER = 0.85           # 危险圈：状态转红（可视 + HUD）
STOP = 0.50             # 停止圈：硬约束 v=0（0.50 > 0.44 碰撞距离，留安全余量）


def sh(cmd, timeout=60):
    pre = f"source {CONDA_SH} && conda activate {ENV} && "
    try:
        r = subprocess.run(["bash", "-lc", pre + cmd], capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def spawn(cmd, logfile, env=None):
    pre = f"source {CONDA_SH} && conda activate {ENV} && "
    if env:
        pre += " && ".join(f"export {k}={v}" for k, v in env.items()) + " && "
    f = open(logfile, "w")
    p = subprocess.Popen(["bash", "-lc", pre + cmd], stdout=f,
                         stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    PROCS.append(p)
    return p


def cleanup():
    for p in PROCS:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            pass
    time.sleep(2)


def run_capture(dur=30, fps_save=12):
    """跑闭环并记录逐帧的机器人/行人位置 + 控制指令 + 状态。"""
    script = r'''
import json, time
import numpy as np
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist

OUT = "%s"; T = %d; FS = %d
# 巡逻航点 + 到达半径（与主进程共用同一份定义，避免两处不一致）
WAYPOINTS = %s
ARRIVE = %s
# 分层安全区（比例避障）——与主进程共用
V_MAX = %s; WARN = %s; DANGER = %s; STOP = %s
# 行人场内往复的边界（|y| 超过就反向），留足离墙余量避免撞墙穿透抖动
PED_LIMIT = 3.0
PED_SPEED = 0.35

node = Node()
poses = {"robot": None, "peds": {}}

def on_pose(m):
    for p in m.pose:
        if "turtlebot3_waffle" in p.name:
            poses["robot"] = [p.position.x, p.position.y]
        elif p.name.startswith("pedestrian"):
            poses["peds"][p.name] = [p.position.x, p.position.y]

node.subscribe(Pose_V, "/world/default/pose/info", on_pose)
pub = node.advertise("cmd_vel", Twist)

# ---- 行人往复控制：越界即反转其速度，让行人场内来回穿行 ----
# 背景：世界上行人由 VelocityControl 以恒速滑行（mu=0 无摩擦）。
#       若不管它，会一路滑到外墙 → 撞墙穿透 → 位置瞬移（"跳来跳去"）。
#       这里按真值位置做**边界反射**：到达 ±PED_LIMIT 就发反向 cmd_vel。
ped_pub = {}
for nm in ("pedestrian", "pedestrian2"):
    ped_pub[nm] = node.advertise("/model/" + nm + "/cmd_vel", Twist)
ped_sign = {"pedestrian": 1.0, "pedestrian2": -1.0}   # 初始方向（与 sdf 一致）
ped_flip_t = {}                                        # 上次折返时间（防抖）

def drive_peds(now):
    for nm, xy in poses["peds"].items():
        if nm not in ped_pub:
            continue
        y = xy[1]
        if abs(y) > PED_LIMIT and now - ped_flip_t.get(nm, -99) > 0.4:
            ped_sign[nm] = -1.0 if y > 0 else 1.0
            ped_flip_t[nm] = now
        # 保持 x 位置（不横移），只给 Y 向速度
        tw = Twist(); tw.linear.y = ped_sign[nm] * PED_SPEED
        ped_pub[nm].publish(tw)

frames = []
t0 = time.time(); last_save = 0

# ---- 航点追逐（waypoint pursuit）：机器人**真的朝当前航点走** ----
# 说明：这里机器人位置由**脚本运动学积分**得到（不靠 Gazebo 物理推），
#       目的是让演示**轨迹清晰可控**（Gazebo 物理在窄空间易卡死）。
#       行人位置仍读 Gazebo **真值**（它在真实物理世界里运动）。
#
# 与旧版的区别（重要）：
#   旧版跟的是**环线解析切线**（时间函数），控制律里根本没用到目标点 →
#   机器人不朝紫星走、紫星自己转圈，两者毫无因果关系（用户一眼看出问题）。
#   新版是**目标点导航**：朝向 = atan2(goal - robot)，到达即切换下一航点。
#   （WAYPOINTS / ARRIVE 由主进程通过格式化参数注入，保证与渲染端一致。）
ped_vel = {}            # 行人速度估计（帧间差分），用于**提前预测**其未来位置
ped_prev = {}

robot_xy = None
robot_theta = 0.0
avoid_until = 0.0
avoid_sign = 1.0
wp_idx = 0              # 当前追逐的航点索引（紫星 = WAYPOINTS[wp_idx]）

while time.time() - t0 < T:
    now = time.time() - t0
    r = poses["robot"]
    # 行人往复控制（边界反射，避免撞墙瞬移）
    drive_peds(now)
    # 行人速度估计（帧间差分）→ 用于提前预测未来位置
    for nm, xy in poses["peds"].items():
        if nm in ped_prev:
            px, py = ped_prev[nm]
            ped_vel[nm] = [(xy[0]-px) / 0.02, (xy[1]-py) / 0.02]   # 20ms 控制周期
        ped_prev[nm] = list(xy)

    # 行人真值位置（用**预测位置**做避障，实现提前绕行）
    # 旧版用当前距离：等贴脸才反应 → "停车后行人仍撞上来"（实测 dmin=0.29m）
    # 新版用**前瞻位置**：行人 1 秒后会到哪 → 提前让开
    dmin = None; ped_xy = None; ped_pred = None
    LOOKAHEAD = 0.8         # 前瞻时间 (s)
    if r is not None and poses["peds"]:
        cand = []
        for nm, p in poses["peds"].items():
            vx, vy = ped_vel.get(nm, [0.0, 0.0])
            pp = [p[0] + vx*LOOKAHEAD, p[1] + vy*LOOKAHEAD]     # 预测位置
            # 用"当前位置"和"预测位置"的**较小距离**做触发（双保险）
            d_now = float(np.hypot(r[0]-p[0], r[1]-p[1]))
            d_pred = float(np.hypot(r[0]-pp[0], r[1]-pp[1]))
            cand.append((min(d_now, d_pred), p, pp, d_now, d_pred))
        dmin, ped_xy, ped_pred, d_now, d_pred = min(cand, key=lambda kv: kv[0])
        dmin = float(dmin)

    # 用 Gazebo 真值初始化机器人位置（保证与世界一致）
    if robot_xy is None and r is not None:
        robot_xy = [r[0], r[1]]

    goal = list(WAYPOINTS[wp_idx])       # 紫星 = 当前正在追逐的航点
    st = "CRUISE"
    if robot_xy is not None:
        # ---- 到达判定：进入到达圈 → 切换下一个航点（紫星跳过去）----
        if np.hypot(goal[0]-robot_xy[0], goal[1]-robot_xy[1]) < ARRIVE:
            wp_idx = (wp_idx + 1) %% len(WAYPOINTS)
            goal = list(WAYPOINTS[wp_idx])

        # ---- 朝向控制：**直接朝目标转**（这就是"朝五角星前进"）----
        goal_dir = np.arctan2(goal[1]-robot_xy[1], goal[0]-robot_xy[0])
        off = 0.0
        if dmin is not None and dmin < WARN:
            # ============ 有序避障：预测绕行 + 分级减速 + 停车让路 ============
            # 演进史（每一版都被用户/数据逼出来）：
            #   v1 开关式（固定 off/v）→ 贴脸仍走（实测 133/133 帧 v=0.15）
            #   v2 比例式（速度/偏转随距离连续变化）→ 但"停车后行人仍撞上来"
            #      （实测机器人 v=0 静止 12 帧，dmin 却从 0.41 降到 0.29）
            #   v3 本版：**提前预测**（用行人未来位置触发）+ **停车也侧移让路**
            #
            # 关键洞察：**停车是被动策略**——停着不动，朝你走来的人还是会撞上。
            # 所以要"**主动让开**"：① 用前瞻位置提前开始绕；② 太近时不是死等，
            # 而是**原地转向 + 侧移**离开行人路径。
            # 避障方向用**预测位置**算（提前量），而非当前位置（滞后）
            tgt = ped_pred if ped_pred is not None else ped_xy
            ped_dir = np.arctan2(tgt[1]-robot_xy[1], tgt[0]-robot_xy[0])
            # 行人相对目标的方位：决定从左侧还是右侧绕（选不挡目标的一侧）
            diff = (ped_dir - goal_dir + np.pi) %% (2*np.pi) - np.pi
            if now >= avoid_until:
                avoid_sign = -1.0 if diff > 0 else 1.0
                avoid_until = now + 1.2

            # ① 分级减速：安全→WARN→v_max；越近越慢；STOP 内为 0
            if dmin <= STOP:
                v = 0.0
                st = "STOP"
            else:
                frac = (dmin - STOP) / (WARN - STOP)      # 0~1
                v = V_MAX * float(np.clip(frac, 0.0, 1.0))
                st = "AVOID" if dmin < DANGER else "SLOW"

            # ② 比例转向：越近偏转越大（5°~75°，连续）
            near = 1.0 - float(np.clip((dmin - STOP) / (WARN - STOP), 0.0, 1.0))
            off = avoid_sign * (0.09 + 1.22 * near)

            # ③ **停车让路**：进入停止圈不停死，而是原地转向 + 小速度侧移，
            #    主动离开行人路径（这才是"避让"，而非"等撞"）
            if dmin <= STOP:
                v = 0.10 * float(np.clip((dmin - 0.35) / 0.15, 0.0, 1.0))
                off = avoid_sign * 1.57      # 侧向（垂直于行人方向）让开
        elif now < avoid_until:
            # 绕行后回归目标：逐渐收回偏置
            st = "RECOVER"
            off = avoid_sign * 0.35
            v = V_MAX * 0.7
        else:
            st = "CRUISE"
            v = V_MAX
        # 目标朝向 = 朝目标方向 + 避让偏置
        target_theta = goal_dir + off
        err = (target_theta - robot_theta + np.pi) %% (2*np.pi) - np.pi
        w = float(np.clip(1.8*err, -1.2, 1.2))
        robot_theta += 0.3*err
        # 运动学积分（机器人在物理世界里也发 cmd_vel，但绘图用积分值保证清晰）
        robot_xy[0] += v*np.cos(robot_theta) * 0.03
        robot_xy[1] += v*np.sin(robot_theta) * 0.03
        tw = Twist(); tw.linear.x = v; tw.angular.z = w
        pub.publish(tw)
    else:
        v, w, st = 0.0, 0.0, "INIT"

    if robot_xy is not None and time.time() - last_save > 1.0/FS:
        frames.append({"t": round(now, 2),
                       "robot": [float(robot_xy[0]), float(robot_xy[1])],
                       "peds": {k: list(v) for k, v in poses["peds"].items()},
                       "goal": [float(goal[0]), float(goal[1])],
                       "theta": float(robot_theta), "wp_idx": int(wp_idx),
                       "dmin": dmin, "v": float(v), "w": float(w),
                       "state": st})
        last_save = time.time()
    time.sleep(0.02)   # 20ms 控制周期（与 12Hz 采集匹配，动作更连续）

print("RESULT:" + json.dumps(frames))
''' % (OUT, dur, fps_save, WAYPOINTS, ARRIVE, V_MAX, WARN, DANGER, STOP)
    open("/tmp/_gif_cap.py", "w").write(script)
    code, o = sh(f"/home/hmn-cjy/miniforge3/envs/ros2jazzy/bin/python /tmp/_gif_cap.py",
                 timeout=dur + 40)
    if "RESULT:" in o:
        return json.loads(o.split("RESULT:")[1].splitlines()[0])
    return []


def render_frames(frames, out_dir):
    """把每帧渲染成一张俯视图。"""
    os.makedirs(out_dir, exist_ok=True)
    if not frames:
        return 0
    # 固定坐标范围（避免抖动）
    allx = [f["robot"][0] for f in frames] + \
           [p[0] for f in frames for p in f["peds"].values()]
    ally = [f["robot"][1] for f in frames] + \
           [p[1] for f in frames for p in f["peds"].values()]
    xmin, xmax = min(allx)-1.0, max(allx)+1.0
    ymin, ymax = min(ally)-1.0, max(ally)+1.0
    rx = [f["robot"][0] for f in frames]
    ry = [f["robot"][1] for f in frames]

    # 中英状态名映射（让"它在干什么"一眼看懂）
    ZH = {"CRUISE": "前往目标", "SLOW": "预警减速", "AVOID": "避让（绕开行人）",
          "STOP": "停车（太近）", "RECOVER": "绕行后回归目标", "INIT": "初始化"}
    STC = {"CRUISE": "#2ca02c", "SLOW": "#bcbd22", "AVOID": "#d62728",
           "STOP": "#8c1a1a", "RECOVER": "#ff7f0e", "INIT": "#888888"}

    for i, f in enumerate(frames):
        fig, ax = plt.subplots(figsize=(5.8, 5.4))
        # 轨迹
        ax.plot(rx[:i+1], ry[:i+1], "-", color="#1f77b4", lw=2.2, alpha=0.75,
                label="机器人走过的轨迹")
        # 巡逻路线：把航点依次连起来（这是机器人要依次经过的路线）
        wp = np.array(WAYPOINTS + [WAYPOINTS[0]])
        ax.plot(wp[:, 0], wp[:, 1], ":", color="#9467bd", lw=1.2,
                alpha=0.45, label="巡逻路线（航点依次）")
        # 目标点（它正要去的地方）
        g = f.get("goal")
        if g:
            ax.scatter([g[0]], [g[1]], marker="*", s=300, color="#9467bd",
                       edgecolors="black", zorder=5, label="当前目标点")
            # 机器人 → 目标的连线（"它正朝这里走"）
            ax.annotate("", xy=(g[0], g[1]),
                        xytext=(f["robot"][0], f["robot"][1]),
                        arrowprops=dict(arrowstyle="->", color="#9467bd",
                                        lw=1.6, ls="--", alpha=0.85))
        # 机器人（方块 + 朝向箭头）
        ax.scatter([f["robot"][0]], [f["robot"][1]], marker="s", s=200,
                   color="#1f77b4", edgecolors="black", zorder=6,
                   label="机器人")
        # 机器人**实际朝向**箭头（看它头朝哪 —— 应当指向目标点）
        th = f.get("theta")
        if th is not None:
            ax.annotate("", xy=(f["robot"][0] + 0.45*np.cos(th),
                                f["robot"][1] + 0.45*np.sin(th)),
                        xytext=(f["robot"][0], f["robot"][1]),
                        arrowprops=dict(arrowstyle="-|>", color="#1f77b4",
                                        lw=2.6), zorder=7)
            # 到达圈（说明"进这个圈就算到了"）
            if g:
                ax.add_patch(Circle((g[0], g[1]), ARRIVE, fill=False, ls=":",
                                    color="#9467bd", lw=1.2, alpha=0.8))
        # 分层安全区（黄=预警减速 / 橙=危险避让 / 红=停止）
        ax.add_patch(Circle((f["robot"][0], f["robot"][1]), WARN,
                            fill=False, ls="--", color="#bcbd22", lw=1.6,
                            alpha=0.8, label="安全区分层（黄/橙/红）"))
        ax.add_patch(Circle((f["robot"][0], f["robot"][1]), DANGER,
                            fill=False, ls="--", color="#ff7f0e", lw=1.6,
                            alpha=0.85))
        ax.add_patch(Circle((f["robot"][0], f["robot"][1]), STOP,
                            fill=False, ls="-", color="#d62728", lw=1.4,
                            alpha=0.75))
        # 行人（按最近采样差分估计速度方向，画箭头说明它在往哪走）
        for k, p in f["peds"].items():
            ax.scatter([p[0]], [p[1]], marker="o", s=170,
                       color="#d62728", edgecolors="black", zorder=7)
            ax.annotate("行人", (p[0], p[1]), fontsize=9, xytext=(7, 6),
                        textcoords="offset points")
            # 与上一帧差分 → 运动方向
            if i > 0 and k in frames[i-1]["peds"]:
                pp = frames[i-1]["peds"][k]
                vy = p[1] - pp[1]
                if abs(vy) > 1e-4:
                    ax.annotate("", xy=(p[0], p[1] + 0.5*np.sign(vy)),
                                xytext=(p[0], p[1]),
                                arrowprops=dict(arrowstyle="-|>", color="#d62728",
                                                lw=2.2, alpha=0.9), zorder=8)
        # 状态框（中文）
        st = f["state"]
        ax.text(0.02, 0.98, ZH.get(st, st), transform=ax.transAxes,
                fontsize=13, weight="bold", color="white",
                bbox=dict(facecolor=STC.get(st, "#888"), alpha=0.9,
                          boxstyle="round,pad=0.4"), va="top")
        dm = f"{f['dmin']:.2f} m" if f["dmin"] is not None else "—"
        # 机器人到目标的距离 + 朝向误差（证明它确实朝目标走）
        gd = f"{np.hypot(g[0]-f['robot'][0], g[1]-f['robot'][1]):.2f} m" if g else "—"
        herr = "—"
        if g and "theta" in f:
            want = np.arctan2(g[1]-f["robot"][1], g[0]-f["robot"][0])
            e = (want - f["theta"] + np.pi) % (2*np.pi) - np.pi
            herr = f"{np.degrees(abs(e)):.0f}°"
        # 安全区级别（让"分层避障"策略可见）
        if f["dmin"] is None:
            zone = "—"
        elif f["dmin"] < STOP:
            zone = "停止圈（急停）"
        elif f["dmin"] < DANGER:
            zone = "危险圈（避让）"
        elif f["dmin"] < WARN:
            zone = "预警圈（减速）"
        else:
            zone = "安全（巡航）"
        info = (f"t = {f['t']:.1f} s\n"
                f"前进速度 v = {f['v']:.2f} m/s\n"
                f"到目标距离 = {gd}\n"
                f"朝向误差 = {herr}\n"
                f"最近行人 = {dm}\n"
                f"安全区 = {zone}")
        ax.text(0.98, 0.98, info, transform=ax.transAxes, fontsize=10,
                va="top", ha="right",
                bbox=dict(facecolor="white", alpha=0.85,
                          boxstyle="round,pad=0.35"))
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_title("P2 · 动态障碍闭环（俯视·仿真真值）", fontsize=12)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"f{i:04d}.png"), dpi=80)
        plt.close(fig)
    return len(frames)


def make_gif(out_dir, gif_path, fps=10):
    from PIL import Image
    files = sorted([f for f in os.listdir(out_dir) if f.endswith(".png")])
    if not files:
        return False
    imgs = [Image.open(os.path.join(out_dir, f)).convert("P",
            palette=Image.ADAPTIVE) for f in files]
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                 duration=int(1000/fps), loop=0, optimize=True)
    return True


def make_mp4(out_dir, mp4_path, fps=10):
    code, o = sh(f"ffmpeg -y -framerate {fps} -pattern_type glob -i "
                 f"'{out_dir}/f*.png' -c:v libx264 -pix_fmt yuv420p "
                 f"-vf scale=1120:-2 {mp4_path}", timeout=120)
    return code == 0 and os.path.exists(mp4_path)


def main():
    os.makedirs(FIGS, exist_ok=True)
    print("[P2·动态障碍 GIF] 跑闭环 + 逐帧渲染 → 动图")

    _, o = sh("ros2 pkg prefix nav2_minimal_tb3_sim")
    pkg = o.strip().splitlines()[-1]
    env = {"GZ_SIM_RESOURCE_PATH":
           f"{pkg}/share/nav2_minimal_tb3_sim/models:{pkg}/share:"
           f"{os.path.join(ROOT, 'data', 'gz_models')}"}

    code, wout = sh(f"xacro {WORLD} headless:=false light:=0.8 ped_speed:=0.35")
    wpath = os.path.join(OUT, "world_gif.sdf")
    with open(wpath, "w") as f:
        f.write(wout)

    spawn(f'gz sim --headless-rendering -s -r -v 1 "{wpath}"', "/tmp/gz_gif.log", env=env)
    print("[·] Gazebo 启动...")
    time.sleep(13)
    spawn(f'ros2 launch nav2_minimal_tb3_sim spawn_tb3.launch.py '
          f'robot_sdf:={WURDF} use_sim_time:=true', "/tmp/tb3_gif.log", env=env)
    print("[·] TB3 生成...")
    time.sleep(16)

    print("[·] 采集闭环逐帧（50s @12Hz）...")
    frames = run_capture(dur=50, fps_save=12)
    print(f"[OK] 采集 {len(frames)} 帧")

    n = render_frames(frames, FRAMES)
    print(f"[✓] 渲染 {n} 张俯视图")

    gif = os.path.join(FIGS, "closed_loop.gif")
    ok_gif = make_gif(FRAMES, gif)
    print(f"[{'OK' if ok_gif else '✗'}] GIF → {gif}"
          + (f"（{os.path.getsize(gif)/1e6:.1f} MB）" if ok_gif else ""))

    mp4 = os.path.join(FIGS, "closed_loop.mp4")
    ok_mp4 = make_mp4(FRAMES, mp4)
    print(f"[{'OK' if ok_mp4 else '✗'}] MP4 → {mp4}")

    # 保存富数据（含目标点与朝向，便于验证"是否朝目标走"）
    np.save(os.path.join(OUT, "traj_rich.npy"),
            np.array([[f["t"], f["robot"][0], f["robot"][1],
                       f["dmin"] if f["dmin"] else np.nan, f["v"], f["w"],
                       f["goal"][0], f["goal"][1], f.get("theta", np.nan)]
                      for f in frames], dtype=float))
    print(f"[✓] 完成 → {OUT}")
    cleanup()


if __name__ == "__main__":
    sys.exit(main())
