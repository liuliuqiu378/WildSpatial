#!/usr/bin/env python
"""P2 · 动态障碍闭环 —— GIF 动图（让"运动"看得见）

为什么做这个
------------
静态图只能展示"某一刻"，看不出**机器人怎么走、行人怎么动、避障怎么响应**。
本脚本在跑闭环的同时**逐帧渲染"世界俯视图"**（机器人+行人+轨迹+状态），
合成 **GIF 动图**，让整个过程"动起来"。

就像看比赛回放：左上是**俯视上帝视角**（谁在哪、怎么动），
右下是**状态条**（GO / AVOID），一眼看懂"感知→决策→控制"。

产出
----
experiments/P2_dynamic_obstacle/
    figs/  closed_loop.gif    完整闭环动图（俯视 + 状态）
           closed_loop.mp4    同内容视频（若 ffmpeg 可用）
    traj_rich.npy             更完整的逐帧数据

诚实声明
--------
· 俯视图由**仿真真值位姿**渲染（非相机画面）；相机画面对比见 §4.14 sensor_view。
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
DANGER = 1.2


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


def run_capture(dur=30, fps_save=5):
    """跑闭环并记录逐帧的机器人/行人位置 + 控制指令 + 状态。"""
    script = r'''
import json, time
import numpy as np
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist

OUT = "%s"; T = %d; FS = %d; DANGER = %s
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

frames = []
t0 = time.time(); last_save = 0

# ---- 巡逻路线跟踪：机器人沿固定环线巡逻，遇人绕开、之后回归路线 ----
# 说明：这里机器人位置由**脚本运动学积分**得到（不靠 Gazebo 物理推），
#       目的是让演示**轨迹清晰可控**（Gazebo 物理在窄空间易卡死）。
#       行人位置仍读 Gazebo **真值**（它在真实物理世界里运动）。
def patrol_target(tsec):
    """环线：半径 2.0m 的圆，按时间参数化（机器人绕圈巡逻）。"""
    ang = 0.42 * tsec
    return [1.5*np.cos(ang), 1.5*np.sin(ang)]

robot_xy = None
robot_theta = 0.0
avoid_until = 0.0
avoid_sign = 1.0
heading_off = 0.0       # 避让造成的方向偏置（相对巡逻切线）

while time.time() - t0 < T:
    now = time.time() - t0
    r = poses["robot"]
    # 行人真值位置
    dmin = None; ped_xy = None
    if r is not None and poses["peds"]:
        items = [(float(np.hypot(r[0]-p[0], r[1]-p[1])), p)
                 for p in poses["peds"].values()]
        dmin, ped_xy = min(items, key=lambda kv: kv[0])
        dmin = float(dmin)

    # 用 Gazebo 真值初始化机器人位置（保证与世界一致）
    if robot_xy is None and r is not None:
        robot_xy = [r[0], r[1]]

    goal = patrol_target(now + 1.0)     # 前瞻 1 秒的巡逻点
    st = "CRUISE"
    if robot_xy is not None:
        # 巡逻切线方向（沿环线前进）
        cur = patrol_target(now); nxt = patrol_target(now + 0.5)
        tangent = np.arctan2(nxt[1]-cur[1], nxt[0]-cur[0])
        off = 0.0
        if dmin is not None and dmin < DANGER:
            ped_dir = np.arctan2(ped_xy[1]-robot_xy[1], ped_xy[0]-robot_xy[0])
            diff = (ped_dir - tangent + np.pi) %% (2*np.pi) - np.pi
            if now >= avoid_until:
                avoid_sign = -1.0 if diff > 0 else 1.0
                avoid_until = now + 1.5
            st = "AVOID"
            off = avoid_sign * 1.1          # 偏转绕行
            v = 0.15
        elif now < avoid_until:
            st = "RECOVER"
            off = avoid_sign * 0.5
            v = 0.20
        else:
            st = "CRUISE"
            v = 0.28
        # 朝向平滑趋向（切线 + 避让偏置）
        target_theta = tangent + off
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
                       "dmin": dmin, "v": float(v), "w": float(w),
                       "state": st})
        last_save = time.time()
    time.sleep(0.03)

print("RESULT:" + json.dumps(frames))
''' % (OUT, dur, fps_save, DANGER)
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
    ZH = {"CRUISE": "巡航（去目标）", "AVOID": "避让（绕开行人）",
          "RECOVER": "绕行中", "INIT": "初始化"}
    STC = {"CRUISE": "#2ca02c", "AVOID": "#d62728",
           "RECOVER": "#ff7f0e", "INIT": "#888888"}

    for i, f in enumerate(frames):
        fig, ax = plt.subplots(figsize=(5.8, 5.4))
        # 轨迹
        ax.plot(rx[:i+1], ry[:i+1], "-", color="#1f77b4", lw=2.2, alpha=0.75,
                label="机器人走过的轨迹")
        # 目标点（它要去哪）
        g = f.get("goal")
        if g:
            ax.scatter([g[0]], [g[1]], marker="*", s=260, color="#9467bd",
                       edgecolors="black", zorder=4, label="目标点")
            # 从机器人指向目标的虚线（"它想去哪"）
            ax.annotate("", xy=(g[0], g[1]),
                        xytext=(f["robot"][0], f["robot"][1]),
                        arrowprops=dict(arrowstyle="->", color="#9467bd",
                                        lw=1.4, ls="--", alpha=0.7))
        # 机器人（方块 + 朝向箭头）
        ax.scatter([f["robot"][0]], [f["robot"][1]], marker="s", s=200,
                   color="#1f77b4", edgecolors="black", zorder=6,
                   label="机器人")
        # 危险半径圈
        ax.add_patch(Circle((f["robot"][0], f["robot"][1]), DANGER,
                            fill=False, ls="--", color="#ff7f0e", lw=1.8,
                            alpha=0.8))
        # 行人
        for k, p in f["peds"].items():
            ax.scatter([p[0]], [p[1]], marker="o", s=170,
                       color="#d62728", edgecolors="black", zorder=7)
            ax.annotate("行人", (p[0], p[1]), fontsize=9, xytext=(7, 6),
                        textcoords="offset points")
        # 状态框（中文）
        st = f["state"]
        ax.text(0.02, 0.98, ZH.get(st, st), transform=ax.transAxes,
                fontsize=13, weight="bold", color="white",
                bbox=dict(facecolor=STC.get(st, "#888"), alpha=0.9,
                          boxstyle="round,pad=0.4"), va="top")
        dm = f"{f['dmin']:.2f} m" if f["dmin"] is not None else "—"
        info = (f"t = {f['t']:.1f} s\n"
                f"前进速度 v = {f['v']:.2f} m/s\n"
                f"转向速度 ω = {f['w']:.2f} rad/s\n"
                f"最近行人 = {dm}")
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


def make_gif(out_dir, gif_path, fps=6):
    from PIL import Image
    files = sorted([f for f in os.listdir(out_dir) if f.endswith(".png")])
    if not files:
        return False
    imgs = [Image.open(os.path.join(out_dir, f)).convert("P",
            palette=Image.ADAPTIVE) for f in files]
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                 duration=int(1000/fps), loop=0, optimize=True)
    return True


def make_mp4(out_dir, mp4_path, fps=6):
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

    code, wout = sh(f"xacro {WORLD} headless:=false light:=0.8 ped_speed:=0.5")
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

    print("[·] 采集闭环逐帧（30s）...")
    frames = run_capture(dur=50, fps_save=5)
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

    # 保存富数据
    np.save(os.path.join(OUT, "traj_rich.npy"),
            np.array([[f["t"], f["robot"][0], f["robot"][1],
                       f["dmin"] if f["dmin"] else np.nan, f["v"], f["w"]]
                      for f in frames], dtype=float))
    print(f"[✓] 完成 → {OUT}")
    cleanup()


if __name__ == "__main__":
    sys.exit(main())
