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
WORLD = os.path.join(ROOT, "data", "gz_models", "worlds", "dynamic_pedestrian.sdf.xacro")
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
v_cur, w_cur = 0.18, 0.0
t0 = time.time(); last_save = 0
while time.time() - t0 < T:
    r = poses["robot"]
    dmin = None
    if r is not None and poses["peds"]:
        ds = [float(np.hypot(r[0]-p[0], r[1]-p[1])) for p in poses["peds"].values()]
        dmin = min(ds) if ds else None
    if dmin is not None and dmin < DANGER:
        v_cmd, w_cmd, st = 0.05, 0.35, "AVOID"
    else:
        v_cmd, w_cmd, st = 0.18, 0.0, "GO"
    v_cur += (v_cmd - v_cur) * 0.2
    w_cur += (w_cmd - w_cur) * 0.2
    tw = Twist(); tw.linear.x = v_cur; tw.angular.z = w_cur
    pub.publish(tw)
    if r is not None and time.time() - last_save > 1.0/FS:
        frames.append({"t": round(time.time()-t0, 2),
                       "robot": [r[0], r[1]],
                       "peds": {k: list(v) for k, v in poses["peds"].items()},
                       "dmin": dmin, "v": v_cur, "w": w_cur, "state": st})
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

    for i, f in enumerate(frames):
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        # 轨迹
        ax.plot(rx[:i+1], ry[:i+1], "-", color="#1f77b4", lw=2, alpha=0.7,
                label="机器人轨迹")
        # 机器人
        ax.scatter([f["robot"][0]], [f["robot"][1]], marker="s", s=190,
                   color="#1f77b4", edgecolors="black", zorder=5, label="机器人")
        # 危险半径圈
        ax.add_patch(Circle((f["robot"][0], f["robot"][1]), DANGER,
                            fill=False, ls="--", color="#ff7f0e", lw=1.8))
        # 行人
        for k, p in f["peds"].items():
            ax.scatter([p[0]], [p[1]], marker="o", s=160,
                       color="#d62728", edgecolors="black", zorder=6)
            ax.annotate("行人", (p[0], p[1]), fontsize=9, xytext=(6, 6),
                        textcoords="offset points")
        # 状态框
        st = f["state"]
        color = "#d62728" if st == "AVOID" else "#2ca02c"
        ax.text(0.02, 0.98, f"状态: {st}", transform=ax.transAxes,
                fontsize=13, weight="bold", color="white",
                bbox=dict(facecolor=color, alpha=0.85, boxstyle="round,pad=0.4"),
                va="top")
        info = (f"t = {f['t']:.1f}s\n"
                f"v = {f['v']:.2f} m/s\n"
                f"ω = {f['w']:.2f} rad/s\n"
                f"最近行人 = {f['dmin']:.2f} m" if f["dmin"] is not None else
                f"t = {f['t']:.1f}s")
        ax.text(0.98, 0.98, info, transform=ax.transAxes, fontsize=10,
                va="top", ha="right",
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"))
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
    frames = run_capture(dur=30, fps_save=5)
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
