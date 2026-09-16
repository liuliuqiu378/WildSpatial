#!/usr/bin/env python
"""P2 · 机器人第一视角送货演示 —— Gazebo 真实相机版（方案 A）

目的
----
把 `p2_delivery_fpv.py` 的"纯合成渲染"升级为 **Gazebo 真实 RGB 相机输出**，
证明在解决 `/dev/dri/renderD128` 权限后，headless EGL 渲染可出真实 3D 画面。

任务
----
TB3 从广场南侧 `(0, -3.8)` 出发，沿固定航点前往北侧送货点 `(0, 3.8)`；
途中需处理：
  · 广场中央静态柱子（绕行/减速）
  · 两侧动态行人（停车让行/错车）
  · 到达目标点后停靠

运行前提
--------
1. 用户已在 `render` 组（本机已执行 `sudo usermod -aG render $USER`）。
2. 当前会话若尚未重新登录，请用 `sg render` 运行：
       sg render -c 'python scripts/p2_delivery_fpv_gazebo.py'
   否则 Gazebo 进程无 render 组权限，相机会灰屏。

产出
----
experiments/P2_delivery_fpv_gazebo/
    figs/delivery_fpv_gazebo.gif   真实相机 FPV 动图（带 HUD）
    figs/delivery_fpv_gazebo.mp4   同内容视频
    figs/keyframes.png             关键帧
    metrics.json                   逐帧元数据
"""

import os
import sys
import json
import time
import math
import signal
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei",
                                          "DejaVu Sans"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDA_SH = "/home/hmn-cjy/miniforge3/etc/profile.d/conda.sh"
ENV = "ros2jazzy"
OUT = os.path.join(ROOT, "experiments", "P2_delivery_fpv_gazebo")
FIGS = os.path.join(OUT, "figs")
FRAMES = os.path.join(OUT, "gif_frames")
WORLD = os.path.join(ROOT, "data", "gz_models", "worlds",
                     "open_plaza_dynamic.sdf.xacro")
WURDF = os.path.join(ROOT, "data", "gz_models", "urdf", "gz_waffle.sdf.xacro")
PROCS = []

# 送货路线（从南到北，刻意经过右侧柱子与行人路径）
WAYPOINTS = [[0.0, -3.8], [1.2, -2.2], [2.3, 0.0],
             [1.5, 2.0], [0.0, 3.8]]
ARRIVE = 0.35
GOAL = WAYPOINTS[-1]
V_MAX = 0.28          # m/s
W_MAX = 1.0           # rad/s
DT = 0.05             # 控制周期 20Hz
DURATION = 55.0       # 演示时长（s），留足余量让仿真慢推也能到终点
SAVE_FPS = 10         # GIF 帧率


def sh(cmd, timeout=90):
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


def make_capture_script(out_dir, frames_dir):
    """生成在 Gazebo 里订阅 /clock + RGB、按仿真时间积分、控制机器人并保存帧的子进程脚本。

    说明：本机 headless rendering 负载高，Gazebo 仿真时间比墙上时间慢，
         因此机器人/行人都用 **/clock 仿真时间** 积分，保证 HUD/minimap 与画面一致。
    """
    return r'''
import os, json, time, math
import numpy as np
from PIL import Image
from gz.transport13 import Node
from gz.msgs10.clock_pb2 import Clock
from gz.msgs10.image_pb2 import Image as GzImage
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist

OUT = "%s"
FRAMES = "%s"
WAYPOINTS = %s
ARRIVE = %s
GOAL = %s
V_MAX = %s
W_MAX = %s
DT = %s
DURATION = %s
SAVE_FPS = %s
os.makedirs(FRAMES, exist_ok=True)

node = Node()
state = {"rgb": None, "robot": None, "sim": 0.0, "wall0": time.time()}

def on_rgb(m):
    w, h, s = m.width, m.height, m.step
    d = np.frombuffer(m.data, dtype=np.uint8)
    arr = d[:h*s].reshape(h, s)[:, :w*3].reshape(h, w, 3).copy()
    state["rgb"] = arr

def on_clock(m):
    state["sim"] = m.sim.sec + m.sim.nsec * 1e-9

def on_pose(m):
    for p in m.pose:
        if "turtlebot3_waffle" in p.name:
            state["robot"] = [p.position.x, p.position.y,
                              2*math.atan2(p.orientation.z, p.orientation.w)]

node.subscribe(Clock, "/clock", on_clock)
node.subscribe(GzImage, "/camera/color/image_raw", on_rgb)
node.subscribe(Pose_V, "/world/default/pose/info", on_pose)
cmd_pub = node.advertise("cmd_vel", Twist)

# 行人运动模型（按世界 xacro 的 VelocityControl 参数，在仿真时间下积分）
PEDS = {
    "pedestrian":  {"x": 2.6,  "y0": -2.5, "vy": 0.35, "limit": 3.0},
    "pedestrian2": {"x": -2.6, "y0": 2.5,  "vy": -0.35, "limit": 3.0},
}
def ped_pos(t, spec):
    """一维 y 往返运动；已知初速度、边界，按仿真时间解析计算。"""
    L = spec["limit"]
    vy = spec["vy"]
    period = 4 * L / abs(vy)
    phase = (vy * t) %% (4 * L)
    if phase < 0:
        phase += 4 * L
    if phase < 2 * L:
        y = spec["y0"] + phase
    else:
        y = spec["y0"] + 4 * L - phase
    # 裁剪到边界（数值安全）
    return spec["x"], max(-L, min(L, y))

robot_xy = None
robot_theta = None
wp_idx = 0
metrics = []
saved_idx = 0
save_every = int(1.0 / (DT * SAVE_FPS) + 0.5)
sim_prev = None

while True:
    wall_now = time.time() - state["wall0"]
    sim_now = state["sim"]
    if sim_prev is None:
        sim_prev = sim_now
    dsim = sim_now - sim_prev
    sim_prev = sim_now

    if wall_now > DURATION + 15:
        break

    # 用 Gazebo 真值初始化一次位姿（后续按仿真时间积分，与画面同步）
    r = state["robot"]
    if r is not None and robot_xy is None:
        robot_xy = [r[0], r[1]]
        robot_theta = r[2]

    v, w = 0.0, 0.0
    scenario = "巡航"
    dgoal = float("inf")
    dmin_ped = float("inf")
    peds_now = {}
    if robot_xy is not None and dsim > 0:
        gx, gy = WAYPOINTS[wp_idx]
        dx, dy = gx - robot_xy[0], gy - robot_xy[1]
        dgoal = math.hypot(dx, dy)
        target_theta = math.atan2(dy, dx)
        err = math.atan2(math.sin(target_theta - robot_theta),
                         math.cos(target_theta - robot_theta))

        # 行人位置（仿真时间）
        for nm, spec in PEDS.items():
            px, py = ped_pos(sim_now, spec)
            peds_now[nm] = [px, py]
            dmin_ped = min(dmin_ped,
                           math.hypot(robot_xy[0]-px, robot_xy[1]-py))

        # 场景判定
        if dgoal < ARRIVE and wp_idx < len(WAYPOINTS) - 1:
            wp_idx += 1
            gx, gy = WAYPOINTS[wp_idx]
            dx, dy = gx - robot_xy[0], gy - robot_xy[1]
            dgoal = math.hypot(dx, dy)
            target_theta = math.atan2(dy, dx)
            err = math.atan2(math.sin(target_theta - robot_theta),
                             math.cos(target_theta - robot_theta))

        near_pillar = False
        for px, py in [(-1.5, -1.5), (1.5, 1.5)]:
            if math.hypot(robot_xy[0]-px, robot_xy[1]-py) < 1.0:
                near_pillar = True
        if near_pillar:
            scenario = "① 静态障碍·绕行"
        if dmin_ped < 1.3 and dmin_ped < dgoal:
            scenario = "② 动态行人·停车让行"
        if math.hypot(robot_xy[0]-GOAL[0], robot_xy[1]-GOAL[1]) < 1.0:
            scenario = "⑥ 送达点·到达"

        # 速度控制
        if scenario.startswith("②"):
            v, w = 0.0, 0.0
        elif scenario.startswith("①"):
            v = V_MAX * 0.5
            w = float(np.clip(err * 2.0, -W_MAX, W_MAX))
        elif scenario.startswith("⑥"):
            v = min(V_MAX * 0.4, dgoal * 0.5)
            w = float(np.clip(err * 1.5, -W_MAX, W_MAX))
        else:
            v = V_MAX
            w = float(np.clip(err * 2.5, -W_MAX, W_MAX))

        # 按仿真时间积分（匹配 Gazebo 实际推进速度）
        robot_xy[0] += v * math.cos(robot_theta) * dsim
        robot_xy[1] += v * math.sin(robot_theta) * dsim
        robot_theta += w * dsim
        robot_theta = math.atan2(math.sin(robot_theta), math.cos(robot_theta))

    tw = Twist(); tw.linear.x = v; tw.angular.z = w
    cmd_pub.publish(tw)

    if state["rgb"] is not None:
        saved_idx += 1
        if saved_idx %% save_every == 0:
            fn = os.path.join(FRAMES, "f_%%05d.png" %% saved_idx)
            Image.fromarray(state["rgb"]).save(fn)
            metrics.append({
                "t": round(sim_now, 3), "frame": saved_idx,
                "rx": round(robot_xy[0], 3) if robot_xy else None,
                "ry": round(robot_xy[1], 3) if robot_xy else None,
                "theta": round(robot_theta, 3) if robot_xy else None,
                "v": round(v, 3), "w": round(w, 3),
                "scenario": scenario,
                "dgoal": round(dgoal, 3),
                "dmin_ped": round(dmin_ped, 3) if dmin_ped != float("inf") else None,
                "wp": wp_idx,
                "peds": {k: [round(vv,3) for vv in peds_now[k]] for k in peds_now},
            })
    time.sleep(DT)

with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print("CAPTURE_DONE", len(metrics), "sim_time", round(state["sim"], 2), flush=True)
''' % (out_dir, frames_dir, repr(WAYPOINTS), ARRIVE, repr(GOAL),
       V_MAX, W_MAX, DT, DURATION, SAVE_FPS)


def draw_hud(rgb, rec):
    """在 RGB 帧上叠加 HUD：场景、速度、到目标距离、小地图。"""
    im = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(im)
    W, H = im.size
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 18)
        font_big = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 22)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 18)
            font_big = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 22)
        except Exception:
            font = ImageFont.load_default()
            font_big = font

    # 顶部场景标签（红底白字）
    scenario = rec.get("scenario", "巡航")
    bbox = draw.textbbox((0, 0), scenario, font=font_big)
    tw = bbox[2] - bbox[0]
    draw.rectangle([10, 10, 18 + tw, 44], fill=(180, 40, 40))
    draw.text((14, 12), scenario, fill=(255, 255, 255), font=font_big)

    # 左下角状态条
    status = (f"速度 {rec.get('v', 0):.2f} m/s\n"
              f"角速度 {rec.get('w', 0):.2f} rad/s\n"
              f"到目标 {rec.get('dgoal', 0):.2f} m\n"
              f"时间 {rec.get('t', 0):.1f} s")
    draw.rectangle([10, H-95, 180, H-10], fill=(20, 20, 20, 180))
    draw.text((14, H-90), status, fill=(255, 255, 255), font=font)

    # 右上角小地图
    mp_sz = 120
    mx0, my0 = W - mp_sz - 10, 10
    draw.rectangle([mx0, my0, mx0 + mp_sz, my0 + mp_sz], fill=(240, 240, 240), outline=(80, 80, 80))
    # 世界范围 ±5m → 小地图
    def wx2sx(x): return mx0 + mp_sz//2 + int(x / 5.0 * (mp_sz//2 - 6))
    def wy2sy(y): return my0 + mp_sz//2 - int(y / 5.0 * (mp_sz//2 - 6))
    # 外墙
    for wall, color in [([-5, 5, 5, 5], (120, 120, 120)),
                        ([-5, -5, 5, -5], (120, 120, 120)),
                        ([5, -5, 5, 5], (120, 120, 120)),
                        ([-5, -5, -5, 5], (120, 120, 120))]:
        draw.line([(wx2sx(wall[0]), wy2sy(wall[1])),
                   (wx2sx(wall[2]), wy2sy(wall[3]))], fill=color, width=2)
    # 柱子
    for px, py in [(-1.5, -1.5), (1.5, 1.5)]:
        r = 5
        draw.ellipse([wx2sx(px)-r, wy2sy(py)-r, wx2sx(px)+r, wy2sy(py)+r],
                     fill=(100, 100, 100))
    # 路线
    for i in range(len(WAYPOINTS)-1):
        draw.line([(wx2sx(WAYPOINTS[i][0]), wy2sy(WAYPOINTS[i][1])),
                   (wx2sx(WAYPOINTS[i+1][0]), wy2sy(WAYPOINTS[i+1][1]))],
                  fill=(0, 150, 200), width=2)
    # 目标
    draw.polygon([(wx2sx(GOAL[0]), wy2sy(GOAL[1])-5),
                  (wx2sx(GOAL[0])-4, wy2sy(GOAL[1])+4),
                  (wx2sx(GOAL[0])+4, wy2sy(GOAL[1])+4)], fill=(160, 32, 240))
    # 行人
    peds = rec.get("peds", {})
    for nm, p in peds.items():
        color = (220, 60, 60) if nm == "pedestrian" else (60, 100, 220)
        draw.ellipse([wx2sx(p[0])-3, wy2sy(p[1])-3, wx2sx(p[0])+3, wy2sy(p[1])+3], fill=color)
    # 机器人
    rx, ry = rec.get("rx", 0), rec.get("ry", 0)
    th = rec.get("theta", 0)
    sx, sy = wx2sx(rx), wy2sy(ry)
    draw.ellipse([sx-4, sy-4, sx+4, sy+4], fill=(0, 100, 220))
    draw.line([(sx, sy), (sx + int(10*math.cos(th)), sy - int(10*math.sin(th)))],
              fill=(0, 80, 180), width=2)
    return np.array(im)


def postprocess():
    """读保存的帧 + metrics，叠加 HUD，生成 GIF/MP4/关键帧。"""
    metrics_path = os.path.join(OUT, "metrics.json")
    if not os.path.exists(metrics_path):
        print("[✗] metrics.json 不存在，后处理失败")
        return False
    with open(metrics_path) as f:
        metrics = json.load(f)
    if not metrics:
        print("[✗] metrics 为空")
        return False

    os.makedirs(FIGS, exist_ok=True)
    files = sorted([f for f in os.listdir(FRAMES) if f.endswith(".png")])
    if len(files) != len(metrics):
        print(f"[!] 帧数 {len(files)} 与 metrics {len(metrics)} 不一致，取较小值")
    n = min(len(files), len(metrics))

    hud_frames = []
    for i in range(n):
        rgb = np.array(Image.open(os.path.join(FRAMES, files[i])))
        hud = draw_hud(rgb, metrics[i])
        hud_frames.append(hud)

    # 关键帧：选巡航、静态障碍、行人、到达各一帧
    selectors = [
        ("巡航", lambda m, idx: m["scenario"] == "巡航" and idx > 5),
        ("① 静态障碍·绕行", lambda m, idx: m["scenario"].startswith("①")),
        ("② 动态行人·停车让行", lambda m, idx: m["scenario"].startswith("②")),
        ("⑥ 送达点·到达", lambda m, idx: m["scenario"].startswith("⑥")),
    ]
    kf_frames = []
    for label, fn in selectors:
        cands = [(j, metrics[j]) for j in range(n) if fn(metrics[j], j)]
        if cands:
            j, _ = cands[len(cands)//2]
            kf_frames.append((label, hud_frames[j]))

    if kf_frames:
        cols = min(4, len(kf_frames))
        rows = (len(kf_frames) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.ravel()
        for ax, (label, fr) in zip(axes, kf_frames):
            ax.imshow(fr)
            ax.set_title(label, fontsize=11, weight="bold")
            ax.axis("off")
        for ax in axes[len(kf_frames):]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGS, "keyframes.png"), dpi=120)
        plt.close()

    # 保存带 HUD 的帧用于 GIF
    hud_dir = os.path.join(OUT, "hud_frames")
    os.makedirs(hud_dir, exist_ok=True)
    for i, fr in enumerate(hud_frames):
        Image.fromarray(fr).save(os.path.join(hud_dir, "h_%05d.png" % i))

    palette_dir = os.path.join(OUT, "pal_frames")
    os.makedirs(palette_dir, exist_ok=True)
    for i, fr in enumerate(hud_frames):
        Image.fromarray(fr).convert("RGB").convert("P", palette=Image.ADAPTIVE,
                                                   colors=128).save(
            os.path.join(palette_dir, "p_%05d.png" % i))

    gif_path = os.path.join(FIGS, "delivery_fpv_gazebo.gif")
    mp4_path = os.path.join(FIGS, "delivery_fpv_gazebo.mp4")
    fps = SAVE_FPS

    # GIF
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i",
                    os.path.join(palette_dir, "p_%05d.png"),
                    "-vf", "split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse",
                    "-loop", "0", gif_path], check=True, capture_output=True)
    # MP4
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i",
                    os.path.join(hud_dir, "h_%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                    mp4_path], check=True, capture_output=True)

    print(f"[✓] GIF: {gif_path} ({n} frames @ {fps} fps)")
    print(f"[✓] MP4: {mp4_path}")
    if kf_frames:
        print(f"[✓] keyframes: {os.path.join(FIGS, 'keyframes.png')}")
    return True


def main():
    os.makedirs(FIGS, exist_ok=True)
    os.makedirs(FRAMES, exist_ok=True)

    # 清理旧帧
    for d in (FRAMES, os.path.join(OUT, "hud_frames"), os.path.join(OUT, "pal_frames")):
        if os.path.isdir(d):
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))

    _, pkg_out = sh("ros2 pkg prefix nav2_minimal_tb3_sim")
    pkg = pkg_out.strip().splitlines()[-1]
    env = {"GZ_SIM_RESOURCE_PATH":
           f"{pkg}/share/nav2_minimal_tb3_sim/models:{pkg}/share:"
           f"{os.path.join(ROOT, 'data', 'gz_models')}"}

    _, sdf_text = sh(f"xacro {WORLD} headless:=false light:=0.8 ped_speed:=0.35")
    sdf_path = os.path.join(OUT, "_delivery_fpv_world.sdf")
    with open(sdf_path, "w") as f:
        f.write(sdf_text)

    cap_script = os.path.join(OUT, "_capture.py")
    with open(cap_script, "w") as f:
        f.write(make_capture_script(OUT, FRAMES))

    # 先启动捕获/控制子进程（订阅要在 Gazebo 启动早期建立）
    cap = subprocess.Popen(
        ["bash", "-lc",
         f"source {CONDA_SH} && conda activate {ENV} && "
         f"/home/hmn-cjy/miniforge3/envs/{ENV}/bin/python {cap_script}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        preexec_fn=os.setsid)
    PROCS.append(cap)
    print("[·] 捕获/控制子进程已启动")
    time.sleep(3)

    spawn(f'gz sim --headless-rendering -s -r -v 1 "{sdf_path}"',
          os.path.join(OUT, "gz.log"), env=env)
    print("[·] Gazebo 启动...")
    time.sleep(12)

    spawn('ros2 launch nav2_minimal_tb3_sim spawn_tb3.launch.py '
          f'robot_sdf:={WURDF} x_pose:=0.0 y_pose:=-3.8 yaw:=1.57 use_sim_time:=true',
          os.path.join(OUT, "tb3.log"), env=env)
    print("[·] TB3 已生成，等待闭环运行...")

    try:
        out, _ = cap.communicate(timeout=DURATION + 40)
        print(out[-1500:])
    except subprocess.TimeoutExpired:
        cap.kill()
        print("[!] 捕获超时")

    cleanup()
    print("[·] 后处理中...")
    postprocess()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        cleanup()
        sys.exit(1)
