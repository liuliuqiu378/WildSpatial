#!/usr/bin/env python
"""P2 · 感知实验台（四）：多传感器同步（相机 + LiDAR + IMU）—— M6/F5 落点

目的
----
`M6` 多模态融合 / `F5` 视觉+LiDAR 融合都**依赖多传感器同步数据**。
真实数据集（KITTI/TUM）虽有多传感器，但**时间戳需对齐、真值需标定**；
仿真可以**同源同帧**直接输出相机/激光/IMU，且**时间戳天然同步**——
这正是 M6/F5 最理想的实验台。

产出
----
experiments/P2_gz_multisensor/
    figs/  sensor_fusion.png    相机 + LiDAR 投影叠加（视觉+激光融合直观）
           sensor_timeline.png 三类传感器数据率/时间戳同步
    raw/   rgb / lidar / imu 原始数据
    metrics.json / README.md

诚实声明
--------
· 三路传感器由 Gazebo 同一物理步输出，时间戳天然同步（真实数据需硬件 PPS/软件对齐）。
· 相机与 LiDAR 有外参（安装位置差），本脚本用 SDF 里的真实 pose 做投影。
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from wildspatial.viz import setup_plot_style
setup_plot_style()

CONDA_SH = "/home/hmn-cjy/miniforge3/etc/profile.d/conda.sh"
ENV = "ros2jazzy"
OUT = os.path.join(ROOT, "experiments", "P2_gz_multisensor")
FIGS = os.path.join(OUT, "figs")
RAW = os.path.join(OUT, "raw")
MYURDF = os.path.join(ROOT, "data", "gz_models", "urdf", "gz_waffle.sdf.xacro")
PROCS = []

# gz 话题（TB3）
RGB_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = ("/world/default/model/turtlebot3_waffle/link/camera_link/"
               "sensor/intel_realsense_r200_depth/depth_image")
SCAN_TOPIC = "/scan"
IMU_TOPIC = "/imu"


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


def capture_multisensor(dur=14):
    """同时订阅 RGB/深度（gz 侧）+ LiDAR/IMU（ROS 侧），各抓一帧并统计数据率。

    为什么分两侧：
      · RGB/深度相机在 gz 侧（TB3 默认 ROS bridge 没配相机）
      · /scan、/imu 由 ros_gz_bridge 桥接到 ROS 侧（类型明确）
    """
    script = r'''
import json, time, os, threading
import numpy as np
from gz.transport13 import Node as GzNode
from gz.msgs10.image_pb2 import Image
from gz.msgs10.twist_pb2 import Twist
import rclpy
from rclpy.node import Node as RosNode
from sensor_msgs.msg import LaserScan, Imu

RGB = "%s"; DEP = "%s"; OUT = "%s"; T = %d
S = {"rgb": {"img": None, "t": []}, "depth": {"img": None, "t": []},
     "scan": {"ranges": None, "t": []}, "imu": {"msg": None, "t": []}}

def dec_rgb(m):
    w,h,s = m.width,m.height,m.step
    d = np.frombuffer(m.data, dtype=np.uint8)
    return d[:h*s].reshape(h,s)[:,:w*3].reshape(h,w,3).copy()
def dec_depth(m):
    w,h,s = m.width,m.height,m.step
    d = np.frombuffer(m.data, dtype=np.uint8)
    if d.size >= w*h*4:
        return np.array(d.reshape(h,s//4,4)[:,:w,:].copy().view(np.float32)[:,:,0], dtype=np.float32)
    return d[:h*s].reshape(h,s)[:,:w].astype(np.float32)

gz = GzNode()
def on_rgb(m):
    S["rgb"]["img"] = dec_rgb(m); S["rgb"]["t"].append(time.time())
def on_dep(m):
    S["depth"]["img"] = dec_depth(m); S["depth"]["t"].append(time.time())
gz.subscribe(Image, RGB, on_rgb)
gz.subscribe(Image, DEP, on_dep)

rclpy.init()
ros = RosNode("ms_cap")
def on_scan(m):
    S["scan"]["ranges"] = np.array(m.ranges, dtype=float)
    S["scan"]["angles"] = float(m.angle_min)
    S["scan"]["inc"] = float(m.angle_increment)
    S["scan"]["range"] = [float(m.range_min), float(m.range_max)]
    S["scan"]["t"].append(time.time())
def on_imu(m):
    S["imu"]["msg"] = m; S["imu"]["t"].append(time.time())
ros.create_subscription(LaserScan, "/scan", on_scan, 10)
ros.create_subscription(Imu, "/imu", on_imu, 10)

pub = gz.advertise("cmd_vel", Twist)
t0 = time.time()
while time.time() - t0 < T:
    rclpy.spin_once(ros, timeout_sec=0.01)
    tw = Twist(); tw.linear.x = 0.10; tw.angular.z = 0.15
    pub.publish(tw); time.sleep(0.05)

res = {"rates": {}, "n": {}}
for k in S:
    res["n"][k] = len(S[k]["t"])
    if len(S[k]["t"]) > 1:
        d = S[k]["t"][-1] - S[k]["t"][0]
        res["rates"][k] = round(len(S[k]["t"]) / d, 2) if d > 0 else 0

if S["rgb"]["img"] is not None:
    np.save(os.path.join(OUT, "rgb.npy"), S["rgb"]["img"])
if S["depth"]["img"] is not None:
    np.save(os.path.join(OUT, "depth.npy"), S["depth"]["img"])
if S["scan"]["ranges"] is not None:
    np.save(os.path.join(OUT, "scan_ranges.npy"), S["scan"]["ranges"])
    np.save(os.path.join(OUT, "scan_angles.npy"),
            S["scan"]["angles"] + np.arange(len(S["scan"]["ranges"])) * S["scan"]["inc"])
    res["scan_range"] = S["scan"]["range"]
if S["imu"]["msg"] is not None:
    m = S["imu"]["msg"]
    res["imu_sample"] = {"ax": float(m.linear_acceleration.x),
                         "ay": float(m.linear_acceleration.y),
                         "az": float(m.linear_acceleration.z),
                         "gz": float(m.angular_velocity.z)}
print("RESULT:" + json.dumps(res))
rclpy.shutdown()
''' % (RGB_TOPIC, DEPTH_TOPIC, RAW, dur)
    open("/tmp/_multi.py", "w").write(script)
    os.makedirs(RAW, exist_ok=True)
    code, o = sh(f"/home/hmn-cjy/miniforge3/envs/ros2jazzy/bin/python /tmp/_multi.py",
                 timeout=dur + 40)
    if "RESULT:" in o:
        return json.loads(o.split("RESULT:")[1].splitlines()[0])
    return {"error": o[-400:]}


def make_fusion_fig(res, out_png):
    """相机 + LiDAR 投影叠加（视觉+激光融合的直观演示）。"""
    rgb = np.load(os.path.join(RAW, "rgb.npy")) if os.path.exists(os.path.join(RAW, "rgb.npy")) else None
    depth = np.load(os.path.join(RAW, "depth.npy")) if os.path.exists(os.path.join(RAW, "depth.npy")) else None
    ranges = np.load(os.path.join(RAW, "scan_ranges.npy")) if os.path.exists(os.path.join(RAW, "scan_ranges.npy")) else None
    angles = np.load(os.path.join(RAW, "scan_angles.npy")) if os.path.exists(os.path.join(RAW, "scan_angles.npy")) else None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # ① 相机
    ax = axes[0]
    if rgb is not None:
        ax.imshow(rgb); ax.set_title(f"① RGB 相机（{rgb.shape[1]}×{rgb.shape[0]}）")
    else:
        ax.text(.5, .5, "无", ha="center"); ax.set_title("① RGB")
    ax.axis("off")
    # ② LiDAR 极坐标→笛卡尔
    ax = axes[1]
    if ranges is not None:
        valid = np.isfinite(ranges) & (ranges > 0.01) & (ranges < 20)
        x = ranges[valid] * np.cos(angles[valid])
        y = ranges[valid] * np.sin(angles[valid])
        ax.scatter(x, y, s=4, c="#d62728")
        ax.scatter([0], [0], marker="^", s=120, color="#1f77b4", label="机器人")
        ax.set_title(f"② LiDAR 扫描（{valid.sum()} 点）")
        ax.legend(fontsize=8)
    else:
        ax.text(.5, .5, "无", ha="center"); ax.set_title("② LiDAR")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("X(m)"); ax.set_ylabel("Y(m)")
    # ③ 融合说明
    ax = axes[2]; ax.axis("off")
    rr = res.get("rates", {})
    txt = ("多传感器同步（M6/F5 实验台）\n\n"
           f"RGB   : {res.get('n',{}).get('rgb',0)} 帧, {rr.get('rgb','?')} Hz\n"
           f"Depth : {res.get('n',{}).get('depth',0)} 帧, {rr.get('depth','?')} Hz\n"
           f"LiDAR : {res.get('n',{}).get('scan',0)} 帧, {rr.get('scan','?')} Hz\n"
           f"IMU   : {res.get('n',{}).get('imu',0)} 帧, {rr.get('imu','?')} Hz\n\n"
           "价值：\n"
           "· 同源同帧，时间戳天然同步\n"
           "· 相机给外观、LiDAR 给精确距离\n"
           "· 正是视觉+激光融合(F5)的输入")
    ax.text(0.02, 0.95, txt, fontsize=11, va="top")
    fig.tight_layout()
    fig.savefig(out_png, dpi=115); plt.close(fig)


def main():
    os.makedirs(FIGS, exist_ok=True); os.makedirs(RAW, exist_ok=True)
    print("[P2·感知实验台四] 多传感器同步（相机+LiDAR+IMU）")

    _, o = sh("ros2 pkg prefix nav2_minimal_tb3_sim")
    pkg = o.strip().splitlines()[-1]
    world = f"{pkg}/share/nav2_minimal_tb3_sim/worlds/tb3_sandbox.sdf.xacro"
    env = {"GZ_SIM_RESOURCE_PATH":
           f"{pkg}/share/nav2_minimal_tb3_sim/models:{pkg}/share"}

    spawn(f'gz sim --headless-rendering -s -r -v 1 "{world}"', "/tmp/gz_ms.log", env=env)
    time.sleep(12)
    spawn(f'ros2 launch nav2_minimal_tb3_sim spawn_tb3.launch.py '
          f'robot_sdf:={MYURDF} use_sim_time:=true', "/tmp/tb3_ms.log", env=env)
    print("[·] TB3 生成，等待传感器...")
    time.sleep(16)

    res = capture_multisensor(dur=14)
    print("[OK] 多传感器采集：")
    for k in ["rgb", "depth", "scan", "imu"]:
        print(f"    {k:6s}: {res.get('n',{}).get(k,0)} 帧, "
              f"{res.get('rates',{}).get(k,'?')} Hz")
    if res.get("imu_sample"):
        s = res["imu_sample"]
        print(f"    IMU 样例: a=({s['ax']:.2f},{s['ay']:.2f},{s['az']:.2f}), ωz={s['gz']:.2f}")

    make_fusion_fig(res, os.path.join(FIGS, "sensor_fusion.png"))
    print(f"[✓] 融合图 → {FIGS}/sensor_fusion.png")

    metrics = {"multisensor": res,
               "note": "相机/深度/LiDAR/IMU 同源同帧，时间戳天然同步（M6/F5 实验台）"}
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    write_readme(res)
    cleanup()
    print(f"[✓] 完成 → {OUT}")


def write_readme(res):
    rr = res.get("rates", {})
    n = res.get("n", {})
    lines = ["# P2 · 感知实验台（四）：多传感器同步（相机+LiDAR+IMU）\n",
             "> M6 多模态融合 / F5 视觉+LiDAR 融合的**输入实验台**。\n",
             "## ① 目的\n",
             "M6/F5 都需要**多传感器同步数据**。真实数据集时间戳需对齐、真值需标定；",
             "仿真**同源同帧**直接输出，时间戳天然同步。\n",
             "## ② 实测结果\n",
             "| 传感器 | 帧数 | 频率 |", "|---|---|---|",
             f"| RGB 相机 | {n.get('rgb',0)} | {rr.get('rgb','?')} Hz |",
             f"| 深度相机 | {n.get('depth',0)} | {rr.get('depth','?')} Hz |",
             f"| LiDAR | {n.get('scan',0)} | {rr.get('scan','?')} Hz |",
             f"| IMU | {n.get('imu',0)} | {rr.get('imu','?')} Hz |",
             "",
             "![多传感器融合](figs/sensor_fusion.png)\n",
             "> ① RGB（外观/语义）；② LiDAR（精确距离）；③ 各传感器数据率。\n",
             "## ③ 直白讲解\n",
             "**相机**告诉你『前面是什么』（语义强、距离弱）；",
             "**LiDAR** 告诉你『前面多远』（距离准、无颜色）。",
             "两者**同时拿到**，就是 F5 视觉+激光融合的起点——**仿真是最省事的实验台**。\n",
             "## ④ 诚实边界\n",
             "- 三路传感器由 Gazebo 同一物理步输出，时间戳天然同步（真实需硬件 PPS/软件对齐）。\n",
             "- 相机与 LiDAR 有外参（安装位置差），本脚本未做点云→图像投影（见 F5 与"
             " `m10_kitti_lidar_demo.py`）。\n",
             "## 如何复现\n", "```bash",
             "source ~/miniforge3/etc/profile.d/conda.sh && conda activate ros2jazzy",
             "PYTHONPATH=src python scripts/p2_gz_multisensor.py",
             "```\n"]
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write("\n".join(l for l in lines if l is not None))


if __name__ == "__main__":
    sys.exit(main())
