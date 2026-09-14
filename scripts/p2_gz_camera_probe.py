#!/usr/bin/env python
"""P2 · 探测 Gazebo TB3 的相机话题（RGB/深度），为"感知实验台"铺路。

目的：确认 Gazebo 仿真里能否抓到**相机 RGB / 深度**，这是"仿真→VGGT 重建"闭环的前提。
"""
import os, sys, json, time, signal, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDA_SH = "/home/hmn-cjy/miniforge3/etc/profile.d/conda.sh"
ENV = "ros2jazzy"
PROCS = []


def sh(cmd, timeout=60):
    full = f"source {CONDA_SH} && conda activate {ENV} && {cmd}"
    try:
        r = subprocess.run(["bash", "-lc", full], capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def spawn(cmd, logfile):
    full = f"source {CONDA_SH} && conda activate {ENV} && {cmd}"
    f = open(logfile, "w")
    p = subprocess.Popen(["bash", "-lc", full], stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    PROCS.append(p); return p


def cleanup():
    for p in PROCS:
        try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception: pass
    time.sleep(2)


_, out = sh("ros2 pkg prefix nav2_minimal_tb3_sim")
pkg = out.strip().splitlines()[-1]
envprefix = (f'export GZ_SIM_RESOURCE_PATH="{pkg}/share/nav2_minimal_tb3_sim/models:{pkg}/share" && ')
world = f"{pkg}/share/nav2_minimal_tb3_sim/worlds/tb3_sandbox.sdf.xacro"

print("[probe] 启动 Gazebo 无头 + TB3")
spawn(f'{envprefix}gz sim --headless-rendering -s -r -v 2 "{world}"', "/tmp/gz_probe.log")
time.sleep(12)
spawn(f'{envprefix}ros2 launch nav2_minimal_tb3_sim spawn_tb3.launch.py use_sim_time:=true',
      "/tmp/tb3_probe.log")
time.sleep(15)

code, out = sh("ros2 topic list", timeout=40)
topics = sorted(out.split())
print("\n=== ROS 侧话题 ===")
for t in topics:
    print("  ", t)

# 关键：列出 **gz 侧**话题（相机等传感器在 gz 侧发布）
code, gout = sh("gz topic -l 2>/dev/null", timeout=40)
gzs = sorted([t for t in gout.split() if t.strip()])
print(f"\n=== GZ 侧话题（{len(gzs)}）===")
for t in gzs:
    print("  ", t)
gz_cam = [t for t in gzs if any(k in t.lower() for k in ("image", "camera", "depth"))]
print(f"\n=== GZ 相机相关（{len(gz_cam)}）===")
for t in gz_cam:
    print("  ", t)

cam = gz_cam
print(f"\n=== 相机相关话题（{len(cam)}）===")
for t in cam:
    print("  ", t)

# 尝试在 gz 侧抓一帧相机数据（确认传感器真的在出数据）
if cam:
    topic = cam[0]
    code, o = sh(f"gz topic -e -t {topic} -n 1 2>&1 | head -20", timeout=30)
    print(f"\n=== gz 相机数据预览（{topic}）===")
    print(o[:800] if o.strip() else "(空/超时)")

cleanup()
