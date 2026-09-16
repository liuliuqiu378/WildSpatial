#!/usr/bin/env python
"""探测 v3：**先建订阅**再 spawn 机器人（避免错过发现窗口），验证 RGB 真实画面。

诊断结论（v1/v2 失败原因推测）：gz.transport13 订阅要在 gz 服务端启动早期建立，
否则错过 topic 发现窗口 → subscribe 返回 True 但收不到数据。
命令行 `gz topic -e` 能收到，说明话题本身有数据。
"""
import os
import sys
import time
import signal
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDA_SH = "/home/hmn-cjy/miniforge3/etc/profile.d/conda.sh"
ENV = "ros2jazzy"
WORLD = os.path.join(ROOT, "data", "gz_models", "worlds",
                     "open_plaza_dynamic.sdf.xacro")
PROCS = []


def sh(cmd, t=90):
    pre = f"source {CONDA_SH} && conda activate {ENV} && "
    r = subprocess.run(["bash", "-lc", pre + cmd], capture_output=True,
                       text=True, timeout=t)
    return r.returncode, r.stdout + r.stderr


def spawn(cmd, lf, env=None):
    pre = f"source {CONDA_SH} && conda activate {ENV} && "
    if env:
        pre += " && ".join(f"export {k}={v}" for k, v in env.items()) + " && "
    f = open(lf, "w")
    p = subprocess.Popen(["bash", "-lc", pre + cmd], stdout=f,
                         stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    PROCS.append(p)
    return p


def main():
    _, o = sh("ros2 pkg prefix nav2_minimal_tb3_sim")
    pkg = o.strip().splitlines()[-1]
    env = {"GZ_SIM_RESOURCE_PATH":
           f"{pkg}/share/nav2_minimal_tb3_sim/models:{pkg}/share:"
           f"{os.path.join(ROOT, 'data', 'gz_models')}"}
    _, w = sh(f"xacro {WORLD} headless:=false light:=0.8 ped_speed:=0.35")
    open("/tmp/_fpv.sdf", "w").write(w)

    # 关键：**先启动订阅进程**（它在后台持续等待），再起 Gazebo + TB3
    script = r'''
import time
import numpy as np
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

node = Node()
imgs = {}
def make_cb(name):
    def cb(m):
        w,h,s = m.width, m.height, m.step
        d = np.frombuffer(m.data, dtype=np.uint8)
        imgs[name] = (w, h, s, d.copy())
    return cb

TOPICS = {
    "rgb": "/camera/color/image_raw",
    "depth": ("/world/default/model/turtlebot3_waffle/link/camera_link/"
              "sensor/intel_realsense_r200_depth/depth_image"),
}
for k, t in TOPICS.items():
    node.subscribe(Image, t, make_cb(k))

# 持续等待直到收到（最多 70s，覆盖 gz+tb3 启动全过程）
t0 = time.time()
last_report = 0
while time.time() - t0 < 70:
    if len(imgs) >= 2:
        time.sleep(2)
        break
    time.sleep(0.5)

print("GOT:", list(imgs.keys()), flush=True)
for k, (w,h,s,d) in imgs.items():
    try:
        a = d[:h*s].reshape(h,s)[:,:w*3].reshape(h,w,3)
        print(f"  {k}: {w}x{h} max={a.max()} mean={a.mean():.1f} "
              f"std={a.std():.1f} ach={a.reshape(-1,3).mean(0).round(1)}", flush=True)
        np.save("/tmp/_fpv_%s.npy" % k, a)
    except Exception as e:
        print(f"  {k}: decode ERR {e}", flush=True)
'''
    open("/tmp/_fpv_cap.py", "w").write(script)
    cap = subprocess.Popen(
        ["bash", "-lc",
         f"source {CONDA_SH} && conda activate {ENV} && "
         f"/home/hmn-cjy/miniforge3/envs/ros2jazzy/bin/python /tmp/_fpv_cap.py"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        preexec_fn=os.setsid)
    PROCS.append(cap)
    print("[·] 订阅进程已启动（先于 Gazebo）")
    time.sleep(3)

    spawn('gz sim --headless-rendering -s -r -v 1 "/tmp/_fpv.sdf"',
          "/tmp/gz_fpv.log", env=env)
    print("[·] Gazebo 启动...")
    time.sleep(12)
    spawn('ros2 launch nav2_minimal_tb3_sim spawn_tb3.launch.py '
          f'robot_sdf:={os.path.join(ROOT, "data/gz_models/urdf/gz_waffle.sdf.xacro")} '
          'use_sim_time:=true', "/tmp/tb3_fpv.log", env=env)
    print("[·] TB3 生成，等订阅进程输出...")

    try:
        out, _ = cap.communicate(timeout=75)
        print(out[-2000:])
    except subprocess.TimeoutExpired:
        cap.kill()
        print("[·] 订阅超时")

    for p in PROCS:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            pass
    time.sleep(2)


if __name__ == "__main__":
    sys.exit(main())
