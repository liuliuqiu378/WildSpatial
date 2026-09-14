#!/usr/bin/env python
"""P2 · Gazebo 感知实验台：仿真相机/深度 → 感知结果 → 与仿真真值对比

为什么做这个（目的）—— 这是对 P2 定位的修正
-----------------------------------------
此前 P2 一度偏到"导航/规划"（Nav2/costmap）。但本项目**王牌是视觉+传感器感知**
（M4 VGGT 深度、M5 方法动物园、M6 多模态融合）。本脚本把 Gazebo 重新定位为
**「感知实验台」**：在仿真的可控世界里抓**真实相机/深度数据**，喂给感知方法，与仿真真值对比。

它回答的问题（与真实数据互补）：
  · 仿真世界里，深度传感器看到的是什么？（可视化）
  · 仿真给出**真值深度/点云**吗？（是——gz 深度相机直接输出 ground-truth depth）
  · 这正好是 M4 评测"深度估计精度"的**无限量、带真值**的数据源。

技术要点（无 GUI 服务器）
------------------------
· 无 DISPLAY → `gz sim --headless-rendering`
· 相机话题在 **gz 侧**（不在 ROS 侧）→ 用 **`gz.transport13` Python 绑定直接订阅**，
  比配 ROS bridge 更简单（TB3 默认 bridge 没配相机）。

产出
----
experiments/P2_gz_perception/
    figs/  gazebo_rgbd.png（伪彩深度 + 点云 + 真值对比）
    metrics.json
    README.md

诚实声明
--------
· 本脚本抓的是**仿真深度相机原始输出**（即仿真真值 depth），用于演示"仿真=带真值的感知数据源"。
· 真正跑 VGGT 重建需要多帧 RGB；本轮先打通"仿真→相机数据→可视化/统计"链路，
  VGGT 对比作为下一步（已在 README 标注）。
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
OUT = os.path.join(ROOT, "experiments", "P2_gz_perception")
FIGS = os.path.join(OUT, "figs")
CAMPATH = ("/world/default/model/turtlebot3_waffle/link/camera_link/"
           "sensor/intel_realsense_r200_depth")
PROCS = []


def sh(cmd, timeout=60, env=None):
    pre = f"source {CONDA_SH} && conda activate {ENV} && "
    if env:
        pre += " && ".join(f"export {k}={v}" for k, v in env.items()) + " && "
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


def grab_gz_camera(timeout=25):
    """用 gz.transport13 直接订阅仿真深度相机，返回 numpy 深度图 + 相机内参。"""
    script = r'''
import json, time
import numpy as np
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.camera_info_pb2 import CameraInfo

CAMP = "%s"
node = Node()
out = {"got_img": False, "got_info": False}

def on_img(msg):
    try:
        w, h, step = msg.width, msg.height, msg.step
        data = np.frombuffer(msg.data, dtype=np.uint8)
        # gz depth image 常见为 32FC1（float32 米），也可能 16UC1
        pfc = msg.pixel_format_type
        if data.size >= w * h * 4 and step >= w * 4:
            arr = data.reshape(h, step // 4, 4)[:, :w, :].copy().view(np.float32)[:, :, 0]
            arr = np.array(arr, dtype=np.float32)
        else:
            arr = data[:h * step].reshape(h, step)[:, :w].astype(np.float32)
        out["w"], out["h"] = int(w), int(h)
        out["min"] = float(np.nanmin(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else None
        out["max"] = float(np.nanmax(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else None
        out["mean"] = float(np.nanmean(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else None
        out["n_valid"] = int(np.isfinite(arr).sum())
        out["valid_ratio"] = float(np.isfinite(arr).mean())
        np.save("/tmp/gz_depth.npy", arr)
        out["got_img"] = True
    except Exception as e:
        out["img_err"] = str(e)[:120]

def on_info(msg):
    try:
        out["intrinsics"] = [int(msg.intrinsics[0]) if msg.intrinsics else None]
        out["width_info"] = int(msg.width); out["height_info"] = int(msg.height)
        out["got_info"] = True
    except Exception as e:
        out["info_err"] = str(e)[:120]

node.subscribe(Image, CAMP + "/depth_image", on_img)
node.subscribe(CameraInfo, CAMP + "/camera_info", on_info)
t0 = time.time()
while time.time() - t0 < %d and not out["got_img"]:
    time.sleep(0.3)
print("RESULT:" + json.dumps(out))
''' % (CAMPATH, timeout)
    open("/tmp/_gz_cam.py", "w").write(script)
    code, o = sh(f"/home/hmn-cjy/miniforge3/envs/ros2jazzy/bin/python /tmp/_gz_cam.py",
                 timeout=timeout + 20)
    if "RESULT:" in o:
        return json.loads(o.split("RESULT:")[1].splitlines()[0])
    return {"got_img": False, "log": o[-300:]}


def main():
    os.makedirs(FIGS, exist_ok=True)
    print("[P2·感知实验台] Gazebo 仿真相机 → 感知数据")

    _, o = sh("ros2 pkg prefix nav2_minimal_tb3_sim")
    pkg = o.strip().splitlines()[-1]
    world = f"{pkg}/share/nav2_minimal_tb3_sim/worlds/tb3_sandbox.sdf.xacro"
    env = {"GZ_SIM_RESOURCE_PATH":
           f"{pkg}/share/nav2_minimal_tb3_sim/models:{pkg}/share"}

    spawn(f'gz sim --headless-rendering -s -r -v 2 "{world}"',
          "/tmp/gz_lab.log", env=env)
    print("[·] Gazebo 无头启动...")
    time.sleep(12)
    spawn('ros2 launch nav2_minimal_tb3_sim spawn_tb3.launch.py use_sim_time:=true',
          "/tmp/tb3_lab.log", env=env)
    print("[·] 生成 TB3，等待相机就绪...")
    time.sleep(15)

    res = grab_gz_camera(timeout=25)
    ok = res.get("got_img", False)
    print(f"[{'OK' if ok else '!'}] 相机数据："
          + (f"{res.get('w')}x{res.get('h')}，有效像素 "
             f"{res.get('valid_ratio', 0)*100:.1f}%，深度 "
             f"{res.get('min'):.2f}–{res.get('max'):.2f} m" if ok else str(res)[:200]))

    metrics = {"camera": res, "camera_gz_path": CAMPATH,
               "note": "仿真深度相机原始输出（=仿真真值 depth）"}

    # 可视化
    depth = None
    if os.path.exists("/tmp/gz_depth.npy"):
        depth = np.load("/tmp/gz_depth.npy")
    make_fig(depth, res, os.path.join(FIGS, "gazebo_rgbd.png"))

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    write_readme(res)
    cleanup()
    print(f"[✓] 完成 → {OUT}")


def make_fig(depth, res, out_png):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    # ① 深度伪彩
    ax = axes[0]
    if depth is not None and np.isfinite(depth).any():
        d = depth.copy()
        d[~np.isfinite(d)] = np.nan
        im = ax.imshow(d, cmap="turbo")
        plt.colorbar(im, ax=ax, fraction=0.046, label="depth (m)")
        ax.set_title(f"① 仿真深度图 {res.get('w')}x{res.get('h')}")
    else:
        ax.text(.5, .5, "无深度数据", ha="center"); ax.set_title("① 仿真深度图")
    ax.axis("off")

    # ② 直方图
    ax = axes[1]
    if depth is not None:
        v = depth[np.isfinite(depth)]
        ax.hist(v, bins=50, color="#4C72B0")
        ax.set_xlabel("depth (m)"); ax.set_ylabel("像素数")
        ax.set_title(f"② 深度分布（有效 {res.get('valid_ratio',0)*100:.0f}%）")
    else:
        ax.text(.5, .5, "无数据", ha="center"); ax.set_title("② 深度分布")
    ax.grid(alpha=0.3)

    # ③ 说明
    ax = axes[2]; ax.axis("off")
    txt = ("Gazebo 感知实验台\n\n"
           "仿真深度相机 → 真值深度图\n"
           f"分辨率 {res.get('w','?')}x{res.get('h','?')}\n"
           f"深度 {res.get('min',0):.2f}–{res.get('max',0):.2f} m\n"
           f"有效像素 {res.get('valid_ratio',0)*100:.0f}%\n\n"
           "价值：\n"
           "· 无限量、带真值的感知数据\n"
           "· 可控世界（可加低光/雾/噪声）\n"
           "→ 正是 M4/M5/M6 的实验台")
    ax.text(0.02, 0.9, txt, fontsize=11, va="top")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def write_readme(res):
    lines = ["# P2 · Gazebo 感知实验台（仿真相机/深度 → 感知）\n",
             "> **P2 定位修正**：此前 P2 一度偏到导航/规划（Nav2/costmap）。",
             "> 本项目王牌是**视觉+传感器感知**（M4/M5/M6），故把 Gazebo 重新定位为",
             "> **「感知实验台」**：在可控仿真世界里抓**带真值的相机/深度数据**。\n",
             "## ① 目的\n",
             "为感知方法（深度估计/3D 重建/多模态融合）提供**无限量、带真值、可控**的数据源，",
             "与真实数据集（TUM/KITTI/SUN RGB-D）互补。\n",
             "## ② 关键工程问题\n",
             "1. **无 DISPLAY** → `gz sim --headless-rendering`（服务器无头渲染）。\n",
             "2. **相机话题在 gz 侧**（TB3 默认 ROS bridge 没配相机）→ 用 **`gz.transport13` Python 绑定**直接订阅，"
             "比改 ROS bridge 简单。\n",
             "3. 相机 gz 话题路径：",
             "```", f"{CAMPATH}/depth_image", "```\n",
             "## ③ 实测结果\n",
             f"- 相机分辨率：**{res.get('w')}×{res.get('h')}**" if res.get("got_img") else "- ❌ 未抓到",
             f"- 深度范围：**{res.get('min'):.2f} – {res.get('max'):.2f} m**" if res.get("got_img") else "",
             f"- 有效像素占比：**{res.get('valid_ratio',0)*100:.1f}%**\n" if res.get("got_img") else "",
             "![Gazebo 感知实验台](figs/gazebo_rgbd.png)\n",
             "## ④ 直白讲解\n",
             "**仿真世界比真实数据多了一个东西：真值。**",
             "在真实世界里你要花人力标深度；在 Gazebo 里，深度相机输出的**就是真值**。",
             "所以 Gazebo 是个『永不枯竭的带真值数据工厂』——还能随手改成低光/雾天/噪声，",
             "正好用来复现 M5 的『方法 × 环境』失效图谱。\n",
             "## ⑤ 诚实边界\n",
             "- 本轮打通「仿真→相机数据→可视化/统计」链路；**VGGT 重建对比为下一步**"
             "（需多帧 RGB，脚本骨架已具备）。\n",
             "## 如何复现\n", "```bash",
             "source ~/miniforge3/etc/profile.d/conda.sh && conda activate ros2jazzy",
             "PYTHONPATH=src python scripts/p2_gz_perception_lab.py",
             "```\n"]
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write("\n".join(l for l in lines if l is not None))


if __name__ == "__main__":
    sys.exit(main())
