#!/usr/bin/env python
"""P2 · 感知实验台（一）：仿真 RGB-D 采集 + 可控退化（M5 落点）

目的（回到感知主线）
------------------
`scripts/p2_gz_perception_lab.py` 只抓了一帧深度。本脚本进一步：
  ① 在 Gazebo 里**驱动 TB3 移动**，采集**多帧 RGB + 真值深度**（像真实数据集一样）；
  ② 对采集的图**施加可控退化**（低光 / 雾 / 噪声）——**复现 M5「方法 × 环境」失效图谱**；
  ③ 输出多张**效果图**，帮初学者建立"仿真→感知数据→退化"的直观概念。

为什么仿真比合成退化更强
------------------------
真实数据 + 合成退化：退化是"事后 P 图"；
仿真：**物理引擎在起作用**（光照/材质/几何都是算出来的），退化更接近真实物理。

产出
----
experiments/P2_gz_sense_degrade/
    figs/  gz_rgbd_frames.png     多帧 RGB + 真值深度（机器人视角）
           degradation_gallery.png 同一帧 × 多种退化（低光/雾/噪声）
           depth_under_degrade.png 退化对深度统计的影响
    raw/   frame_*.npy（RGB+depth 原始数据，供 VGGT 下一步用）
    metrics.json / README.md

诚实声明
--------
· RGB 相机是**注入**到 TB3 SDF 的（原 TB3 只有深度相机）——真实机器人都有 RGB，属合理补全。
· 退化为**在仿真 RGB 上合成施加**（照明/雾/噪声），已在图注标注。
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
OUT = os.path.join(ROOT, "experiments", "P2_gz_sense_degrade")
FIGS = os.path.join(OUT, "figs")
RAW = os.path.join(OUT, "raw")
MYURDF = os.path.join(ROOT, "data", "gz_models", "urdf", "gz_waffle.sdf.xacro")
PROCS = []

DEPTH_TOPIC = ("/world/default/model/turtlebot3_waffle/link/camera_link/"
               "sensor/intel_realsense_r200_depth/depth_image")
RGB_TOPIC = "/camera/color/image_raw"


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


def capture_frames(n_frames=4, drive=True):
    """订阅仿真 RGB+深度，边驱动边抓多帧。返回 metrics + 存 npy。"""
    script = r'''
import json, time, os
import numpy as np
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.twist_pb2 import Twist

RGB = "%s"
DEP = "%s"
NF = %d
DRIVE = %s
OUT = "%s"

node = Node()
rgb_holder = {"img": None, "t": 0}
dep_holder = {"img": None, "t": 0}

def dec_rgb(msg):
    w, h, step = msg.width, msg.height, msg.step
    d = np.frombuffer(msg.data, dtype=np.uint8)
    return d[:h*step].reshape(h, step)[:,:w*3].reshape(h, w, 3).copy()

def dec_depth(msg):
    w, h, step = msg.width, msg.height, msg.step
    d = np.frombuffer(msg.data, dtype=np.uint8)
    if d.size >= w*h*4:
        arr = d.reshape(h, step//4, 4)[:, :w, :].copy().view(np.float32)[:, :, 0]
    else:
        arr = d[:h*step].reshape(h, step)[:, :w].astype(np.float32)
    return np.array(arr, dtype=np.float32)

def on_rgb(m): rgb_holder["img"] = dec_rgb(m); rgb_holder["t"] += 1
def on_dep(m): dep_holder["img"] = dec_depth(m); dep_holder["t"] += 1

node.subscribe(Image, RGB, on_rgb)
node.subscribe(Image, DEP, on_dep)

# 驱动：切到 cmd_vel，缓慢前进+小转向，让视角变化
pub = None
if DRIVE:
    pub = node.advertise("cmd_vel", Twist)

frames = []
t0 = time.time()
saved = 0
last_save = 0
while time.time() - t0 < 40 and saved < NF:
    if pub is not None:
        tw = Twist()
        tw.linear.x = 0.08
        tw.angular.z = 0.15
        pub.publish(tw)
    if rgb_holder["img"] is not None and dep_holder["img"] is not None:
        # 每 5 秒存一帧（视角已变化）
        if time.time() - last_save > 5:
            np.save(os.path.join(OUT, f"frame_%%02d_rgb.npy" %% saved), rgb_holder["img"])
            np.save(os.path.join(OUT, f"frame_%%02d_depth.npy" %% saved), dep_holder["img"])
            frames.append({"idx": saved,
                           "rgb_shape": list(rgb_holder["img"].shape),
                           "depth_shape": list(dep_holder["img"].shape),
                           "depth_min": float(np.nanmin(dep_holder["img"][np.isfinite(dep_holder["img"])])) if np.isfinite(dep_holder["img"]).any() else None,
                           "depth_max": float(np.nanmax(dep_holder["img"][np.isfinite(dep_holder["img"])])) if np.isfinite(dep_holder["img"]).any() else None})
            saved += 1
            last_save = time.time()
    time.sleep(0.2)
print("RESULT:" + json.dumps({"n_rgb": rgb_holder["t"], "n_depth": dep_holder["t"],
                              "frames": frames}))
''' % (RGB_TOPIC, DEPTH_TOPIC, n_frames, "True" if drive else "False", RAW)
    open("/tmp/_gz_frames.py", "w").write(script)
    code, o = sh(f"/home/hmn-cjy/miniforge3/envs/ros2jazzy/bin/python /tmp/_gz_frames.py",
                 timeout=90)
    if "RESULT:" in o:
        return json.loads(o.split("RESULT:")[1].splitlines()[0])
    return {"error": o[-300:]}


# ------------------------------------------------------------------ 退化套件
def degrade_lowlight(rgb, sev=1.0):
    return np.clip(rgb.astype(np.float32) * (1.0 - 0.8 * sev), 0, 255).astype(np.uint8)


def degrade_fog(rgb, sev=1.0):
    """简化雾模型：向白色按距离/强度混合。"""
    f = rgb.astype(np.float32)
    white = 235.0
    out = f * (1 - 0.6 * sev) + white * (0.6 * sev)
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade_noise(rgb, sev=1.0):
    n = np.random.default_rng(0).normal(0, 60 * sev, rgb.shape)
    return np.clip(rgb.astype(np.float32) + n, 0, 255).astype(np.uint8)


def main():
    os.makedirs(FIGS, exist_ok=True); os.makedirs(RAW, exist_ok=True)
    print("[P2·感知实验台一] 仿真 RGB-D 采集 + 可控退化")

    _, o = sh("ros2 pkg prefix nav2_minimal_tb3_sim")
    pkg = o.strip().splitlines()[-1]
    world = f"{pkg}/share/nav2_minimal_tb3_sim/worlds/tb3_sandbox.sdf.xacro"
    env = {"GZ_SIM_RESOURCE_PATH":
           f"{pkg}/share/nav2_minimal_tb3_sim/models:{pkg}/share"}

    spawn(f'gz sim --headless-rendering -s -r -v 1 "{world}"', "/tmp/gz_sd.log", env=env)
    print("[·] Gazebo 启动...")
    time.sleep(12)
    spawn(f'ros2 launch nav2_minimal_tb3_sim spawn_tb3.launch.py '
          f'robot_sdf:={MYURDF} use_sim_time:=true', "/tmp/tb3_sd.log", env=env)
    print("[·] TB3（含注入 RGB 相机）生成...")
    time.sleep(18)

    res = capture_frames(n_frames=4, drive=True)
    print(f"[OK] 采集：RGB {res.get('n_rgb')} 帧到达，存 {len(res.get('frames', []))} 帧")
    for f in res.get("frames", []):
        print(f"    frame{f['idx']}: rgb{f['rgb_shape']} depth{f['depth_shape']} "
              f"{f['depth_min']:.2f}-{f['depth_max']:.2f}m")

    # ---- 出图 1：多帧 RGB + 真值深度 ----
    frames = res.get("frames", [])
    if frames:
        fig, axes = plt.subplots(2, len(frames), figsize=(3.4 * len(frames), 6))
        if len(frames) == 1:
            axes = axes.reshape(2, 1)
        for i, f in enumerate(frames):
            rgb = np.load(os.path.join(RAW, f"frame_{f['idx']:02d}_rgb.npy"))
            dep = np.load(os.path.join(RAW, f"frame_{f['idx']:02d}_depth.npy"))
            axes[0, i].imshow(rgb)
            axes[0, i].set_title(f"RGB 帧{i}", fontsize=9); axes[0, i].axis("off")
            d = dep.copy(); d[~np.isfinite(d)] = np.nan
            im = axes[1, i].imshow(d, cmap="turbo")
            axes[1, i].set_title(f"真值深度 {f['depth_min']:.1f}-{f['depth_max']:.1f}m",
                                 fontsize=9)
            axes[1, i].axis("off")
        fig.suptitle("① Gazebo 感知实验台：机器人边走边采（上=RGB，下=真值深度）",
                     fontsize=13, y=1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "gz_rgbd_frames.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[✓] 多帧图 → {FIGS}/gz_rgbd_frames.png")

    # ---- 出图 2：退化图库 ----
    if frames:
        base = np.load(os.path.join(RAW, f"frame_{frames[0]['idx']:02d}_rgb.npy"))
        variants = [
            ("原始（仿真）", base),
            ("低光 sev=0.5", degrade_lowlight(base, 0.5)),
            ("低光 sev=1.0", degrade_lowlight(base, 1.0)),
            ("雾天 sev=0.5", degrade_fog(base, 0.5)),
            ("雾天 sev=1.0", degrade_fog(base, 1.0)),
            ("高斯噪声 sev=1.0", degrade_noise(base, 1.0)),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.4))
        for ax, (t, im) in zip(axes.ravel(), variants):
            ax.imshow(im); ax.set_title(t, fontsize=10); ax.axis("off")
        fig.suptitle("② 可控退化图库（同一仿真帧 × 6 种退化）——M5 失效图谱的原料",
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "degradation_gallery.png"), dpi=110)
        plt.close(fig)
        print(f"[✓] 退化图库 → {FIGS}/degradation_gallery.png")

    # ---- 出图 3：退化对"感知难度"的影响（亮度 + 对比度量化）----
    if frames:
        base = np.load(os.path.join(RAW, f"frame_{frames[0]['idx']:02d}_rgb.npy"))
        cases = [("原始", base), ("低光0.5", degrade_lowlight(base, .5)),
                 ("低光1.0", degrade_lowlight(base, 1.0)),
                 ("雾0.5", degrade_fog(base, .5)), ("雾1.0", degrade_fog(base, 1.0)),
                 ("噪声1.0", degrade_noise(base, 1.0))]
        names = [c[0] for c in cases]
        bright = [float(c[1].mean()) for c in cases]
        contrast = [float(c[1].std()) for c in cases]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        axes[0].bar(names, bright, color="#4C72B0")
        axes[0].set_title("退化对图像亮度的影响（越低越暗）")
        axes[0].set_ylabel("平均亮度"); axes[0].tick_params(axis="x", rotation=20)
        axes[1].bar(names, contrast, color="#DD8452")
        axes[1].set_title("退化对图像对比度的影响（越低越模糊）")
        axes[1].set_ylabel("标准差"); axes[1].tick_params(axis="x", rotation=20)
        for ax in axes:
            ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "degradation_stats.png"), dpi=110)
        plt.close(fig)
        print(f"[✓] 退化统计 → {FIGS}/degradation_stats.png")

    metrics = {
        "capture": res,
        "rgb_topic": RGB_TOPIC, "depth_topic": DEPTH_TOPIC,
        "injected_rgb": True,
        "degradations": ["low_light(0.5/1.0)", "fog(0.5/1.0)", "gaussian_noise(1.0)"],
        "note": "RGB 相机为注入（原 TB3 仅深度）；退化为仿真 RGB 上合成施加",
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    write_readme(metrics)
    cleanup()
    print(f"[✓] 完成 → {OUT}")


def write_readme(metrics):
    fr = metrics["capture"].get("frames", [])
    lines = ["# P2 · 感知实验台（一）：仿真 RGB-D 采集 + 可控退化\n",
             "> 回到**视觉/传感器主线**：在 Gazebo 里边走边采 RGB-D，再施加可控退化。\n",
             "## ① 目的\n",
             "为感知方法（M4 深度 / M5 失效图谱 / M6 融合）提供**带真值、可控、无限量**的数据。\n",
             "## ② 关键工程问题\n",
             "1. **TB3 原模型只有深度相机**（`intel_realsense_r200_depth`）→ 本脚本**注入了一个 RGB 相机**"
             "（640×480，原 SDF 副本在 `data/gz_models/urdf/`），用 `robot_sdf:=` 参数覆盖。\n",
             "2. 用 `gz.transport13` 直接订阅 gz 话题（比配 ROS bridge 简单）。\n",
             "3. 通过 `cmd_vel` 驱动 TB3 缓慢移动，**边动边采**（视角变化）。\n",
             "## ③ 实测结果\n",
             f"- 采集帧数：**{len(fr)}**" if fr else "- ❌ 未采到帧",
             f"- RGB 分辨率：**{fr[0]['rgb_shape']}**、深度：**{fr[0]['depth_shape']}**" if fr else "",
             f"- 深度范围：**{fr[0]['depth_min']:.2f} – {fr[0]['depth_max']:.2f} m**\n" if fr else "",
             "![多帧 RGB + 真值深度](figs/gz_rgbd_frames.png)\n",
             "> 上行 = 机器人视角 RGB；下行 = 对应**真值深度**（伪彩）。",
             "> 注意机器人**边移动边采**，视角在变化——这正是一段『仿真录制的数据集』。\n",
             "![可控退化图库](figs/degradation_gallery.png)\n",
             "> 同一帧 × 6 种退化：低光、雾天、高斯噪声——**这正是 M5『方法 × 环境』失效图谱的原料**。\n",
             "![退化统计](figs/degradation_stats.png)\n",
             "> 量化退化对『感知难度』的影响：亮度↓（低光）、对比度↓（雾/噪声）。\n",
             "## ④ 直白讲解\n",
             "**真实数据 + 合成退化**：退化是事后『P 图』，光照/材质是假的；",
             "**仿真**：物理引擎在算，光照/材质/几何都是真的——退化更接近真实物理。",
             "所以 Gazebo 是『**比合成退化更真、比真实数据更可控**』的中间地带。\n",
             "## ⑤ 诚实边界\n",
             "- RGB 相机为**注入**（真实 TB3 waffle 常配 RealSense RGB-D，属合理补全，已标注）。\n",
             "- 退化为**仿真 RGB 上合成施加**（非仿真引擎内的雾）；下一步可直接在 SDF 里加雾/改光照。\n",
             "## 如何复现\n", "```bash",
             "source ~/miniforge3/etc/profile.d/conda.sh && conda activate ros2jazzy",
             "PYTHONPATH=src python scripts/p2_gz_sense_degrade.py",
             "```\n"]
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write("\n".join(l for l in lines if l is not None))


if __name__ == "__main__":
    sys.exit(main())
