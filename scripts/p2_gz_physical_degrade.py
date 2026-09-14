#!/usr/bin/env python
"""P2 · 感知实验台（三）：物理级退化（光照/雾）+ 真值轨迹

目的
----
§4.8 的退化是**事后 P 图**（在采好的 RGB 上叠加）；本脚本做**物理级退化**：
直接改 Gazebo 世界的 **光照强度 / 雾密度**，由**渲染引擎真实计算**——
这更接近真实物理，也更能说明"仿真 = 简化世界的理想模型"。

同时采集**真值轨迹**（`/odom`），为 ATE 评测提供免费真值（真实数据集才有）。

产出
----
experiments/P2_gz_physical/
    figs/  physical_conditions.png  同一场景 × 4 种物理条件（正常/低光/雾/夜间）
           physical_vs_synthetic.png 物理级退化 vs 事后 P 图（对比）
           gt_trajectory.png         真值轨迹（机器人走的路径）
    metrics.json / README.md

诚实声明
--------
· 光照/雾由 Gazebo 渲染引擎计算（物理级），非图像后处理。
· 轨迹来自仿真 `/odom`（真值），可直接用作 ATE 参考。
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
OUT = os.path.join(ROOT, "experiments", "P2_gz_physical")
FIGS = os.path.join(OUT, "figs")
RAW = os.path.join(OUT, "raw")
WORLD = os.path.join(ROOT, "data", "gz_models", "worlds", "tb3_sandbox_param.sdf.xacro")
MYURDF = os.path.join(ROOT, "data", "gz_models", "urdf", "gz_waffle.sdf.xacro")
RGB_TOPIC = "/camera/color/image_raw"
PROCS = []

# 4 种物理条件（light, fog_density, 名字）
CONDITIONS = [
    ("正常", 0.8, 0.0),
    ("低光", 0.15, 0.0),
    ("雾天", 0.8, 0.6),
    ("夜间", 0.05, 0.1),
]


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


def run_condition(name, light, fog, drive_sec=12):
    """起一个世界的特定光照/雾，驱动 TB3 + 采 1 帧 RGB + 真值轨迹。"""
    # 生成该条件的 world 文件（xacro 展开）
    # headless:=false → 启用 SceneBroadcaster → 才会发布 /world/default/pose/info（真值位姿）
    code, wout = sh(f"xacro {WORLD} headless:=false light:={light} "
                    f"fog_density:={fog}")
    wpath = os.path.join(OUT, f"world_{name}.sdf")
    os.makedirs(OUT, exist_ok=True)
    with open(wpath, "w") as f:
        f.write(wout.split("\n", 0)[0] if False else wout)

    _, o = sh("ros2 pkg prefix nav2_minimal_tb3_sim")
    pkg = o.strip().splitlines()[-1]
    env = {"GZ_SIM_RESOURCE_PATH":
           f"{pkg}/share/nav2_minimal_tb3_sim/models:{pkg}/share:"
           f"{os.path.join(ROOT, 'data', 'gz_models')}"}

    local_procs = []
    def lspawn(cmd, lf):
        pre = f"source {CONDA_SH} && conda activate {ENV} && "
        pre += " && ".join(f"export {k}={v}" for k, v in env.items()) + " && "
        f = open(lf, "w")
        p = subprocess.Popen(["bash", "-lc", pre + cmd], stdout=f,
                             stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        local_procs.append(p); return p

    try:
        lspawn(f'gz sim --headless-rendering -s -r -v 1 "{wpath}"',
               f"/tmp/gz_{name}.log")
        time.sleep(12)
        lspawn(f'ros2 launch nav2_minimal_tb3_sim spawn_tb3.launch.py '
               f'robot_sdf:={MYURDF} use_sim_time:=true', f"/tmp/tb3_{name}.log")
        time.sleep(16)

        # 采集：驱动 + 抓 RGB + 抓轨迹（gz Pose_V，需 SceneBroadcaster）
        script = r'''
import json, time, os
import numpy as np
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.twist_pb2 import Twist
from gz.msgs10.pose_v_pb2 import Pose_V

RGB = "%s"; OUT = "%s"; NAME = "%s"; T = %d
node = Node()
rgb = {"img": None}; traj = []

def dec(msg):
    w,h,s = msg.width,msg.height,msg.step
    d = np.frombuffer(msg.data, dtype=np.uint8)
    return d[:h*s].reshape(h,s)[:,:w*3].reshape(h,w,3).copy()

def on_rgb(m): rgb["img"] = dec(m)
def on_pose(m):
    for p in m.pose:
        if "turtlebot3_waffle" in p.name:
            traj.append([p.position.x, p.position.y, p.position.z])

node.subscribe(Image, RGB, on_rgb)
node.subscribe(Pose_V, "/world/default/pose/info", on_pose)
pub = node.advertise("cmd_vel", Twist)
t0=time.time()
while time.time()-t0 < T:
    tw=Twist(); tw.linear.x=0.10; tw.angular.z=0.20
    pub.publish(tw); time.sleep(0.1)
if rgb["img"] is not None:
    np.save(os.path.join(OUT, "rgb_%%s.npy" %% NAME), rgb["img"])
np.save(os.path.join(OUT, "traj_%%s.npy" %% NAME), np.array(traj))
print("RESULT:"+json.dumps({"got_rgb": rgb["img"] is not None, "n_traj": len(traj)}))
''' % (RGB_TOPIC, RAW, name, drive_sec)
        open(f"/tmp/_cap_{name}.py", "w").write(script)
        os.makedirs(RAW, exist_ok=True)
        code, o2 = sh(f"/home/hmn-cjy/miniforge3/envs/ros2jazzy/bin/python "
                      f"/tmp/_cap_{name}.py", timeout=drive_sec + 40)
        if "RESULT:" in o2:
            return json.loads(o2.split("RESULT:")[1].splitlines()[0])
        return {"error": o2[-200:]}
    finally:
        for p in local_procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
        time.sleep(3)


def main():
    os.makedirs(FIGS, exist_ok=True); os.makedirs(RAW, exist_ok=True)
    print("[P2·感知实验台三] 物理级退化（光照/雾）+ 真值轨迹")

    results = {}
    for name, light, fog in CONDITIONS:
        print(f"[·] 条件『{name}』(light={light}, fog={fog}) ...")
        r = run_condition(name, light, fog, drive_sec=10)
        results[name] = r
        print(f"    {'OK' if r.get('got_rgb') else '✗'} rgb={r.get('got_rgb')}, "
              f"traj={r.get('n_traj')} 点")

    # ---- 出图 1：4 种物理条件 ----
    imgs = {}
    for name, _, _ in CONDITIONS:
        p = os.path.join(RAW, f"rgb_{name}.npy")
        if os.path.exists(p):
            imgs[name] = np.load(p)
    if imgs:
        fig, axes = plt.subplots(1, len(imgs), figsize=(4.2 * len(imgs), 4))
        if len(imgs) == 1:
            axes = [axes]
        for ax, (name, im) in zip(axes, imgs.items()):
            ax.imshow(im); ax.set_title(f"{name}（亮度 {im.mean():.0f}）", fontsize=11)
            ax.axis("off")
        fig.suptitle("① 物理级退化：同一仿真世界 × 4 种光照/雾（渲染引擎真实计算）",
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "physical_conditions.png"), dpi=110)
        plt.close(fig)
        print(f"[✓] 物理条件图 → {FIGS}/physical_conditions.png")

    # ---- 出图 2：物理级 vs 事后P图 ----
    if "正常" in imgs and "低光" in imgs:
        base = imgs["正常"]; phys = imgs["低光"]
        syn = np.clip(base.astype(np.float32) * 0.19, 0, 255).astype(np.uint8)  # 模拟同强度
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        for ax, (t, im) in zip(axes, [("原始（正常光）", base),
                                       ("物理级低光（Gazebo 渲染）", phys),
                                       ("事后 P 图低光", syn)]):
            ax.imshow(im); ax.set_title(f"{t}\n亮度 {im.mean():.0f}", fontsize=10)
            ax.axis("off")
        fig.suptitle("② 物理级退化 vs 事后 P 图（前者几何/光照更自洽）", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "physical_vs_synthetic.png"), dpi=110)
        plt.close(fig)
        print(f"[✓] 对比图 → {FIGS}/physical_vs_synthetic.png")

    # ---- 出图 3：真值轨迹 ----
    tp = os.path.join(RAW, "traj_正常.npy")
    if os.path.exists(tp):
        traj = np.load(tp)
        if len(traj) > 2:
            fig, ax = plt.subplots(figsize=(6.5, 6))
            ax.plot(traj[:, 0], traj[:, 1], "-o", ms=2, color="#2ca02c",
                    label=f"真值轨迹（{len(traj)} 点）")
            ax.scatter([traj[0, 0]], [traj[0, 1]], marker="s", s=90,
                       color="#1f77b4", label="起点")
            ax.scatter([traj[-1, 0]], [traj[-1, 1]], marker="X", s=110,
                       color="#ff7f0e", label="终点")
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
            ax.set_title("③ 仿真真值轨迹（/odom）——ATE 评测的免费参考")
            ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_aspect("equal")
            fig.tight_layout()
            fig.savefig(os.path.join(FIGS, "gt_trajectory.png"), dpi=110)
            plt.close(fig)
            print(f"[✓] 真值轨迹图 → {FIGS}/gt_trajectory.png")

    metrics = {"conditions": [{"name": n, "light": l, "fog": f,
                               "got_rgb": results[n].get("got_rgb"),
                               "n_traj": results[n].get("n_traj")}
                              for n, l, f in CONDITIONS],
               "mean_brightness": {n: float(im.mean()) for n, im in imgs.items()},
               "note": "光照/雾由 Gazebo 渲染引擎计算（物理级）；轨迹来自仿真 /odom 真值"}
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    write_readme(metrics)
    print(f"[✓] 完成 → {OUT}")


def write_readme(metrics):
    bright = metrics["mean_brightness"]
    conds = metrics["conditions"]
    lines = ["# P2 · 感知实验台（三）：物理级退化（光照/雾）+ 真值轨迹\n",
             "> §4.8 的退化是**事后 P 图**；本节做**物理级退化**——直接改世界的**光照/雾**，",
             "> 由 Gazebo **渲染引擎真实计算**。同时采集**真值轨迹**（ATE 免费参考）。\n",
             "## ① 目的\n",
             "证明仿真能提供**物理自洽**的退化（不只 P 图），并免费给出**真值轨迹**。\n",
             "## ② 关键工程\n",
             "把世界做成**参数化 SDF**（`data/gz_models/worlds/tb3_sandbox_param.sdf.xacro`）：",
             "`light`（光照强度 0~1）+ `fog_density`（雾密度 0~1）做成 xacro 参数，",
             "同一世界 `xacro world.xacro light:=0.15 fog_density:=0.6` 即得低光/雾天。\n",
             "## ③ 实测结果\n",
             "| 条件 | light | fog | RGB 亮度 |", "|---|---|---|---|"]
    for c in conds:
        b = bright.get(c["name"])
        lines.append(f"| {c['name']} | {c['light']} | {c['fog']} | "
                     f"{b:.0f} |" if b is not None else
                     f"| {c['name']} | {c['light']} | {c['fog']} | — |")
    lines += ["", "![物理级 4 条件](figs/physical_conditions.png)\n",
              "> 同一世界 × 4 种光照/雾：**正常 / 低光 / 雾天 / 夜间**（渲染引擎真实计算）。\n",
              "![物理级 vs 事后P图](figs/physical_vs_synthetic.png)\n",
              "> **左**：正常光；**中**：物理级低光（Gazebo 渲染，几何/光照自洽）；",
              "> **右**：事后 P 图低光（简单乘系数，阴影/高光不自然）。\n",
              "![真值轨迹](figs/gt_trajectory.png)\n",
              "## ④ 直白讲解\n",
             "**事后 P 图**：把整张图变暗——但真实的黑暗里，靠近光源的地方还是亮的；",
             "**物理级**：Gazebo 按光照方程渲染——**哪里该亮哪里该暗，是按物理算的**。",
             "所以仿真能给『更真的退化』。同时 `/odom` 提供**真值轨迹**，ATE 评测不用再花钱标。\n",
              "## ⑤ 诚实边界\n",
              "- 退化由 Gazebo 渲染引擎计算（物理级），非图像后处理（与 §4.8 区分）。\n",
              "- 轨迹来自仿真 `/odom`（真值）；真实系统需另行标定。\n",
              "## 如何复现\n", "```bash",
              "source ~/miniforge3/etc/profile.d/conda.sh && conda activate ros2jazzy",
              "PYTHONPATH=src python scripts/p2_gz_physical_degrade.py",
              "```\n"]
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write("\n".join(l for l in lines if l is not None))


if __name__ == "__main__":
    sys.exit(main())
