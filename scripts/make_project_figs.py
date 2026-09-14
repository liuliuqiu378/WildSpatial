"""生成 P0 实战项目的「公开数据集例图」拼图（图文结合用）。

把四个项目的真实/合成输入帧抽一帧，拼成一张 montage，便于教程里
直接挂「公开数据集长什么样」的例图。覆盖：
  - TUM fr1/desk RGB（室内真实采集）
  - TUM fr1/desk 深度图（16-bit → 米，viridis 着色）
  - 4Seasons oldtown_night RGB（夜间驾驶真实采集）
  - TUM fr1/desk + M3 合成退化（低光+运动模糊，搜救式退化）
  - 合成巡检序列一帧（零下载，演示框架与数据来源解耦）

注意色序：cv2.imread/imdecode 返回 BGR，matplotlib 显示需 BGR→RGB；
合成序列是程序构造的 RGB，不做转换。

用法：
    PYTHONPATH=src python scripts/make_project_figs.py
产出：experiments/projects/figs/dataset_samples.png
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from wildspatial.projects.loaders import get_sequence


def _bgr2rgb(img):
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    out_dir = os.path.join(ROOT, "experiments", "projects", "figs")
    os.makedirs(out_dir, exist_ok=True)

    # --- 抽帧 ---
    tum = get_sequence("tum", seq="fr1/desk", max_frames=1, stride=1)
    tum_rgb = _bgr2rgb(tum.rgbs[0])
    tum_depth = tum.depths[0]
    f4 = get_sequence("4seasons", split="oldtown_night", start=400, n=1)
    f4_rgb = _bgr2rgb(f4.rgbs[0])
    deg = get_sequence("tum_degraded", seq="fr1/desk", max_frames=1, stride=1,
                       degrade="low_light,motion_blur", degrade_sev=0.6)
    deg_rgb = _bgr2rgb(deg.rgbs[0])
    syn = get_sequence("synthetic", n_frames=1)
    syn_rgb = syn.rgbs[0]          # 程序构造，RGB
    syn_depth = syn.depths[0]

    # --- 深度图着色（仅显示有效深度）---
    def depth_img(d, vmax=5.0):
        m = np.ma.masked_where((d <= 0) | ~np.isfinite(d), d)
        return m, vmax

    # --- 拼图：2 行 × 3 列 ---
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    fig.suptitle("Datasets / input samples used by the projects", fontsize=13, y=0.98)

    ax = axes[0, 0]
    ax.imshow(tum_rgb); ax.set_title("1 TUM fr1/desk - RGB\n(indoor, real capture)", fontsize=9)
    ax.axis("off")

    ax = axes[0, 1]
    dm, vmax = depth_img(tum_depth)
    ax.imshow(dm, cmap="viridis", vmin=0, vmax=vmax)
    ax.set_title("2 TUM fr1/desk - Depth\n(16-bit->m, viridis)", fontsize=9)
    ax.axis("off")

    ax = axes[0, 2]
    ax.imshow(f4_rgb); ax.set_title("3 4Seasons oldtown_night - RGB\n(night driving, real)", fontsize=9)
    ax.axis("off")

    ax = axes[1, 0]
    ax.imshow(deg_rgb)
    ax.set_title("4 TUM + M3 synthetic degrade\n(low_light+motion_blur)", fontsize=9)
    ax.axis("off")

    ax = axes[1, 1]
    ax.imshow(syn_rgb); ax.set_title("5 Synthetic inspection - frame\n(no download, floating cubes)", fontsize=9)
    ax.axis("off")

    ax = axes[1, 2]
    sm, _ = depth_img(syn_depth, vmax=8.0)
    ax.imshow(sm, cmap="plasma", vmin=0, vmax=8.0)
    ax.set_title("6 Synthetic - depth\n(procedural ground-truth)", fontsize=9)
    ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(out_dir, "dataset_samples.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[ok] dataset samples -> {out}")


if __name__ == "__main__":
    main()
