"""ModelNet40-C 点云退化 demo：量化「3D 数据退化」对几何的影响。

============================ 目的 ============================
项目主线是「方法 × 环境退化的失效图谱」。但此前退化都在**图像**侧
（M3/M5 的运动模糊/噪声/低光）。ModelNet40-C 让我们把同一套「退化 → 影响」
的研究搬到一个**全新的模态：3D 点云**——直观回答：

    · 3D 点云被"损坏"之后长什么样？（视觉）
    · 几何形状被破坏到什么程度？（量化：Chamfer / 体积 / 点数）

============================ 方法原理 ============================
ModelNet40-C 提供 15 种损坏类型 × 5 严重级（每种 1000 个点云）：
    background / cutout / density(_inc) / distortion(_rbf/_rbf_inv) / gaussian /
    impulse / lidar / occlusion / rotation / shear / uniform / upsampling
本 demo 从**同一批物体**取 original 与各损坏版本，量化：
    · 点数变化（up/down sampling、cutout 会改点数）
    · 点云包围盒体积变化（shear/rotation/distortion 会变形）
    · 与 original 的 Chamfer 距离（形状偏移）——需点数一致时算

============================ 用法 ============================
    conda activate wildspatial
    PYTHONPATH=src python scripts/modelnet40c_corruption_demo.py
产出 experiments/ModelNet40C_corruption/figs/{corruption_gallery,chamfer_curve}.png + metrics.json
"""
import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from wildspatial.viz import setup_plot_style  # noqa: E402

setup_plot_style()
DATA = os.path.join(ROOT, "data", "raw", "ms", "OmniData__ModelNet40-C", "raw", "modelnet40_c")

CORRUPTIONS = ["background", "cutout", "density", "density_inc", "distortion",
               "distortion_rbf", "gaussian", "impulse", "lidar", "occlusion",
               "rotation", "shear", "uniform", "upsampling"]


def load_pts(path, sample_idx=0):
    arr = np.load(path, allow_pickle=True)
    a = np.asarray(arr[sample_idx])
    return a


def bbox_volume(pts):
    rng = pts.max(0) - pts.min(0)
    return float(np.prod(np.maximum(rng, 1e-6)))


def chamfer(a, b):
    """对称 Chamfer 距离（下采样以控速）"""
    def _cd(x, y):
        x = x if len(x) <= 2000 else x[np.random.RandomState(0).choice(len(x), 2000, replace=False)]
        y = y if len(y) <= 2000 else y[np.random.RandomState(0).choice(len(y), 2000, replace=False)]
        d = np.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(-1))
        return d.min(1).mean()
    return 0.5 * (_cd(a, b) + _cd(b, a))


def main():
    if not os.path.exists(DATA):
        print(f"[✗] 未找到 {DATA}（请先 fetch_datasets.py --tier A）"); return 1
    out_dir = os.path.join(ROOT, "experiments", "ModelNet40C_corruption")
    figs = os.path.join(out_dir, "figs"); os.makedirs(figs, exist_ok=True)

    orig = load_pts(os.path.join(DATA, "data_original.npy"), 0)

    # ---- 图1：退化图库（original + 若干损坏类型，同一物体）----
    show = ["original", "gaussian_3", "cutout_3", "shear_3", "occlusion_3",
            "upsampling_3", "distortion_rbf_3", "lidar_3"]
    fig = plt.figure(figsize=(3.0 * len(show), 3.2))
    for i, tag in enumerate(show):
        p = os.path.join(DATA, f"data_{tag}.npy")
        if not os.path.exists(p):
            continue
        pts = load_pts(p, 0)
        ax = fig.add_subplot(1, len(show), i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.6)
        ax.set_title(tag.replace("_", "\n"), fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.suptitle("ModelNet40-C：同一 3D 物体的 8 种「损坏」形态（severity=3）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "corruption_gallery.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- 图2：各损坏类型 severity 递增 → Chamfer 距离曲线 ----
    metrics = {"note": "ModelNet40-C 点云损坏；Chamfer 相对 original（低=形状保持好）",
               "curves": {}}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sev = [1, 2, 3, 4, 5]
    for c in CORRUPTIONS:
        vals = []
        ok = True
        for s in sev:
            p = os.path.join(DATA, f"data_{c}_{s}.npy")
            if not os.path.exists(p):
                ok = False; break
            try:
                pts = load_pts(p, 0)
                vals.append(chamfer(orig, pts))
            except Exception:
                ok = False; break
        if ok and len(vals) == 5:
            ax.plot(sev, vals, marker="o", label=c)
            metrics["curves"][c] = [round(v, 4) for v in vals]

    ax.set_xlabel("损坏严重级 (severity 1→5)")
    ax.set_ylabel("Chamfer 距离（相对 original，↑=形状破坏越重）")
    ax.set_title("3D 点云损坏 → 形状偏移曲线（15 种损坏类型）")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "chamfer_curve.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[✓] 产出 → {out_dir}")
    print(f"    物点数(original)={len(orig)} | 损坏类型={len(metrics['curves'])}")
    for c, v in list(metrics["curves"].items())[:6]:
        print(f"    {c:20s} chamfer: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
