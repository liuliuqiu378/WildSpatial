"""M5 · 由 metrics.json 重绘「失效图谱」热力图（避免重跑四方法，秒级出图）。

用法：
    PYTHONPATH=src python scripts/m5_make_figs.py

读 experiments/M5_method_atlas/metrics.json → 重绘 figs/atlas_ate.png。
图要点（对应教程 B4 打磨）：图内自带「怎么读这张图」注记，让热力图自解释。
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from wildspatial.viz import setup_plot_style  # noqa: E402

setup_plot_style()


def main():
    out = os.path.join(ROOT, "experiments", "M5_method_atlas")
    mpath = os.path.join(out, "metrics.json")
    if not os.path.exists(mpath):
        print(f"[✗] 找不到 {mpath}"); return 1
    with open(mpath) as f:
        d = json.load(f)

    conds = d["conditions"]
    matrix = d["matrix"]
    methods = list(matrix.keys())

    raw = np.full((len(methods), len(conds)), np.nan)
    for mi, m in enumerate(methods):
        for ci, c in enumerate(conds):
            v = matrix[m].get(c, {}).get("ate_rmse_m")
            if v is not None:
                raw[mi, ci] = v

    fig, ax = plt.subplots(figsize=(max(10, len(conds) * 0.82), len(methods) * 1.35 + 2.6))
    masked = np.ma.masked_invalid(raw)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("lightgray")
    vmax = np.nanmax(raw) if np.nanmax(raw) > 0 else 1.0
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    for mi in range(len(methods)):
        for ci in range(len(conds)):
            v = raw[mi, ci]
            txt = "—" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(ci, mi, txt, ha="center", va="center", fontsize=7,
                    color="white" if (np.isfinite(v) and v > vmax * 0.5) else "black")
    ax.set_title(f"M5 失效图谱 · ATE (m) · {d['seq']} · {d['n_frames']} 帧"
                 f"（颜色越黄=越差；每格数字=该 方法×条件 的 ATE）")
    ax.text(0.0, -0.30,
            "读图：横轴=13 个条件（干净 + 4 类退化×3 档）；纵轴=四方法。\n"
            "紫色≈准；黄/绿=误差大甚至塌缩。VGGT 一整行几乎全紫（免疫退化）；\n"
            "COLMAP 仅「高斯噪声」两格发绿（传统几何死穴）；自研 VO 多个黄格（轨迹塌缩）。",
            transform=ax.transAxes, fontsize=8, va="top", color="dimgray",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="lightgray", alpha=0.9))
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="ATE RMSE (m)")
    fig.tight_layout()
    figs = os.path.join(out, "figs")
    os.makedirs(figs, exist_ok=True)
    fig.savefig(os.path.join(figs, "atlas_ate.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] 重绘 → {os.path.join(figs, 'atlas_ate.png')}")
    print(f"    {len(methods)} 方法 × {len(conds)} 条件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
