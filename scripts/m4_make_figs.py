"""从已有的 depth_metrics.json 直接生成 M4 的图与 README（不重跑 VGGT）。

用法：
    PYTHONPATH=src python scripts/m4_make_figs.py --out experiments/M4_depth_atlas
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wildspatial.viz import setup_plot_style
setup_plot_style()

# 复用 m5 脚本的退化定义顺序
from m5_method_sweep import KINDS, LEVELS  # noqa

SEQ_DEFAULT = "fr1/desk"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/M4_depth_atlas")
    ap.add_argument("--depth-metrics", default=None,
                    help="覆盖路径，默认 <out>/depth_metrics.json")
    args = ap.parse_args()

    mpath = args.depth_metrics or os.path.join(args.out, "depth_metrics.json")
    d = json.load(open(mpath))
    depth = d["depth"]
    conds = d["conditions"]
    labels = [l for l, _ in LEVELS]

    figdir = os.path.join(args.out, "figs")
    os.makedirs(figdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, ["rmse", "absrel"]):
        for kind in KINDS:
            ys, xs = [], []
            for li, (lab, _) in enumerate(LEVELS):
                cname = f"{KINDS[kind].split('（')[0]}-{lab}"
                v = depth.get(cname)
                ys.append(v.get(metric) if v else np.nan)
                xs.append(li)
            cv = depth.get("clean")
            clean_v = cv.get(metric) if cv else np.nan
            ax.plot([-0.3] + xs, [clean_v] + ys, "o-",
                    label=KINDS[kind].split('（')[0])
        ax.set_xticks([-0.3, 0, 1, 2])
        ax.set_xticklabels(["clean"] + labels, fontsize=8)
        ax.set_xlim(-0.6, 2.4)
        ax.set_title(f"VGGT 深度 {metric.upper()} vs 退化严重度（{d.get('seq',SEQ_DEFAULT)}）")
        ax.set_xlabel("退化严重度（轻/中/重）")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "depth_vs_degradation.png"), dpi=130)
    plt.close(fig)

    # ---- README ----
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(f"# M4 深化 · VGGT 深度 / 点云质量失效图谱\n\n")
        f.write(f"**序列**：{d.get('seq',SEQ_DEFAULT)}（{d.get('n_frames','?')} 帧）\n\n")
        f.write("## 目的\nM5 只比了轨迹 ATE；本实验用 VGGT 的稠密深度 / 世界点，在 13 退化条件下量化"
                "「几何重建质量」是否也随退化崩坏。\n\n")
        f.write("## 方法\nVGGT(return_dense) 直接出深度图与 world_points；深度用中位比例与真值对齐后算 "
                "RMSE/AbsRel/δ1；world_points 借相机中心轨迹的 Sim3 对齐到真值系，与真值点云算 Chamfer。\n\n")
        f.write("## 结果（深度，详见 depth_metrics.json）\n")
        f.write("| 条件 | RMSE(m) | AbsRel | δ1 |\n|---|---|---|---|\n")
        for c in conds:
            v = depth.get(c)
            if v:
                f.write(f"| {c} | {v['rmse']:.4f} | {v['absrel']:.4f} | {v['delta1']:.3f} |\n")
            else:
                f.write(f"| {c} | — | — | — |\n")
        if d.get("chamfer_all"):
            f.write(f"\n**点云 Chamfer（全条件合并）**: {d['chamfer_all'].get('all', float('nan')):.4f} m\n")
        f.write("\n> ⚠️ 退化为合成施加（data/degrade.py）。\n")

    print(f"[✓] 图与 README 已生成 → {args.out}")


if __name__ == "__main__":
    main()
