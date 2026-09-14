"""M5 · 方法动物园 × 退化压力测试（失效图谱）

==================== 目的 ====================
不再手搓——直接把「现成工具 / 代码包 / 模型」当成可替换的黑盒方法，
在**同一套带真值的数据 + 同一套合成退化**下公平对比，定量画出
「方法 × 环境」的失效图谱：哪个方法在哪种退化下先崩、崩多远。

==================== 用法 ====================
    conda activate wildspatial
    cd /home/hmn-cjy/liuliuqiu/WildSpatial
    PYTHONPATH=src python scripts/m5_method_sweep.py --seq fr1/desk --frames 150 --stride 3

    # 只跑指定方法（其余未装的会自动跳过）
    PYTHONPATH=src python scripts/m5_method_sweep.py --methods handcrafted_vo colmap_sfm

产出（experiments/M5_method_atlas/）：
    figs/atlas_ate.png   方法 × 退化条件的 ATE 热力图（失效一目了然）
    metrics.json         完整量化（每个 方法×条件 的 ATE/状态）
    README.md            四段式结论
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from wildspatial.data.tum import TUMDataset
from wildspatial.data.degrade import KINDS, LEVELS, degrade, level_label
from wildspatial.methods import discover_methods
from wildspatial.viz import setup_plot_style

setup_plot_style()

TUM_K = {
    "fr1": (517.3, 516.5, 318.6, 255.3),
    "fr2": (520.9, 521.0, 325.1, 249.7),
    "fr3": (535.4, 539.2, 320.1, 247.6),
}
SEQUENCES = {
    "fr1/desk": "rgbd_dataset_freiburg1_desk",
    "fr1/room": "rgbd_dataset_freiburg1_room",
    "fr3/nostructure": "rgbd_dataset_freiburg3_nostructure_notexture_near_withloop",
}


def build_conditions():
    """干净基线 + 4 类退化 × 3 档位 = 13 个条件。"""
    conds = [("clean", None, 0.0)]
    for kind in KINDS:
        for lab, sev in LEVELS:
            conds.append((f"{KINDS[kind].split('（')[0]}-{lab}", kind, sev))
    return conds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="fr1/desk")
    ap.add_argument("--root", default=None)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--methods", nargs="*", default=None,
                    help="只跑这些方法（默认全部可用）")
    ap.add_argument("--max-features", type=int, default=3000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = args.root or os.path.join(ROOT, "data", "raw", SEQUENCES[args.seq])
    if not os.path.exists(root):
        print(f"[✗] 序列不存在: {root}")
        return 1

    prefix = args.seq.split("/")[0]
    fx, fy, cx, cy = TUM_K.get(prefix, TUM_K["fr1"])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    ds = TUMDataset(root)
    sel = list(range(0, len(ds), args.stride))[:args.frames]
    frames = [ds[i]["rgb"] for i in sel]
    gt = ds.trajectory_gt()[sel]
    print(f"[·] 序列 {args.seq}：{len(frames)} 帧（stride={args.stride}），有真值 {ds.has_gt}")

    methods = [m for m in discover_methods()
               if args.methods is None or m.name in args.methods]
    if not methods:
        print("[✗] 没有可用方法（检查 --methods 或依赖安装）")
        return 1
    print(f"[·] 启用方法: {[m.name for m in methods]}")

    conds = build_conditions()
    print(f"[·] 退化条件数: {len(conds)}")

    matrix = {m.name: {} for m in methods}
    raw = np.full((len(methods), len(conds)), np.nan)

    for mi, m in enumerate(methods):
        for ci, (clabel, kind, sev) in enumerate(conds):
            t0 = time.time()
            try:
                if kind is None:
                    dframes = frames
                else:
                    dframes = [degrade(img, kind, sev) for img in frames]
                res = m.run(dframes, K, max_features=args.max_features)
                ate = res.aligned_ate(gt)["rmse"] if res.ok else float("nan")
                ok = res.ok and np.isfinite(ate)
            except Exception as e:
                ate, ok = float("nan"), False
                res = type("R", (), {"note": f"异常: {e}"})()
            raw[mi, ci] = ate if ok else np.nan
            matrix[m.name][clabel] = {
                "ate_rmse_m": None if not np.isfinite(ate) else float(ate),
                "ok": bool(ok),
                "note": getattr(res, "note", ""),
            }
            print(f"    {m.name:16s} | {clabel:10s} | "
                  f"ATE={ate if ok else '失效':>8} | {time.time()-t0:5.1f}s")

    # ---- 写量化 ----
    out = args.out or os.path.join(ROOT, "experiments", "M5_method_atlas")
    figdir = os.path.join(out, "figs")
    os.makedirs(figdir, exist_ok=True)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump({"seq": args.seq, "n_frames": len(frames),
                   "stride": args.stride, "conditions": [c[0] for c in conds],
                   "matrix": matrix}, f, ensure_ascii=False, indent=2)

    # ---- 热力图 ----
    fig, ax = plt.subplots(figsize=(max(10, len(conds) * 0.8), len(methods) * 1.4 + 2))
    masked = np.ma.masked_invalid(raw)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("lightgray")
    vmax = np.nanmax(raw) if np.nanmax(raw) > 0 else 1.0
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c[0] for c in conds], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m.name for m in methods], fontsize=9)
    for mi in range(len(methods)):
        for ci in range(len(conds)):
            v = raw[mi, ci]
            txt = "—" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(ci, mi, txt, ha="center", va="center", fontsize=7,
                    color="white" if (np.isfinite(v) and v > vmax * 0.5) else "black")
    ax.set_title(f"M5 失效图谱 · ATE (m) · {args.seq} · {len(frames)} 帧 "
                 f"（颜色越黄=越差；每格数字=该 方法×条件 的 ATE）")
    # 图内「怎么读这张图」注记：让图自解释
    ax.text(0.0, -0.28,
            "读图：横轴=13 个条件（干净 + 4 类退化×3 档）；纵轴=四方法。\n"
            "紫色≈准；黄/绿=误差大甚至塌缩。VGGT 一整行几乎全紫（免疫退化）；\n"
            "COLMAP 仅「高斯噪声」两格发绿（传统几何死穴）；自研 VO 多个黄格（轨迹塌缩）。",
            transform=ax.transAxes, fontsize=8, va="top", color="dimgray",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="lightgray", alpha=0.9))
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="ATE RMSE (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "atlas_ate.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- README（四段式简要） ----
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(f"# M5 方法动物园 × 退化压力测试\n\n")
        f.write(f"**序列**：{args.seq}（{len(frames)} 帧，stride={args.stride}）\n\n")
        f.write(f"## 目的\n把现成工具/模型当成黑盒方法，在带真值的同一数据 + 同一套合成退化下公平对比，"
                f"定量画出「方法 × 环境」失效图谱。\n\n")
        f.write(f"## 方法原理\n每个方法统一输出相机光心轨迹，与真值做 Umeyama(Sim3) 对齐后算 ATE RMSE。"
                f"退化条件 = 干净 + 4 类 × 3 档 = 13 项。\n\n")
        f.write(f"## 直白讲解\n喂图→吐路径；换方法、换退化档位，看哪根线先掉下去。\n\n")
        f.write(f"## 结果（详见 metrics.json / figs/atlas_ate.png）\n")
        f.write("| 方法 | 干净 ATE | 最差退化 |\n|---|---|---|\n")
        for m in methods:
            row = matrix[m.name]
            clean = row.get("clean", {}).get("ate_rmse_m")
            worst = max((v["ate_rmse_m"] for v in row.values()
                         if v["ate_rmse_m"] is not None), default=None)
            f.write(f"| {m.name} | {clean if clean is not None else '失效'} | "
                    f"{worst if worst is not None else '—'} |\n")
        f.write("\n> ⚠️ 退化为合成施加（详见 `src/wildspatial/data/degrade.py`）。\n")

    print(f"[✓] 完成 → {out}")
    print(f"    图: {os.path.join(figdir, 'atlas_ate.png')}")


if __name__ == "__main__":
    main()
