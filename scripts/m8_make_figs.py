"""M8 · 由 metrics.json 重绘「端侧三角权衡」图（避免重跑 VO，秒级出图）。

用法：
    PYTHONPATH=src python scripts/m8_make_figs.py

读 experiments/M8_edge/metrics.json → 重绘 figs/tradeoff.png。
图要点（对应教程 B2 打磨）：
  · 有效档（Large/Medium）画成散点；
  · FAIL 档（Small/Tiny）画在「失败带」，标 ✗ 与失败原因（特征地板 / 分辨率悬崖）；
  · 标出「★ 最佳性价比点」。
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
    out_dir = os.path.join(ROOT, "experiments", "M8_edge")
    mpath = os.path.join(out_dir, "metrics.json")
    if not os.path.exists(mpath):
        print(f"[✗] 找不到 {mpath}"); return 1
    with open(mpath) as f:
        data = json.load(f)
    results = data["results"]
    seq = data.get("sequence", "fr1/desk")

    valid = [r for r in results if r.get("ate_rmse_m") is not None]
    failed = [r for r in results if r.get("ate_rmse_m") is None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
    colors = {"Large": "navy", "Medium": "green", "Small": "darkorange", "Tiny": "crimson"}

    valid_ates = [r["ate_rmse_m"] for r in valid]
    fail_y = (min(valid_ates) - 0.06) if valid_ates else 0.30

    def reason_short(r):
        fr = r.get("fail_reason", "")
        if "地板" in fr:
            return "特征地板"
        if "悬崖" in fr:
            return "分辨率悬崖"
        return "越界"

    # FAIL 档位置接近 → 标签上下交替错位，避免文字重叠
    fail_dy = [14, -30, 26, -42]
    for ax, xf in ((ax1, lambda r: r["rel_compute_budget"]),
                   (ax2, lambda r: r["latency_median_ms"])):
        for k, r in enumerate(failed):
            c = colors.get(r["tier"], "gray")
            x = xf(r)
            ax.axvspan(x * 0.88, x * 1.12, color=c, alpha=0.12, zorder=0)
            ax.scatter(x, fail_y, s=170, c=c, marker="x", zorder=5)
            dy = fail_dy[k % len(fail_dy)]
            ax.annotate(f"{r['tier']} 失败：{reason_short(r)}",
                        (x, fail_y), textcoords="offset points", xytext=(9, dy),
                        fontsize=8, color=c, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec=c, alpha=0.85, lw=0.6),
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.6))

    for r in valid:
        c = colors.get(r["tier"], "gray")
        ax1.scatter(r["rel_compute_budget"], r["ate_rmse_m"], s=140, c=c, zorder=3)
        ax1.annotate(r["tier"], (r["rel_compute_budget"], r["ate_rmse_m"]),
                     textcoords="offset points", xytext=(8, 6), fontsize=9)
        ax2.scatter(r["latency_median_ms"], r["ate_rmse_m"], s=140, c=c, zorder=3)
        rt = "[实时]" if r["real_time_30fps"] else "[非实时]"
        ax2.annotate(f"{r['tier']} {rt}", (r["latency_median_ms"], r["ate_rmse_m"]),
                     textcoords="offset points", xytext=(8, 6), fontsize=9, zorder=3)

    med = next((r for r in valid if r["tier"] == "Medium"), None)
    if med:
        ax1.annotate("★ 最佳性价比点（0.5x ORB1000）",
                     (med["rel_compute_budget"], med["ate_rmse_m"]),
                     textcoords="offset points", xytext=(8, -16), fontsize=9,
                     color="green", fontweight="bold")
        ax2.annotate("★", (med["latency_median_ms"], med["ate_rmse_m"]),
                     textcoords="offset points", xytext=(8, -16), fontsize=13, color="green")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.set_ylabel("ATE RMSE (m) ↓越好")
        ax.axhline(fail_y, color="gray", ls=":", lw=1, alpha=0.6)
        # 给失败带留出标注空间
        ylo = fail_y - 0.06
        yhi = (max(valid_ates) + 0.02) if valid_ates else 0.55
        ax.set_ylim(ylo, yhi)
    ax1.set_xlabel("相对算力预算（功耗代理，Large=1.0，越小越省）")
    ax1.set_title("① 算力预算 vs 精度：可行域是个「窗口」", fontsize=12)
    ax2.set_xlabel("延迟 (ms/帧，单核串行) ↓越好")
    ax2.set_title("② 延迟 vs 精度：越界直接 FAIL，非平滑退化", fontsize=12)
    ax1.text(0.30, 0.02,
             "虚线=失败带：越过「特征地板 / 分辨率悬崖」→ VO 直接初始化失败（标叉档）",
             transform=ax1.transAxes, fontsize=7.5, color="gray", va="bottom",
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
    plt.suptitle(f"M8 端侧模拟 · {seq} · 算力预算=分辨率²×特征数×算子代价", fontsize=13)
    plt.tight_layout()
    figs = os.path.join(out_dir, "figs")
    os.makedirs(figs, exist_ok=True)
    plt.savefig(os.path.join(figs, "tradeoff.png"), dpi=130)
    plt.close()
    print(f"[✓] 重绘 → {os.path.join(figs, 'tradeoff.png')}")
    print(f"    有效档 {len(valid)} 个，FAIL 档 {len(failed)} 个（已标 ✗ 与原因）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
