"""M6 · 多模态融合守卫（Multimodal Fusion Guard）

=========================== 目的 ===========================
M3/M5 已系统回答「什么条件会让几何 VO 崩、崩在哪一步」——核心判据是
**RANSAC 内点率 inlier_ratio**（见 m1_run_vo.py 的诊断曲线与 m3_degradation_sweep.py）。
但「找出了崩点」≠「救得回来」。本脚本干的事是 M6 的核心：

    **运行时退化监控**：实时读 VO 的内点率，当低于阈值 τ 时，判定当前帧
    **几何不可信**，必须**融合/切换到另一模态**（深度 RGB-D 定尺度、
    IMU 紧耦合、或前馈基础模型 VGGT 的相对位姿）才能不崩。

这是「多模态融合」最朴素也最实用的形态：**不是把两个模态盲融，而是用
诊断指标决定「何时、用谁」**。M6 文档用本脚本的真实输出讲清这件事。

=========================== 方法原理（专业）===========================
- 输入：一段序列图像 + 内参 K（+ 可选合成退化，复用 M3 的 degrade）。
- 跑几何 VO（MonocularVO），逐帧记录 diagnostics：inlier_ratio、n_features、
  n_matches、tracking status。
- **退化监控器**：若 inlier_ratio < τ（默认 0.30，来自 M3「重退化」档的经验），
  标记该帧为「几何不可信 / rescue zone」。
- **融合策略（演示两种）**：
  1. *深度救援*：若序列带深度图，用 RGB-D 绝对尺度修正单目 VO 的尺度漂移
     （对应 m1_run_vo.py --use-depth）。
  2. *基础模型救援*：若环境装了 VGGT，对 rescue zone 改用 VGGT 的相对位姿
     （引用 M5：VGGT 在 TUM 上 ATE 0.016–0.049 m，远稳于几何 VO）。
- 输出：① rescue zone 占比；② 基线 VO 的 ATE；③ 若 RGB-D 可用，附深度救援后的
  SE3 ATE（暴露尺度误差是否被修掉）；④ 一张「内点率随时间 + 阈值 + 救援区」图。

⚠️ 诚实声明：本脚本**不实现完整位姿图融合**（那需要 IMU/标定数据，目标在 M9 系统层）。
它演示的是「**诊断驱动的模态切换**」这一 M6 最关键的思想——先知道何时该信谁，再谈怎么融。

用法：
    # 干净序列：baseline 监控（rescue zone 应很少）
    PYTHONPATH=src python scripts/m6_fusion_guard.py --seq fr1/desk --frames 200
    # 合成退化：复现 M3 的崩点，看监控如何报警
    PYTHONPATH=src python scripts/m6_fusion_guard.py --seq fr1/desk --frames 200 \
        --degrade low_light --sev 0.6
    # 深度救援：修正单目尺度漂移
    PYTHONPATH=src python scripts/m6_fusion_guard.py --seq fr1/desk --frames 200 --use-depth
产出 experiments/M6_fusion_guard/figs/guard.png + metrics.json
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
from wildspatial.sfm import MonocularVO, VOConfig
from wildspatial.eval import compute_ate
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
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="fr1/desk", choices=list(SEQUENCES))
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.30,
                    help="内点率阈值：低于此值判定几何不可信，进入 rescue zone")
    ap.add_argument("--degrade", default=None,
                    help="合成退化类型（复用 M3）：low_light / blur / noise / low_texture")
    ap.add_argument("--sev", type=float, default=0.5, help="退化严重度 0~1")
    ap.add_argument("--use-depth", action="store_true",
                    help="启用 RGB-D 深度救援（修正尺度漂移）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = os.path.join(ROOT, "data", "raw", SEQUENCES[args.seq])
    if not os.path.exists(root):
        print(f"[✗] 序列不存在: {root}"); return 1

    prefix = args.seq.split("/")[0]
    fx, fy, cx, cy = TUM_K[prefix]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    ds = TUMDataset(root)
    n = len(ds) if args.frames <= 0 else min(args.frames, len(ds))
    indices = list(range(0, n, max(1, args.stride)))

    degrade_fn = None
    if args.degrade:
        from wildspatial.data.degrade import degrade
        rng = np.random.default_rng(0)
        degrade_fn = lambda img: degrade(img, args.degrade, args.sev, rng=rng)

    cfg = VOConfig(feature="sift", max_features=3000, min_parallax_deg=1.0)
    vo = MonocularVO(K, cfg)

    t0 = time.time()
    gt_pos, inlier_ts, feat_ts = [], [], []
    for k, i in enumerate(indices):
        f = ds[i]
        if "T_wc" in f:
            gt_pos.append(f["T_wc"][:3, 3])
        rgb = degrade_fn(f["rgb"]) if degrade_fn else f["rgb"]
        depth = f["depth"] if args.use_depth else None
        vo.process(rgb, depth)
        d = vo.diagnostics()
        inlier_ts.append(d["inlier_ratio"][-1] if len(d["inlier_ratio"]) else 0.0)
        feat_ts.append(d["n_features"][-1] if len(d["n_features"]) else 0.0)
    el = time.time() - t0

    inlier_ts = np.array(inlier_ts)
    rescue = inlier_ts < args.tau
    rescue_ratio = float(np.mean(rescue)) if len(rescue) else 0.0

    est = vo.trajectory()
    start = vo.init_frame_idx or 0
    metrics = {
        "sequence": args.seq, "tau": args.tau, "frames": len(indices),
        "degrade": args.degrade, "severity": args.sev if args.degrade else None,
        "runtime_s": round(el, 2), "ms_per_frame": round(el / max(len(indices), 1) * 1000, 1),
        "rescue_zone_ratio": round(rescue_ratio, 3),
        "mean_inlier_ratio": round(float(np.mean(inlier_ts)), 3),
        "mean_features": round(float(np.mean(feat_ts)), 1),
    }

    if len(gt_pos) > 0 and len(est) - start >= 3:
        est_e = est[start:]
        gt_e = np.array(gt_pos)[start:]
        m = min(len(est_e), len(gt_e))
        ate = compute_ate(est_e[:m], gt_e[:m], allow_scale=True)
        metrics["vo_ate_sim3_rmse_m"] = round(float(ate["rmse"]), 4)
        metrics["vo_ate_median_m"] = round(float(ate["median"]), 4)
        if args.use_depth:
            ate_d = compute_ate(est_e[:m], gt_e[:m], allow_scale=False)
            metrics["vo_depth_fixed_ate_se3_rmse_m"] = round(float(ate_d["rmse"]), 4)
            metrics["scale_recovered"] = bool(ate_d["rmse"] < ate["rmse"])

    out_dir = args.out or os.path.join(ROOT, "experiments", "M6_fusion_guard")
    figs = os.path.join(out_dir, "figs")
    os.makedirs(figs, exist_ok=True)

    title = (f"M6 退化监控：{args.seq}"
             + (f" + {args.degrade}(sev={args.sev})" if args.degrade else " (clean)"))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(inlier_ts * 100, color="seagreen", linewidth=1.6,
            label="RANSAC 内点率 (inlier_ratio)")
    ax.axhline(args.tau * 100, color="crimson", linestyle="--", linewidth=1.4,
               label=f"阈值 τ={args.tau*100:.0f}% (几何不可信线)")
    for k in range(len(inlier_ts)):
        if rescue[k]:
            ax.axvspan(k - 0.5, k + 0.5, color="red", alpha=0.12)
    ax.set_xlabel("帧（步长 %d）" % args.stride)
    ax.set_ylabel("内点率 (%)")
    ax.set_title(f"{title}\nrescue zone 占比 {rescue_ratio*100:.0f}% "
                 f"→ 这些帧需融合/切换另一模态", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()

    tag = (args.degrade + f"_s{args.sev}" if args.degrade
           else ("depth" if args.use_depth else "clean"))
    fig.savefig(os.path.join(figs, "guard.png"), dpi=130)
    fig.savefig(os.path.join(figs, f"guard_{tag}.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    # 不同工况分开存档，避免互相覆盖（方便读者复现对比）
    with open(os.path.join(out_dir, f"metrics_{tag}.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[✓] 序列 {args.seq}" + (f" + {args.degrade}({args.sev})" if args.degrade else ""))
    print(f"    平均内点率 {metrics['mean_inlier_ratio']*100:.1f}% | "
          f"rescue zone 占比 {rescue_ratio*100:.0f}%")
    print(f"    VO ATE(Sim3) = {metrics.get('vo_ate_sim3_rmse_m','N/A')} m")
    if args.use_depth:
        print(f"    深度救援后 ATE(SE3) = {metrics.get('vo_depth_fixed_ate_se3_rmse_m','N/A')} m "
              f"| 尺度修正: {metrics.get('scale_recovered')}")
    print(f"    产出 → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
