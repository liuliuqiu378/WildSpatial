"""M1 · 在真实数据（TUM RGB-D）上跑手搓单目 VO 并评估

用法：
    conda activate wildspatial
    cd /home/hmn-cjy/liuliuqiu/WildSpatial
    PYTHONPATH=src python scripts/m1_run_vo.py --seq fr1/desk --frames 200
    PYTHONPATH=src python scripts/m1_run_vo.py --seq fr3/nostructure --frames 150
    PYTHONPATH=src python scripts/m1_run_vo.py --seq fr1/desk --use-depth   # RGB-D 定尺度

产出（experiments/M1_vo_<seq>/）：
    figs/trajectory.png     轨迹对比（XY + 三轴随时间）
    figs/diagnostics.png    每帧诊断曲线（M3 失效归因的数据源）
    figs/ate_curve.png      ATE 随时间的漂移
    figs/trajectory_ba.png 局部 BA 前后轨迹对比（加 --ba 时生成）
    metrics.json            量化结果
    README.md               结论与反思
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from wildspatial.data.tum import TUMDataset
from wildspatial.sfm import MonocularVO, VOConfig
from wildspatial.eval import compute_ate, compute_rpe, TrajectoryEval
from wildspatial.viz import setup_plot_style

setup_plot_style()

# TUM RGB-D 各序列的标准内参
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", default="fr1/desk", choices=list(SEQUENCES) + ["custom"])
    p.add_argument("--root", default=None, help="自定义序列根目录（与 --seq custom 配合）")
    p.add_argument("--frames", type=int, default=200, help="处理前 N 帧（0=全部）")
    p.add_argument("--stride", type=int, default=1,
                   help="跳帧步长。TUM 是 30fps，相邻帧位移极小（实测视差仅 0.4°），"
                        "跳帧可增大基线。建议 3~5")
    p.add_argument("--min-parallax", type=float, default=1.0,
                   help="三角化的最小视差角（度）")
    p.add_argument("--feature", default="sift", choices=["sift", "orb", "akaze"])
    p.add_argument("--max-features", type=int, default=3000)
    p.add_argument("--use-depth", action="store_true",
                   help="用深度图定尺度（RGB-D 模式；否则单目，需 Sim3 对齐）")
    p.add_argument("--ba", action="store_true",
                   help="跑完 VO 后追加局部 BA 重优化（消除累积漂移，通常显著降 ATE）")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    root = args.root or os.path.join(ROOT, "data", "raw", SEQUENCES[args.seq])
    if not os.path.exists(root):
        print(f"[✗] 序列不存在: {root}\n    请先下载（见 PROGRESS.md 数据集台账）")
        return 1

    prefix = args.seq.split("/")[0]
    fx, fy, cx, cy = TUM_K.get(prefix, TUM_K["fr1"])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    print(f"[·] 序列: {args.seq}\n[·] 内参: fx={fx} fy={fy} cx={cx} cy={cy}")

    ds = TUMDataset(root)
    print(f"[·] 有效帧数: {len(ds)}；有真值: {ds.has_gt}")

    n = len(ds) if args.frames <= 0 else min(args.frames, len(ds))

    cfg = VOConfig(feature=args.feature, max_features=args.max_features,
                   min_parallax_deg=args.min_parallax)
    vo = MonocularVO(K, cfg)

    indices = list(range(0, n, max(1, args.stride)))
    print(f"[·] 处理 {len(indices)} 帧（步长 {args.stride}，特征={args.feature}，"
          f"深度定尺度={args.use_depth}）...")
    t0 = time.time()
    gt_pos_sel, gt_pose_sel = [], []
    for k, i in enumerate(indices):
        f = ds[i]
        if "T_wc" in f:
            gt_pos_sel.append(f["T_wc"][:3, 3])
            gt_pose_sel.append(f["T_wc"])
        depth = f["depth"] if args.use_depth else None
        info = vo.process(f["rgb"], depth)
        if (k + 1) % 10 == 0 or k == 0:
            print(f"    [{k+1:4d}/{len(indices)}] feat={info['n_features']:4d} "
                  f"match={info.get('n_matches',0):4d} "
                  f"inlier={info.get('inlier_ratio',0)*100:5.1f}% "
                  f"pnp={info.get('pnp_points',0):3d} "
                  f"map={info.get('n_map',0):5d}  {info['status']}")
    el = time.time() - t0
    print(f"[✓] 完成，用时 {el:.1f}s（{el/max(n,1)*1000:.0f} ms/帧）")

    # ---------------- 评估 ----------------
    est = vo.trajectory()
    out_dir = args.out or os.path.join(ROOT, "experiments",
                                       f"M1_vo_{args.seq.replace('/', '_')}")
    os.makedirs(os.path.join(out_dir, "figs"), exist_ok=True)

    metrics = {
        "sequence": args.seq, "frames": int(n), "feature": args.feature,
        "max_features": args.max_features, "use_depth_scale": args.use_depth,
        "runtime_s": round(el, 2), "ms_per_frame": round(el / max(n, 1) * 1000, 1),
        "initialized": bool(vo.initialized),
    }

    # 从**成功初始化的那一帧**开始评估（初始化前的帧本就没有有效估计）
    start = vo.init_frame_idx if vo.init_frame_idx is not None else 0
    metrics["init_frame_idx"] = int(vo.init_frame_idx) if vo.init_frame_idx else None

    if len(gt_pos_sel) > 0 and len(est) - start >= 3:
        est_e = est[start:]
        gt_e = np.array(gt_pos_sel)[start:]
        m = min(len(est_e), len(gt_e))
        est_e, gt_e = est_e[:m], gt_e[:m]
        # 单目 → Sim3；用了真实深度定尺度 → 可用 SE3（能暴露尺度误差）
        allow_scale = not args.use_depth
        ate = compute_ate(est_e, gt_e, allow_scale=allow_scale)
        metrics["ate"] = {k: v for k, v in ate.items()}

        # RPE（用完整位姿）
        est_poses = [np.linalg.inv(f.T_cw) for f in vo.frames[start:start + m]
                     if f.T_cw is not None]
        gt_poses = gt_pose_sel[start:start + m]
        m2 = min(len(est_poses), len(gt_poses))
        if m2 > 3:
            rpe = compute_rpe(np.array(est_poses[:m2]), np.array(gt_poses[:m2]), delta=1)
            metrics["rpe"] = rpe

        print("\n===== 结果 =====")
        print(f"对齐方式     : {'Sim3（单目，含尺度）' if allow_scale else 'SE3（已定尺度）'}")
        print(f"ATE  RMSE    : {ate['rmse']:.4f} m   (median {ate['median']:.4f}, "
              f"max {ate['max']:.4f})")
        if allow_scale:
            print(f"对齐尺度 s   : {ate['scale']:.4f}")
        if "rpe" in metrics:
            print(f"RPE  平移    : {metrics['rpe']['trans_rmse']:.4f} m   "
                  f"旋转 {metrics['rpe']['rot_rmse_deg']:.3f}°")

        from wildspatial.eval import align_trajectory
        aligned, _ = align_trajectory(est_e, gt_e, allow_scale)

        # 图1：轨迹对比
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        axes[0].plot(gt_e[:, 0], gt_e[:, 2], "k-", label="真值", linewidth=2)
        axes[0].plot(aligned[:, 0], aligned[:, 2], "r--",
                     label="估计（对齐后）", linewidth=1.5)
        axes[0].set_xlabel("X (m)"); axes[0].set_ylabel("Z (m)")
        axes[0].set_title("轨迹俯视对比 (X-Z)", fontsize=12)
        axes[0].legend(); axes[0].axis("equal")

        ax2 = axes[1]
        for k, lab in enumerate(["X", "Y", "Z"]):
            ax2.plot(gt_e[:, k], "k-", alpha=0.5, linewidth=2 if k == 0 else 1)
            ax2.plot(aligned[:, k], "--", alpha=0.9, linewidth=1.2,
                     label=f"估计 {lab}" if k == 0 else None)
        ax2.set_xlabel("帧"); ax2.set_ylabel("位置 (m)")
        ax2.set_title("三轴位置随时间（黑=真值）", fontsize=12)
        ax2.legend()
        plt.suptitle(f"{args.seq} · 单目 VO 轨迹（ATE RMSE = {ate['rmse']:.3f} m）", fontsize=13)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/figs/trajectory.png", dpi=130)
        plt.close()

        # 图3：ATE 随时间的漂移曲线
        err_t = np.linalg.norm(aligned - gt_e, axis=1)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(err_t, color="crimson", linewidth=1.5)
        ax.fill_between(range(len(err_t)), 0, err_t, color="crimson", alpha=0.15)
        ax.set_xlabel("帧"); ax.set_ylabel("位置误差 (m)")
        ax.set_title("ATE 随时间的漂移（典型现象：误差单调累积）", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/figs/ate_curve.png", dpi=130)
        plt.close()

        # -------- 可选：局部 BA 精修（消除累积漂移）--------
        metrics["ba"] = {"enabled": bool(args.ba)}
        if args.ba:
            ba_info = vo.refine(max_iter=20, verbose=False)
            metrics["ba"].update({k: v for k, v in ba_info.items()})
            if ba_info.get("status") == "ok":
                est_after = vo.trajectory()
                est_after_e = est_after[start:]
                m_a = min(len(est_after_e), len(gt_e))
                est_after_e = est_after_e[:m_a]
                aligned_after, _ = align_trajectory(est_after_e, gt_e[:m_a], allow_scale)
                ate_after = compute_ate(est_after_e, gt_e[:m_a], allow_scale=allow_scale)
                metrics["ate_after"] = {k: v for k, v in ate_after.items()}

                rb, ra = ba_info.get("rmse_before"), ba_info.get("rmse_after")
                print(f"\n[BA] 重投影 RMSE: {rb:.3f} → {ra:.3f} px")
                print(f"[BA] ATE  RMSE   : {ate['rmse']:.4f} → {ate_after['rmse']:.4f} m")

                # 图4：BA 前后轨迹对比（俯视 X-Z）
                fig, ax = plt.subplots(figsize=(7, 6))
                ax.plot(gt_e[:m_a, 0], gt_e[:m_a, 2], "k-", linewidth=2, label="真值")
                ax.plot(aligned[:m_a, 0], aligned[:m_a, 2], "b--", linewidth=1.4,
                        label=f"BA 前 (ATE={ate['rmse']:.3f}m)")
                ax.plot(aligned_after[:, 0], aligned_after[:, 2], "r-", linewidth=1.4,
                        label=f"BA 后 (ATE={ate_after['rmse']:.3f}m)")
                ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
                ax.set_title("局部 BA 前后轨迹对比（俯视 X-Z）", fontsize=12)
                ax.legend(); ax.axis("equal")
                plt.tight_layout()
                plt.savefig(f"{out_dir}/figs/trajectory_ba.png", dpi=130)
                plt.close()
            else:
                print(f"[BA] 跳过：{ba_info.get('status')}")

        metrics["ate_curve_mean"] = float(np.mean(err_t))
    else:
        print("[!] 无真值，跳过评估")

    # 图2：诊断曲线（M3 的核心数据源）
    d = vo.diagnostics()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(d["n_features"], color="steelblue")
    axes[0, 0].set_title("每帧特征点数"); axes[0, 0].set_xlabel("帧")
    axes[0, 1].plot(d["n_matches"], color="darkorange")
    axes[0, 1].set_title("每帧匹配数"); axes[0, 1].set_xlabel("帧")
    axes[1, 0].plot(d["inlier_ratio"] * 100, color="seagreen")
    axes[1, 0].set_title("RANSAC 内点率 (%)  ← 最关键的失效指标")
    axes[1, 0].set_xlabel("帧"); axes[1, 0].set_ylim(0, 101)
    axes[1, 1].plot(d["n_map"], color="purple")
    axes[1, 1].set_title("地图点数"); axes[1, 1].set_xlabel("帧")
    plt.suptitle("VO 每帧诊断：这些过程量比最终 ATE 更能定位失效根因", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/figs/diagnostics.png", dpi=130)
    plt.close()

    metrics["diagnostics_mean"] = {
        "n_features": float(np.mean(d["n_features"])),
        "n_matches": float(np.mean(d["n_matches"])),
        "inlier_ratio": float(np.mean(d["inlier_ratio"])),
        "n_map_final": float(d["n_map"][-1] if len(d["n_map"]) else 0),
    }
    status_count = {}
    for s in d["status"]:
        status_count[s] = status_count.get(s, 0) + 1
    metrics["status_count"] = status_count

    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] 产出目录: {out_dir}")
    print("    figs/trajectory.png  figs/diagnostics.png  figs/ate_curve.png"
          + ("  figs/trajectory_ba.png" if args.ba else "") + "  metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
