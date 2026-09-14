"""M4 深化：VGGT 深度 / 点云质量在退化下的鲁棒性（不止轨迹）

==================== 目的 ====================
VGGT 一次前向不仅出相机轨迹，还出 **稠密深度图** 与 **每像素 3D 世界坐标（world_points）**。
M5 的失效图谱只比较了「轨迹 ATE」，但前馈模型的真正价值可能更体现在
**几何重建质量**（深度准不准、点云像不像）上——尤其在传统几何最先崩的退化条件下。

本脚本在 **同一套 13 退化条件** 下跑 VGGT（return_dense=True），量化：
  1. 深度质量：预测深度 vs TUM 真值深度 → RMSE / AbsRel / δ1（按中位比例对齐尺度）
  2. 点云质量：world_points 用轨迹 Sim3 对齐到真值系后，与真值点云算 **Chamfer 距离**
产出 experiments/M4_depth_atlas/：depth_metrics.json + figs/depth_vs_degradation.png + README.md。

⚠️ 纪律：退化为合成施加（见 data/degrade.py），图注须标注。
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
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from wildspatial.data.tum import TUMDataset
from wildspatial.data.degrade import KINDS, LEVELS, degrade
from wildspatial.methods import get_method
from wildspatial.eval import align_trajectory
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
    conds = [("clean", None, 0.0)]
    for kind in KINDS:
        for lab, sev in LEVELS:
            conds.append((f"{KINDS[kind].split('（')[0]}-{lab}", kind, sev))
    return conds


def depth_metrics(pred, gt, max_depth=10.0):
    """pred, gt: (H,W) 同分辨率。返回 per-pixel 深度误差指标。"""
    mask = (gt > 0.1) & (gt < max_depth) & (pred > 1e-3)
    if mask.sum() < 10:
        return None
    # 中位比例对齐（VGGT 深度可能存在全局尺度歧义）
    ratio = np.median(gt[mask] / pred[mask])
    p = pred * ratio
    g = gt[mask]
    err = p[mask] - g
    rmse = float(np.sqrt(np.mean(err ** 2)))
    absrel = float(np.mean(np.abs(err) / g))
    d1 = float(np.mean(np.maximum(p[mask] / g, g / p[mask]) < 1.25))
    silog = float(np.sqrt(np.mean((np.log(p[mask]) - np.log(g)) ** 2)))
    return {"rmse": rmse, "absrel": absrel, "delta1": d1, "silog": silog,
            "scale": float(ratio), "n_valid": int(mask.sum())}


def backproject(gt_depth, K, T_wc, step=6, max_pts=4000):
    """从真值深度 + 真值位姿反投影点云 (GT 世界系)。"""
    H, W = gt_depth.shape
    us, vs = np.meshgrid(np.arange(0, W, step), np.arange(0, H, step))
    us, vs = us.ravel(), vs.ravel()
    d = gt_depth[vs, us]
    ok = (d > 0.1) & (d < 10.0)
    us, vs, d = us[ok], vs[ok], d[ok]
    if len(d) == 0:
        return np.zeros((0, 3))
    pts_cam = d[:, None] * (np.linalg.inv(K) @ np.stack([us, vs, np.ones_like(d)], 0)).T
    pts_w = (T_wc[:3, :3] @ pts_cam.T).T + T_wc[:3, 3]
    if len(pts_w) > max_pts:
        pts_w = pts_w[np.random.default_rng(0).choice(len(pts_w), max_pts, False)]
    return pts_w


def chamfer(A, B, n=3000):
    """双向 Chamfer 距离（内存安全，块化最近邻）。"""
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    rng = np.random.default_rng(0)

    def samp(X):
        if len(X) > n:
            X = X[rng.choice(len(X), n, False)]
        return X

    A, B = samp(A), samp(B)

    def nn_mean(X, Y):
        dsum, cnt = 0.0, 0
        bs = 400
        for i in range(0, len(X), bs):
            xi = X[i:i + bs]
            d = np.linalg.norm(xi[:, None, :] - Y[None, :, :], axis=2)
            dsum += d.min(axis=1).sum()
            cnt += d.shape[0]
        return dsum / max(cnt, 1)

    return float(nn_mean(A, B) + nn_mean(B, A))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="fr1/desk")
    ap.add_argument("--root", default=None)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--no-pc", action="store_true", help="跳过点云 Chamfer（更快）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = args.root or os.path.join(ROOT, "data", "raw", SEQUENCES[args.seq])
    prefix = args.seq.split("/")[0]
    fx, fy, cx, cy = TUM_K.get(prefix, TUM_K["fr1"])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    ds = TUMDataset(root)
    sel = list(range(0, len(ds), args.stride))[:args.frames]
    frames = [ds[i]["rgb"] for i in sel]
    gt_depths = [ds[i]["depth"] for i in sel]
    gt_Twc = [ds[i]["T_wc"] for i in sel]
    print(f"[·] {args.seq}：{len(frames)} 帧；有深度 {all(d is not None for d in gt_depths)}")

    method = get_method("vggt")
    if not method.available():
        print("[✗] VGGT 不可用（检查 vggt/torch/cuda）")
        return 1

    conds = build_conditions()
    depth_res = {}      # cond -> aggregated depth metrics
    pc_res = {}        # cond -> chamfer
    pred_cloud_all, gt_cloud_all = [], []

    for clabel, kind, sev in conds:
        t0 = time.time()
        if kind is None:
            dframes = frames
        else:
            dframes = [degrade(img, kind, sev) for img in frames]
        try:
            r = method.run(dframes, K, return_dense=True)
        except Exception as e:
            print(f"    {clabel:12s} | 运行失败: {e}")
            depth_res[clabel] = None
            continue

        if not r.ok or "depth" not in r.extra:
            depth_res[clabel] = None
            print(f"    {clabel:12s} | 无稠密输出")
            continue

        pred_depth = r.extra["depth"]              # (S,H,W) VGGT 分辨率
        pred_wp = r.extra.get("world_points")       # (S,H,W,3) 同分辨率
        H0, W0 = gt_depths[0].shape

        # ---- 深度质量 ----
        per_frame = []
        for i in range(len(pred_depth)):
            ph, pw = pred_depth[i].shape
            p = cv2.resize(pred_depth[i].astype(np.float32), (W0, H0),
                           interpolation=cv2.INTER_NEAREST)
            m = depth_metrics(p, gt_depths[i], max_depth=args.max_depth)
            if m:
                per_frame.append(m)
        if per_frame:
            agg = {k: float(np.mean([m[k] for m in per_frame]))
                   for k in ["rmse", "absrel", "delta1", "silog", "scale"]}
            agg["n_frames"] = len(per_frame)
        else:
            agg = None
        depth_res[clabel] = agg

        # ---- 点云质量 ----
        if not args.no_pc and pred_wp is not None:
            # 用相机中心轨迹把 VGGT 世界系对齐到真值系（Sim3）
            centers_gt = np.array([T[:3, 3] for T in gt_Twc])
            _, tf = align_trajectory(r.positions, centers_gt)
            s, R, t = tf["scale"], tf["R"], tf["t"]
            wp = pred_wp  # (S,H,W,3)
            for i in range(len(wp)):
                Xv = wp[i].reshape(-1, 3)
                Xw = (s * (R @ Xv.T)).T + t
                d_i = pred_depth[i].reshape(-1)
                mask = d_i > 1e-3
                Xw = Xw[mask][::8]  # 子采样
                if len(Xw) > 4000:
                    Xw = Xw[np.random.default_rng(i).choice(len(Xw), 4000, False)]
                pred_cloud_all.append(Xw)
                g = backproject(gt_depths[i], K, gt_Twc[i], step=6, max_pts=4000)
                gt_cloud_all.append(g)

        print(f"    {clabel:12s} | depth_rmse={agg['rmse'] if agg else 'NA':>7} | "
              f"{time.time()-t0:5.1f}s")

    if pred_cloud_all:
        pred_cloud = np.concatenate(pred_cloud_all, 0)
        gt_cloud = np.concatenate(gt_cloud_all, 0)
        ch = chamfer(pred_cloud, gt_cloud)
        pc_res["all"] = ch
        print(f"[·] 点云 Chamfer (全条件合并) = {ch:.4f} m")

    # ---- 写量化 ----
    out = args.out or os.path.join(ROOT, "experiments", "M4_depth_atlas")
    figdir = os.path.join(out, "figs")
    os.makedirs(figdir, exist_ok=True)
    with open(os.path.join(out, "depth_metrics.json"), "w") as f:
        json.dump({"seq": args.seq, "n_frames": len(frames),
                   "conditions": [c[0] for c in conds],
                   "depth": depth_res, "chamfer_all": pc_res},
                  f, ensure_ascii=False, indent=2)

    # ---- 图：各退化类型下深度 RMSE / AbsRel 随严重度变化 ----
    kinds = list(KINDS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = [l for l, _ in LEVELS]
    for ax, metric in zip(axes, ["rmse", "absrel"]):
        for kind in kinds:
            ys, xs = [], []
            for li, (lab, _) in enumerate(LEVELS):
                cname = f"{KINDS[kind].split('（')[0]}-{lab}"
                v = depth_res.get(cname, {})
                ys.append(v.get(metric, np.nan) if v else np.nan)
                xs.append(li)
            cclean = depth_res.get("clean", {})
            clean_v = cclean.get(metric) if cclean else np.nan
            ax.plot([ -0.3] + xs, [clean_v] + ys, "o-",
                    label=KINDS[kind].split('（')[0])
        ax.set_xticks([-0.3, 0, 1, 2])
        ax.set_xticklabels(["clean"] + labels, fontsize=8)
        ax.set_xlim(-0.6, 2.4)
        ax.set_title(f"VGGT 深度 {metric.upper()} vs 退化严重度（{args.seq}）")
        ax.set_xlabel("退化严重度（轻/中/重）"); ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "depth_vs_degradation.png"), dpi=130)
    plt.close(fig)

    # ---- README ----
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(f"# M4 深化 · VGGT 深度 / 点云质量失效图谱\n\n")
        f.write(f"**序列**：{args.seq}（{len(frames)} 帧，stride={args.stride}）\n\n")
        f.write("## 目的\nM5 只比了轨迹 ATE；本实验进一步用 VGGT 的稠密深度 / 世界点，")
        f.write("在 13 退化条件下量化「几何重建质量」是否也随退化崩坏。\n\n")
        f.write("## 方法\nVGGT(return_dense) 直接出深度图与 world_points；")
        f.write("深度用中位比例与真值对齐后算 RMSE/AbsRel/δ1；")
        f.write("world_points 借相机中心轨迹的 Sim3 对齐到真值系，与真值点云算 Chamfer。\n\n")
        f.write("## 结果（深度，详见 depth_metrics.json）\n")
        f.write("| 条件 | RMSE(m) | AbsRel | δ1 |\n|---|---|---|---|\n")
        for c in conds:
            v = depth_res.get(c[0])
            if v:
                f.write(f"| {c[0]} | {v['rmse']:.4f} | {v['absrel']:.4f} | "
                        f"{v['delta1']:.3f} |\n")
            else:
                f.write(f"| {c[0]} | — | — | — |\n")
        if pc_res:
            f.write(f"\n**点云 Chamfer（全条件合并）**: {pc_res['all']:.4f} m\n")
        f.write("\n> ⚠️ 退化为合成施加（data/degrade.py）。\n")

    print(f"[✓] 完成 → {out}")


if __name__ == "__main__":
    main()
