"""
m10_kitti_lidar_demo.py — F5 升级：KITTI 真实激光雷达 + 相机融合（真实数据）
=================================================================================
对应 docs/F5_vision_lidar_fusion.md §4。原 m10_lidar_fusion_demo.py 用 TUM RGB-D 深度
**反投影**点云演示融合几何（已诚实标注"非真实 LiDAR"）。本脚本用 **KITTI depth completion**
基准的**真实激光雷达**数据，把"模拟 LiDAR"替换为真实传感器：

  - `image/`            : 真实车载相机 RGB（左目 image_02）
  - `velodyne_raw/`     : **真实稀疏激光雷达深度**（投影到图像，16-bit PNG，depth=val/256 m）
  - `groundtruth_depth/`: **真实稠密深度**（多帧激光累积，作为"理想 3D 几何"真值）
  - `intrinsics/`       : 相机内参 P2（3×4）→ 取 fx/fy/cx/cy

演示「相机（语义强） × 真实激光（几何强）」融合：
  ① 相机 RGB               —— 看得懂但没距离
  ② 真实稀疏激光 → 图像    —— 把真实 LiDAR 点按距离着色投回图像（稀疏扫描线，正是真雷达形态）
  ③ 真实稠密深度图         —— 真实激光累积出的"每像素多远"
  ④ 真实深度 → 彩色 3D 点云 —— 外观 × 真实几何 的融合体

数据：data/raw/ms/OmniData__KITTI_depth_completion/raw/KITTI_depth_completion/
      depth_selection/val_selection_cropped/

运行（wildspatial 环境，纯 cv2/numpy/matplotlib）：
  PYTHONPATH=src python scripts/m10_kitti_lidar_demo.py
"""
import os
import re
import glob
import json

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(
    ROOT, "data/raw/ms/OmniData__KITTI_depth_completion/raw/KITTI_depth_completion",
    "depth_selection", "val_selection_cropped")
IMAGE_DIR = os.path.join(BASE, "image")
VELO_DIR = os.path.join(BASE, "velodyne_raw")
GT_DIR = os.path.join(BASE, "groundtruth_depth")
INTR_DIR = os.path.join(BASE, "intrinsics")

OUT = os.path.join(ROOT, "experiments", "M10_kitti_lidar")
FIGS = os.path.join(OUT, "figs")
os.makedirs(FIGS, exist_ok=True)

N_SHOW = 4  # 展示样本数


def decode_kitti_depth(png_path):
    """KITTI 深度图为 16-bit PNG：depth_m = png / 256.0，0 为无效。"""
    img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    depth = img.astype(np.float32) / 256.0
    depth[depth <= 0] = np.nan
    return depth


def load_intrinsics(txt_path):
    # KITTI depth_selection/intrinsics 为单行 9 个数的 3×3 内参矩阵：
    #   [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    with open(txt_path) as f:
        nums = [float(x) for x in f.read().split()]
    K = np.array(nums[:9]).reshape(3, 3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return fx, fy, cx, cy


def main():
    imgs = sorted(glob.glob(os.path.join(IMAGE_DIR, "*_image_02.png")))
    if not imgs:
        raise RuntimeError(f"未找到 KITTI 图像：{IMAGE_DIR}（先解压 data_depth_selection.zip）")
    # 均匀抽 N_SHOW 张
    idx = np.linspace(0, len(imgs) - 1, N_SHOW, dtype=int)
    picks = [imgs[i] for i in idx]

    panels = []  # (rgb, velo_depth, gt_depth, fx,fy,cx,cy, name)
    for fpath in picks:
        fname = os.path.basename(fpath)
        m = re.match(r"(2011\S+?)_image_(\d+)_image_02\.png", fname)
        if not m:
            continue
        drive, frame = m.group(1), m.group(2)
        velo_path = os.path.join(VELO_DIR, f"{drive}_velodyne_raw_{frame}_image_02.png")
        gt_path = os.path.join(GT_DIR, f"{drive}_groundtruth_depth_{frame}_image_02.png")
        intr_path = os.path.join(INTR_DIR, f"{drive}_image_{frame}_image_02.txt")
        if not (os.path.exists(velo_path) and os.path.exists(gt_path) and os.path.exists(intr_path)):
            continue
        rgb = cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)
        velo = decode_kitti_depth(velo_path)
        gt = decode_kitti_depth(gt_path)
        fx, fy, cx, cy = load_intrinsics(intr_path)
        panels.append((rgb, velo, gt, fx, fy, cx, cy, fname))

    if not panels:
        raise RuntimeError("未能匹配到任何 (图像, 激光, 真值, 内参) 四元组")

    # ---------- 汇总指标 ----------
    metrics = {"dataset": "KITTI depth completion (val_selection_cropped, image_02)",
               "n_panels": len(panels), "depth_decode": "png/256.0 (m), 0=invalid",
               "samples": []}
    # ---------- 画 N_SHOW × 4 图 ----------
    fig, axes = plt.subplots(len(panels), 4, figsize=(20, 5 * len(panels)))
    if len(panels) == 1:
        axes = axes[None, :]
    for ri, (rgb, velo, gt, fx, fy, cx, cy, name) in enumerate(panels):
        H, W = rgb.shape[:2]
        # 真实激光有效点
        vmask = ~np.isnan(velo)
        vvalid = int(vmask.sum())
        vmin, vmax = (np.nanmin(velo), np.nanmax(velo)) if vvalid else (0, 1)

        # ① RGB
        ax = axes[ri, 0]
        ax.imshow(rgb); ax.axis("off")
        ax.set_title(f"① Camera RGB (real)\n{name}\nsemantic-strong, no depth", fontsize=8)

        # ② 真实稀疏激光 → 图像
        ax = axes[ri, 1]
        ax.imshow(rgb, alpha=0.55)
        if vvalid:
            uu, vv = np.meshgrid(np.arange(W), np.arange(H))
            sc = ax.scatter(uu[vmask], vv[vmask], c=velo[vmask], s=4,
                           cmap="turbo", vmin=0, vmax=np.nanpercentile(velo, 98),
                           edgecolors="none")
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("LiDAR Z (m)")
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
        ax.set_title(f"② Real sparse LiDAR → image\n{vvalid} real LiDAR points (color=distance)", fontsize=8)

        # ③ 真实稠密深度图
        ax = axes[ri, 2]
        if gt is not None and np.nansum(~np.isnan(gt)) > 0:
            gmask = ~np.isnan(gt)
            d = np.where(gmask, gt, np.nan)
            im = ax.imshow(d, cmap="turbo", vmin=0, vmax=np.nanpercentile(gt, 98))
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("depth (m)")
        ax.axis("off")
        ax.set_title("③ Real dense depth (multi-frame LiDAR GT)\ngeometry-strong, light-invariant", fontsize=8)

        # ④ 真实深度 → 彩色 3D 点云
        ax = axes[ri, 3]
        if gt is not None:
            gmask = ~np.isnan(gt)
            vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            Z = gt[gmask]
            X = (uu[gmask] - cx) * Z / fx
            Y = (vv[gmask] - cy) * Z / fy
            cols = rgb[gmask] / 255.0
            # 下采样控制点数
            n = len(Z)
            if n > 40000:
                sel = np.random.RandomState(0).choice(n, 40000, replace=False)
                X, Y, Z, cols = X[sel], Y[sel], Z[sel], cols[sel]
            ax.scatter(X, Z, c=cols, s=0.4, marker=".")
            ax.set_xlabel("X right (m)"); ax.set_ylabel("Z forward (m)")
            ax.set_title("④ Real depth → colored point cloud (top-view X-Z)\nappearance x real geometry (fusion)", fontsize=8)

        # 指标
        gvalid = int((~np.isnan(gt)).sum()) if gt is not None else 0
        metrics["samples"].append({
            "name": name, "size": [W, H], "fx_fy_cx_cy": [round(fx, 1), round(fy, 1), round(cx, 1), round(cy, 1)],
            "lidar_sparse_points": vvalid, "lidar_depth_m": [round(float(vmin), 2), round(float(vmax), 2)],
            "gt_dense_points": gvalid})

    fig.suptitle("F5 · Real LiDAR + camera fusion — KITTI real scene (real sensor data)", fontsize=13)
    fig.tight_layout()
    out_png = os.path.join(FIGS, "kitti_lidar_fusion.png")
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    metrics["note"] = ("KITTI 真实 LiDAR 替代原 demo 的 RGB-D 反投影点云；"
                       "velodyne_raw=真实稀疏激光、groundtruth_depth=真实稠密几何真值")
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[✓] KITTI 真实 LiDAR 融合 demo → {OUT}")
    print(f"    样本数={len(panels)} | 首图稀疏激光点={metrics['samples'][0]['lidar_sparse_points']} | "
          f"深度范围={metrics['samples'][0]['lidar_depth_m']} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
