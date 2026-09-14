#!/usr/bin/env python
"""P2 · 感知实验台（二）：仿真 RGB → VGGT 重建 → 对比仿真真值深度（M4 落点）

目的
----
仿真的最大价值 = **带真值**。本脚本把 Gazebo 采到的多帧 RGB 喂给 **VGGT**
（前馈 3D 基础模型），输出深度/点云，再与**仿真真值深度**对比——
这正是 M4「前馈模型深度鲁棒性」在**可控仿真世界**里的复现。

与真实数据的区别
----------------
真实数据（TUM/SUN RGB-D）标深度昂贵、场景固定；
仿真可以**随手改世界/光照**，且**真值免费**——是 M4/M5 的"无限量实验台"。

产出
----
experiments/P2_gz_vggt/
    figs/  vggt_vs_gt_depth.png   VGGT 深度 vs 仿真真值 + 误差图
           vggt_pointcloud.png    重建点云
    metrics.json / README.md

诚实声明
--------
· VGGT 输出为**相对尺度**；与真值对比前按中位数比例对齐（同 M4 做法）。
· 仿真 RGB 的"材质"较简单（纯色方块），与真实纹理差异大，结论仅代表仿真域。
"""

import os
import sys
import json
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from wildspatial.viz import setup_plot_style
setup_plot_style()

SRC = os.path.join(ROOT, "experiments", "P2_gz_sense_degrade", "raw")
OUT = os.path.join(ROOT, "experiments", "P2_gz_vggt")
FIGS = os.path.join(OUT, "figs")


def main():
    os.makedirs(FIGS, exist_ok=True)
    print("[P2·感知实验台二] 仿真 RGB → VGGT 重建 → 对比真值")

    rgbs = sorted(glob.glob(os.path.join(SRC, "*_rgb.npy")))
    depths = sorted(glob.glob(os.path.join(SRC, "*_depth.npy")))
    if len(rgbs) < 2:
        print(f"[✗] 需要 ≥2 帧仿真 RGB（先跑 p2_gz_sense_degrade.py），"
              f"当前 {len(rgbs)}")
        return 1
    frames = [np.load(p) for p in rgbs]           # BGR? gz 输出 RGB
    gt_depths = [np.load(p) for p in depths]
    print(f"[·] 载入 {len(frames)} 帧仿真 RGB + 真值深度 "
          f"({frames[0].shape})")

    # ---- 调 VGGT ----
    from wildspatial.methods.vggt import VGGT
    m = VGGT()
    if not m.available():
        print("[✗] VGGT 不可用（需 vggt + torch cuda）")
        return 1
    print("[·] 运行 VGGT（首次加载 5GB 权重，耐心等）...")
    # VGGT 期望 BGR（opencv 写盘），gz 输出是 RGB → 转一下
    import cv2
    bgr_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if f.shape[2] == 3 else f
                  for f in frames]
    res = m.run(bgr_frames, K=None, return_dense=True)
    if not res.ok:
        print(f"[✗] VGGT 失败：{res.note}")
        return 1
    print(f"[OK] VGGT 输出：{res.note}")

    vggt_depth = res.extra.get("depth")      # (S,H,W)
    if vggt_depth is None:
        print("[✗] 未取到 VGGT 深度")
        return 1
    print(f"[·] VGGT 深度 shape={vggt_depth.shape}；真值深度 "
          f"shape={gt_depths[0].shape}（不同分辨率，需重采样对比）")

    # ---- 对比：把 VGGT 深度上采样到真值分辨率，中位比例对齐尺度 ----
    S = min(len(gt_depths), vggt_depth.shape[0])
    rows = []
    for i in range(S):
        vd = vggt_depth[i]
        gd = gt_depths[i].astype(np.float32)
        # 真值可能含无效值
        gd = np.where(np.isfinite(gd) & (gd > 0), gd, np.nan)
        # VGGT 深度上采样到真值尺寸
        vd_up = cv2.resize(vd, (gd.shape[1], gd.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        mask = np.isfinite(gd) & np.isfinite(vd_up) & (vd_up > 0)
        if mask.sum() < 100:
            continue
        # 中位比例对齐（VGGT 相对尺度 → 真值尺度）
        s = float(np.median(gd[mask]) / np.median(vd_up[mask]))
        vd_s = vd_up * s
        err = np.abs(vd_s[mask] - gd[mask])
        rel = err / np.maximum(gd[mask], 1e-6)
        rows.append({
            "frame": i, "scale": round(s, 4),
            "rmse": round(float(np.sqrt((err ** 2).mean())), 4),
            "absrel": round(float(rel.mean()), 4),
            "n_valid": int(mask.sum()),
        })
        # 存对齐后的 VGGT 深度供出图
        np.save(os.path.join(OUT, f"vggt_depth_aligned_{i:02d}.npy"), vd_s)

    if not rows:
        print("[✗] 无有效对比帧")
        return 1
    for r in rows:
        print(f"    frame{r['frame']}: scale={r['scale']}, "
              f"RMSE={r['rmse']}m, AbsRel={r['absrel']}")

    # ---- 出图 1：VGGT 深度 vs 真值 + 误差 ----
    make_depth_fig(frames, gt_depths, rows, os.path.join(FIGS, "vggt_vs_gt_depth.png"))
    print(f"[✓] 深度对比图 → {FIGS}/vggt_vs_gt_depth.png")

    # ---- 出图 2：点云 ----
    wp = res.extra.get("world_points")
    if wp is not None:
        make_pointcloud_fig(wp, os.path.join(FIGS, "vggt_pointcloud.png"))
        print(f"[✓] 点云图 → {FIGS}/vggt_pointcloud.png")

    metrics = {
        "n_frames": len(frames), "vggt": True,
        "comparison": rows,
        "mean_rmse": round(float(np.mean([r["rmse"] for r in rows])), 4),
        "mean_absrel": round(float(np.mean([r["absrel"] for r in rows])), 4),
        "note": "VGGT 相对尺度按中位比例对齐；仿真域材质简单",
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    write_readme(metrics)
    print(f"[✓] 完成 → {OUT}")


def make_depth_fig(frames, gt_depths, rows, out_png):
    """三行：RGB / VGGT 深度 / 真值深度（选前几帧）。"""
    n = min(3, len(rows))
    fig, axes = plt.subplots(3, n, figsize=(3.8 * n, 9))
    if n == 1:
        axes = axes.reshape(3, 1)
    for j in range(n):
        i = rows[j]["frame"]
        axes[0, j].imshow(frames[i]); axes[0, j].set_title(f"RGB 帧{i}", fontsize=10)
        axes[0, j].axis("off")
        vd = np.load(os.path.join(OUT, f"vggt_depth_aligned_{i:02d}.npy"))
        im1 = axes[1, j].imshow(vd, cmap="turbo", vmin=0, vmax=5)
        axes[1, j].set_title(f"VGGT 深度（对齐后）RMSE={rows[j]['rmse']}m",
                             fontsize=10)
        axes[1, j].axis("off")
        gd = gt_depths[i].copy(); gd[~np.isfinite(gd)] = np.nan
        axes[2, j].imshow(gd, cmap="turbo", vmin=0, vmax=5)
        axes[2, j].set_title("仿真真值深度", fontsize=10); axes[2, j].axis("off")
    fig.suptitle("P2 感知实验台：VGGT 重建深度 vs 仿真真值（中位尺度对齐）",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)


def make_pointcloud_fig(wp, out_png):
    """world_points (S,H,W,3) → 稀疏采样散点（俯视+侧视）。"""
    pts = wp.reshape(-1, 3)
    m = np.isfinite(pts).all(1)
    pts = pts[m]
    if len(pts) > 60000:
        idx = np.random.default_rng(0).choice(len(pts), 60000, replace=False)
        pts = pts[idx]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    z = pts[:, 2]
    zc = np.clip(z, np.percentile(z, 2), np.percentile(z, 98))
    axes[0].scatter(pts[:, 0], pts[:, 1], c=zc, s=0.6, cmap="turbo")
    axes[0].set_title("俯视 X-Y（颜色=Z）"); axes[0].set_aspect("equal")
    axes[0].set_xlabel("X"); axes[0].set_ylabel("Y")
    axes[1].scatter(pts[:, 0], pts[:, 2], c=zc, s=0.6, cmap="turbo")
    axes[1].set_title("侧视 X-Z"); axes[1].set_aspect("equal")
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Z")
    fig.suptitle("VGGT 重建点云（world_points）", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110); plt.close(fig)


def write_readme(metrics):
    rows = metrics["comparison"]
    lines = ["# P2 · 感知实验台（二）：仿真 RGB → VGGT 重建 → 对比真值\n",
             "> M4 落点：把前馈 3D 基础模型（VGGT）放到**可控仿真世界**里评测。\n",
             "## ① 目的\n",
             "仿真的核心优势是**真值免费**。本脚本验证：VGGT 在仿真域能否重建出与真值一致的深度。\n",
             "## ② 方法\n",
             "1. 载入 `p2_gz_sense_degrade.py` 采的**多帧仿真 RGB**；",
             "2. 喂给 VGGT（`return_dense=True`）→ 输出稠密深度 + 点云；",
             "3. VGGT 深度为**相对尺度** → 按**中位比例对齐**到仿真真值；",
             "4. 算 RMSE / AbsRel。\n",
             "## ③ 实测结果\n",
             f"- 帧数：**{metrics['n_frames']}**",
             f"- 平均 RMSE：**{metrics['mean_rmse']} m**",
             f"- 平均 AbsRel：**{metrics['mean_absrel']}**\n",
             "| 帧 | 尺度系数 | RMSE(m) | AbsRel | 有效像素 |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['frame']} | {r['scale']} | {r['rmse']} | "
                     f"{r['absrel']} | {r['n_valid']} |")
    lines += ["", "![VGGT vs 真值深度](figs/vggt_vs_gt_depth.png)\n",
              "> 三行：RGB / **VGGT 重建深度** / **仿真真值深度**。",
              "> 可对比结构是否一致（近处柱子、远处墙）。\n",
              "![VGGT 点云](figs/vggt_pointcloud.png)\n",
             "## ④ 直白讲解\n",
             "VGGT 像『一个只看过图片就懂了立体感的模型』：给它几帧 RGB，它直接『脑补』出每个像素多远。",
             "仿真世界给它**免费的真值**来打分——这就是『可控实验台』的意义：随便改光照/世界，真值永远免费。\n",
              "## ⑤ 诚实边界\n",
              "- VGGT 输出为相对尺度 → 用中位比例对齐（同 M4 真实数据做法）。\n",
              "- 仿真材质简单（纯色方块），纹理远不如真实，**结论仅代表仿真域**。\n",
              "## 如何复现\n", "```bash",
              "# 先采仿真数据，再跑 VGGT",
              "PYTHONPATH=src python scripts/p2_gz_sense_degrade.py",
              "PYTHONPATH=src python scripts/p2_gz_vggt_eval.py",
              "```\n"]
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write("\n".join(l for l in lines if l is not None))


if __name__ == "__main__":
    sys.exit(main())
