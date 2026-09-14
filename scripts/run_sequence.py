"""通用序列评测：把「任意真实序列」接入方法动物园（Task 3 落地管线）

==================== 目的 ====================
M5 的失效图谱用的是「TUM + 合成退化」。但项目真正关心的是**真实恶劣环境**
（地下/烟雾/水下，见 SubT-MRS、DUO 等）。本脚本把"一个图像文件夹 + 一条真值轨迹"
直接喂给方法动物园，算各方法 ATE——**无需合成退化**，用来检验
"VGGT 在真实烟雾/水下是否仍碾压几何法"这一命题。

==================== 用法 ====================
    PYTHONPATH=src python scripts/run_sequence.py \
        --img-dir data/raw/subt/forward/rgb \
        --gt     data/raw/subt/forward/groundtruth.txt \
        --calib  517.3 516.5 318.6 255.3 \
        --out    experiments/M5_subt_forward

真值轨迹支持两种格式（--gt-format）：
    tum   : 每行 "timestamp tx ty tz qx qy qz qw"（忽略时间戳，按行序对应图像序）
    xyzqw : 每行 "tx ty tz qx qy qz qw"（与图像按文件名排序后一一对应）
图像按文件名排序后与真值逐行对应（真实数据集通常已对齐）。

产出（experiments/<out>/）：
    metrics.json        各方法 ATE（Sim3 对齐）
    figs/ate_bar.png    方法 ATE 柱状图
    README.md           四段式结论
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

from wildspatial.methods import discover_methods
from wildspatial.eval import compute_ate
from wildspatial.viz import setup_plot_style

setup_plot_style()


def load_images(img_dir):
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    paths = sorted(
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.lower().endswith(exts))
    return paths


def load_gt(path, fmt):
    poses = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if fmt == "tum":
                p = p[1:]  # 去掉 timestamp
            # p: tx ty tz qx qy qz qw
            t = np.array([float(x) for x in p[:3]])
            q = np.array([float(x) for x in p[3:7]])
            # 四元数 (x,y,z,w) -> 旋转矩阵
            R = _quat_to_R(q)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            poses.append(T)
    return poses


def _quat_to_R(q):
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--gt-format", default="tum", choices=["tum", "xyzqw"])
    ap.add_argument("--calib", required=True,
                    help="fx fy cx cy，或含这四数的文件")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0=全部")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # 内参
    if os.path.isfile(args.calib):
        nums = open(args.calib).read().split()
    else:
        nums = args.calib.split()
    fx, fy, cx, cy = [float(x) for x in nums[:4]]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    img_paths = load_images(args.img_dir)
    if not img_paths:
        print(f"[✗] 图像目录为空: {args.img_dir}")
        return 1
    import cv2
    frames = [cv2.imread(p, cv2.IMREAD_COLOR) for p in img_paths]
    if args.max_frames:
        frames = frames[:args.max_frames]

    gt_poses = load_gt(args.gt, args.gt_format)
    # 取与帧数对齐的真值（按行序；不足则截断图像）
    n = min(len(frames), len(gt_poses))
    frames, gt_poses = frames[:n], gt_poses[:n]
    gt_pos = np.array([T[:3, 3] for T in gt_poses])
    print(f"[·] {len(frames)} 帧，内参 fx={fx} fy={fy} cx={cx} cy={cy}")

    methods = [m for m in discover_methods()
               if args.methods is None or m.name in args.methods]
    if not methods:
        print("[✗] 没有可用方法")
        return 1
    print(f"[·] 方法: {[m.name for m in methods]}")

    results = {}
    raw = []
    for m in methods:
        t0 = time.time()
        try:
            r = m.run(frames, K)
            ate = r.aligned_ate(gt_pos)["rmse"] if r.ok else float("nan")
        except Exception as e:
            import traceback
            traceback.print_exc()
            ate = float("nan")
            r = type("R", (), {"note": f"异常: {e}"})()
        ok = r.ok and np.isfinite(ate)
        results[m.name] = {
            "ate_rmse_m": None if not np.isfinite(ate) else float(ate),
            "ok": bool(ok),
            "note": getattr(r, "note", ""),
        }
        raw.append(ate if np.isfinite(ate) else np.nan)
        print(f"    {m.name:16s} | ATE={ate if ok else '失效':>8} | {time.time()-t0:5.1f}s")
    raw = np.array(raw)

    out = args.out
    figdir = os.path.join(out, "figs")
    os.makedirs(figdir, exist_ok=True)
    names = [m.name for m in methods]
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump({"img_dir": args.img_dir, "n_frames": len(frames),
                   "calib": [fx, fy, cx, cy], "results": results},
                  f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 4))
    vals = [results[n]["ate_rmse_m"] or 0 for n in names]
    bars = ax.bar(names, vals,
                  color=["#2ca02c" if results[n]["ate_rmse_m"] else "#999" for n in names])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("ATE RMSE (m)")
    ax.set_title(f"真实序列方法对比 · {len(frames)} 帧（无合成退化）")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "ate_bar.png"), dpi=130)
    plt.close(fig)

    best = min((results[n]["ate_rmse_m"] for n in names if results[n]["ate_rmse_m"]),
               default=None)
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(f"# 真实序列方法对比（无合成退化）\n\n")
        f.write(f"**序列**：`{args.img_dir}`（{len(frames)} 帧）\n\n")
        f.write("## 目的\n把真实恶劣环境序列接入方法动物园，检验 VGGT 是否仍碾压几何法"
                "（对照 M5 合成退化结论）。\n\n")
        f.write("## 方法原理\n各方法统一输出相机光心轨迹，与真值 Sim3 对齐后算 ATE。\n\n")
        f.write("## 结果\n| 方法 | ATE(m) | 状态 |\n|---|---|---|\n")
        for n in names:
            v = results[n]["ate_rmse_m"]
            f.write(f"| {n} | {v if v is not None else '失效'} | "
                    f"{'OK' if v else '失败'} |\n")
        if best is not None:
            f.write(f"\n**最优**：{names[vals.index(best)]}（{best:.3f}m）\n")
        f.write("\n> 真值来自数据集自带轨迹；若序列无真值则无法算 ATE。\n")

    print(f"[✓] 完成 → {out}")


if __name__ == "__main__":
    main()
