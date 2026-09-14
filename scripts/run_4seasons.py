"""4Seasons 真实恶劣天气序列 → 方法动物园 ATE 评测（Task 3 落地管线，单命令）

============ 目的 ============
把 4Seasons 的**真实退化**序列（夜间/雨雾/季节变化）直接喂给方法动物园，
检验 "VGGT 在真实恶劣条件下是否仍碾压几何法"（对照 M5 合成退化结论）。

4Seasons 自带：去畸变图像 undistorted_images/cam0、真值 reference_poses
（GNSSPoses.txt，全局优化位姿）、calibration（undistorted_calib_0.txt 内参）。
本脚本直接从 stereo_images_undistorted.zip 内读图（不落地整包），按时间戳把
图像帧对齐到 GNSSPoses 真值关键帧，逐方法算 Sim3 对齐后的 ATE。

============ 用法 ============
    PYTHONPATH=src python scripts/run_4seasons.py \
        --zip   data/raw/4seasons/oldtown_night/stereo.zip \
        --gt    data/raw/4seasons/oldtown_night/ref/recording_*/GNSSPoses.txt \
        --calib data/raw/4seasons/calibration/calibration/undistorted_calib_0.txt \
        --out   experiments/M5_4seasons_oldtown_night \
        --start 400 --n 220 --methods vggt handcrafted_vo lightglue_vo colmap

产出 experiments/<out>/：metrics.json, figs/ate_bar.png, README.md
"""

import os
import re
import sys
import json
import time
import argparse
import zipfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from wildspatial.methods import discover_methods
from wildspatial.viz import setup_plot_style

setup_plot_style()


def parse_times(raw_bytes):
    """times.txt: 每行 frame_id, timestamp_ns, exposure_ms → {int_id: ts_ns}"""
    table = {}
    for line in raw_bytes.decode(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            fid = int(parts[0])
            ts = int(parts[1])
            table[fid] = ts
        except ValueError:
            continue
    return table


def parse_gnss(raw_bytes):
    """GNSSPoses.txt: # ts_ns,tx,ty,tz,qx,qy,qz,qw,scale,fq,v3
    返回 (ts_array[int], pos_array[N,3])"""
    ts, pos = [], []
    for line in raw_bytes.decode(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split(",")
        if len(p) < 8:
            continue
        try:
            ts.append(int(p[0]))
            pos.append([float(p[1]), float(p[2]), float(p[3])])
        except ValueError:
            continue
    return np.array(ts, dtype=np.int64), np.array(pos, dtype=float)


def load_slice(zip_path, cam="cam0", start=0, n=0):
    """从 zip 直接读 cam0 图像切片 + 对应时间戳"""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist()
                 if re.search(rf"{cam}/[^/]+\.(png|jpg|jpeg)$", n, re.I)]
        names.sort()
        tnames = [n for n in z.namelist()
                  if re.search(rf"{cam}/times\.txt$", n, re.I)]
        times = parse_times(z.read(tnames[0])) if tnames else {}
        sel = names[start:start + n] if n else names[start:]
        frames, stamps = [], []
        for nm in sel:
            data = z.read(nm)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            frames.append(img)
            fid = int(os.path.basename(nm).split(".")[0])
            stamps.append(times.get(fid, 0))
    return frames, np.array(stamps, dtype=np.int64)


def nearest_pos(gt_ts, gt_pos, stamps):
    """每个图像帧时间戳 → 最近 GNSS 关键帧位置（Sim3 对齐会吸收残差）"""
    order = np.argsort(gt_ts)
    gts, gp = gt_ts[order], gt_pos[order]
    idx = np.searchsorted(gts, stamps)
    idx = np.clip(idx, 1, len(gts) - 1)
    left = idx - 1
    right = idx
    dleft = np.abs(gts[left] - stamps)
    dright = np.abs(gts[right] - stamps)
    pick = np.where(dleft <= dright, left, right)
    return gp[pick]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="stereo_images_undistorted.zip")
    ap.add_argument("--gt", required=True, help="GNSSPoses.txt")
    ap.add_argument("--calib", required=True, help="undistorted_calib_0.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=0, help="0=全部")
    ap.add_argument("--methods", nargs="*", default=None)
    args = ap.parse_args()

    # 内参
    with open(args.calib) as f:
        first = f.readline().split()
    # 行格式: Model fx fy cx cy d1..d4\n W H\n ...
    fx, fy, cx, cy = [float(x) for x in first[1:5]]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    frames, stamps = load_slice(args.zip, "cam0", args.start, args.n)
    if not frames:
        print("[✗] 没有读到图像")
        return 1
    gt_ts, gt_pos = parse_gnss(open(args.gt, "rb").read())
    gt_sel = nearest_pos(gt_ts, gt_pos, stamps)
    print(f"[·] {len(frames)} 帧 | K=({fx:.1f},{fy:.1f},{cx:.1f},{cy:.1f}) | "
          f"GT 关键帧 {len(gt_ts)}")

    methods = [m for m in discover_methods()
               if args.methods is None or m.name in args.methods]
    print(f"[·] 方法: {[m.name for m in methods]}")

    results = {}
    for m in methods:
        t0 = time.time()
        try:
            r = m.run(frames, K)
            ate = r.aligned_ate(gt_sel)["rmse"] if r.ok else float("nan")
        except Exception:
            import traceback
            traceback.print_exc()
            ate = float("nan")
            r = type("R", (), {"note": "异常", "ok": False})()
        ok = getattr(r, "ok", False) and np.isfinite(ate)
        results[m.name] = {
            "ate_rmse_m": None if not np.isfinite(ate) else float(ate),
            "ok": bool(ok), "note": getattr(r, "note", ""),
        }
        print(f"    {m.name:16s} | ATE={ate if ok else '失效':>8} | {time.time()-t0:5.1f}s")

    out = args.out
    figdir = os.path.join(out, "figs")
    os.makedirs(figdir, exist_ok=True)
    names = [m.name for m in methods]
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump({"zip": args.zip, "n_frames": len(frames),
                   "calib": [fx, fy, cx, cy], "results": results},
                  f, ensure_ascii=False, indent=2)

    vals = [results[n]["ate_rmse_m"] or 0 for n in names]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 4))
    bars = ax.bar(names, vals, color=["#2ca02c" if results[n]["ate_rmse_m"] else "#999"
                                      for n in names])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("ATE RMSE (m)")
    ax.set_title(f"4Seasons 真实退化序列 · {len(frames)} 帧（Sim3 对齐）")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "ate_bar.png"), dpi=130)
    plt.close(fig)

    best = min((results[n]["ate_rmse_m"] for n in names if results[n]["ate_rmse_m"]),
               default=None)
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("# 4Seasons 真实退化序列 · 方法对比\n\n")
        f.write(f"**序列**：`{os.path.basename(args.zip)}`（{len(frames)} 帧，cam0 单目）\n")
        f.write(f"**内参**：fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}\n\n")
        f.write("## 目的\n把真实恶劣天气序列接入方法动物园，检验 VGGT 是否仍碾压几何法"
                "（对照 M5 合成退化结论）。\n\n")
        f.write("## 方法原理\n各方法输出相机光心轨迹，与 GNSS 真值按时间戳对齐后 Sim3 "
                "对齐算 ATE。\n\n")
        f.write("## 结果\n| 方法 | ATE(m) | 状态 |\n|---|---|---|\n")
        for n in names:
            v = results[n]["ate_rmse_m"]
            f.write(f"| {n} | {v if v is not None else '失效'} | "
                    f"{'OK' if v else '失败'} |\n")
        if best is not None:
            f.write(f"\n**最优**：{names[vals.index(best)]}（{best:.3f}m）\n")

    print(f"[✓] 完成 → {out}")


if __name__ == "__main__":
    main()
