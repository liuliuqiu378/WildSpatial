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
        --start 400 --n 220 --methods vggt handcrafted_vo lightglue_vo colmap_sfm

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
from wildspatial.eval.trajectory import align_trajectory, compute_ate

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
    """从 zip 直接读 cam0 图像切片 + 对应时间戳。

    4Seasons 的 undistorted 图像文件名即纳秒时间戳
    （如 `cam0/1620675660292607232.png`），与 GNSSPoses.txt 的 ts_ns 同尺度，
    因此**优先用文件名当时间戳**（最稳，不依赖 times.txt 的路径）。
    times.txt（位于 recording 根目录而非 cam0 子目录）仅作兜底。
    """
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist()
                 if re.search(rf"{cam}/[^/]+\.(png|jpg|jpeg)$", n, re.I)]
        names.sort()
        tnames = [n for n in z.namelist() if re.search(r"times\.txt$", n, re.I)]
        times = parse_times(z.read(tnames[0])) if tnames else {}
        sel = names[start:start + n] if n else names[start:]
        frames, stamps = [], []
        for nm in sel:
            data = z.read(nm)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            frames.append(img)
            fid = os.path.basename(nm).split(".")[0]
            # 文件名是纯数字时间戳 → 直接用；否则回退 times.txt / 0
            try:
                stamps.append(int(fid))
            except ValueError:
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

    # 真值轨迹长度（与「每个方法实际输出的帧」对齐后算，才有可比性）
    def _path_len(pts):
        pts = np.asarray(pts, float).reshape(-1, 3)
        if len(pts) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    gt_full_len = _path_len(gt_sel)  # 整段 GT 路径长度

    results = {}
    raw = {}          # 保留 MethodResult 原对象（含 positions），供画轨迹图
    for m in methods:
        t0 = time.time()
        try:
            r = m.run(frames, K)
            if r.ok:
                # 关键：部分方法（如 COLMAP）返回的 frame_indices 是其自身注册顺序，
                # 并非时间顺序。轨迹长度与绘图必须按帧时间排序，否则会把点连成乱序折线；
                # 而 Umeyama(Sim3) 对齐本身与顺序无关，ATE 不受影响。
                fi = np.asarray(r.frame_indices)
                order = np.argsort(fi)
                fi = fi[order]
                pos_s = np.asarray(r.positions)[order]
                gt_for_est = np.asarray(gt_sel)[fi]
                # 单目方法输出带任意尺度，必须用 Sim3 对齐后的长度判断「是否真动了」，
                # 否则会把 VGGT 这种尺度不定但形状正确的结果误判为退化。
                aligned_est, _ = align_trajectory(pos_s, gt_for_est,
                                                 allow_scale=True)
                aligned_len = _path_len(aligned_est)
                gt_len = _path_len(gt_for_est)
                est_len = aligned_len          # 报告对齐后的物理路径长
                ate = compute_ate(pos_s, gt_for_est, allow_scale=True)["rmse"]
                # 退化判定：对齐后轨迹仍几乎没动（< GT 的 5%）→ 跟踪丢失/冻结，
                # 此时 ATE 只反映「没动」而非「定位偏差」，必须单独标记以免误导。
                degenerate = (aligned_len < 0.05 * max(gt_len, 1e-6))
            else:
                ate, est_len, gt_len, degenerate = float("nan"), 0.0, 0.0, False
        except Exception:
            import traceback
            traceback.print_exc()
            ate, est_len, gt_len, degenerate = float("nan"), 0.0, 0.0, False
            r = type("R", (), {"note": "异常", "ok": False})()
        ok = getattr(r, "ok", False) and np.isfinite(ate) and not degenerate
        status = "OK" if ok else ("跟踪丢失/退化" if degenerate else "失败")
        results[m.name] = {
            "ate_rmse_m": None if not np.isfinite(ate) else float(ate),
            "est_path_len_m": float(est_len),
            "gt_path_len_m": float(gt_len),
            "degenerate": bool(degenerate),
            "ok": bool(ok), "note": getattr(r, "note", ""),
        }
        print(f"    {m.name:16s} | ATE={ate if np.isfinite(ate) else '—':>8} | "
              f"est_len={est_len:7.2f} gt_len={gt_len:7.2f} | {status} | "
              f"{time.time()-t0:5.1f}s")
        raw[m.name] = r

    out = args.out
    figdir = os.path.join(out, "figs")
    os.makedirs(figdir, exist_ok=True)
    names = [m.name for m in methods]
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump({"zip": args.zip, "n_frames": len(frames),
                   "calib": [fx, fy, cx, cy], "gt_full_path_len_m": gt_full_len,
                   "results": results},
                  f, ensure_ascii=False, indent=2)

    # 配色：OK=绿，跟踪丢失/退化=橙（并非算法"偏了"，而是"根本没动"），失败=灰
    def _color(n):
        if not results[n]["ok"]:
            return "#999999" if not results[n]["degenerate"] else "#ff7f0e"
        return "#2ca02c"
    vals = [results[n]["ate_rmse_m"] or 0 for n in names]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 4))
    bars = ax.bar(names, vals, color=[_color(n) for n in names])
    for b, n in zip(bars, names):
        v = vals[names.index(n)]
        label = f"{v:.3f}" if results[n]["ok"] else (
            "跟踪丢失" if results[n]["degenerate"] else "失败")
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), label,
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("ATE RMSE (m)")
    ax.set_title(f"4Seasons 真实夜间序列 · {len(frames)} 帧（Sim3 对齐）\n"
                 f"GT 路径长 {gt_full_len:.1f}m")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "ate_bar.png"), dpi=130)
    plt.close(fig)

    # 俯视轨迹对比图：把每个「有轨迹」的方法对齐后叠到 GT 上（视觉证明 VGGT 跟住了）
    plotted = 0
    for n in names:
        r = raw.get(n)
        if r is None or not getattr(r, "ok", False):
            continue
        pos = np.asarray(r.positions, float).reshape(-1, 3)
        fi = np.asarray(r.frame_indices)
        if len(pos) < 4:
            continue
        order = np.argsort(fi)
        pos = pos[order]
        g = np.asarray(gt_sel)[fi[order]]
        a_est, _ = align_trajectory(pos, g, allow_scale=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(g[:, 0], g[:, 2], "-o", color="#444", ms=3, lw=1.5,
                label=f"GNSS 真值 (len {_path_len(g):.1f}m)")
        ax.plot(a_est[:, 0], a_est[:, 2], "-", color="#2ca02c", lw=1.8,
                label=f"{n} 对齐后 (len {_path_len(a_est):.1f}m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
        ax.set_title(f"{n} 俯视轨迹 vs 真值（ATE={results[n]['ate_rmse_m']:.3f}m）")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, f"traj_{n}.png"), dpi=130)
        plt.close(fig)
        plotted += 1
    if plotted:
        print(f"[·] 已出 {plotted} 张俯视轨迹对比图 → {figdir}")

    best = min((results[n]["ate_rmse_m"] for n in names if results[n]["ate_rmse_m"]),
               default=None)
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("# 4Seasons 真实退化序列 · 方法对比\n\n")
        f.write(f"**序列**：`{os.path.basename(args.zip)}`（{len(frames)} 帧，cam0 单目）\n")
        f.write(f"**内参**：fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}\n")
        f.write(f"**GT 整段路径长**：{gt_full_len:.1f} m\n\n")
        f.write("## 目的\n把真实恶劣天气序列接入方法动物园，检验 VGGT 是否仍碾压几何法"
                "（对照 M5 合成退化结论）。\n\n")
        f.write("## 方法原理\n各方法输出相机光心轨迹，与 GNSS 真值按时间戳对齐后 Sim3 "
                "对齐算 ATE。\n\n")
        f.write("## 结果\n| 方法 | ATE(m) | 估计路径长(m) | GT对应路径长(m) | 状态 |\n")
        f.write("|---|---|---|---|---|\n")
        for n in names:
            v = results[n]["ate_rmse_m"]
            r = results[n]
            st = ("OK" if r["ok"] else ("跟踪丢失/退化" if r["degenerate"] else "失败"))
            f.write(f"| {n} | {v if v is not None else '—'} | "
                    f"{r['est_path_len_m']:.2f} | {r['gt_path_len_m']:.2f} | {st} |\n")
        if best is not None:
            f.write(f"\n**最优（有效）**：{names[vals.index(best)]}（{best:.3f}m）\n")
        f.write("\n> 注：标记为「跟踪丢失/退化」的方法，其估计轨迹几乎没动"
                "（路径长 << GT），ATE 只反映「没动」而非「定位偏差」——"
                "这是经典几何法在夜间低纹理上丢失跟踪的典型表现，并非算法算错了位置。\n")
        # 轨迹对比图（俯视）
        traj_pngs = [n for n in names
                     if os.path.exists(os.path.join(figdir, f"traj_{n}.png"))]
        if traj_pngs:
            f.write("\n## 俯视轨迹对比（对齐后 vs GNSS 真值）\n")
            for n in traj_pngs:
                f.write(f"\n### {n}\n![]({os.path.relpath(os.path.join(figdir, f'traj_{n}.png'), out)})\n")

    print(f"[✓] 完成 → {out}")


if __name__ == "__main__":
    main()
