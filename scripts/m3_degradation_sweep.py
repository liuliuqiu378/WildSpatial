"""M3 · 退化建模与失效归因（Degradation Sweep）

=========================== 目的（为什么做这个）===========================
前面 M1 把系统**跑通**了（干净数据 ATE ≈ 0.16~0.51 m）。但本项目真正关心的是
**恶劣物理环境**下的可靠性：水下、地下、矿山、烟雾、低光照。
所以这里要系统回答三个问题：
    1. **什么条件**会让系统崩？（模糊？噪声？暗光？分辨率？）
    2. **崩到什么程度**？（多严重开始崩 —— 定量给出临界点）
    3. **崩在哪一步**？（特征 → 匹配 → 位姿 → 建图，哪一环先断）

=========================== 方法原理 ============================
**受控对照实验（Controlled Experiment）**：固定算法与全部参数，只改变**成像条件**这一个变量，
观察各项指标如何退化。这是定位失效根因的标准做法。

困难在于：带**真值标签**的恶劣环境公开数据集极其稀缺。因此采用
**"带真值数据 + 合成退化"**的方案：

        干净序列（有 ground truth）──施加合成退化──▶ 退化序列（**仍保留真值**）

这样每一档都能定量算 ATE / RPE，从而画出"失效曲线"。
⚠️ 所有结论都标注"退化为合成施加"，不代表真实采集数据。

=========================== 观测指标（对应管线的每一环）===========================
    n_features   特征点数   ← 第 1 环：还能不能找到特征
    n_matches    匹配数     ← 第 2 环：还能不能配上
    inlier_ratio 内点率     ← 第 3 环：配上的是否几何自洽（误匹配多不多）
    n_map        地图点数   ← 第 4 环：还能不能建出图
    ATE / RPE    轨迹误差   ← 最终结果：位姿估计准不准
    status       跟踪状态   ← 崩的方式（初始化失败 / 丢失 / 位姿跳变被拒）

产出（experiments/M3_degradation_sweep/）
    figs/failure_atlas.png     失效图谱：各指标随退化严重度的变化曲线
    figs/sample_images.png     各退化档位下"算法实际看到的图"+特征点
    figs/status_breakdown.png  失效方式分解（崩在哪一步）
    metrics.json
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2

from wildspatial.data.tum import TUMDataset
from wildspatial.data.degrade import KINDS, LEVELS, degrade, level_label
from wildspatial.sfm import MonocularVO, VOConfig
from wildspatial.sfm.features import extract_features, draw_keypoints
from wildspatial.eval.trajectory import compute_ate, compute_rpe
from wildspatial.viz import setup_plot_style

setup_plot_style()

TUM_K = {"fr1": (517.3, 516.5, 318.6, 255.3),
         "fr2": (520.9, 521.0, 325.1, 249.7),
         "fr3": (535.4, 539.2, 320.1, 247.6)}
SEQ_DIR = {"fr1/desk": "rgbd_dataset_freiburg1_desk"}


def K_of(seq):
    fx, fy, cx, cy = TUM_K.get(seq.split("/")[0], TUM_K["fr1"])
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def run_once(root, K, frames, stride, kind, sev):
    """在给定退化条件下跑一遍 VO，返回量化结果"""
    ds = TUMDataset(root)
    n = len(ds) if frames <= 0 else min(frames, len(ds))
    cfg = VOConfig(feature="sift", max_features=3000, min_parallax_deg=1.0)
    vo = MonocularVO(K, cfg)
    gt = []
    rng = np.random.default_rng(0)
    for i in range(0, n, max(1, stride)):
        f = ds[i]
        if "T_wc" in f:
            gt.append(f["T_wc"][:3, 3])
        vo.process(degrade(f["rgb"], kind, sev, rng=rng), None)

    est = vo.trajectory()
    gt = np.array(gt) if len(gt) else np.zeros((0, 3))
    d = vo.diagnostics()

    res = {
        "kind": kind, "severity": float(sev), "level": level_label(sev),
        "initialized": bool(vo.initialized),
        "n_features": float(np.mean(d["n_features"])) if len(d["n_features"]) else 0.0,
        "n_matches": float(np.mean(d["n_matches"])) if len(d["n_matches"]) else 0.0,
        "inlier_ratio": float(np.mean(d["inlier_ratio"])) if len(d["inlier_ratio"]) else 0.0,
        "n_map": float(len(vo.points_3d)),
        "ate": float("nan"), "rpe_trans": float("nan"), "rpe_rot": float("nan"),
    }
    status = {}
    for s in d["status"]:
        status[s] = status.get(s, 0) + 1
    res["status"] = status

    if vo.initialized and len(gt) > 5 and len(est) > 5:
        m = min(len(est), len(gt))
        try:
            res["ate"] = float(compute_ate(est[:m], gt[:m], allow_scale=True)["rmse"])
        except Exception:
            pass
    return res


# ------------------------------------------------------------------ 图1 失效图谱
def fig_failure_atlas(results, path):
    """各指标随退化严重度的变化 —— 这是"失效图谱"的主体"""
    kinds = [k for k in KINDS]
    metrics = [("ate", "ATE RMSE (m) ↓越小越好", True),
               ("n_features", "平均特征点数", False),
               ("inlier_ratio", "平均内点率", False),
               ("n_map", "最终地图点数", False)]
    clean = next((r for r in results if r["kind"] == "clean"), None)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, title, lower_better) in zip(axes.ravel(), metrics):
        # 干净基线：画一条水平虚线作参照，才看得出"退化了多少"
        if clean is not None and np.isfinite(clean.get(key, float("nan"))):
            ax.axhline(clean[key], color="black", linestyle="--", linewidth=1.2,
                       alpha=0.7, label="干净基线" if key == "ate" else None)
        for kind in kinds:
            rows = sorted([r for r in results if r["kind"] == kind],
                          key=lambda r: r["severity"])
            if not rows:
                continue
            xs = [r["severity"] for r in rows]
            ys = [r[key] for r in rows]
            ax.plot(xs, ys, "o-", linewidth=1.8, markersize=6, label=KINDS[kind])
        ax.set_xlabel("退化严重度"); ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=8, loc="upper left")
    plt.suptitle("图 · 失效图谱：各项指标随退化严重度的变化"
                 "（TUM fr1/desk + 合成退化，每点均为一次完整实跑）", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


# ------------------------------------------------------------------ 图2 样本图
def fig_sample_images(root, path, frame_idx=None):
    """各退化档位下，算法"实际看到的图"与能提取的特征点"""
    ds = TUMDataset(root)
    img0 = ds[frame_idx if frame_idx is not None else len(ds) // 3]["rgb"]
    kinds = list(KINDS.keys())
    fig, axes = plt.subplots(len(kinds), len(LEVELS) + 1,
                             figsize=(4.0 * (len(LEVELS) + 1), 3.6 * len(kinds)))
    for r, kind in enumerate(kinds):
        # 第一列：干净原图
        ax = axes[r, 0] if len(kinds) > 1 else axes[0]
        fs = extract_features(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY), "sift",
                              max_features=2000)
        ax.imshow(cv2.cvtColor(draw_keypoints(img0, fs), cv2.COLOR_BGR2RGB))
        ax.set_title("干净\n" + (f"特征={len(fs)}" if r == 0 else f"{len(fs)}"),
                     fontsize=11)
        ax.axis("off")
        for c, (lname, sev) in enumerate(LEVELS, start=1):
            ax = axes[r, c] if len(kinds) > 1 else axes[c]
            img = degrade(img0, kind, sev)
            fs = extract_features(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), "sift",
                                  max_features=2000)
            ax.imshow(cv2.cvtColor(draw_keypoints(img, fs), cv2.COLOR_BGR2RGB))
            ax.set_title(f"{lname}（sev={sev:.2f}）\n特征={len(fs)}", fontsize=11)
            ax.axis("off")
        # 行标题
        axes[r, 0].set_ylabel(KINDS[kind].split("（")[0], fontsize=11, rotation=90,
                              labelpad=12)
    plt.suptitle("图 · 各退化档位下算法实际看到的图像与可提取特征点"
                 "（TUM fr1/desk + 合成退化）", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


# ------------------------------------------------------------------ 图3 失效方式
def fig_status_breakdown(results, path):
    """崩的方式：跟踪状态分布 —— 定位"崩在哪一步" """
    # 只取"重"档 + 干净，避免图太挤
    sel = [r for r in results if r["kind"] == "clean" or r["level"] == "重"]
    labels = ["clean" if r["kind"] == "clean"
              else f"{KINDS[r['kind']].split('（')[0]}-{r['level']}" for r in sel]
    all_status = sorted({s for r in sel for s in r["status"]})
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(sel))
    for i, st in enumerate(all_status):
        vals = np.array([r["status"].get(st, 0) for r in sel], dtype=float)
        tot = np.array([max(1, sum(r["status"].values())) for r in sel], dtype=float)
        ax.bar(range(len(sel)), vals / tot, bottom=bottom, label=st,
               color=cmap(i / max(1, len(all_status) - 1)))
        bottom += vals / tot
    ax.set_xticks(range(len(sel)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("帧占比"); ax.set_ylim(0, 1.05)
    ax.set_title("跟踪状态分布：看系统「崩在哪一步」", fontsize=12)
    ax.legend(fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="fr1/desk", choices=list(SEQ_DIR))
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "experiments", "M3_degradation_sweep")
    figs = os.path.join(out_dir, "figs")
    os.makedirs(figs, exist_ok=True)

    root = os.path.join(ROOT, "data", "raw", SEQ_DIR[args.seq])
    K = K_of(args.seq)
    if not os.path.exists(root):
        print(f"[✗] 数据不存在: {root}")
        return 1

    results = []
    # 1) 干净基线
    print("[·] 干净基线...")
    r = run_once(root, K, args.frames, args.stride, "clean", 0.0)
    r["kind"] = "clean"
    results.append(r)
    print(f"    ATE={r['ate']:.3f}m 特征={r['n_features']:.0f} 地图点={r['n_map']:.0f}")

    # 2) 各退化类型 × 各档位
    for kind in KINDS:
        for lname, sev in LEVELS:
            print(f"[·] {KINDS[kind]} · {lname} (sev={sev:.2f}) ...")
            r = run_once(root, K, args.frames, args.stride, kind, sev)
            results.append(r)
            ate = f"{r['ate']:.3f}m" if np.isfinite(r["ate"]) else "失效"
            print(f"    ATE={ate:>8s}  特征={r['n_features']:6.0f}  "
                  f"匹配={r['n_matches']:6.0f}  内点率={r['inlier_ratio']*100:5.1f}%  "
                  f"地图点={r['n_map']:6.0f}  初始化={r['initialized']}")

    print("[·] 绘图...")
    fig_failure_atlas(results, f"{figs}/failure_atlas.png")
    fig_sample_images(root, f"{figs}/sample_images.png")
    fig_status_breakdown(results, f"{figs}/status_breakdown.png")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] 产出目录: {out_dir}")
    for p in sorted(os.listdir(figs)):
        print(f"    figs/{p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
