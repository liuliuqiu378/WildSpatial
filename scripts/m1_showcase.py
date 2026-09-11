"""M1 · 算法效果展示图（公开数据集 TUM RGB-D 实拍）

为什么要有这个脚本？
--------------------
指标（ATE / 重投影误差）能证明"好不好"，但看不出"**在干什么**"。
这个脚本在**公开真实数据**上生成一组肉眼可见的效果图，回答：
这套 SfM/VO 管线每一阶段到底对图像做了什么、效果如何、在什么数据上会崩。

产出（experiments/M1_showcase/）
    figs/fig1_features.png        真实图像上的特征点（有纹理 vs 无纹理）
    figs/fig2_matching.png        特征匹配：RANSAC 前（大量误匹配）vs 后（干净内点）
    figs/fig3_epipolar_real.png   真实图像对上的对极约束（左图点 → 右图一条线）
    figs/fig4_map3d.png           重建结果：稀疏 3D 地图点 + 估计轨迹 vs 真值
    figs/fig5_dataset_compare.png 跨数据集对比：干净序列 vs 弱纹理序列（失效对照）
    README.md / metrics.json

用法：
    PYTHONPATH=src python scripts/m1_showcase.py
    PYTHONPATH=src python scripts/m1_showcase.py --frames 300 --stride 3
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
from wildspatial.sfm.features import extract_features, draw_keypoints
from wildspatial.sfm.matching import match_ratio_test
from wildspatial.sfm.ransac import ransac_essential
from wildspatial.sfm import MonocularVO, VOConfig
from wildspatial.geometry import epipolar
from wildspatial.eval.trajectory import align_trajectory, compute_ate
from wildspatial.viz import setup_plot_style

setup_plot_style()

TUM_K = {
    "fr1": (517.3, 516.5, 318.6, 255.3),
    "fr2": (520.9, 521.0, 325.1, 249.7),
    "fr3": (535.4, 539.2, 320.1, 247.6),
}

SEQUENCES = {
    "fr1/desk": "rgbd_dataset_freiburg1_desk",
    "fr3/nostructure": "rgbd_dataset_freiburg3_nostructure_notexture_near_withloop",
}


def to_rgb(img):
    """cv2 读进来是 BGR，matplotlib 显示要 RGB"""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def K_of(seq):
    fx, fy, cx, cy = TUM_K.get(seq.split("/")[0], TUM_K["fr1"])
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def _line_endpoints(l, W, H):
    """把极线 ax+by+c=0 裁剪到图像边界内的两个端点"""
    a, b, c = l
    pts = []
    if abs(b) > 1e-9:
        for x in (0.0, float(W)):
            y = -(a * x + c) / b
            if -1 <= y <= H + 1:
                pts.append((x, y))
    if abs(a) > 1e-9:
        for y in (0.0, float(H)):
            x = -(b * y + c) / a
            if -1 <= x <= W + 1:
                pts.append((x, y))
    # 去重取前两个
    uniq = []
    for p in pts:
        if all(np.linalg.norm(np.array(p) - np.array(q)) > 1e-6 for q in uniq):
            uniq.append(p)
    return uniq[:2]


def apply_degradation(img, mode):
    """对真实图像施加**合成退化**，模拟恶劣成像条件

    为什么这么做：本项目真正关心的是"恶劣环境下系统怎么崩"。
    手头可用的公开序列 fr1/desk 是干净数据，所以对它施加可控退化来做对照实验 ——
    好处是**每一档都保留真值**，可以定量算 ATE 对比（fr3/nostructure 那种
    弱纹理序列既缺 rgb.txt 又缺真值，做不了定量对比）。
    """
    if mode is None or mode == "clean":
        return img
    rng = np.random.default_rng(0)
    if mode == "motion_blur":
        k = np.zeros((15, 15))
        k[7, :] = 1.0 / 15.0
        return cv2.filter2D(img, -1, k)
    if mode == "low_light":
        dark = np.clip(img.astype(np.float32) * 0.30, 0, 255)
        noise = rng.normal(0, 6, img.shape).astype(np.float32)
        return np.clip(dark + noise, 0, 255).astype(np.uint8)
    if mode == "noise":
        noise = rng.normal(0, 25, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------- 图1 特征点
def fig_features(root, path):
    """同一套 SIFT，在**干净**与**退化**真实图像上的差别

    直白讲：VO 的燃料就是特征点。图像一模糊 / 一变暗，
    SIFT 能稳定提出并描述的点就大幅减少 → 后面匹配、位姿、建图全部被削弱
    甚至直接崩掉（这正是 M3 失效归因 / M5 压力测试要系统研究的事）。
    """
    ds = TUMDataset(root)
    img0 = ds[len(ds) // 3]["rgb"]
    conditions = [("干净（原始图像）", "clean"),
                  ("运动模糊（快门/抖动）", "motion_blur"),
                  ("低光照（暗光+噪声）", "low_light")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
    info = {}
    for ax, (name, mode) in zip(axes, conditions):
        img = apply_degradation(img0, mode)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fs = extract_features(gray, "sift", max_features=2000)
        vis = draw_keypoints(img, fs)
        ax.imshow(to_rgb(vis))
        ax.set_title(f"{name}\nSIFT 特征点 = {len(fs)}", fontsize=12)
        ax.axis("off")
        info[mode] = int(len(fs))
    plt.suptitle("图1 · 特征检测：同一套 SIFT 在干净 / 退化真实图像上的产出差异"
                 "（TUM fr1/desk，退化为合成施加）", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    return info


# ---------------------------------------------------------------- 图2 匹配
def fig_matching(root, K, path, idx_a=0,
                 candidates=(8, 12, 16, 20, 25, 30, 40, 60)):
    """RANSAC 前 vs 后：这是"外点剔除"最直观的一张图

    直白讲：描述子匹配本身会产出大量**错误配对**（比值检验只能滤掉一部分）。
    RANSAC 用"两帧之间必须满足对极几何"这个约束，把错误配对全部踢掉，
    只留下几何上自洽的内点 —— 位姿估计只在这些内点上做，才不会被带偏。
    """
    ds = TUMDataset(root)
    ia = min(idx_a, len(ds) - 1)
    ga = cv2.cvtColor(ds[ia]["rgb"], cv2.COLOR_BGR2GRAY)
    fa = extract_features(ga, "sift", max_features=2000)

    # ⚠️ 自动挑帧对：不同序列运动快慢差别极大，间隔太大会"匹配不上"
    # （实测 fr1/desk 隔 60 帧只剩 4 组匹配，RANSAC 直接报"点数不足 8"）。
    # 这里从小到大试间隔，取内点最多且足够多的一对。
    pool = []
    for gap in candidates:
        ib = min(ia + gap, len(ds) - 1)
        if ib == ia:
            continue
        gb = cv2.cvtColor(ds[ib]["rgb"], cv2.COLOR_BGR2GRAY)
        fb = extract_features(gb, "sift", max_features=2000)
        m_try = match_ratio_test(fa.descriptors, fb.descriptors, ratio=0.8)
        if len(m_try) < 8:
            continue
        pa = fa.keypoints[m_try[:, 0]]
        pb = fb.keypoints[m_try[:, 1]]
        E_try, mask_try, stats_try = ransac_essential(
            pa, pb, K, thresh_px=1.5, max_iters=1000, seed=0, return_stats=True)
        if E_try is None:
            continue
        pool.append((fb, m_try, mask_try.astype(bool), E_try, stats_try, ib, gap))

    if not pool:
        print("        [!] 未找到可用帧对，跳过匹配/对极图")
        return {"n_matches": 0, "n_inliers": 0, "inlier_ratio": 0.0}, None

    # 在内点足够多（≥60，保证结果可靠）的前提下，优先选**外点最多**的一对 ——
    # 这样"RANSAC 前（满屏乱线）vs 后（干净）"的对比才最直观、最有说服力。
    good = [p for p in pool if int(p[2].sum()) >= 60]
    if good:
        best = max(good, key=lambda p: int(len(p[1]) - p[2].sum()))
    else:
        best = max(pool, key=lambda p: int(p[2].sum()))
    fb, m, inl, E, stats, ib, gap = best
    print(f"        帧对: {ia} ↔ {ib}（间隔 {gap} 帧）"
          f" 匹配 {len(m)} → 内点 {int(inl.sum())}")

    img_a = to_rgb(ds[ia]["rgb"])
    img_b = to_rgb(ds[ib]["rgb"])
    h = max(img_a.shape[0], img_b.shape[0])
    w1 = img_a.shape[1]

    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    for ax, (sel, title, color) in zip(axes, [
            (np.ones(len(m), bool), f"RANSAC 前：全部 {len(m)} 组匹配（含大量误匹配）", "yellow"),
            (inl, f"RANSAC 后：{int(inl.sum())} 组内点"
                  f"（内点率 {stats.get('inlier_ratio', 0) * 100:.1f}%）", "lime")]):
        canvas = np.zeros((h, w1 + img_b.shape[1], 3), dtype=np.uint8)
        canvas[:img_a.shape[0], :w1] = img_a
        canvas[:img_b.shape[0], w1:] = img_b
        ax.imshow(canvas)
        for a, b in m[sel]:
            x1, y1 = fa.keypoints[a]
            x2, y2 = fb.keypoints[b]
            ax.plot([x1, x2 + w1], [y1, y2], color=color, linewidth=0.7, alpha=0.85)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    plt.suptitle("图2 · 特征匹配与外点剔除（TUM fr1/desk 真实帧对）", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    return ({"n_matches": int(len(m)), "n_inliers": int(inl.sum()),
             "inlier_ratio": float(stats.get("inlier_ratio", 0.0))},
            {"fa": fa, "fb": fb, "m": m, "inl": inl, "ia": ia, "ib": ib,
             "E": E, "stats": stats})


# ---------------------------------------------------------------- 图3 对极线
def fig_epipolar(root, K, path, pair_info, n_show=10):
    """真实图像上的对极约束

    直白讲：左图一个点，在右图里**不可能乱跑** —— 它必须落在一条确定的直线（极线）上。
    这就是对极几何的全部内容，也是"已知左图点，右图只需沿一条线搜"的由来（2D 搜索降为 1D）。
    """
    fa, fb, m, inl = pair_info["fa"], pair_info["fb"], pair_info["m"], pair_info["inl"]
    ia, ib, E = pair_info["ia"], pair_info["ib"], pair_info["E"]
    ds = TUMDataset(root)
    img_a = to_rgb(ds[ia]["rgb"])
    img_b = to_rgb(ds[ib]["rgb"])
    pa = fa.keypoints[m[inl, 0]]
    pb = fb.keypoints[m[inl, 1]]

    if len(pa) < 8:
        return None
    # F = K^{-T}·E·K^{-1}（用标定内参，几何上更准）；
    # 若上游 E 缺失，则直接在已剔除外点的内点上用**八点法**拟合 F。
    if E is not None:
        F = epipolar.fundamental_from_essential(E, K, K)
    else:
        F = epipolar.eight_point_fundamental(pa, pb)

    rng = np.random.default_rng(0)
    sel = rng.choice(len(pa), size=min(n_show, len(pa)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    axes[0].imshow(img_a)
    axes[1].imshow(img_b)
    H, W = img_b.shape[:2]
    cmap = plt.get_cmap("tab10")
    for k, i in enumerate(sel):
        c = cmap(k % 10)
        u1, v1 = pa[i]
        axes[0].plot(u1, v1, "o", color=c, markersize=8, markeredgecolor="white")
        # 左图对应点在右图的极线
        l = epipolar.epipolar_line(F, pa[i])
        ends = _line_endpoints(l, W, H)
        if len(ends) == 2:
            (x0, y0), (x1, y1) = ends
            axes[1].plot([x0, x1], [y0, y1], "-", color=c, linewidth=1.6, alpha=0.95)
        u2, v2 = pb[i]
        axes[1].plot(u2, v2, "o", color=c, markersize=8, markeredgecolor="white",
                     markeredgewidth=1.0)
    axes[0].set_title("左图：随机取若干特征点", fontsize=12)
    axes[1].set_title("右图：对应点必须落在同色极线上（对极约束）", fontsize=12)
    for ax in axes:
        ax.axis("off")
    plt.suptitle("图3 · 真实图像上的对极约束（颜色一一对应）", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    return {"n_points": int(len(sel))}


# ---------------------------------------------------------------- 图4 三维重建
def run_vo(root, K, frames, stride, use_depth=False, degrade=None):
    """跑一遍 VO，返回 vo / 估计轨迹 / 真值位置

    degrade: 对每帧图像施加的合成退化模式（None / clean / motion_blur / low_light / noise）
    """
    ds = TUMDataset(root)
    n = len(ds) if frames <= 0 else min(frames, len(ds))
    cfg = VOConfig(feature="sift", max_features=3000, min_parallax_deg=1.0)
    vo = MonocularVO(K, cfg)
    gt, idxs = [], []
    for i in range(0, n, max(1, stride)):
        f = ds[i]
        if "T_wc" in f:
            gt.append(f["T_wc"][:3, 3])
        depth = f.get("depth") if use_depth else None
        vo.process(apply_degradation(f["rgb"], degrade), depth)
        idxs.append(i)
    est = vo.trajectory()
    gt = np.array(gt) if gt else np.zeros((0, 3))
    return vo, est, gt


def fig_map3d(vo, est, gt, path):
    """重建结果：稀疏 3D 地图点 + 估计轨迹 vs 真值轨迹

    直白讲：这就是 VO 最终交出来的东西 —— 一堆 3D 点云（地图）+ 一条相机轨迹。
    灰点是三角化出来的地图点，黑线是真值，红线是算法估计。
    """
    if len(gt) < 5 or len(est) < 5:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.text(0.5, 0.5, "数据不足，无法绘制", ha="center", va="center")
        plt.savefig(path, dpi=130); plt.close()
        return {}

    m = min(len(est), len(gt))
    est_e, gt_e = est[:m], gt[:m]
    aligned, params = align_trajectory(est_e, gt_e, allow_scale=True)
    ate = compute_ate(est_e, gt_e, allow_scale=True)

    # 用同一套 Sim3 把地图点也变换到真值坐标系，才能和轨迹画在一起
    s, R, t = params["scale"], params["R"], params["t"]
    X = vo.points_3d
    if len(X) > 20000:
        rng = np.random.default_rng(0)
        X = X[rng.choice(len(X), size=20000, replace=False)]
    X_al = (s * (R @ X.T).T) + t

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    views = [(0, 2, "X (m)", "Z (m)", "俯视 X-Z"),
             (0, 1, "X (m)", "Y (m)", "正视 X-Y")]
    for ax, (ix, iy, xl, yl, name) in zip(axes, views):
        ax.scatter(X_al[:, ix], X_al[:, iy], s=1.2, c="lightgray",
                   alpha=0.5, label="三角化地图点")
        ax.plot(gt_e[:, ix], gt_e[:, iy], "k-", linewidth=2.2, label="真值轨迹")
        ax.plot(aligned[:, ix], aligned[:, iy], "r-", linewidth=1.6,
                label=f"估计轨迹 (ATE={ate['rmse']:.3f}m)")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"{name}", fontsize=12)
        ax.legend(loc="upper right", fontsize=9)
        ax.axis("equal")
    plt.suptitle(f"图4 · 重建结果：{len(vo.points_3d)} 个地图点 + 轨迹"
                 f"（TUM fr1/desk 真实数据，Sim3 对齐后）", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    return {"ate_rmse": float(ate["rmse"]), "n_points": int(len(vo.points_3d))}


# ---------------------------------------------------------------- 图5 跨数据集
def fig_dataset_compare(stats, path):
    """干净 vs 退化条件：算法在哪儿会崩、崩到什么程度

    直白讲：同一套代码、同一组参数，只改**成像条件**，系统就从「能跑」变成「崩掉」。
    这就是 M3（失效归因）要说的事：**失效往往不是 bug，而是数据本身缺信息** ——
    模糊让特征点消失，暗光让描述子不可靠，最后位姿和地图一起崩。
    """
    names = list(stats.keys())
    keys = [("ate", "ATE RMSE (m) ↓越小越好"), ("n_features", "平均特征点数"),
            ("n_matches", "平均匹配数"), ("inlier_ratio", "平均内点率 (%)"),
            ("n_map", "最终地图点数")]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.3 * len(keys), 4.6))
    colors = plt.cm.Set2(np.linspace(0, 0.8, max(len(names), 1)))
    for ax, (k, label) in zip(np.atleast_1d(axes), keys):
        vals, bad = [], []
        for nm in names:
            v = float(stats[nm].get(k, 0.0))
            if k == "inlier_ratio":
                v *= 100.0
            bad.append(not np.isfinite(v))
            vals.append(0.0 if not np.isfinite(v) else v)
        bars = ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("/", "\n") for n in names], fontsize=9)
        ax.set_title(label, fontsize=11)
        for b, v, isbad in zip(bars, vals, bad):
            txt = "失效" if isbad else (f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}")
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), txt,
                    ha="center", va="bottom", fontsize=9)
    plt.suptitle("图5 · 条件对比：同一套算法在干净 / 退化真实数据上的表现差异"
                 "（TUM fr1/desk + 合成退化）", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def _write_readme(out_dir, metrics):
    """写一份「术语 + 直白讲解 + 结果图」三段式的展示说明"""
    lines = [
        "# M1 · 算法效果展示（公开数据集 TUM RGB-D 实拍）",
        "",
        "数据：**TUM RGB-D `fr1/desk`**（公开标准基准，含真值轨迹）。",
        "退化档位为**合成施加**（运动模糊 / 低光照），用于可控对照 —— 因为原始弱纹理序列",
        "`fr3/nostructure` 缺 `rgb.txt` 与真值，做不了定量对比。",
        "",
        "## 图1 · 特征检测（SIFT）",
        "",
        "**术语**：SIFT 特征点 —— 图像中**尺度不变、旋转不变**的稳定局部结构，",
        "是整套 VO 的燃料：没有它，后面的匹配、位姿、建图全都无从谈起。",
        "",
        "**直白讲解**：同一套算法、同一张真实照片，一旦模糊或变暗，能稳定提出来的点就大幅减少。",
        "这就是恶劣环境（水下浑浊、矿山粉尘、夜间低照度）下 SLAM 失效的**第一环**。",
        "",
        "![图1](figs/fig1_features.png)",
        "",
        "## 图2 · 特征匹配与外点剔除（RANSAC）",
        "",
        "**术语**：描述子匹配 + RANSAC —— 先用描述子找候选配对，",
        "再用「两帧必须满足对极几何」这一约束剔除不满足的**外点（outlier）**。",
        "",
        "**直白讲解**：光靠描述子匹配会产生大量**错误配对**（看下图上半部分的乱线）。",
        "RANSAC 相当于「用几何常识投票」：只留下彼此自洽的那部分（下半部分）。",
        "位姿估计只用这些内点，才不会被错误配对带偏。",
        "",
        "![图2](figs/fig2_matching.png)",
        "",
        "## 图3 · 对极约束（真实图像）",
        "",
        "**术语**：对极约束 —— 左图一个点 p₁，在右图中的对应点 p₂ 必须落在一条确定的直线",
        "（**极线** l = F·p₁）上。F 为基础矩阵。",
        "",
        "**直白讲解**：这把「在整张图上找点」压缩成「**沿一条线找点**」（2D 搜索 → 1D 搜索）。",
        "图上同色：左边的点，一定落在右边同色的那条线上。",
        "",
        "![图3](figs/fig3_epipolar_real.png)",
        "",
        "## 图4 · 重建结果（稀疏地图 + 轨迹）",
        "",
        "**术语**：三角化 —— 已知两帧位姿与同名点像素，反求该点的 3D 坐标。",
        "所有 3D 点构成**稀疏地图**，相机位置构成**轨迹**。",
        "",
        "**直白讲解**：这就是 VO 最终交出来的东西。灰色小点是重建出的 3D 结构，",
        "黑线是真值轨迹，红线是算法估计的轨迹（已做 Sim3 对齐，因为单目有尺度歧义）。",
        "",
        "![图4](figs/fig4_map3d.png)",
        "",
        "## 图5 · 条件对比（干净 vs 退化）",
        "",
        "**术语**：受控对照实验 —— 固定算法与参数，只改变**成像条件**，观察指标如何退化。",
        "",
        "**直白讲解**：同一套代码，只把图像弄模糊或弄暗，系统就从「能跑」变成「崩掉」。",
        "**失效往往不是 bug，而是数据本身缺信息**（M3 失效归因 / M5 压力测试的核心命题）。",
        "",
        "![图5](figs/fig5_dataset_compare.png)",
        "",
        "## 量化结果",
        "",
        "```json",
        json.dumps(metrics.get("condition_stats", {}), indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("\n".join(lines))


def mean_diag(vo, key):
    d = vo.diagnostics()
    arr = d.get(key, np.zeros(1))
    return float(np.mean(arr)) if len(arr) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "experiments", "M1_showcase")
    figs = os.path.join(out_dir, "figs")
    os.makedirs(figs, exist_ok=True)
    metrics = {}

    print("[·] 生成效果图（公开数据集 TUM RGB-D 实拍）...")
    root_desk = os.path.join(ROOT, "data", "raw", SEQUENCES["fr1/desk"])
    K = K_of("fr1/desk")

    # ---- 图1：特征点 ----
    print("    [1/5] 特征检测...")
    metrics["features"] = fig_features(root_desk, f"{figs}/fig1_features.png")

    # ---- 图2 / 图3：匹配与对极线（fr1/desk）----
    print("    [2/5] 特征匹配 + RANSAC...")
    mstats, pair_info = fig_matching(root_desk, K, f"{figs}/fig2_matching.png")
    metrics["matching"] = mstats
    print("    [3/5] 对极约束...")
    if pair_info is not None:
        estat = fig_epipolar(root_desk, K, f"{figs}/fig3_epipolar_real.png", pair_info)
        if estat:
            metrics["epipolar"] = estat

    # ---- 图4：三维重建（fr1/desk，有真值）----
    print("    [4/5] 跑 VO + 建图（fr1/desk）...")
    vo, est, gt = run_vo(root_desk, K, args.frames, args.stride)
    metrics["map3d"] = fig_map3d(vo, est, gt, f"{figs}/fig4_map3d.png")
    if len(gt) > 5:
        m = min(len(est), len(gt))
        metrics["ate_desk"] = {k: float(v) for k, v in
                               compute_ate(est[:m], gt[:m], allow_scale=True).items()}

    # ---- 图5：条件对比（干净 vs 合成退化）----
    print("    [5/5] 条件对比（干净 / 运动模糊 / 低光照）...")
    conditions = [("干净", "clean"), ("运动模糊", "motion_blur"),
                  ("低光照", "low_light")]
    stats = {}
    for name, mode in conditions:
        if mode == "clean":
            v, e, g = vo, est, gt          # 复用第 4 步已跑完的结果
        else:
            v, e, g = run_vo(root_desk, K, args.frames, args.stride, degrade=mode)
        st = {
            "n_features": mean_diag(v, "n_features"),
            "n_matches": mean_diag(v, "n_matches"),
            "inlier_ratio": mean_diag(v, "inlier_ratio"),
            "n_map": float(len(v.points_3d)),
            "initialized": bool(v.initialized),
        }
        if len(g) > 5 and len(e) > 5:
            mm = min(len(e), len(g))
            st["ate"] = float(compute_ate(e[:mm], g[:mm], allow_scale=True)["rmse"])
        else:
            st["ate"] = float("nan")
        stats[name] = st
        ate_s = f"{st['ate']:.3f}m" if np.isfinite(st["ate"]) else "失效(未初始化)"
        print(f"        {name:6s}: ATE={ate_s:>12s}  特征={st['n_features']:6.0f}"
              f"  地图点={st['n_map']:6.0f}")
    fig_dataset_compare(stats, f"{figs}/fig5_dataset_compare.png")
    metrics["condition_stats"] = stats

    _write_readme(out_dir, metrics)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] 产出目录: {out_dir}")
    for p in sorted(os.listdir(figs)):
        print(f"    figs/{p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
