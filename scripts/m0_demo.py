"""M0 演示：把抽象的几何变成看得见的图

运行：
    cd /home/hmn-cjy/liuliuqiu/WildSpatial
    PYTHONPATH=src python scripts/m0_demo.py

产出：experiments/M0_geometry_foundation/figs/*.png
每张图都对应 docs/M0_geometry_foundation.md 里的一个知识点。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from wildspatial.geometry import lie, camera, epipolar, triangulation, pnp
from wildspatial.viz import setup_plot_style

setup_plot_style()   # 中文字体 + 全局风格

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "experiments", "M0_geometry_foundation", "figs")
os.makedirs(OUT, exist_ok=True)

K = camera.make_K(525.0, 525.0, 319.5, 239.5)
CAM = camera.PinholeCamera.from_K(K, width=640, height=480)


def make_scene(n=300, seed=0, baseline=0.5, rot_deg=10.0):
    """标准两视图场景"""
    rr = np.random.default_rng(seed)
    T1 = np.eye(4)
    T2 = np.eye(4)
    T2[:3, :3] = lie.expSO3(np.array([0.0, np.deg2rad(rot_deg), 0.0]))
    T2[:3, 3] = [baseline, 0.05, 0.1]
    X = np.stack([rr.uniform(-2, 2, n),
                  rr.uniform(-1.5, 1.5, n),
                  rr.uniform(2.0, 6.0, n)], axis=1)
    x1 = CAM.project_world(X, T1)
    x2 = CAM.project_world(X, T2)
    return T1, T2, X, x1, x2


# ---------------------------------------------------------------- 图1
def fig1_projection():
    """针孔相机投影 + 两视图几何"""
    T1, T2, X, x1, x2 = make_scene(n=200)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (x, T, title) in zip(axes, [(x1, T1, "相机1 (世界原点)"), (x2, T2, "相机2 (右移0.5m, 转10°)")]):
        ax.scatter(x[:, 0], x[:, 1], s=8, c=X[:, 2], cmap="viridis")
        ax.set_xlim(0, 640); ax.set_ylim(480, 0)
        ax.set_title(f"{title}\n颜色=深度", fontsize=11)
        ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
        ax.grid(alpha=0.3)
    plt.colorbar(axes[0].collections[0], ax=axes[1], label="Z (m)")
    plt.suptitle("图1 · 针孔投影：同一批 3D 点在两个相机下的成像", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig1_projection.png", dpi=130)
    plt.close()
    print("✓ fig1_projection.png")


# ---------------------------------------------------------------- 图2
def fig2_epipolar_lines():
    """极线：p1 确定后，p2 必落在第二幅图的一条直线上"""
    T1, T2, X, x1, x2 = make_scene(n=200)
    R = T2[:3, :3] @ T1[:3, :3].T
    t = T2[:3, 3] - R @ T1[:3, 3]
    E = lie.hat3(t) @ R
    F = epipolar.fundamental_from_essential(E, K, K)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(x1[:, 0], x1[:, 1], s=6, c="steelblue")
    axes[1].scatter(x2[:, 0], x2[:, 1], s=6, c="lightgray")

    rng = np.random.default_rng(3)
    idx = rng.choice(len(x1), 6, replace=False)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for k, i in enumerate(idx):
        c = colors[k]
        axes[0].plot(*x1[i], "o", color=c, markersize=10, markeredgecolor="k")
        # 在第二幅图画对应极线
        l = epipolar.epipolar_line(F, x1[i])          # a u + b v + c = 0
        a, b, cc = l
        xs = np.array([0.0, 640.0])
        if abs(b) > 1e-9:
            ys = -(a * xs + cc) / b
        else:
            xs = np.array([-cc / a, -cc / a])
            ys = np.array([0.0, 480.0])
        axes[1].plot(xs, ys, "-", color=c, linewidth=1.6)
        axes[1].plot(*x2[i], "o", color=c, markersize=10, markeredgecolor="k")

    for ax, ttl in zip(axes, ["图像1：选定的点", "图像2：对应的极线（点必落在极线上）"]):
        ax.set_xlim(0, 640); ax.set_ylim(480, 0)
        ax.set_title(ttl, fontsize=11); ax.grid(alpha=0.3)
    plt.suptitle("图2 · 对极约束：2D 匹配搜索 → 1D 极线搜索", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig2_epipolar_lines.png", dpi=130)
    plt.close()
    print("✓ fig2_epipolar_lines.png")


# ---------------------------------------------------------------- 图3
def fig3_triangulation_vs_noise():
    """三角化精度随像素噪声的退化"""
    T1, T2, X, x1, x2 = make_scene(n=300, seed=5)
    sigmas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    med, p90 = [], []
    rr = np.random.default_rng(0)
    for s in sigmas:
        x1n = x1 + rr.normal(0, s, x1.shape)
        x2n = x2 + rr.normal(0, s, x2.shape)
        Xe, valid, _ = triangulation.triangulate_points(K, T1, T2, x1n, x2n)
        err = np.linalg.norm(Xe[valid] - X[valid], axis=1)
        med.append(np.median(err)); p90.append(np.percentile(err, 90))

    plt.figure(figsize=(8, 5))
    plt.plot(sigmas, med, "o-", label="中位误差")
    plt.plot(sigmas, p90, "s--", label="P90 误差")
    plt.xlabel("像素噪声 σ (px)"); plt.ylabel("3D 重建误差 (m)")
    plt.title("图3 · 三角化误差随像素噪声线性增长", fontsize=13)
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig3_triangulation_vs_noise.png", dpi=130)
    plt.close()
    print(f"✓ fig3_triangulation_vs_noise.png  (σ=0.5px → 中位误差 {med[2]:.3f} m)")


# ---------------------------------------------------------------- 图4
def fig4_parallax_vs_depth_error():
    """核心工程结论：视差角决定深度精度"""
    T1 = np.eye(4)
    X = np.array([[0.0, 0.0, 5.0]])
    baselines = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    angles, errs = [], []
    rr = np.random.default_rng(0)
    for b in baselines:
        T2 = np.eye(4)
        T2[:3, 3] = [b, 0, 0]
        x1 = CAM.project_world(X, T1)
        x2 = CAM.project_world(X, T2)
        angles.append(triangulation.parallax_angle(K, T1, T2, X))
        e = []
        for _ in range(200):
            x1n = x1 + rr.normal(0, 0.5, x1.shape)
            x2n = x2 + rr.normal(0, 0.5, x2.shape)
            Xe, valid, _ = triangulation.triangulate_points(K, T1, T2, x1n, x2n)
            if valid[0]:
                e.append(np.linalg.norm(Xe[0] - X[0]))
        errs.append(np.median(e) if e else np.nan)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(baselines, angles, "o-", color="darkorange")
    ax[0].set_xlabel("基线 (m)"); ax[0].set_ylabel("视差角 (°)")
    ax[0].set_title("基线 → 视差角", fontsize=11); ax[0].grid(alpha=0.3)

    ax[1].semilogy(angles, errs, "o-", color="crimson")
    ax[1].set_xlabel("视差角 (°)"); ax[1].set_ylabel("深度误差 (m，对数轴)")
    ax[1].set_title("视差角 → 深度误差", fontsize=11); ax[1].grid(alpha=0.3, which="both")
    for a, e in zip(angles, errs):
        ax[1].annotate(f"{e:.2f}", (a, e), textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.suptitle("图4 · 工程铁律：视差不足 → 深度必然崩（σ=0.5px, 距离 5m）", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig4_parallax_vs_depth_error.png", dpi=130)
    plt.close()
    print("✓ fig4_parallax_vs_depth_error.png")


# ---------------------------------------------------------------- 图5
def fig5_pure_rotation_degenerate():
    """纯旋转：E 退化为零矩阵，几何上不可解"""
    T1 = np.eye(4)
    T2 = np.eye(4)
    T2[:3, :3] = lie.expSO3(np.array([0.0, np.deg2rad(15), 0.0]))   # 纯旋转，无平移
    R = T2[:3, :3]
    t = np.zeros(3)
    E = lie.hat3(t) @ R

    rr = np.random.default_rng(2)
    X = np.stack([rr.uniform(-2, 2, 100), rr.uniform(-1.5, 1.5, 100), rr.uniform(2, 6, 100)], axis=1)
    x1 = CAM.project_world(X, T1)
    x2 = CAM.project_world(X, T2)

    plt.figure(figsize=(8, 5))
    plt.scatter(x1[:, 0], x1[:, 1], s=10, alpha=0.6, label="相机1 观测")
    plt.scatter(x2[:, 0], x2[:, 1], s=10, alpha=0.6, label="相机2 观测（原地转 15°）")
    plt.xlim(0, 640); plt.ylim(480, 0); plt.legend(); plt.grid(alpha=0.3)
    plt.xlabel("u (px)"); plt.ylabel("v (px)")
    plt.title(f"图5 · 纯旋转退化：E = [t]×R = 0（‖E‖={np.linalg.norm(E):.1e}）\n"
              "对极约束变成 0=0 → 单目 VO 原地转圈必丢位姿", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig5_pure_rotation_degenerate.png", dpi=130)
    plt.close()
    print("✓ fig5_pure_rotation_degenerate.png")


# ---------------------------------------------------------------- 图6
def fig6_pnp_refine():
    """PnP：DLT 初值 + 高斯-牛顿精化"""
    T1, T2, X, x1, x2 = make_scene(n=80, seed=21)
    rr = np.random.default_rng(9)
    x2n = x2 + rr.normal(0, 1.5, x2.shape)
    T_c2c1 = T2 @ np.linalg.inv(T1)

    T_dlt = pnp.pnp_dlt(X, x2n, K)
    _, rmse_dlt = triangulation.reprojection_error(K, T_dlt, X, x2n)

    T_gt = T_c2c1
    _, rmse_gt = triangulation.reprojection_error(K, T_gt, X, x2n)

    curve = [rmse_dlt]
    T = T_dlt
    for _ in range(20):
        T = pnp.refine_pnp(X, x2n, K, T, iters=1)
        _, r = triangulation.reprojection_error(K, T, X, x2n)
        curve.append(r)

    plt.figure(figsize=(8, 5))
    plt.plot(curve, "o-", color="seagreen", label="高斯-牛顿迭代")
    plt.axhline(rmse_gt, color="crimson", linestyle="--", label=f"真值位姿 RMSE={rmse_gt:.3f}px")
    plt.xlabel("迭代次数"); plt.ylabel("重投影 RMSE (px)")
    plt.title("图6 · PnP 精化：DLT 粗解 → 李代数高斯-牛顿 → 逼近噪声下限", fontsize=12)
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig6_pnp_refine.png", dpi=130)
    plt.close()
    print(f"✓ fig6_pnp_refine.png  (DLT {rmse_dlt:.3f}px → 精化 {curve[-1]:.3f}px，噪声下限 {rmse_gt:.3f}px)")


if __name__ == "__main__":
    print(f"输出目录: {OUT}\n")
    fig1_projection()
    fig2_epipolar_lines()
    fig3_triangulation_vs_noise()
    fig4_parallax_vs_depth_error()
    fig5_pure_rotation_degenerate()
    fig6_pnp_refine()
    print("\n全部完成。")
