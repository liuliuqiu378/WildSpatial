"""M1-BA 自检

BA 的雅可比极容易写错（符号、转置、hat 的正负号）。
而且**写错了不会报错**——只表现为"优化收敛慢"或"结果略差"，很难察觉。
所以必须用**有限差分逐元素对照**。
"""

import numpy as np
import pytest

from wildspatial.geometry import lie, camera
from wildspatial.sfm.ba import BundleAdjuster, ba_residual_jacobian, local_ba

K = camera.make_K(525.0, 525.0, 319.5, 239.5)


def make_ba_problem(n_poses=5, n_points=40, noise_pose=0.05, noise_point=0.05,
                    pixel_noise=0.0, seed=0):
    """构造一个标准 BA 问题：真值位姿/点 → 生成观测 → 加扰动作为初值"""
    rng = np.random.default_rng(seed)

    # 真值位姿：沿一条弧线运动
    poses_gt = []
    for i in range(n_poses):
        T = np.eye(4)
        T[:3, :3] = lie.expSO3(rng.normal(0, 0.15, 3))
        T[:3, 3] = [0.3 * i, 0.1 * np.sin(i), 0.05 * i]
        poses_gt.append(T)

    # 真值地图点：在相机前方
    pts_gt = np.stack([rng.uniform(-1.5, 1.5, n_points),
                       rng.uniform(-1.0, 1.0, n_points),
                       rng.uniform(2.0, 6.0, n_points)], axis=1)

    # 生成观测（每个点尽量被多个相机看到）
    obs = []
    for j in range(n_points):
        for i in range(n_poses):
            cam = camera.PinholeCamera.from_K(K)
            uv = cam.project_world(pts_gt[j], poses_gt[i])
            if 0 <= uv[0] < 640 and 0 <= uv[1] < 480:
                uv = uv + rng.normal(0, pixel_noise, 2)
                obs.append((i, j, uv))

    # 加扰动作为优化初值
    poses0 = []
    for T in poses_gt:
        Tn = np.eye(4)
        Tn[:3, :3] = lie.expSO3(rng.normal(0, noise_pose, 3)) @ T[:3, :3]
        Tn[:3, 3] = T[:3, 3] + rng.normal(0, noise_pose, 3)
        poses0.append(Tn)
    pts0 = pts_gt + rng.normal(0, noise_point, pts_gt.shape)

    return poses_gt, pts_gt, poses0, pts0, obs


# ------------------------------------------------------------------ 雅可比
def test_jacobian_matches_finite_difference():
    """🔴 最关键的一条：解析雅可比 vs 有限差分"""
    _, _, poses0, pts0, obs = make_ba_problem(n_poses=4, n_points=15,
                                              pixel_noise=0.0, seed=1)
    obs = obs[:40]
    r0, J_ana = ba_residual_jacobian(K, poses0, pts0, obs)

    n_p, n_x = len(poses0), len(pts0)
    x0 = np.concatenate([np.zeros(6 * n_p), pts0.ravel()])

    def fun(x):
        ba = BundleAdjuster(K, n_fixed_poses=0)
        return ba._residual(x, poses0, pts0, obs, None, n_p)

    eps = 1e-6
    J_num = np.zeros_like(J_ana)
    for c in range(J_num.shape[1]):
        xp = x0.copy(); xp[c] += eps
        xm = x0.copy(); xm[c] -= eps
        J_num[:, c] = (fun(xp) - fun(xm)) / (2 * eps)

    err = np.abs(J_ana - J_num)
    assert err.max() < 1e-4, f"雅可比与数值差分不符，最大误差 {err.max():.2e}"

    # 检查**稀疏结构**：每个残差只依赖 1 个位姿(6) + 1 个点(3)
    # 注意不是恰好 9 个非零 —— 存在结构性零：∂u/∂t_y ≡ 0（u 与 y 平移无关）
    n_pose_cols = 6 * n_p
    for k, (i, j, _) in enumerate(obs):
        for row in (2 * k, 2 * k + 1):
            nz = set(np.where(np.abs(J_ana[row]) > 1e-12)[0])
            allowed = set(range(6 * i, 6 * i + 6)) | \
                      set(range(n_pose_cols + 3 * j, n_pose_cols + 3 * j + 3))
            assert nz <= allowed, f"第 {k} 个观测的残差依赖了不该依赖的参数"
            assert len(nz) >= 7, f"非零元素过少（{len(nz)}），雅可比可能缺失"


def test_jacobian_sign_convention():
    """确认残差定义为 (观测 − 预测) 时的符号

    u 增大 → 残差减小，所以 ∂r_u/∂X（沿相机 x 轴移动点，使 u 增大）应为负。
    """
    T = np.eye(4)
    X = np.array([[0.0, 0.0, 3.0]])
    obs = [(0, 0, np.array([319.5, 239.5]))]      # 观测在主点
    r, J = ba_residual_jacobian(K, [T], X, obs)
    # 点在光轴上，投影正好在主点 → 残差为 0
    assert np.allclose(r, 0.0, atol=1e-6)
    # 把点沿 +x 移动 → u 增大 → 残差 u 分量应变负
    d = J[0, 6 + 0]        # 第 0 个残差(u)，对 X[0] 的偏导
    assert d < 0, f"符号约定错误：∂r_u/∂X = {d}，应为负"


# ------------------------------------------------------------------ 优化
def test_ba_reduces_reprojection_error():
    """BA 应显著降低重投影误差

    ⚠️ BA 因参数尺度差异（旋转/平移/点深度）收敛偏慢，需要足够多的函数评估
    （least_squares 的 max_nfev 是评估次数而非迭代次数）。这里给足迭代预算。
    """
    _, _, poses0, pts0, obs = make_ba_problem(n_poses=5, n_points=40,
                                              noise_pose=0.03, noise_point=0.05,
                                              pixel_noise=0.0, seed=2)
    _, _, info = local_ba(K, poses0, pts0, obs, n_fixed_poses=1, max_iter=120)
    assert info["status"] == "ok"
    assert info["cost_after"] < info["cost_before"] * 0.5, \
        f"BA 未有效降低代价: {info['cost_before']:.3f} → {info['cost_after']:.3f}"
    assert info["rmse_after"] < info["rmse_before"]
    # BA 应当把重投影误差压到一个很小的量级（干净的仿真设定下应趋近 0）
    assert info["rmse_after"] < info["rmse_before"] * 0.5, \
        f"BA 收敛不足: {info['rmse_before']:.3f} → {info['rmse_after']:.3f}"


def _centers(poses):
    return np.array([BundleAdjuster.camera_center(T) for T in poses])


def test_ba_recovers_poses_from_noisy_init():
    """从带噪声的初值出发，BA 应把位姿拉回真值附近

    ⚠️ 单目 BA 的解带有规范自由度（含尺度），不能直接拿去和真值比平移误差。
    正确做法：先把待比较的轨迹用 Sim3（Umeyama）对齐到真值，再比相对误差。
    """
    from wildspatial.eval import align_trajectory

    poses_gt, pts_gt, poses0, pts0, obs = make_ba_problem(
        n_poses=6, n_points=60, noise_pose=0.05, noise_point=0.08,
        pixel_noise=0.3, seed=3)
    poses_opt, pts_opt, info = local_ba(K, poses0, pts0, obs,
                                        n_fixed_poses=1, max_iter=120)

    def pose_err(A, B, start=1):
        """对齐到真值后的平均相对平移误差"""
        ca, _ = align_trajectory(_centers(A), _centers(B), allow_scale=True)
        cgt = _centers(B)
        errs = []
        m = min(len(ca), len(cgt))
        for i in range(start, m):
            errs.append(np.linalg.norm(ca[i] - cgt[i]))
        return np.mean(errs)

    before = pose_err(poses0, poses_gt)
    after = pose_err(poses_opt, poses_gt)
    assert after < before, f"BA 后位姿误差应减小: {before:.4f} → {after:.4f}"


def test_gauge_freedom_first_pose_fixed():
    """固定第一个位姿以消除 gauge freedom：它必须**完全不变**"""
    _, _, poses0, pts0, obs = make_ba_problem(n_poses=5, n_points=30,
                                              noise_pose=0.05, seed=4)
    poses_opt, _, info = local_ba(K, poses0, pts0, obs, n_fixed_poses=1, max_iter=20)
    assert np.allclose(poses_opt[0], poses0[0], atol=1e-12), \
        "第一个位姿应被固定，不应改变"


def test_ba_handles_degenerate_gracefully():
    """观测太少时不应崩溃"""
    _, _, poses0, pts0, obs = make_ba_problem(n_poses=3, n_points=5, seed=5)
    poses_opt, pts_opt, info = local_ba(K, poses0, pts0, obs, max_iter=10)
    assert poses_opt.shape == (3, 4, 4)
    assert np.all(np.isfinite(poses_opt))
