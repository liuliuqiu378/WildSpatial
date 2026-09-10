"""M0 几何库数值自检

为什么几何库一定要自检？
    几何代码的 bug 不会报错，只会让你跑出来的轨迹"看起来有点歪但说不上哪不对"，
    等你三天后发现是某个转置写反了，前面的实验全废。
    所以每个函数都配**解析解对照** + **数值差分对照**。

运行：
    python -m pytest tests/ -q
"""

import numpy as np
import pytest

from wildspatial.geometry import lie, camera, epipolar, triangulation, pnp

rng = np.random.default_rng(42)


# ------------------------------------------------------------------ 李群
def test_hat_vee_roundtrip():
    for _ in range(20):
        v = rng.normal(size=3)
        assert np.allclose(lie.vee3(lie.hat3(v)), v, atol=1e-12)


def test_hat3_is_cross_product():
    for _ in range(20):
        v, u = rng.normal(size=3), rng.normal(size=3)
        assert np.allclose(lie.hat3(v) @ u, np.cross(v, u), atol=1e-12)


def test_exp_log_SO3_roundtrip():
    for _ in range(50):
        phi = rng.normal(size=3) * 0.8
        R = lie.expSO3(phi)
        # 必须是正交阵且 det=1
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
        assert np.allclose(np.linalg.det(R), 1.0, atol=1e-10)
        assert np.allclose(lie.logSO3(R), phi, atol=1e-8)


def test_expSO3_matches_rodrigues_axis_angle():
    """绕 z 轴转 90° 的经典结果"""
    R = lie.expSO3(np.array([0, 0, np.pi / 2]))
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    assert np.allclose(R, expected, atol=1e-12)


def test_expSO3_small_angle_taylor():
    """θ→0 分支：exp 应趋近 I + φ^"""
    phi = np.array([1e-10, -2e-10, 3e-10])
    assert np.allclose(lie.expSO3(phi), np.eye(3) + lie.hat3(phi), atol=1e-18)


def test_exp_log_SE3_roundtrip():
    for _ in range(50):
        xi = rng.normal(size=6) * 0.5
        T = lie.expSE3(xi)
        assert np.allclose(lie.logSE3(T), xi, atol=1e-7)


def test_expSE3_translation_uses_left_jacobian():
    """关键：t = J_l(φ)·ρ，不是 ρ"""
    phi = np.array([0.3, -0.2, 0.5])
    rho = np.array([1.0, 2.0, 3.0])
    T = lie.expSE3(np.concatenate([rho, phi]))
    expected_t = lie.left_jacobian_SO3(phi) @ rho
    assert np.allclose(T[:3, 3], expected_t, atol=1e-12)


def test_left_jacobian_matches_numerical():
    """J_l 的数值验证：exp(φ+δ)^ ≈ exp(J_l δ)^ · exp(φ)^

    这是左雅可比的定义，用有限差分对照最可靠。
    """
    for _ in range(10):
        phi = rng.normal(size=3) * 0.6
        J = lie.left_jacobian_SO3(phi)
        eps = 1e-6
        J_num = np.zeros((3, 3))
        for i in range(3):
            d = np.zeros(3)
            d[i] = eps
            # 李代数上的增量 dJ·Δ 应等于李群上左乘的增量
            R_plus = lie.expSO3(phi + d)
            R_base = lie.expSO3(phi)
            # log( R_plus · R_base^{-1} ) ≈ J_l(φ) · d
            delta = lie.logSO3(R_plus @ R_base.T)
            J_num[:, i] = delta / eps
        assert np.allclose(J, J_num, atol=1e-5)


def test_left_jacobian_inv():
    for _ in range(10):
        phi = rng.normal(size=3) * 0.7
        J = lie.left_jacobian_SO3(phi)
        Ji = lie.left_jacobian_SO3_inv(phi)
        assert np.allclose(J @ Ji, np.eye(3), atol=1e-8)


# ------------------------------------------------------------------ 相机
def test_camera_project_unproject_roundtrip():
    cam = camera.PinholeCamera(fx=525.0, fy=525.0, cx=319.5, cy=239.5,
                               width=640, height=480)
    P_c = np.array([[0.1, -0.2, 2.5], [1.0, 0.5, 5.0]])
    uv = cam.project(P_c)
    # 用真实深度反投影应回到原点
    back = cam.unproject(uv, depth=P_c[:, 2])
    assert np.allclose(back, P_c, atol=1e-9)


def test_camera_project_world():
    """投影一个已知点，验证与手工计算一致"""
    K = camera.make_K(500, 500, 320, 240)
    cam = camera.PinholeCamera.from_K(K)
    T_cw = np.eye(4)
    T_cw[:3, 3] = [0, 0, 0]
    P_w = np.array([0.0, 0.0, 2.0])
    uv = cam.project_world(P_w, T_cw)
    assert np.allclose(uv, [320, 240], atol=1e-9)   # 光轴上的点 → 主点

    P_w = np.array([1.0, 0.0, 2.0])
    uv = cam.project_world(P_w, T_cw)
    assert np.allclose(uv, [320 + 500 * 1.0 / 2.0, 240], atol=1e-9)


def test_normalize_points():
    K = camera.make_K(500, 500, 320, 240)
    uv = np.array([[320, 240], [820, 240]])
    xn = camera.normalize_points(uv, K)
    assert np.allclose(xn[:, :2], [[0, 0], [1, 0]], atol=1e-12)


def test_projection_jacobian_numerical():
    """投影雅可比的数值对照（BA 会用到）"""
    fx, fy = 525.0, 525.0
    P_c = np.array([0.3, -0.4, 3.0])
    J_ana = camera.project_jacobian(P_c, fx, fy)
    eps = 1e-6
    J_num = np.zeros((2, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        f_plus = np.array([fx * (P_c[0] + d[0]) / (P_c[2] + d[2]),
                           fy * (P_c[1] + d[1]) / (P_c[2] + d[2])])
        f_minus = np.array([fx * (P_c[0] - d[0]) / (P_c[2] - d[2]),
                            fy * (P_c[1] - d[1]) / (P_c[2] - d[2])])
        J_num[:, i] = (f_plus - f_minus) / (2 * eps)
    assert np.allclose(J_ana, J_num, atol=1e-6)


# ------------------------------------------------------------------ 对极几何
def _synthetic_two_view(n_points=200, seed=0):
    """构造一个标准的两视图场景：随机 3D 点 + 已知位姿 → 投影到两幅图"""
    rr = np.random.default_rng(seed)
    K = camera.make_K(525.0, 525.0, 319.5, 239.5)
    # 相机1 在世界原点，相机2 沿 x 平移 0.5m 并绕 y 转 10°
    T_cw1 = np.eye(4)
    T_cw2 = np.eye(4)
    T_cw2[:3, :3] = lie.expSO3(np.array([0.0, np.deg2rad(10), 0.0]))
    T_cw2[:3, 3] = [0.5, 0.05, 0.1]

    # 3D 点：在相机1 前方 2~6m
    X_w = np.stack([
        rr.uniform(-2, 2, n_points),
        rr.uniform(-1.5, 1.5, n_points),
        rr.uniform(2.0, 6.0, n_points),
    ], axis=1)
    cam = camera.PinholeCamera.from_K(K)
    x1 = cam.project_world(X_w, T_cw1)
    x2 = cam.project_world(X_w, T_cw2)
    return K, T_cw1, T_cw2, X_w, x1, x2


def test_essential_from_ground_truth_pose():
    """验证 E = t^ R 满足对极约束"""
    K, T1, T2, X_w, x1, x2 = _synthetic_two_view()
    R = T2[:3, :3] @ T1[:3, :3].T
    t = T2[:3, 3] - R @ T1[:3, 3]
    E = lie.hat3(t) @ R

    xn1 = camera.normalize_points(x1, K)
    xn2 = camera.normalize_points(x2, K)
    resid = np.sum(xn2 * (E @ xn1.T).T, axis=1)
    assert np.abs(resid).max() < 1e-8
    # E 的奇异值应为 (σ, σ, 0)
    s = np.linalg.svd(E, compute_uv=False)
    assert s[2] < 1e-8
    assert abs(s[0] - s[1]) < 1e-8


def test_eight_point_recovers_essential():
    """无噪声下八点法应精确恢复 E（up to scale）"""
    K, T1, T2, X_w, x1, x2 = _synthetic_two_view(n_points=50)
    R_gt = T2[:3, :3] @ T1[:3, :3].T
    t_gt = T2[:3, 3] - R_gt @ T1[:3, 3]
    E_gt = lie.hat3(t_gt) @ R_gt

    xn1 = camera.normalize_points(x1, K)
    xn2 = camera.normalize_points(x2, K)
    E = epipolar.eight_point_essential(xn1, xn2)

    # 尺度不定，比较归一化后
    a = E / np.linalg.norm(E)
    b = E_gt / np.linalg.norm(E_gt)
    # 符号可能相反（E 与 -E 等价）
    err = min(np.abs(a - b).max(), np.abs(a + b).max())
    assert err < 1e-6, f"E 恢复误差 {err}"


def test_recover_pose_from_E():
    """从 E 恢复的 R,t 应与真值一致（旋转角误差 < 1e-6 rad，平移方向误差 < 1e-6）"""
    K, T1, T2, X_w, x1, x2 = _synthetic_two_view(n_points=100)
    R_gt = T2[:3, :3] @ T1[:3, :3].T
    t_gt = T2[:3, 3] - R_gt @ T1[:3, 3]
    t_gt = t_gt / np.linalg.norm(t_gt)

    E = lie.hat3(t_gt) @ R_gt
    R, t, mask, n_inliers = epipolar.recover_pose(E, x1, x2, K)

    assert n_inliers > 90, f"内点数太少：{n_inliers}"
    ang = np.linalg.norm(lie.logSO3(R @ R_gt.T))
    assert ang < 1e-6, f"旋转误差 {ang}"
    # 平移方向（可能整体反号）
    dir_err = min(np.linalg.norm(t - t_gt), np.linalg.norm(t + t_gt))
    assert dir_err < 1e-6, f"平移方向误差 {dir_err}"


def test_eight_point_with_noise_degrades_gracefully():
    """加噪声后仍能恢复大致正确的位姿（这是 RANSAC 要解决的问题的动机）"""
    K, T1, T2, X_w, x1, x2 = _synthetic_two_view(n_points=200, seed=3)
    rr = np.random.default_rng(1)
    x1n = x1 + rr.normal(0, 0.5, x1.shape)
    x2n = x2 + rr.normal(0, 0.5, x2.shape)
    E = epipolar.eight_point_essential(
        camera.normalize_points(x1n, K), camera.normalize_points(x2n, K))
    R, t, mask, n_inliers = epipolar.recover_pose(E, x1n, x2n, K)
    R_gt = T2[:3, :3] @ T1[:3, :3].T
    ang = np.degrees(np.linalg.norm(lie.logSO3(R @ R_gt.T)))
    assert ang < 5.0, f"0.5px 噪声下旋转误差应 <5°，实际 {ang:.2f}°"


def test_sampson_error_zero_for_perfect_data():
    K, T1, T2, X_w, x1, x2 = _synthetic_two_view()
    R = T2[:3, :3] @ T1[:3, :3].T
    t = T2[:3, 3] - R @ T1[:3, 3]
    E = lie.hat3(t) @ R
    F = epipolar.fundamental_from_essential(E, K, K)
    err = epipolar.sampson_error(F, x1, x2)
    assert err.max() < 1e-6


def test_pure_rotation_is_degenerate():
    """纯旋转（t=0）时 E 退化 —— 这是单目 VO 的经典失效模式

    数学上：E = t^ R，若 t = 0 则 E = 0，对极约束变成 0 = 0，无法解出位姿。
    工程后果：机器人原地旋转时，单目 VO 会丢失位姿。
    """
    K = camera.make_K(525, 525, 320, 240)
    R_only = lie.expSO3(np.array([0.0, 0.3, 0.0]))
    t_zero = np.zeros(3)
    E = lie.hat3(t_zero) @ R_only
    assert np.allclose(E, 0), "纯旋转时本质矩阵应为零矩阵，几何上不可解"


def test_small_baseline_degrades_pose_accuracy():
    """基线越短（视差越小），位姿估计越不准 —— 单目 VO 的核心工程约束。

    工程含义：机器人缓慢前行或远距离观测时，单目位姿估计会急剧恶化。
    这也是为什么工业上要给 VO 加 IMU：IMU 提供尺度与短基线下缺失的激励。
    """
    K, T1, T2, X_w, x1, x2 = _synthetic_two_view(n_points=300, seed=7)
    R_gt = T2[:3, :3] @ T1[:3, :3].T
    t_full = T2[:3, 3] - R_gt @ T1[:3, 3]
    cam = camera.PinholeCamera.from_K(K)

    angles = []
    for scale in [1.0, 0.2, 0.03]:
        T2s = np.eye(4)
        T2s[:3, :3] = R_gt
        T2s[:3, 3] = R_gt @ T1[:3, 3] + t_full * scale
        x2s = cam.project_world(X_w, T2s)

        rr = np.random.default_rng(0)
        x1n = x1 + rr.normal(0, 0.3, x1.shape)
        x2n = x2s + rr.normal(0, 0.3, x2s.shape)

        E = epipolar.eight_point_essential(
            camera.normalize_points(x1n, K), camera.normalize_points(x2n, K))
        R, t, mask, n = epipolar.recover_pose(E, x1n, x2n, K)
        angles.append(np.degrees(np.linalg.norm(lie.logSO3(R @ R_gt.T))))

    assert angles[0] < angles[-1], \
        f"基线缩短后位姿误差应增大，实际 {['%.2f' % a for a in angles]}°"
    assert angles[-1] > 5 * max(angles[0], 1e-6), \
        f"恶化幅度不明显: {['%.2f' % a for a in angles]}°"


# ------------------------------------------------------------------ 三角化
def test_triangulation_recovers_3d():
    K, T1, T2, X_w, x1, x2 = _synthetic_two_view(n_points=50)
    X_est, valid, _ = triangulation.triangulate_points(K, T1, T2, x1, x2)
    assert valid.all(), "所有点都应有效"
    # 三角化结果在第一相机系下，与真值（第一相机就在原点）比较
    assert np.allclose(X_est, X_w, atol=1e-6)


def test_triangulation_with_noise():
    K, T1, T2, X_w, x1, x2 = _synthetic_two_view(n_points=200, seed=11)
    rr = np.random.default_rng(5)
    x1n = x1 + rr.normal(0, 0.5, x1.shape)
    x2n = x2 + rr.normal(0, 0.5, x2.shape)
    X_est, valid, _ = triangulation.triangulate_points(K, T1, T2, x1n, x2n)
    err = np.linalg.norm(X_est[valid] - X_w[valid], axis=1)
    assert err.mean() < 0.15, f"0.5px 噪声下平均 3D 误差应 <0.15m，实际 {err.mean():.3f}"


def test_parallax_angle_matches_translation():
    """视差角应随基线增大而增大"""
    K = camera.make_K(525, 525, 320, 240)
    X = np.array([[0.0, 0.0, 5.0]])
    angles = []
    for b in [0.05, 0.5, 2.0]:
        T1 = np.eye(4)
        T2 = np.eye(4)
        T2[:3, 3] = [b, 0, 0]
        a = triangulation.parallax_angle(K, T1, T2, X)
        angles.append(a)
    assert angles[0] < angles[1] < angles[2]
    # 基线 b=0.5，距离 5m → 视差角约 atan(0.5/5) ≈ 5.7°
    assert 4.0 < angles[1] < 7.5, f"视差角 {angles[1]:.2f}° 不合理"


# ------------------------------------------------------------------ PnP
def test_pnp_dlt_recovers_pose():
    K, T1, T2, X_w, x1, x2 = _synthetic_two_view(n_points=30)
    # X_w 在"相机1 坐标系"（因 T1=I，世界系即相机1 系）；
    # 我们要解的是把点从相机1 系转到相机2 系：P_c2 = T_c2c1 · P_c1
    T_c2c1 = T2 @ np.linalg.inv(T1)
    T_est = pnp.pnp_dlt(X_w, x2, K)
    ang = np.degrees(np.linalg.norm(lie.logSO3(T_est[:3, :3] @ T_c2c1[:3, :3].T)))
    assert ang < 1e-4, f"PnP 旋转误差 {ang}"
    assert np.allclose(T_est[:3, 3], T_c2c1[:3, 3], atol=1e-6)


def test_pnp_refine_reduces_error():
    """加噪声后，GN 精化应显著降低重投影误差"""
    K, T1, T2, X_w, x1, x2 = _synthetic_two_view(n_points=60, seed=21)
    rr = np.random.default_rng(9)
    x2n = x2 + rr.normal(0, 1.0, x2.shape)
    T21 = np.linalg.inv(T2) @ T1

    T_dlt = pnp.pnp_dlt(X_w, x2n, K)
    _, rmse_dlt = triangulation.reprojection_error(K, T_dlt, X_w, x2n)

    T_ref = pnp.refine_pnp(X_w, x2n, K, T_dlt, iters=30)
    _, rmse_ref = triangulation.reprojection_error(K, T_ref, X_w, x2n)

    assert rmse_ref < rmse_dlt, "GN 精化后误差应下降"
    # 且应接近噪声水平
    assert rmse_ref < 1.5, f"精化后 RMSE={rmse_ref:.3f}px 仍过大"
