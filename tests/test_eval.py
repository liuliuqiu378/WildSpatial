"""轨迹评估自检：Sim3 对齐 / ATE / RPE

这些是最容易"算错但看起来合理"的指标 —— 必须自检。
"""

import numpy as np
import pytest

from wildspatial.geometry.lie import expSO3
from wildspatial.eval import align_umeyama, compute_ate, compute_rpe


def _rand_traj(n=200, seed=0):
    rng = np.random.default_rng(seed)
    # 平滑轨迹（随机游走 + 平滑），更像真实相机运动
    steps = rng.normal(0, 0.02, (n, 3))
    return np.cumsum(steps, axis=0) + rng.normal(0, 0.5, 3)


def test_umeyama_recovers_similarity():
    """给定已知的 (s, R, t)，Umeyama 应精确恢复"""
    src = _rand_traj(200, seed=1)
    s_gt = 2.7
    R_gt = expSO3(np.array([0.3, -0.2, 0.6]))
    t_gt = np.array([1.5, -2.0, 0.7])
    dst = (s_gt * (R_gt @ src.T).T) + t_gt

    s, R, t = align_umeyama(src, dst, allow_scale=True)
    assert abs(s - s_gt) < 1e-8, f"尺度错误: {s} vs {s_gt}"
    assert np.allclose(R, R_gt, atol=1e-8)
    assert np.allclose(t, t_gt, atol=1e-8)


def test_umeyama_se3_ignores_scale():
    """allow_scale=False 时不应拟合尺度"""
    src = _rand_traj(200, seed=2)
    R_gt = expSO3(np.array([0.1, 0.4, -0.3]))
    t_gt = np.array([-1.0, 2.0, 3.0])
    dst = (3.3 * (R_gt @ src.T).T) + t_gt   # 故意带尺度

    s, R, t = align_umeyama(src, dst, allow_scale=False)
    assert abs(s - 1.0) < 1e-12, "SE3 对齐不应引入尺度"
    # 此时 R 会被尺度干扰（这正是"SE3 能暴露尺度漂移"的原因）


def test_ate_zero_for_perfect_trajectory():
    gt = _rand_traj(100, seed=3)
    res = compute_ate(gt.copy(), gt.copy(), allow_scale=True)
    assert res["rmse"] < 1e-8
    assert abs(res["scale"] - 1.0) < 1e-6


def test_ate_ignores_global_similarity():
    """单目 VO 的典型情形：轨迹整体差一个相似变换，对齐后 ATE 应仍为 0"""
    gt = _rand_traj(150, seed=4)
    est = (2.5 * (expSO3([0.2, 0.3, -0.1]) @ gt.T).T) + np.array([3, -1, 2])
    res = compute_ate(est, gt, allow_scale=True)
    assert res["rmse"] < 1e-6, "Sim3 对齐后应完全重合"
    assert abs(res["scale"] - 1 / 2.5) < 1e-6

    # 但若用 SE3（不允许缩放），误差应明显变大
    res_se3 = compute_ate(est, gt, allow_scale=False)
    assert res_se3["rmse"] > 0.1, "SE3 无法吸收尺度差异，误差应显著"


def test_ate_detects_drift():
    """漂移越大，ATE 越大"""
    gt = _rand_traj(200, seed=5)
    errs = []
    for amp in [0.0, 0.01, 0.05]:
        rng = np.random.default_rng(7)
        est = gt + np.cumsum(rng.normal(0, amp, gt.shape), axis=0)
        errs.append(compute_ate(est, gt)["rmse"])
    assert errs[0] < errs[1] < errs[2], f"ATE 应随漂移增大: {errs}"


def test_rpe_on_known_motion():
    """匀速直线运动：每帧相对位姿相同，RPE 应≈0"""
    n = 60
    poses = []
    for i in range(n):
        T = np.eye(4)
        T[:3, 3] = [0.0, 0.0, 0.1 * i]      # 每帧前进 0.1m
        poses.append(T)
    poses = np.array(poses)
    # 估计与真值完全一致 → RPE = 0
    res = compute_rpe(poses, poses, delta=1)
    assert res["trans_rmse"] < 1e-9
    assert res["rot_rmse_deg"] < 1e-6

    # 若每帧多走 0.01m，RPE 平移误差应≈0.01
    est = poses.copy()
    for i in range(n):
        est[i][:3, 3] = [0.0, 0.0, 0.11 * i]
    res2 = compute_rpe(est, poses, delta=1)
    assert abs(res2["trans_rmse"] - 0.01) < 1e-6, f"RPE 应为 0.01，实际 {res2['trans_rmse']}"


def test_rpe_vs_ate_differentiate_drift_types():
    """ATE 与 RPE 的分工：纯抖动 → RPE 大但 ATE 小；纯漂移 → ATE 大"""
    n = 100
    gt = np.array([np.eye(4) for _ in range(n)])
    for i in range(n):
        gt[i][:3, 3] = [0.1 * i, 0, 0]
    gt_pos = gt[:, :3, 3]

    # A) 每帧独立抖动（不累积）
    rng = np.random.default_rng(0)
    jitter = gt.copy()
    for i in range(n):
        jitter[i][:3, 3] += rng.normal(0, 0.05, 3)
    ate_a = compute_ate(jitter[:, :3, 3], gt_pos)["rmse"]
    rpe_a = compute_rpe(jitter, gt, delta=1)["trans_rmse"]

    # B) 累积漂移（**非线性**，否则会被 Sim3 对齐吸收掉）
    # ⚠️ 这里有个重要教训：线性漂移 [0,0,k*i] 等价于对整个轨迹做一次线性变换，
    #    Sim3 对齐会把它完全吸收，ATE 反而≈0 —— 测不出问题。
    #    真实 SLAM 的漂移是随机游走/非线性累积，形状对不上，才会在 ATE 上暴露。
    drift = gt.copy()
    for i in range(n):
        drift[i][:3, 3] += [0.0, 0.0, 0.0005 * i * i]
    ate_b = compute_ate(drift[:, :3, 3], gt_pos)["rmse"]
    rpe_b = compute_rpe(drift, gt, delta=1)["trans_rmse"]

    assert rpe_a > rpe_b, "抖动的 RPE 应大于平滑漂移"
    assert ate_b > ate_a, f"非线性累积漂移的 ATE 应更大: {ate_b:.4f} vs {ate_a:.4f}"
