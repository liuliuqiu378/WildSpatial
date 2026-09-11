"""M1-PGO 自检：位姿图优化 / 回环

位姿图优化的雅可比（Adjoint、符号）极易写错，且错了会"看起来在收敛其实方向反了"。
这里用三个可解析验证的用例锁住正确性：
  1. Adjoint 的定义性质  T·exp(ξ)·T^{-1} = exp(Ad_T·ξ)
  2. 无漂移 + 无回环 → 优化应当几乎不动
  3. 有漂移 + 加回环 → 优化应当显著降低轨迹误差（回环的价值所在）
"""

import numpy as np

from wildspatial.geometry.lie import expSE3, logSE3
from wildspatial.sfm.pgo import se3_adjoint, pose_graph_optimize


def _T(pos, rot=None):
    """构造 T_cw：旋转为 rot（默认单位阵），相机光心位于 pos"""
    T = np.eye(4)
    R = np.eye(3) if rot is None else rot
    T[:3, :3] = R
    T[:3, 3] = -R @ np.asarray(pos, float)
    return T


def _center(T):
    return -T[:3, :3].T @ T[:3, 3]


def test_se3_adjoint_definition():
    """Adjoint 的定义性质：T·exp(ξ)·T^{-1} = exp(Ad_T · ξ)"""
    rng = np.random.default_rng(0)
    for _ in range(5):
        xi = rng.normal(scale=0.3, size=6)
        T = expSE3(rng.normal(scale=0.4, size=6))
        lhs = logSE3(T @ expSE3(xi) @ np.linalg.inv(T))
        rhs = se3_adjoint(T) @ xi
        assert np.allclose(lhs, rhs, atol=1e-6)


def test_pgo_no_drift_no_loop_is_identity():
    """里程计自洽（无漂移）且无回环时，PGO 不应改动位姿"""
    gt_pos = np.array([[0, 0, 0], [1, 0, 0], [2, 0.2, 0], [3, 0.1, 0.1]], float)
    poses0 = [_T(p) for p in gt_pos]
    odo = [(i, i + 1, poses0[i + 1] @ np.linalg.inv(poses0[i]))
           for i in range(len(poses0) - 1)]
    poses_opt, info = pose_graph_optimize(poses0, odo, (), fix_first=True)
    for i in range(len(poses0)):
        assert np.allclose(_center(poses_opt[i]), gt_pos[i], atol=1e-6)
    assert info["rmse_after"] < 1e-6


def test_pgo_loop_closure_reduces_drift():
    """核心用例：链式漂移 + 一条回环约束 → 误差应显著下降

    这正是"为什么 SLAM 要回环"的最小可验证版本：
    回环边的误差不随轨迹长度累积，优化后能把漂移分摊掉。
    """
    gt_pos = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],      # 回到起点（真值）
    ], float)
    # 初值：沿链累积漂移，最后一帧漂到 (0.35, 0.32, 0)
    drift_pos = np.array([
        [0.00, 0.00, 0.0],
        [1.05, 0.02, 0.0],
        [1.10, 1.06, 0.0],
        [0.10, 1.10, 0.0],
        [0.35, 0.32, 0.0],
    ], float)
    poses0 = [_T(p) for p in drift_pos]
    odo = [(i, i + 1, poses0[i + 1] @ np.linalg.inv(poses0[i]))
           for i in range(len(poses0) - 1)]
    # 回环：检测到第 4 帧回到起点，给出绝对位姿（真值）
    priors = [(4, _T(gt_pos[4]))]

    poses_opt, info = pose_graph_optimize(poses0, odo, priors, fix_first=True)

    err_before = np.mean([np.linalg.norm(_center(poses0[i]) - gt_pos[i])
                          for i in range(len(gt_pos))])
    err_after = np.mean([np.linalg.norm(_center(poses_opt[i]) - gt_pos[i])
                         for i in range(len(gt_pos))])
    assert info["cost_after"] < info["cost_before"]
    assert err_after < err_before * 0.7, (
        f"回环后轨迹误差应显著下降: {err_before:.4f} → {err_after:.4f}")


def test_pgo_fixes_first_pose():
    """固定第 0 帧（消规范自由度）：优化后第 0 帧必须不变"""
    gt_pos = np.array([[0.5, 0.2, 0.1], [1.5, 0.2, 0.1], [1.5, 1.2, 0.1]], float)
    poses0 = [_T(p) for p in gt_pos]
    odo = [(i, i + 1, poses0[i + 1] @ np.linalg.inv(poses0[i]))
           for i in range(len(poses0) - 1)]
    priors = [(2, _T(np.array([1.6, 1.3, 0.1])))]
    poses_opt, _ = pose_graph_optimize(poses0, odo, priors, fix_first=True)
    assert np.allclose(poses_opt[0], poses0[0], atol=1e-9)
