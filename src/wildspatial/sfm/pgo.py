"""位姿图优化（Pose Graph Optimization, PGO）

=========================== 它在解决什么问题 ===========================
VO 逐帧估计位姿，误差**沿轨迹累积**（漂移）。BA 能优化重投影误差，但在
**没有回环约束的开阔轨迹**上消不掉整体偏差（实测：BA 把重投影 1.54px→0.20px，
ATE 却 0.506→0.506 m 纹丝不动）。

PGO 的思路是把问题抽象成一张**图**：
    节点 = 相机位姿 T_i
    边   = 位姿之间的**相对约束**
         · 里程计边：相邻帧之间由 VO 给出的相对位姿
         · 回环边：检测到"回到旧地"时，两帧之间**直接测量**的相对/绝对约束
优化目标：调整所有节点位姿，让所有边都尽量满足。

        minimize  Σ_edges  ‖ log( M_ij^{-1} · T_j · T_i^{-1} ) ‖²

**为什么加一条回环边就能大幅降 ATE**：里程计边的误差是逐帧累积的（走 N 步累积 N 份误差）；
而回环边是"当前帧 ↔ 很久以前的帧"之间**一次直接测量**，只含一份误差、不累积。
优化时这条"准确的长程约束"会把累积误差**分摊回整条轨迹**，于是漂移被摊平。
这就是 SLAM（ORB-SLAM）精度远胜纯 VO 的根本原因。

=========================== 数学推导（务必看懂）===========================
残差（每条边 i→j，测量 M = T_j T_i^{-1}，约定 P_cj = M · P_ci）：

    e_ij = log( M^{-1} · T_j · T_i^{-1} )        （一致时为单位阵，log = 0）

对位姿做**左乘扰动** T ← exp(η)·T，推导雅可比：
    E(η) = M^{-1}·exp(η_j)·T_j·T_i^{-1}·exp(-η_i)
         = exp(Ad_{M^{-1}} η_j) · B · exp(-η_i)      B ≜ M^{-1} T_j T_i^{-1}
         = exp(Ad_{M^{-1}} η_j) · exp(-Ad_B η_i) · B
         ≈ exp( Ad_{M^{-1}} η_j − Ad_B η_i ) · B      （BCH 一阶）

而 log(exp(δ)·B) ≈ log(B) + J_l^{-1}(log B)·δ，于是

    ∂e/∂η_j = J_l^{-1}(e) · Ad_{M^{-1}}
    ∂e/∂η_i = −J_l^{-1}(e) · Ad_B

回环先验（绝对位姿观测 T_abs，残差 e = log(T_abs^{-1}·T_i)）同理：
    ∂e/∂η_i = J_l^{-1}(e) · Ad_{T_abs^{-1}}

⚠️ **教学简化（诚实声明）**：本实现取 **J_l^{-1}(e) ≈ I**。
生产库（g2o / GTSAM）用完整的 SE(3) 左雅可比逆（含 ρ-φ 耦合项 Q）。
当残差 e 较小时 J_l^{-1} ≈ I 是一阶有效的，配合 LM 迭代收敛良好；
代价是残差很大时收敛略慢。这里为了代码可读性与推导透明而简化。
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, csr_matrix

from ..geometry.lie import expSE3, logSE3, hat3

__all__ = ["se3_adjoint", "pose_graph_optimize"]


def se3_adjoint(T):
    """SE(3) 伴随矩阵 Ad_T (6x6)

        Ad_T = [[ R      t^·R ]
                [ 0      R    ]]

    作用：T·exp(ξ)·T^{-1} = exp(Ad_T · ξ)。
    雅可比推导里到处都是它（把"某个坐标系下的扰动"搬到另一个坐标系下）。
    """
    R = T[:3, :3]
    t = T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[3:, 3:] = R
    A[:3, 3:] = hat3(t) @ R
    return A


def _inv(T):
    return np.linalg.inv(T)


def pose_graph_optimize(poses0, odo_edges, loop_priors=(), fix_first=True,
                        loop_weight=1.0, max_iter=50, verbose=False):
    """位姿图优化（SE(3)）

    参数
        poses0      : (N,4,4) 初始位姿 T_cw
        odo_edges   : list of (i, j, M_ij)，M_ij = T_j·T_i^{-1}（相机 i → j 的相对位姿）
        loop_priors : list of (i, T_abs) 或 (i, T_abs, w)
                      —— 第 i 帧的**绝对**位姿观测（回环给出），w 为权重
        fix_first   : 固定第 0 帧（消除 6 维规范自由度）
        loop_weight : 回环先验的默认权重

    返回
        poses_opt (N,4,4), info dict
    """
    poses0 = [np.asarray(T, float).reshape(4, 4) for T in poses0]
    n = len(poses0)
    if n == 0:
        return np.array(poses0), {"status": "empty"}

    # 规范化回环先验为 (i, T_abs, w)
    priors = []
    for p in loop_priors:
        if len(p) == 2:
            priors.append((int(p[0]), np.asarray(p[1], float).reshape(4, 4),
                           float(loop_weight)))
        else:
            priors.append((int(p[0]), np.asarray(p[1], float).reshape(4, 4),
                           float(p[2])))
    odo = [(int(i), int(j), np.asarray(M, float).reshape(4, 4))
           for i, j, M in odo_edges]

    n_fixed = 1 if fix_first else 0
    n_free = n - n_fixed

    def block_of(i):
        return i - n_fixed          # 自由节点在 x 中的块索引

    def build_poses(x):
        poses = [T.copy() for T in poses0]
        for b in range(n_free):
            i = b + n_fixed
            poses[i] = expSE3(x[6 * b: 6 * b + 6]) @ poses0[i]
        return poses

    # 预计算常量
    odo_Minv = [_inv(M) for (_, _, M) in odo]
    pri_Tinv = [_inv(T) for (_, T, _) in priors]

    n_res = 6 * (len(odo) + len(priors))

    def residual(x):
        poses = build_poses(x)
        out = np.empty(n_res)
        k = 0
        for (i, j, _), Minv in zip(odo, odo_Minv):
            E = Minv @ poses[j] @ _inv(poses[i])
            out[6 * k: 6 * k + 6] = logSE3(E)
            k += 1
        for (i, _, w), Tinv in zip(priors, pri_Tinv):
            E = Tinv @ poses[i]
            out[6 * k: 6 * k + 6] = w * logSE3(E)
            k += 1
        return out

    def jacobian(x):
        poses = build_poses(x)
        J = lil_matrix((n_res, 6 * n_free))
        k = 0
        # 里程计边：∂e/∂η_j = Ad_{M^{-1}}，∂e/∂η_i = −Ad_B
        for (i, j, _), Minv in zip(odo, odo_Minv):
            B = Minv @ poses[j] @ _inv(poses[i])
            Ad_B = se3_adjoint(B)
            Ad_Minv = se3_adjoint(Minv)
            if j >= n_fixed:
                c0 = 6 * block_of(j)
                J[6 * k: 6 * k + 6, c0: c0 + 6] = Ad_Minv
            if i >= n_fixed:
                c0 = 6 * block_of(i)
                J[6 * k: 6 * k + 6, c0: c0 + 6] = -Ad_B
            k += 1
        # 回环先验：∂e/∂η_i = w · Ad_{T_abs^{-1}}
        for (i, _, w), Tinv in zip(priors, pri_Tinv):
            if i >= n_fixed:
                c0 = 6 * block_of(i)
                J[6 * k: 6 * k + 6, c0: c0 + 6] = w * se3_adjoint(Tinv)
            k += 1
        return csr_matrix(J)

    x0 = np.zeros(6 * n_free)
    r0 = residual(x0)
    cost0 = float(np.sum(r0 ** 2))

    sol = least_squares(residual, x0, jac=jacobian, method="trf",
                        max_nfev=max_iter * 30, verbose=2 if verbose else 0)

    x = sol.x
    poses_opt = build_poses(x)
    cost1 = float(np.sum(sol.fun ** 2))

    info = {
        "status": "ok", "success": bool(sol.success),
        "cost_before": cost0, "cost_after": cost1,
        "rmse_before": float(np.sqrt(cost0 / max(1, len(r0)))),
        "rmse_after": float(np.sqrt(cost1 / max(1, len(sol.fun)))),
        "n_poses": n, "n_odo_edges": len(odo), "n_loop_priors": len(priors),
        "n_iter": int(sol.nfev),
    }
    return np.array(poses_opt), info
