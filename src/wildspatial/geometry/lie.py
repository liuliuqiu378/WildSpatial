"""李群 / 李代数：SO(3) 与 SE(3)

为什么需要这一层？
--------------------
旋转矩阵 R 有 9 个数但只有 3 个自由度，且必须满足 R^T R = I, det(R)=1 的约束。
如果直接对 R 的 9 个元素做优化，结果会跑出约束空间之外，需要不断"拉回来"。

李代数的做法：用一个 3 维向量 φ（旋转向量，方向=旋转轴，长度=旋转角）表示旋转，
通过指数映射 exp: so(3) → SO(3)（即 Rodrigues 公式）得到 R。
这样优化变量从"带约束的 9 维"变成"无约束的 3 维"，这就是 SLAM 里所有位姿优化的基础。

关键公式（务必记住，后面推导雅可比全靠它）：
    exp(φ) = I + (sinθ/θ)·φ^ + ((1-cosθ)/θ²)·(φ^)²        θ = |φ|   [Rodrigues]

约定（全项目统一，写错一个转置 debug 三天）：
    * 点用**列向量**表示
    * 旋转向量 φ ∈ R³，φ^ 是其反对称矩阵（hat）
    * SO(3): R = exp(φ^)，李代数元素是 R³ 向量 φ（不是矩阵）
    * SE(3): T = [[R, t],[0,1]]，表示 **T_cw（从世界到相机）** 的变换：P_c = R·P_w + t
    * se(3) 元素 ξ = [ρ, φ] ∈ R⁶，其中 ρ 是平移部分（注意：ρ 不等于 t！t = J_l(φ)·ρ）
"""

import numpy as np

__all__ = [
    "hat3", "vee3", "hat6", "vee6",
    "expSO3", "logSO3", "expSE3", "logSE3",
    "left_jacobian_SO3", "left_jacobian_SO3_inv",
    "right_jacobian_SO3",
    "skew_symmetric_basis",
]


def _safe_theta(theta, eps=1e-12):
    return float(theta), eps


def hat3(v):
    """R³ → so(3)：把三维向量变成 3x3 反对称矩阵（叉积矩阵）。

    v^ 定义为满足 v^ · u = v × u 的矩阵。
    """
    v = np.asarray(v, dtype=float).reshape(3)
    x, y, z = v
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]])


def vee3(M):
    """so(3) → R³：hat3 的逆。"""
    M = np.asarray(M, dtype=float).reshape(3, 3)
    return np.array([M[2, 1] - M[1, 2],
                     M[0, 2] - M[2, 0],
                     M[1, 0] - M[0, 1]]) / 2.0


def skew_symmetric_basis():
    """返回 so(3) 的三个生成元 G_i，满足 hat(v) = Σ v_i · G_i。

    用途：数值/解析雅可比推导时非常好用。
    """
    G1 = np.array([[0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    G2 = np.array([[0, 0, 1.0], [0, 0, 0], [-1.0, 0, 0]])
    G3 = np.array([[0, -1.0, 0], [1.0, 0, 0], [0.0, 0, 0]])
    return np.stack([G1, G2, G3])


def expSO3(phi):
    """so(3) → SO(3)，Rodrigues 公式。

        R = I + (sinθ/θ)·φ^ + ((1-cosθ)/θ²)·(φ^)²,   θ = |φ|

    θ→0 时两个系数分别趋于 1 和 1/2，用泰勒展开避免除零。
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = np.linalg.norm(phi)
    K = hat3(phi)
    if theta < 1e-8:
        # 泰勒：sinθ/θ = 1 - θ²/6, (1-cosθ)/θ² = 1/2 - θ²/24
        return np.eye(3) + K + 0.5 * (K @ K)
    K2 = K @ K
    return np.eye(3) + (np.sin(theta) / theta) * K + ((1.0 - np.cos(theta)) / (theta ** 2)) * K2


def logSO3(R):
    """SO(3) → so(3)：expSO3 的逆，返回旋转向量 φ。

    两种算法：
      1) θ = arccos((tr(R)-1)/2)，轴 = (R-R^T)^∨ / (2 sinθ)
      2) 更稳的做法：直接由反对称部分取 vee，再用 atan2 定 θ
    θ 接近 0 时用泰勒：φ ≈ (R-R^T)^∨
    θ 接近 π 时数值不稳定，这里给出稳定分支。
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # 反对称部分给出 sinθ · 轴
    w = vee3(R)                       # = sinθ · a
    sin_theta = np.linalg.norm(w)

    if sin_theta < 1e-8:
        if cos_theta > 0:
            # θ ≈ 0：φ ≈ (R - R^T)^∨ / (1 + cosθ) 的极限
            return vee3(R - R.T) / (1.0 + cos_theta) if cos_theta > -0.999 else w
        else:
            # θ ≈ π：sinθ≈0 但 θ 很大，需特殊处理
            # 取 R 对角线上最大的分量构造轴
            diag = np.diag(R)
            i = int(np.argmax(diag))
            a = np.zeros(3)
            a[i] = 1.0
            # 用 R + I 的某一列（避开数值误差）求轴
            axis = (R + np.eye(3))[:, i]
            axis = axis / np.linalg.norm(axis)
            if np.dot(axis, a) < 0:
                axis = -axis
            return axis * np.arccos(cos_theta)

    theta = np.arctan2(sin_theta, cos_theta)
    return (theta / sin_theta) * w


def left_jacobian_SO3(phi):
    """SO(3) 左雅可比 J_l(φ)，满足 exp(φ+Δφ) ≈ exp(J_l·Δφ)^ · exp(φ)。

    公式（φ = θa，|a|=1）：
        J_l = (sinθ/θ)·I + (1 - sinθ/θ)·a a^T + ((1-cosθ)/θ)·a^

    直觉：它把"李代数上的微小增量"换算成"李群上的微小增量（左乘）"。
    在 SE(3) 的 exp 里，平移部分 t = J_l(φ)·ρ 也用它。
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = np.linalg.norm(phi)
    K = hat3(phi)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * K
    a = phi / theta
    aaT = np.outer(a, a)
    s = np.sin(theta) / theta
    c = (1.0 - np.cos(theta)) / theta
    return s * np.eye(3) + (1.0 - s) * aaT + c * hat3(a)


def left_jacobian_SO3_inv(phi):
    """J_l^{-1}(φ)。

        J_l^{-1} = (θ/2)cot(θ/2)·I + (1 - (θ/2)cot(θ/2))·a a^T - (θ/2)·a^
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = np.linalg.norm(phi)
    if theta < 1e-8:
        return np.eye(3) - 0.5 * hat3(phi)
    a = phi / theta
    aaT = np.outer(a, a)
    half = theta / 2.0
    cot = np.cos(half) / np.sin(half)
    return half * cot * np.eye(3) + (1.0 - half * cot) * aaT - half * hat3(a)


def right_jacobian_SO3(phi):
    """右雅可比 J_r(φ)，满足 exp(φ+Δφ) ≈ exp(φ)^ · exp(J_r·Δφ)^。
    性质：J_r(φ) = J_l(-φ)
    """
    return left_jacobian_SO3(-np.asarray(phi, dtype=float).reshape(3))


def hat6(xi):
    """R⁶ → se(3)：ξ = [ρ(3), φ(3)] → 4x4 矩阵 [[φ^, ρ],[0,0]]"""
    xi = np.asarray(xi, dtype=float).reshape(6)
    rho, phi = xi[:3], xi[3:]
    M = np.zeros((4, 4))
    M[:3, :3] = hat3(phi)
    M[:3, 3] = rho
    return M


def vee6(M):
    """se(3) → R⁶：hat6 的逆"""
    M = np.asarray(M, dtype=float).reshape(4, 4)
    return np.concatenate([M[:3, 3], vee3(M[:3, :3])])


def expSE3(xi):
    """se(3) → SE(3)。

        T = [[exp(φ^),  J_l(φ)·ρ],
             [0 0 0,      1     ]]

    ⚠️ 高频考点：平移部分是 J_l(φ)·ρ 而不是 ρ。
    """
    xi = np.asarray(xi, dtype=float).reshape(6)
    rho, phi = xi[:3], xi[3:]
    R = expSO3(phi)
    t = left_jacobian_SO3(phi) @ rho
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def logSE3(T):
    """SE(3) → se(3)：expSE3 的逆。ρ = J_l^{-1}(φ)·t"""
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    phi = logSO3(R)
    rho = left_jacobian_SO3_inv(phi) @ t
    return np.concatenate([rho, phi])
