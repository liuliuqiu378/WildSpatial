"""PnP：已知 3D 点和它们的 2D 投影，求相机位姿

为什么需要 PnP？
----------------
VO 的流程是：
  1) 前两帧用对极几何（E/F）初始化，三角化出一批 3D 点
  2) 之后每一帧，已经有一堆已知 3D 点（地图）了，只要求当前帧位姿
     → 这就是 PnP（Perspective-n-Point），比重新解 E 更快更稳

最小解：P3P（3 个点，最多 4 解，需要第 4 个点消歧）
线性解：DLT（≥6 点，快但不最小化几何误差，无噪声时精确）
最优解：EPnP / 迭代 LM 最小化重投影误差 —— OpenCV 的 solvePnP 默认走这条

本模块实现：
  * pnp_dlt          线性解（教学用，理解原理）
  * refine_pnp       以李代数扰动做高斯-牛顿迭代，最小化重投影误差（理解 BA 的雏形）
"""

import numpy as np

from .camera import make_K
from .lie import expSO3, logSO3

__all__ = ["pnp_dlt", "refine_pnp", "pnp_with_refine"]


def pnp_dlt(X_w, uv, K):
    """DLT 求解 PnP：直接把 R 和 t 的 12 个未知量当线性方程组解。

    由 s·[u,v,1]^T = K [R|t] X 展开，消去 s 得到每个点 2 个方程：
        (P1^T - u·P3^T) · M = 0
        (P2^T - v·P3^T) · M = 0      M = [R|t] 拉成 12 维
    需要 ≥6 个点。

    返回 T_cw (4,4)
    """
    X_w = np.asarray(X_w, dtype=float).reshape(-1, 3)
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    K = np.asarray(K, dtype=float).reshape(3, 3)
    assert len(X_w) == len(uv) >= 6, "DLT PnP 至少需要 6 个点"

    # ⚠️ 关键：DLT 方程建立在**归一化坐标**上（u = X/Z 形式），
    # 所以必须先用 K^{-1} 去掉内参，否则解出来的 R 会被内参污染。
    Kinv = np.linalg.inv(K)
    uvh = np.concatenate([uv, np.ones((len(uv), 1))], axis=1)
    uv_n = (Kinv @ uvh.T).T[:, :2]

    Xh = np.concatenate([X_w, np.ones((len(X_w), 1))], axis=1)   # (N,4)
    A = []
    for (u, v), xh in zip(uv_n, Xh):
        A.append(np.concatenate([xh, np.zeros(4), -u * xh]))
        A.append(np.concatenate([np.zeros(4), xh, -v * xh]))
    A = np.array(A)                                              # (2N, 12)
    _, _, Vt = np.linalg.svd(A)
    M = Vt[-1].reshape(3, 4)

    # ⚠️ 符号不定性：M 与 -M 给出完全相同的方程残差，SVD 无法区分。
    # 用"点必须在相机前方（z > 0）"来消歧 —— 这是 chirality 约束。
    depths = (M @ Xh.T)[2]
    if np.mean(depths) < 0:
        M = -M

    # 恢复尺度：M 的尺度不定，用 R 的行模长归一化
    scale = np.linalg.norm(M[:3, :3], axis=1).mean()
    if scale < 1e-12:
        raise RuntimeError("PnP DLT 退化")
    M = M / scale

    R = M[:3, :3]
    # 把 R 正交化（SVD 投影到 SO(3)）
    U, _, Vt2 = np.linalg.svd(R)
    if np.linalg.det(U @ Vt2) < 0:
        Vt2 = -Vt2
    R = U @ Vt2
    t = M[:3, 3]

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def refine_pnp(X_w, uv, K, T_init, iters=30, verbose=False):
    """高斯-牛顿迭代优化位姿，最小化重投影误差（李代数扰动）。

    这是 BA 的最小形态：
        残差 r = 观测像素 - 投影像素
        待优化变量 ξ ∈ se(3)（6 维，无约束）
        每次迭代：J = ∂r/∂ξ，解 (J^T J) Δξ = -J^T r，更新 T ← exp(Δξ) · T

    雅可比推导（链式法则，务必搞懂，BA 完全同构）：
        ∂r/∂ξ = ∂r/∂P_c · ∂P_c/∂ξ
        ∂r/∂P_c = [[-fx/Z, 0, fx·X/Z²], [0, -fy/Z, fy·Y/Z²]]   (2x3，注意符号)
        ∂P_c/∂ξ = [I, -P_c^]   (3x6)  对平移扰动和旋转扰动
    """
    X_w = np.asarray(X_w, dtype=float).reshape(-1, 3)
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    K = np.asarray(K, dtype=float).reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    T = np.asarray(T_init, dtype=float).reshape(4, 4).copy()

    prev_cost = None
    for it in range(iters):
        R, t = T[:3, :3], T[:3, 3]
        Pc = (R @ X_w.T).T + t
        Z = Pc[:, 2]
        valid = np.abs(Z) > 1e-9
        if valid.sum() < 3:
            break

        # 预测像素
        proj_u = fx * Pc[:, 0] / Z + cx
        proj_v = fy * Pc[:, 1] / Z + cy
        r = np.stack([uv[:, 0] - proj_u, uv[:, 1] - proj_v], axis=1)   # (N,2)
        r = r[valid].reshape(-1)
        cost = float(np.sum(r ** 2))
        if verbose:
            print(f"  iter {it:02d}  cost={cost:.6f}")
        if prev_cost is not None and abs(prev_cost - cost) < 1e-12:
            break
        prev_cost = cost

        # 组装雅可比
        J = np.zeros((valid.sum() * 2, 6))
        idx = 0
        for i in np.where(valid)[0]:
            X, Y, Zi = Pc[i]
            # ∂r/∂P_c  (2x3)
            dr_dPc = np.array([[-fx / Zi, 0.0, fx * X / (Zi * Zi)],
                               [0.0, -fy / Zi, fy * Y / (Zi * Zi)]])
            # ∂P_c/∂ξ  (3x6): 平移部分 I，旋转部分 -P_c^
            dPc_dxi = np.zeros((3, 6))
            dPc_dxi[:, :3] = np.eye(3)
            dPc_dxi[:, 3:] = -np.array([[0.0, -Zi, Y],
                                        [Zi, 0.0, -X],
                                        [-Y, X, 0.0]])
            J[idx:idx + 2] = dr_dPc @ dPc_dxi
            idx += 2

        # 高斯-牛顿（带阻尼，处理 J^T J 奇异）
        H = J.T @ J
        g = -J.T @ r
        try:
            delta = np.linalg.solve(H + 1e-6 * np.eye(6), g)
        except np.linalg.LinAlgError:
            break

        # 更新：T ← exp(δξ) · T ，δξ = [ρ, φ]，注意 expSE3 的顺序
        dT = np.eye(4)
        # 这里直接用 [t_perturb; phi] 的形式（t 与 ρ 在小量下近似相同）
        dT[:3, :3] = expSO3(delta[3:])
        dT[:3, 3] = delta[:3]
        T = dT @ T

    return T


def pnp_with_refine(X_w, uv, K, iters=30):
    """DLT 初值 + GN 精化"""
    T = pnp_dlt(X_w, uv, K)
    return refine_pnp(X_w, uv, K, T, iters=iters)
