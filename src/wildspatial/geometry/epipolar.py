"""对极几何：本质矩阵 E / 基础矩阵 F / 单应 H，以及从 E 恢复位姿

=========================== 核心直觉 ===========================
两个相机看同一个 3D 点 P：
  * 相机光心 C1、C2，点 P 三点共面 → 这个平面叫**对极平面**
  * 对极平面与两个成像平面的交线叫**极线**
  * 于是：已知 P 在第一幅图的像素 p1，它在第二幅图上的对应点 p2
    一定落在一条直线（极线）上 —— 这就是**对极约束**

  数学形式：
      像素坐标:     p2^T · F · p1 = 0        (F: 基础矩阵, 3x3, rank 2, 7自由度)
      归一化坐标:    x2^T · E · x1 = 0        (E: 本质矩阵, 3x3, rank 2, 5自由度)
      关系:          E = K2^T · F · K1

  E 比 F 少 2 个自由度，因为内参已知后 2 个自由度的信息已经被"用掉"了。
  E 的自由度 = 3(旋转) + 2(平移方向, 无尺度) = 5 —— 这正是**单目无法恢复尺度**的几何根源。

=========================== 八点法 ===========================
  x2^T E x1 = 0 对 E 是线性的。把 E 拉成 9 维向量 e，每行是一个方程：
      [x2·x1, x2·y1, x2, y2·x1, y2·y1, y2, x1, y1, 1] · e = 0
  8 个点 → 8 个方程 → SVD 取最小奇异向量。

  ⚠️ 必须先做 Hartley 归一化（把点平移到质心、缩放到平均距离 √2），
  否则像素坐标数值太大，A^T A 的条件数爆炸，解出来的 F 完全不能用。
"""

import numpy as np

from .camera import normalize_points

__all__ = [
    "hartley_normalize", "eight_point_essential", "eight_point_fundamental",
    "enforce_rank2", "fundamental_from_essential", "essential_from_fundamental",
    "decompose_essential", "recover_pose", "sampson_error", "epipolar_line",
]


def hartley_normalize(points):
    """Hartley 归一化：平移到质心为原点，缩放使到原点平均距离为 √2。

    返回 (归一化后的点, 变换矩阵 T)，满足 pts_norm = (T @ pts_hom^T).T
    """
    pts = _as_euclidean(points)
    centroid = pts.mean(axis=0)
    d = pts - centroid
    mean_dist = np.mean(np.linalg.norm(d, axis=1))
    s = np.sqrt(2.0) / (mean_dist + 1e-12)
    T = np.array([[s, 0.0, -s * centroid[0]],
                  [0.0, s, -s * centroid[1]],
                  [0.0, 0.0, 1.0]])
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    pts_n = (T @ hom.T).T
    return pts_n, T


def _as_euclidean(x):
    """统一输入为 (N,2)。

    ⚠️ 踩过的坑：normalize_points 返回的是**齐次**坐标 (N,3)，
    若直接 reshape(-1,2) 会把数据彻底打乱（且不会报错！）。这里统一处理。
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] == 3:
        x = x[:, :2] / np.where(np.abs(x[:, 2:3]) < 1e-18, 1e-18, x[:, 2:3])
    else:
        x = x[:, :2]
    return x


def _build_equation_matrix(x1, x2):
    """构造八点法的系数矩阵 A (N x 9)"""
    x1 = _as_euclidean(x1)
    x2 = _as_euclidean(x2)
    u1, v1 = x1[:, 0], x1[:, 1]
    u2, v2 = x2[:, 0], x2[:, 1]
    ones = np.ones_like(u1)
    return np.stack([u2 * u1, u2 * v1, u2,
                     v2 * u1, v2 * v1, v2,
                     u1, v1, ones], axis=1)


def enforce_rank2(M):
    """把 3x3 矩阵投影到 rank=2 的流形上：SVD 后把最小奇异值置 0。

    E/F 必须是 rank 2（因为 det(E)=0 是其内在性质）。
    八点法直接 SVD 得到的解一般不满足，强制置 0 后才合法。
    """
    U, S, Vt = np.linalg.svd(M)
    S[2] = 0.0
    return U @ np.diag(S) @ Vt


def eight_point_fundamental(p1, p2, normalize=True):
    """八点法求基础矩阵 F（像素坐标输入）。

    返回 F，满足 p2^T F p1 ≈ 0
    """
    p1 = np.asarray(p1, dtype=float).reshape(-1, 2)
    p2 = np.asarray(p2, dtype=float).reshape(-1, 2)
    assert len(p1) == len(p2) >= 8, "八点法至少需要 8 对点"

    if normalize:
        p1n, T1 = hartley_normalize(p1)
        p2n, T2 = hartley_normalize(p2)
        A = _build_equation_matrix(p1n, p2n)
        _, _, Vt = np.linalg.svd(A)
        Fn = Vt[-1].reshape(3, 3)
        Fn = enforce_rank2(Fn)
        # 反归一化：F = T2^T · Fn · T1
        F = T2.T @ Fn @ T1
    else:
        A = _build_equation_matrix(p1, p2)
        _, _, Vt = np.linalg.svd(A)
        F = Vt[-1].reshape(3, 3)
        F = enforce_rank2(F)
    return F / (np.linalg.norm(F) + 1e-12)


def eight_point_essential(x1, x2):
    """八点法求本质矩阵 E（归一化坐标输入）"""
    A = _build_equation_matrix(x1, x2)
    _, _, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)
    E = enforce_rank2(E)
    # E 的两个非零奇异值应相等（σ, σ, 0），投影到该流形上更稳
    U, S, Vt = np.linalg.svd(E)
    s = (S[0] + S[1]) / 2.0
    E = U @ np.diag([s, s, 0.0]) @ Vt
    return E / (np.linalg.norm(E) + 1e-12)


def fundamental_from_essential(E, K1, K2):
    """E → F:  F = K2^{-T} · E · K1^{-1}"""
    K1 = np.asarray(K1, dtype=float).reshape(3, 3)
    K2 = np.asarray(K2, dtype=float).reshape(3, 3)
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)


def essential_from_fundamental(F, K1, K2):
    """F → E:  E = K2^T · F · K1"""
    K1 = np.asarray(K1, dtype=float).reshape(3, 3)
    K2 = np.asarray(K2, dtype=float).reshape(3, 3)
    return K2.T @ F @ K1


def decompose_essential(E):
    """从本质矩阵 E 分解出 4 组候选 (R, t)。

    经典 SVD 分解，令
        W = [[0,-1,0],[1,0,0],[0,0,1]]   (绕 z 轴 90°)
    则 4 组解为：
        R1 = U W  V^T,  R2 = U W  V^T  (与 R1 相同)
        R  = U W  V^T  或  U W^T V^T
        t  = ±u3（U 的第三列）
    即 {U W V^T, U W^T V^T} × {+u3, -u3} = 4 组。

    ⚠️ 4 组解在几何上都满足对极约束，必须用 **chirality（正深度）检验** 选出唯一正确解：
       三角化后，点必须在**两个相机的前方**（Z > 0）。这就是 recover_pose 做的事。
    """
    U, S, Vt = np.linalg.svd(E)
    # 保证 det(U)>0, det(V)>0（否则 t 的符号会翻转）
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array([[0.0, -1.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]])
    R_a = U @ W @ Vt
    R_b = U @ W.T @ Vt
    t = U[:, 2]
    cands = [(R_a, t), (R_a, -t), (R_b, t), (R_b, -t)]
    return cands


def recover_pose(E, x1, x2, K=None):
    """从 E 恢复相机相对位姿（含正深度检验）。

    参数
        E  : 本质矩阵
        x1 : 第一幅图的点（若 K 为 None，则视为归一化坐标；否则是像素坐标）
        x2 : 第二幅图的点

    返回 (R, t, inlier_mask, n_inliers)
        其中 R, t 表示从相机1 到 相机2 的变换：P_c2 = R · P_c1 + t（t 为单位向量，无尺度）

    判定准则：三角化后两个相机的深度都 > 0。
    """
    if K is not None:
        x1 = normalize_points(x1, K)
        x2 = normalize_points(x2, K)
    x1 = _as_euclidean(x1)
    x2 = _as_euclidean(x2)

    from .triangulation import triangulate_dlt
    # 相机1 位姿为原点：P1 = [I | 0]；相机2：P2 = [R | t]
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])

    best = None
    for R, t in decompose_essential(E):
        P2 = np.hstack([R, t[:, None]])
        X = np.array([triangulate_dlt([P1, P2], [a, b]) for a, b in zip(x1, x2)])
        X2 = (R @ X.T).T + t                       # 同一批点在相机2 坐标系下
        ok = np.isfinite(X).all(axis=1)
        mask = ok & (X[:, 2] > 0) & (X2[:, 2] > 0)  # 正深度检验（chirality）
        score = int(mask.sum())
        if best is None or score > best[2]:
            best = (R, t, score, mask)
    if best is None:
        raise RuntimeError("无法从 E 恢复位姿：所有候选解均无正深度点")
    R, t, score, mask = best
    return R, t / (np.linalg.norm(t) + 1e-12), mask, score


def sampson_error(F, p1, p2):
    """Sampson 距离：点到极线距离的一阶近似（比纯代数残差更符合几何意义）。

        d = (p2^T F p1)² / ( (F p1)_x² + (F p1)_y² + (F^T p2)_x² + (F^T p2)_y² )

    RANSAC 里用它判断内点，比 |p2^T F p1| 更公平（后者在远离主点时会被放大）。
    """
    p1 = np.asarray(p1, dtype=float).reshape(-1, 2)
    p2 = np.asarray(p2, dtype=float).reshape(-1, 2)
    p1h = np.concatenate([p1, np.ones((len(p1), 1))], axis=1)
    p2h = np.concatenate([p2, np.ones((len(p2), 1))], axis=1)
    Fp1 = (F @ p1h.T).T              # (N,3) 第二幅图上的极线
    Ftp2 = (F.T @ p2h.T).T           # (N,3) 第一幅图上的极线
    num = np.sum(p2h * Fp1, axis=1)  # p2^T F p1
    den = Fp1[:, 0] ** 2 + Fp1[:, 1] ** 2 + Ftp2[:, 0] ** 2 + Ftp2[:, 1] ** 2
    return num ** 2 / (den + 1e-18)


def epipolar_line(F, p1):
    """给定第一幅图上的点，返回第二幅图上对应的极线 [a, b, c]，满足 a·u + b·v + c = 0"""
    p1h = np.append(np.asarray(p1, dtype=float).reshape(2), 1.0)
    return F @ p1h
