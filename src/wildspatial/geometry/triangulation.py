"""三角化：从多视图的 2D 对应点恢复 3D 点

原理
------
已知相机投影矩阵 P1, P2（3x4）和对应像素 x1, x2，求空间点 X（4x1 齐次）满足：
    x1 × (P1 X) = 0        叉积为 0 表示共线（点在从光心出发的射线上）
    x2 × (P2 X) = 0
把叉积展开成矩阵，每个视图给出 2 个独立方程（第 3 行是前两行的线性组合，冗余）：
    [ u·p3^T - p1^T ]
    [ v·p3^T - p2^T ]  X = 0        (p_i^T 是 P 的第 i 行)
两个视图 → 4 个方程 → 齐次方程组 → SVD 取最小奇异值对应的向量。

三角化的精度准则（非常重要，工程上常用来筛掉烂点）
--------------------------------------------------
1. 正深度：点必须在两个相机前方
2. 视差角（parallax angle）θ：两条射线夹角。
   - θ 太小（< 1°）→ 深度不确定度爆炸，三角化结果不可信
   - 经验阈值：θ > 2~5° 才接受
3. 重投影误差：把 X 投影回去，看与观测差多少
"""

import numpy as np

from .camera import normalize_points

__all__ = [
    "triangulate_dlt", "triangulate_two_view",
    "triangulate_points", "parallax_angle", "reprojection_error",
]


def _build_rows(P, x):
    """为一个视图构造 2 行方程"""
    P = np.asarray(P, dtype=float).reshape(3, 4)
    u, v = float(x[0]), float(x[1])
    return np.stack([u * P[2] - P[0],
                     v * P[2] - P[1]])


def triangulate_dlt(Ps, xs):
    """多视图线性三角化（DLT）。

    参数
        Ps: 投影矩阵列表 [(3,4), ...]，长度 N≥2
        xs: 对应的 2D 点 [(2,), ...]，长度 N（像素坐标或归一化坐标，需与 P 一致）
    返回
        X: (3,) 非齐次 3D 点（在第一台相机的坐标系下）
    """
    rows = [ _build_rows(P, x) for P, x in zip(Ps, xs) ]
    A = np.vstack(rows)                       # (2N, 4)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-14:
        return np.full(3, np.nan)
    return X[:3] / X[3]


def triangulate_two_view(P1, P2, x1, x2):
    """两视图三角化的便捷封装"""
    return triangulate_dlt([P1, P2], [x1, x2])


def triangulate_points(K, T_cw1, T_cw2, pts1, pts2):
    """从相机内参 + 两个位姿 + 像素对应点，批量三角化。

    参数
        K      : (3,3) 内参
        T_cw1/2: (4,4) 世界→相机变换
        pts1/2 : (N,2) 像素坐标
    返回
        X_c1   : (N,3) 在**第一台相机坐标系**下的 3D 点
        valid  : (N,) 布尔掩码（正深度 + 有限值）
    """
    K = np.asarray(K, dtype=float).reshape(3, 3)
    T1 = np.asarray(T_cw1, dtype=float).reshape(4, 4)
    T2 = np.asarray(T_cw2, dtype=float).reshape(4, 4)
    pts1 = np.asarray(pts1, dtype=float).reshape(-1, 2)
    pts2 = np.asarray(pts2, dtype=float).reshape(-1, 2)

    # 以第一台相机为参考系构造投影矩阵：P = K [R | t]
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])          # 相机1 为世界原点
    R_rel = T2[:3, :3] @ T1[:3, :3].T
    t_rel = T2[:3, 3] - R_rel @ T1[:3, 3]
    P2 = K @ np.hstack([R_rel, t_rel[:, None]])

    out = []
    for a, b in zip(pts1, pts2):
        out.append(triangulate_dlt([P1, P2], [a, b]))
    X = np.asarray(out)
    # 检查在两相机下的深度
    X2 = (R_rel @ X.T).T + t_rel
    valid = np.isfinite(X).all(axis=1) & (X[:, 2] > 0) & (X2[:, 2] > 0)
    return X, valid, (R_rel, t_rel)


def parallax_angle(K, T_cw1, T_cw2, X_c1):
    """视差角：同一 3D 点在两个相机下的视线夹角（度）。

    用途：判断三角化是否可信。角度太小说明两条射线几乎平行，深度极不确定。
    """
    K = np.asarray(K, dtype=float).reshape(3, 3)
    T1 = np.asarray(T_cw1, dtype=float).reshape(4, 4)
    T2 = np.asarray(T_cw2, dtype=float).reshape(4, 4)
    X = np.asarray(X_c1, dtype=float).reshape(1, 3)

    R_rel = T2[:3, :3] @ T1[:3, :3].T
    t_rel = T2[:3, 3] - R_rel @ T1[:3, 3]
    X2 = (R_rel @ X.T).T + t_rel
    Kinv = np.linalg.inv(K)
    # 视线方向（第一相机系下：就是 X 本身；第二相机下需转回）
    d1 = X[0] / (np.linalg.norm(X[0]) + 1e-12)
    d2_in_c2 = X2[0] / (np.linalg.norm(X2[0]) + 1e-12)
    d2 = R_rel.T @ d2_in_c2
    cos = np.clip(np.dot(d1, d2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def reprojection_error(K, T_cw, X_c, pts):
    """重投影误差：把 3D 点投影回图像，与观测像素的欧氏距离（像素）。

    返回 (N,) 每个点的误差，以及 RMSE 标量
    """
    K = np.asarray(K, dtype=float).reshape(3, 3)
    T_cw = np.asarray(T_cw, dtype=float).reshape(4, 4)
    X = np.asarray(X_c, dtype=float).reshape(-1, 3)
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)

    Xc = (T_cw[:3, :3] @ X.T).T + T_cw[:3, 3]
    proj = (K @ Xc.T).T
    uv = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(uv - pts, axis=1)
    return err, float(np.sqrt(np.mean(err ** 2))) if len(err) else 0.0
