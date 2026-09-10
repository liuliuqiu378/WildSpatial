"""RANSAC：在外点海洋里找出正确的几何模型

为什么需要 RANSAC？
    比值检验后仍可能有 20~40% 的误匹配。最小二乘对离群点极其敏感——
    哪怕只有一个外点，解出的 E 也会完全跑偏。RANSAC 的思路是"投票"：
    随机采最小样本 → 估模型 → 数内点 → 重复 → 选内点最多的模型。

关键细节（工程上决定成败）
--------------------------
1. **迭代次数不是拍脑袋定的**：
       N = log(1 - p) / log(1 - w^s)
   其中 p 是期望成功率（如 0.999），w 是当前内点率，s 是最小样本数（八点法 s=8）。
   内点率降到 50% 时，8 点法需要 ~1177 次迭代！这就是为什么要**自适应更新 N**。

2. **内点判据用 Sampson 距离**，不要用代数残差 |x2^T E x1|
   （后者在远离主点处被系统性放大，会误杀远处的正确匹配）。

3. **最后要用所有内点重新估一次模型**（refine），否则精度浪费了。

4. ⚠️ **RANSAC 会失效的场景**（M3 的重点）：
   当内点率 w 很低（< 10~20%）时，需要的迭代次数爆炸，在给定的 max_iters 内基本找不到正确模型。
   弱纹理、重复结构、动态物体都会把 w 打下去。
"""

import numpy as np

from ..geometry.camera import normalize_points
from ..geometry.epipolar import (
    eight_point_essential, eight_point_fundamental, sampson_error, _as_euclidean,
)

__all__ = ["ransac_essential", "ransac_fundamental", "required_iterations"]


def required_iterations(inlier_ratio, sample_size, confidence=0.9999):
    """达到给定置信度所需的 RANSAC 迭代次数。

        N = log(1 - p) / log(1 - w^s)

    当 w→0 时 N→∞ —— 这解释了为什么内点率崩了以后 RANSAC 救不回来。
    """
    w = float(np.clip(inlier_ratio, 1e-6, 1 - 1e-6))
    denom = np.log(1.0 - w ** sample_size)
    if denom >= -1e-12:
        return 10 ** 9
    return int(np.ceil(np.log(1.0 - confidence) / denom))


def _fit_essential(xn1, xn2):
    """八点法估计 E（归一化坐标）"""
    try:
        return eight_point_essential(xn1, xn2)
    except np.linalg.LinAlgError:
        return None


def ransac_essential(pts1, pts2, K=None, thresh_px=1.5, max_iters=2000,
                     confidence=0.9999, seed=0, return_stats=False):
    """RANSAC 估计本质矩阵 E。

    参数
        pts1, pts2 : (N,2) 或 (N,3) 对应点（像素坐标）
        K          : 内参。给定时内部转到归一化坐标（推荐）
        thresh_px  : 内点阈值（像素）。会按焦距换算到归一化平面
        max_iters  : 最大迭代次数（自适应可能提前停）
    返回
        E           : (3,3) 本质矩阵（归一化坐标下）；失败返回 None
        inlier_mask : (N,) bool
        stats       : dict（可选）
    """
    p1 = _as_euclidean(pts1)
    p2 = _as_euclidean(pts2)
    n = len(p1)
    assert len(p2) == n

    if K is not None:
        K = np.asarray(K, dtype=float).reshape(3, 3)
        x1 = _as_euclidean(normalize_points(p1, K))
        x2 = _as_euclidean(normalize_points(p2, K))
        # 像素阈值 → 归一化平面阈值（用平均焦距换算）
        fx = (K[0, 0] + K[1, 1]) / 2.0
        thresh = thresh_px / fx
    else:
        x1, x2 = p1, p2
        thresh = thresh_px

    if n < 8:
        return (None, np.zeros(n, bool), {"reason": "点数不足 8"}) if return_stats \
            else (None, np.zeros(n, bool))

    rng = np.random.default_rng(seed)
    best_E, best_mask, best_score = None, None, -1
    iters = max_iters
    it = 0
    # 采样时用像素坐标判断退化（避免选到共线的点）
    while it < iters:
        it += 1
        idx = rng.choice(n, 8, replace=False)
        E = _fit_essential(x1[idx], x2[idx])
        if E is None or not np.all(np.isfinite(E)):
            continue

        err = sampson_error(E, x1, x2)          # E 在归一化坐标下等价于 F
        mask = err < thresh ** 2                # sampson_error 返回的是平方距离
        score = int(mask.sum())
        if score > best_score:
            best_score, best_E, best_mask = score, E, mask
            w = score / n
            iters = min(max_iters, required_iterations(w, 8, confidence))

    if best_E is None:
        out = (None, np.zeros(n, bool))
        return out + ({"iterations": it, "inlier_ratio": 0.0},) if return_stats else out

    # 用全部内点重新估计（refine），显著提升精度
    if best_mask.sum() >= 8:
        E_ref = _fit_essential(x1[best_mask], x2[best_mask])
        if E_ref is not None:
            err = sampson_error(E_ref, x1, x2)
            m2 = err < thresh ** 2
            if m2.sum() >= best_mask.sum():
                best_E, best_mask = E_ref, m2

    if return_stats:
        stats = {"iterations": it,
                 "inlier_ratio": float(best_mask.sum()) / n,
                 "n_inliers": int(best_mask.sum()),
                 "n_matches": n}
        return best_E, best_mask, stats
    return best_E, best_mask


def ransac_fundamental(pts1, pts2, thresh_px=1.5, max_iters=2000,
                       confidence=0.9999, seed=0, return_stats=False):
    """RANSAC 估计基础矩阵 F（像素坐标，无需内参）。

    用于**未标定**场景：只有图像、不知道 K。此时只能得到 F，无法直接恢复度量位姿。
    """
    p1 = _as_euclidean(pts1)
    p2 = _as_euclidean(pts2)
    n = len(p1)
    if n < 8:
        return (None, np.zeros(n, bool), {"reason": "点数不足 8"}) if return_stats \
            else (None, np.zeros(n, bool))

    rng = np.random.default_rng(seed)
    best_F, best_mask, best_score = None, None, -1
    iters = max_iters
    it = 0
    while it < iters:
        it += 1
        idx = rng.choice(n, 8, replace=False)
        try:
            F = eight_point_fundamental(p1[idx], p2[idx])
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(F)):
            continue
        err = sampson_error(F, p1, p2)
        mask = err < thresh_px ** 2
        score = int(mask.sum())
        if score > best_score:
            best_score, best_F, best_mask = score, F, mask
            iters = min(max_iters, required_iterations(score / n, 8, confidence))

    if best_F is None:
        out = (None, np.zeros(n, bool))
        return out + ({"iterations": it, "inlier_ratio": 0.0},) if return_stats else out

    if best_mask.sum() >= 8:
        try:
            F_ref = eight_point_fundamental(p1[best_mask], p2[best_mask])
            err = sampson_error(F_ref, p1, p2)
            m2 = err < thresh_px ** 2
            if m2.sum() >= best_mask.sum():
                best_F, best_mask = F_ref, m2
        except np.linalg.LinAlgError:
            pass

    if return_stats:
        stats = {"iterations": it,
                 "inlier_ratio": float(best_mask.sum()) / n,
                 "n_inliers": int(best_mask.sum()),
                 "n_matches": n}
        return best_F, best_mask, stats
    return best_F, best_mask
