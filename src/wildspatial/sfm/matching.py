"""描述子匹配

核心问题：最近邻不够，必须做**比值检验（ratio test）**
---------------------------------------------------------
只取最近邻会产生大量误匹配（尤其在重复纹理、白墙、弱纹理区域——正是我们 M3 要研究的场景）。

Lowe's ratio test：
    设最近邻距离 d1，次近邻距离 d2。若 d1/d2 < ratio（典型 0.7~0.8），才接受。
    直觉：**好的特征应该是"独一无二的"**——它跟最佳匹配明显比跟第二匹配近得多。
    如果 d1 ≈ d2（比如都在一片重复纹理里），说明这个点不具有区分性，直接丢弃。

⚠️ 这个检验在恶劣环境下会"过度拒绝"：
    弱纹理/模糊/低光照下 d1/d2 普遍偏大 → 大量正确匹配被误杀 → 匹配数骤降 → VO 崩。
    **这是 M3 失效归因要量化的关键指标之一**（内点率、匹配数）。
"""

import cv2
import numpy as np

__all__ = ["match_descriptors", "match_ratio_test", "filter_matches_by_motion"]


def _norm_type(desc):
    return cv2.NORM_HAMMING if desc.dtype == np.uint8 else cv2.NORM_L2


def match_ratio_test(desc1, desc2, ratio=0.8, cross_check=True, **kw):
    """比值检验 + 可选的互最近邻检查。

    返回 matches: (M,2) int32，每行是 [idx_in_1, idx_in_2]
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    if len(desc1) < 2 or len(desc2) < 2:
        return np.zeros((0, 2), dtype=np.int32)

    bf = cv2.BFMatcher(_norm_type(desc1), crossCheck=False)
    knn = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append((m.queryIdx, m.trainIdx, m.distance))

    if not good:
        return np.zeros((0, 2), dtype=np.int32)

    matches = np.array([[q, t] for q, t, _ in good], dtype=np.int32)

    if cross_check:
        # 反向再匹配一次，只保留双向一致的（更严格，误匹配更少）
        bf_rev = cv2.BFMatcher(_norm_type(desc2), crossCheck=False)
        knn_rev = bf_rev.knnMatch(desc2, desc1, k=2)
        rev = {}
        for pair in knn_rev:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                rev[m.queryIdx] = m.trainIdx     # desc2 索引 -> desc1 索引
        keep = np.array([rev.get(t, -1) == q for q, t in matches], dtype=bool)
        matches = matches[keep]

    return matches


# 默认入口
match_descriptors = match_ratio_test


def filter_matches_by_motion(kp1, kp2, matches, max_px=200.0):
    """用运动平滑性粗筛：相邻帧匹配点的位移不应过大。

    工程用途：帧间运动有上界，用它先砍掉明显离谱的匹配，能大幅降低 RANSAC 迭代次数。
    """
    if len(matches) == 0:
        return matches
    d = np.linalg.norm(kp1[matches[:, 0]] - kp2[matches[:, 1]], axis=1)
    return matches[d < max_px]
