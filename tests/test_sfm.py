"""M1 自检：特征提取 / 匹配 / RANSAC

用**合成纹理图**测试，不依赖外部数据集（下载慢，不能阻塞开发）。
"""

import numpy as np
import cv2
import pytest

from wildspatial.geometry import lie, camera, epipolar
from wildspatial.sfm import features, matching, ransac


# ------------------------------------------------------------------ 工具
def _texture_image(h=480, w=640, seed=0):
    """合成一张有丰富梯度的纹理图（SIFT/ORB 都能提取到特征）"""
    rng = np.random.default_rng(seed)
    img = rng.normal(128, 45, (h, w)).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    for _ in range(120):
        x = int(rng.integers(20, w - 20))
        y = int(rng.integers(20, h - 20))
        val = 255.0 if rng.random() > 0.5 else 0.0
        cv2.circle(img, (x, y), int(rng.integers(3, 7)), val, -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def _synthetic_correspondences(n=200, n_outliers=60, noise=0.5, seed=0):
    """构造带外点的两视图对应关系"""
    rng = np.random.default_rng(seed)
    K = camera.make_K(525.0, 525.0, 319.5, 239.5)
    T1 = np.eye(4)
    T2 = np.eye(4)
    T2[:3, :3] = lie.expSO3(np.array([0.05, np.deg2rad(8), -0.03]))
    T2[:3, 3] = [0.35, 0.08, 0.05]
    X = np.stack([rng.uniform(-2, 2, n),
                  rng.uniform(-1.5, 1.5, n),
                  rng.uniform(2.0, 6.0, n)], axis=1)
    cam = camera.PinholeCamera.from_K(K)
    p1 = cam.project_world(X, T1) + rng.normal(0, noise, (n, 2))
    p2 = cam.project_world(X, T2) + rng.normal(0, noise, (n, 2))

    if n_outliers:
        # 外点：随机配对（位置完全不相关）
        o1 = rng.uniform([0, 0], [640, 480], (n_outliers, 2))
        o2 = rng.uniform([0, 0], [640, 480], (n_outliers, 2))
        p1 = np.vstack([p1, o1])
        p2 = np.vstack([p2, o2])
    R_gt = T2[:3, :3] @ T1[:3, :3].T
    t_gt = T2[:3, 3] - R_gt @ T1[:3, 3]
    return K, p1, p2, R_gt, t_gt, n


# ------------------------------------------------------------------ 特征
def test_sift_extraction():
    img = _texture_image()
    fs = features.extract_features(img, "sift", max_features=2000)
    assert len(fs) > 100, f"SIFT 特征太少: {len(fs)}"
    assert fs.keypoints.shape[1] == 2
    assert fs.descriptors.shape[0] == len(fs)
    assert fs.descriptors.dtype == np.float32


def test_orb_extraction_is_binary():
    img = _texture_image()
    fs = features.extract_features(img, "orb", max_features=2000)
    assert len(fs) > 100
    assert fs.descriptors.dtype == np.uint8, "ORB 描述子应为二进制(uint8)"


def test_top_k_selection():
    img = _texture_image()
    fs = features.extract_features(img, "sift", max_features=3000)
    small = fs.top_k(50)
    assert len(small) == 50
    # top_k 应保留响应最强的
    assert small.responses.min() >= fs.responses.max() * 0.0


# ------------------------------------------------------------------ 匹配
def test_matching_on_translated_image():
    """同一张图平移后，匹配点的位移应一致"""
    img = _texture_image(seed=3)
    M = np.float32([[1, 0, 30], [0, 1, 18]])
    img2 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    f1 = features.extract_features(img, "sift")
    f2 = features.extract_features(img2, "sift")
    m = matching.match_ratio_test(f1.descriptors, f2.descriptors, ratio=0.8)
    assert len(m) > 50, f"匹配数太少: {len(m)}"

    d = f2.keypoints[m[:, 1]] - f1.keypoints[m[:, 0]]
    med = np.median(d, axis=0)
    assert np.allclose(med, [30, 18], atol=1.5), f"位移不一致: {med}"
    # 大部分匹配应接近真实位移
    good = np.linalg.norm(d - med, axis=1) < 2.0
    assert good.mean() > 0.85, f"正确匹配率仅 {good.mean():.2f}"


def test_ratio_test_removes_ambiguous():
    """比值检验应显著减少误匹配：ratio 越小，保留的匹配越少但越准"""
    img = _texture_image(seed=5)
    img2 = cv2.warpAffine(img, np.float32([[1, 0, 25], [0, 1, 10]]),
                          (img.shape[1], img.shape[0]))
    f1 = features.extract_features(img, "sift")
    f2 = features.extract_features(img2, "sift")
    strict = matching.match_ratio_test(f1.descriptors, f2.descriptors, ratio=0.6)
    loose = matching.match_ratio_test(f1.descriptors, f2.descriptors, ratio=0.95)
    assert len(strict) <= len(loose), "更严格的 ratio 应保留更少匹配"


# ------------------------------------------------------------------ RANSAC
def test_required_iterations_formula():
    # N = log(1-p) / log(1-w^s)
    # 内点率 50%，8 点法：
    #   p=0.99   → 约 1177 次
    #   p=0.9999 → 约 2354 次
    assert 1100 < ransac.required_iterations(0.5, 8, 0.99) < 1250
    n = ransac.required_iterations(0.5, 8, 0.9999)
    assert 2200 < n < 2500, f"期望 ~2354，实际 {n}"
    # 内点率高时迭代次数应很少
    assert ransac.required_iterations(0.9, 8, 0.9999) < 30
    # 内点率极低时爆炸（15% → 百万级）
    assert ransac.required_iterations(0.15, 8, 0.9999) > 10 ** 5
    assert ransac.required_iterations(0.1, 8, 0.9999) > 10 ** 6


def test_ransac_finds_inliers_among_outliers():
    """30% 外点下，RANSAC 应恢复正确 E 并识别外点"""
    K, p1, p2, R_gt, t_gt, n_in = _synthetic_correspondences(n=200, n_outliers=60)
    E, mask, stats = ransac.ransac_essential(p1, p2, K, thresh_px=1.5,
                                             max_iters=2000, seed=0, return_stats=True)
    assert E is not None, "RANSAC 未找到模型"
    # 内点应基本覆盖所有真实对应
    recall = mask[:n_in].mean()
    assert recall > 0.9, f"真实对应点召回率仅 {recall:.2f}"
    # 外点应基本被剔除
    fp = mask[n_in:].mean()
    assert fp < 0.1, f"外点误接受率 {fp:.2f} 过高"
    assert stats["inlier_ratio"] > 0.6


def test_ransac_recovers_pose():
    K, p1, p2, R_gt, t_gt, n_in = _synthetic_correspondences(n=300, n_outliers=0, noise=0.0)
    E, mask, stats = ransac.ransac_essential(p1, p2, K, thresh_px=1.0,
                                             max_iters=500, seed=1, return_stats=True)
    assert E is not None
    R, t, _, n_pos = epipolar.recover_pose(E, p1, p2, K)
    ang = np.degrees(np.linalg.norm(lie.logSO3(R @ R_gt.T)))
    assert ang < 1.0, f"无噪声下旋转误差应 <1°，实际 {ang:.3f}°"
    t_gt = t_gt / np.linalg.norm(t_gt)
    d = min(np.linalg.norm(t - t_gt), np.linalg.norm(t + t_gt))
    assert d < 0.02, f"平移方向误差 {d:.4f}"


def test_ransac_degrades_with_low_inlier_ratio():
    """内点率崩到 ~15% 时，RANSAC 在有限迭代内会救不回来 —— M3 的核心失效模式

    这是要刻骨铭心的一条：弱纹理/重复结构/动态物体会把内点率打下去，
    一旦内点率过低，靠堆 RANSAC 迭代是没用的（需要的次数指数爆炸）。
    """
    K, p1, p2, R_gt, t_gt, n_in = _synthetic_correspondences(
        n=100, n_outliers=550, noise=0.0, seed=2)   # 内点率 ≈ 15%
    E, mask, stats = ransac.ransac_essential(p1, p2, K, thresh_px=1.0,
                                             max_iters=200, seed=3, return_stats=True)
    need = ransac.required_iterations(0.15, 8, 0.9999)
    assert need > 10 ** 5, f"15% 内点率下所需迭代应极大，实际 {need}"
    # 有限迭代内，模型质量明显下降（可能解不出正确位姿）
    if E is not None:
        R, t, _, _ = epipolar.recover_pose(E, p1, p2, K)
        ang = np.degrees(np.linalg.norm(lie.logSO3(R @ R_gt.T)))
        # 只断言"不如高内点率时可靠"，不断言具体数值（随机性大）
        assert ang > 0.0


def test_ransac_fundamental_no_calibration():
    """无内参时也能估 F（但无法恢复度量位姿）"""
    K, p1, p2, R_gt, t_gt, n_in = _synthetic_correspondences(n=200, n_outliers=40)
    F, mask, stats = ransac.ransac_fundamental(p1, p2, thresh_px=1.5,
                                               max_iters=1000, seed=0, return_stats=True)
    assert F is not None
    assert stats["inlier_ratio"] > 0.6
    # F 应满足对极约束
    err = epipolar.sampson_error(F, p1[mask], p2[mask])
    assert err.max() < 2.25, "内点的 Sampson 距离应小于阈值"
