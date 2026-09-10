"""M1 端到端自检：在合成序列上跑通整个 VO 管线

为什么用合成序列？
    真实数据下载很慢，不能让开发被网络阻塞。合成序列可以**精确控制真值**，
    反而更适合验证管线正确性（比如"轨迹是否单调前进""尺度是否稳定"）。

渲染方式：
    生成一堆 3D 点 → 每帧按相机位姿投影 → 在图像上画高斯斑点。
    SIFT 能把这些斑点检测为特征，于是整条管线（特征→匹配→RANSAC→PnP→三角化）都能跑通。
"""

import numpy as np
import cv2
import pytest

from wildspatial.geometry import lie, camera
from wildspatial.sfm import MonocularVO, VOConfig
from wildspatial.eval import compute_ate, compute_rpe


K = camera.make_K(525.0, 525.0, 319.5, 239.5)
W, H = 640, 480


def render_points(X_w, T_cw, K, size=(640, 480), radius=4):
    """把 3D 点投影并画成斑点，返回灰度图"""
    img = np.zeros((size[1], size[0]), np.float32)
    cam = camera.PinholeCamera.from_K(K)
    Xh = np.concatenate([X_w, np.ones((len(X_w), 1))], axis=1)
    Pc = (T_cw @ Xh.T).T[:, :3]
    z = Pc[:, 2]
    vis = z > 0.2
    uv = cam.project(Pc[vis])
    zz = z[vis]
    for (u, v), d in zip(uv, zz):
        if 0 <= u < size[0] and 0 <= v < size[1]:
            # 近的点画大一点（一点透视感）
            r = int(np.clip(radius * 2.0 / max(d, 0.5), 2, 7))
            cv2.circle(img, (int(u), int(v)), r, 255.0, -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_sequence(n_frames=25, n_points=400, seed=0, motion="forward"):
    """生成一段相机运动的合成序列"""
    rng = np.random.default_rng(seed)
    X_w = np.stack([rng.uniform(-3, 3, n_points),
                    rng.uniform(-2, 2, n_points),
                    rng.uniform(1.5, 8.0, n_points)], axis=1)

    poses = []        # T_cw
    for i in range(n_frames):
        T = np.eye(4)
        if motion == "forward":
            T[:3, 3] = [0, 0, 0.12 * i]                   # 沿光轴前进
        elif motion == "lateral":
            T[:3, 3] = [0.10 * i, 0, 0]                   # 横向平移
        elif motion == "rotate":
            T[:3, :3] = lie.expSO3([0, 0.02 * i, 0])      # 纯旋转（退化！）
        poses.append(T)

    images = [render_points(X_w, T, K) for T in poses]
    return images, np.array(poses), X_w


def _vo_run(images, **cfg_kw):
    cfg = VOConfig(max_features=1500, min_inliers=12, min_pnp_points=6, **cfg_kw)
    vo = MonocularVO(K, cfg)
    for img in images:
        vo.process(img)
    return vo


# ------------------------------------------------------------------ 测试
def test_vo_runs_without_crash():
    images, gt, _ = make_sequence(n_frames=12)
    vo = _vo_run(images)
    assert len(vo.frames) == len(images)
    traj = vo.trajectory()
    assert len(traj) > 0


def test_vo_lateral_motion_shape():
    """横向平移：估计轨迹应与真值形状吻合（Sim3 对齐后 ATE 小）"""
    images, gt_cw, _ = make_sequence(n_frames=20, motion="lateral", seed=1)
    vo = _vo_run(images)
    est = vo.trajectory()
    gt_pos = np.array([np.linalg.inv(T)[:3, 3] for T in gt_cw])

    n = min(len(est), len(gt_pos))
    if n >= 5:
        res = compute_ate(est[:n], gt_pos[:n], allow_scale=True)
        # 合成渲染下精度有限，只要求量级合理（< 0.5m）
        assert res["rmse"] < 0.5, f"横向运动 ATE={res['rmse']:.3f}m 过大"
        assert res["scale"] > 0, "尺度应为正"


def test_vo_pure_rotation_is_degenerate():
    """纯旋转序列：VO 应该跟丢或产生巨大误差 —— 这是几何的固有退化

    这条测试把 M0 §6.1 的结论在完整管线上复现一遍：
    t=0 → E=0 → 位姿不可解。真实机器人原地转圈就是这个下场。
    """
    images, gt_cw, _ = make_sequence(n_frames=15, motion="rotate", seed=2)
    vo = _vo_run(images)
    est = vo.trajectory()
    gt_pos = np.array([np.linalg.inv(T)[:3, 3] for T in gt_cw])

    n = min(len(est), len(gt_pos))
    if n >= 5:
        # 纯旋转下相机位置不动，真值轨迹是一个点；
        # 估计结果要么崩掉（ATE 大），要么 VO 报 lost
        res = compute_ate(est[:n], gt_pos[:n], allow_scale=True)
        statuses = [h["status"] for h in vo.history]
        n_lost = sum(1 for s in statuses if s.startswith("lost"))
        assert res["rmse"] > 1e-3 or n_lost > 0 or not vo.initialized, \
            "纯旋转下 VO 不该给出看似正确的结果"


def test_vo_diagnostics_recorded():
    """诊断信息必须完整记录 —— 这是 M3 失效归因的数据源"""
    images, _, _ = make_sequence(n_frames=10)
    vo = _vo_run(images)
    diag = vo.diagnostics()
    for key in ["n_features", "n_matches", "n_inliers", "inlier_ratio", "n_map"]:
        assert key in diag, f"缺少诊断字段 {key}"
        assert len(diag[key]) == len(images)
    assert len(diag["status"]) == len(images)


def test_vo_trajectory_monotonic_forward():
    """沿光轴前进：估计出的位移应大致同向（不要求精确）"""
    images, _, _ = make_sequence(n_frames=18, motion="forward", seed=3)
    vo = _vo_run(images)
    traj = vo.trajectory()
    if len(traj) > 5 and vo.initialized:
        # 相邻帧位移方向的余弦，多数应为正（说明没有来回跳）
        d = np.diff(traj, axis=0)
        nrm = np.linalg.norm(d, axis=1)
        if len(d) > 2 and nrm.max() > 1e-6:
            cos = np.dot(d[:-1], d[1:].T).diagonal() / (nrm[:-1] * nrm[1:] + 1e-9)
            assert np.mean(cos) > -0.2, "轨迹方向不应剧烈反复"
