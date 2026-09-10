"""轨迹评估：ATE / RPE / Sim3 对齐

两个必须理解的点
----------------
1. **单目 VO 必须先做 Sim3 对齐再算 ATE**
   因为单目有尺度歧义（M0 §1.3），估计出的轨迹与真值之间差一个**相似变换**。
   直接比位置会得到一个巨大且毫无意义的误差。
   对齐方式：Umeyama 算法求最优 (s, R, t)，使 `gt ≈ s·R·est + t`。
   - 单目/VIO 初期：allow_scale=True（Sim3）
   - RGB-D/双目/带 IMU：allow_scale=False（SE3），这样能额外暴露尺度漂移

2. **ATE vs RPE 的区别**
   - ATE（Absolute Trajectory Error）：全局一致性，反映**累积漂移**
   - RPE（Relative Pose Error）：固定间隔内的相对误差，反映**局部抖动/每帧精度**
   一条轨迹可以 ATE 很大但 RPE 很小（典型漂移），也可能反过来（局部抖但没累积）。

参考：Sturm et al., "A Benchmark for the Evaluation of RGB-D SLAM Systems", IROS 2012
"""

import numpy as np

from ..geometry.lie import logSO3

__all__ = ["align_umeyama", "align_trajectory", "compute_ate", "compute_rpe",
           "TrajectoryEval"]


def align_umeyama(src, dst, allow_scale=True):
    """Umeyama 算法：求 (s, R, t) 使 dst ≈ s·R·src + t

    参数
        src, dst : (N,3) 对应点集
        allow_scale : True → Sim3（7自由度）；False → SE3（6自由度）
    返回
        s (float), R (3,3), t (3,)
    """
    src = np.asarray(src, dtype=float).reshape(-1, 3)
    dst = np.asarray(dst, dtype=float).reshape(-1, 3)
    assert len(src) == len(dst) >= 3

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d

    sigma = (src_c ** 2).sum() / len(src)
    if sigma < 1e-12:
        return 1.0, np.eye(3), mu_d - mu_s

    S = (dst_c.T @ src_c) / len(src)          # (3,3)
    U, D, Vt = np.linalg.svd(S)

    # 防止反射（det 必须为 +1）
    Sgn = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sgn[2, 2] = -1.0
    R = U @ Sgn @ Vt

    if allow_scale:
        s = np.trace(np.diag(D) @ Sgn) / sigma
    else:
        s = 1.0

    t = mu_d - s * (R @ mu_s)
    return float(s), R, t


def align_trajectory(est, gt, allow_scale=True):
    """把 est 轨迹对齐到 gt，返回对齐后的 est 与变换参数"""
    s, R, t = align_umeyama(est, gt, allow_scale)
    aligned = (s * (R @ np.asarray(est).T).T) + t
    return aligned, {"scale": s, "R": R, "t": t}


def compute_ate(est, gt, allow_scale=True, return_aligned=False):
    """绝对轨迹误差（位置 RMSE，单位：米）

    返回 dict: rmse / mean / median / std / max / min
    """
    est = np.asarray(est, dtype=float).reshape(-1, 3)
    gt = np.asarray(gt, dtype=float).reshape(-1, 3)
    n = min(len(est), len(gt))
    est, gt = est[:n], gt[:n]

    aligned, tf = align_trajectory(est, gt, allow_scale)
    err = np.linalg.norm(aligned - gt, axis=1)

    res = {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "std": float(np.std(err)),
        "max": float(np.max(err)),
        "min": float(np.min(err)),
        "n": int(n),
        "scale": tf["scale"],
    }
    if return_aligned:
        return res, aligned
    return res


def _relative_pose(T1, T2):
    """T1 → T2 的相对变换：T_12 = inv(T1) · T2"""
    return np.linalg.inv(T1) @ T2


def compute_rpe(est_poses, gt_poses, delta=1, allow_scale=False):
    """相对位姿误差。

    参数
        est_poses, gt_poses : (N,4,4) 位姿序列（同一个坐标系约定）
        delta : 间隔帧数（TUM 官方常用 1）
    返回 dict: trans_rmse / rot_rmse_deg / n
    """
    est_poses = np.asarray(est_poses, dtype=float)
    gt_poses = np.asarray(gt_poses, dtype=float)
    n = min(len(est_poses), len(gt_poses))

    trans_err, rot_err = [], []
    for i in range(0, n - delta):
        E_est = _relative_pose(est_poses[i], est_poses[i + delta])
        E_gt = _relative_pose(gt_poses[i], gt_poses[i + delta])
        # 误差变换：E_err = inv(E_gt) · E_est
        E_err = np.linalg.inv(E_gt) @ E_est
        trans_err.append(np.linalg.norm(E_err[:3, 3]))
        rot_err.append(np.linalg.norm(logSO3(E_err[:3, :3])))

    if not trans_err:
        return {"trans_rmse": float("nan"), "rot_rmse_deg": float("nan"), "n": 0}

    trans_err = np.array(trans_err)
    rot_err = np.array(rot_err)
    return {
        "trans_rmse": float(np.sqrt(np.mean(trans_err ** 2))),
        "trans_mean": float(np.mean(trans_err)),
        "rot_rmse_deg": float(np.degrees(np.sqrt(np.mean(rot_err ** 2)))),
        "n": len(trans_err),
        "delta": delta,
    }


class TrajectoryEval:
    """便捷封装：一次算完 ATE + RPE"""

    def __init__(self, est_positions, gt_positions, est_poses=None, gt_poses=None,
                 allow_scale=True):
        self.est = np.asarray(est_positions, float).reshape(-1, 3)
        self.gt = np.asarray(gt_positions, float).reshape(-1, 3)
        self.ate, self.aligned = compute_ate(self.est, self.gt, allow_scale,
                                             return_aligned=True)
        self.rpe = None
        if est_poses is not None and gt_poses is not None:
            self.rpe = compute_rpe(est_poses, gt_poses, delta=1)

    def summary(self):
        lines = [
            f"ATE  RMSE = {self.ate['rmse']:.4f} m   "
            f"(median {self.ate['median']:.4f}, max {self.ate['max']:.4f})",
            f"     对齐尺度 s = {self.ate['scale']:.4f}  "
            f"{'（单目：尺度不定）' if abs(self.ate['scale'] - 1) > 0.05 else ''}",
        ]
        if self.rpe:
            lines.append(f"RPE  平移 RMSE = {self.rpe['trans_rmse']:.4f} m   "
                         f"旋转 RMSE = {self.rpe['rot_rmse_deg']:.3f}°")
        return "\n".join(lines)
