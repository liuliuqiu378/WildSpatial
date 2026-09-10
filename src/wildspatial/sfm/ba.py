"""Bundle Adjustment（光束法平差）

=========================== 它在解决什么问题 ===========================
VO 逐帧估计位姿，误差会**累积**（这就是"漂移"）。BA 的思路是：
    不要只往前看，而是**回过头联合优化**——同时调整所有相机位姿和所有地图点，
    使得所有观测的重投影误差之和最小。

    minimize  Σ_i Σ_j  ρ( ‖ z_ij − π(K, T_i, X_j) ‖² )
              T_i, X_j

其中 z_ij 是第 j 个地图点在第 i 个相机上的观测像素，π 是投影函数。
"Bundle" 指的就是**从每个光心出发射向空间点的一束光线（bundle of rays）**，
"Adjustment" 就是整体调整这些光束使它们尽可能交于一点。

=========================== 为什么 BA 是 SLAM 的核心 ===========================
1. 它是**最大后验估计**（在观测噪声为高斯时），统计上最优
2. 它能**消除累积漂移**——把误差"摊平"到整条轨迹上，而不是让它单调增长
3. 现代 SLAM（ORB-SLAM、VINS、COLMAP）的精度，主要就来自 BA + 回环

=========================== 关键难点：稀疏性 ===========================
假设 100 帧、每帧看 200 个点：
    参数量 = 100×6 + 10000×3 = 30,600
    Hessian H = J^T J 是 30600×30600 —— 稠密存储要 7.5 GB！

但 H 有**天然的稀疏块结构**：
    H = [ B   E  ]      B: 位姿-位姿块（对角块 + 少量非零）
        [ E^T C  ]      C: 点-点块（**严格块对角**，因为点之间不直接耦合）
                        E: 位姿-点交叉块（很稀疏，只有被观测才非零）

利用这个结构做 **Schur 补（舒尔补 / marginalization）**：
    先消掉地图点： (B − E·C⁻¹·E^T)·Δx_cam = v − E·C⁻¹·w
    得到一个只含位姿的 600×600 小系统（"reduced camera system"），
    解出位姿后再回代求点。这就是所有实用 BA 的做法。

本模块为了教学清晰，先用 scipy 的稀疏 LM 求解器（它内部处理稀疏线性代数），
但**雅可比是我们手写解析式**的，并且显式构造稀疏结构。

=========================== 心腹大患：Gauge Freedom（规范自由度）===========================
单目 BA 有 **7 维**规范自由度，不是一个能糊弄过去的细节：
    3 旋转 + 3 平移 + **1 尺度**。
对所有点 X→sX、位姿平移 t_i→s·t_i 同时缩放，投影结果完全不变：
    P_c = R(sX) + s·t = s(RX + t)   →   u = fx·(s·Pc_x)/(s·Pc_z) = 原值
所以 H 秩亏 7，解不唯一。

**本实现的处理（关键，也是上一次踩坑的地方）**：
- 固定第 0 帧（6 DOF）→ 消掉旋转+平移共 6 维。
- 但固定第 0 帧**并没有**消掉尺度！因为全局尺度同时缩放所有点深度和所有基线，
  重投影残差对尺度是**平**的（沿尺度方向梯度≈0），优化器会沿这个平坦方向自由漂移
  （实测漂到 1.19×，导致位姿误差 0.0573 → 0.0683 反而变大）。
- 因此必须再钉死**第 7 维**：用一个**尺度锚（scale anchor）**——
  把"第 0 帧光心到首个自由帧光心的基线长度"固定为初值时的基线长度。
  这是 1 个标量约束，正好消掉最后的 1 维尺度自由度。
- ⚠️ 注意：不能像某些实现那样"优化完再对所有平移乘一个尺度 s"——
  那样会把第 0 帧（本应固定）的平移也缩放了，导致 `pose[0]` 改变、规范自由度没消干净。
  正确做法是**在优化过程中**用锚约束，让优化器自己把尺度停在正确位置。
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, csr_matrix

from ..geometry.lie import expSE3, hat3

__all__ = ["project_point", "ba_residual_jacobian", "local_ba", "BundleAdjuster"]


def project_point(K, T_cw, X_w):
    """把世界点投影到像素，同时返回相机坐标（求雅可比需要）"""
    R = T_cw[:3, :3]
    t = T_cw[:3, 3]
    Pc = R @ X_w + t
    p = K @ Pc
    if abs(p[2]) < 1e-12:
        return np.array([np.nan, np.nan]), Pc
    return np.array([p[0] / p[2], p[1] / p[2]]), Pc


class BundleAdjuster:
    """局部 BA 求解器

    参数化：
        位姿：在**李代数上做增量**  T_i ← exp(ξ_i^) · T_i       （6 自由度，无约束）
        地图点：直接用世界坐标                                   （3 自由度）

    ⚠️ Gauge freedom 处理见模块顶部 docstring：固定第 0 帧（6 DOF）+ 尺度锚（1 DOF）。
    """

    def __init__(self, K, n_fixed_poses=1, robust="huber", huber_f=1.0,
                 fix_scale=True, scale_w=1.0):
        self.K = np.asarray(K, dtype=float).reshape(3, 3)
        self.n_fixed = n_fixed_poses
        self.robust = robust
        self.huber_f = huber_f
        self.fix_scale = fix_scale
        self.scale_w = scale_w
        self._anchor_target_val = None   # 在 optimize() 中根据初值计算

    @staticmethod
    def camera_center(T_cw):
        """相机在世界系下的位置：C = −R^T t"""
        return -T_cw[:3, :3].T @ T_cw[:3, 3]

    # ------------------------------------------------------------------ 锚
    def _anchor_enabled(self, n_p):
        """是否启用尺度锚：需要 fix_scale 且至少有 1 个固定帧 + 1 个自由帧"""
        return self.fix_scale and self.n_fixed >= 1 and n_p >= self.n_fixed + 1

    def _anchor_target(self, poses0):
        """尺度锚目标：第 0 帧光心到首个自由帧光心的基线长度（取自初值）"""
        C0 = self.camera_center(poses0[0])
        Ck = self.camera_center(poses0[self.n_fixed])
        return float(np.linalg.norm(Ck - C0))

    def _anchor_val(self, poses_list):
        """尺度锚残差：w·(当前基线 − 目标基线)"""
        if self._anchor_target_val is None:
            return 0.0
        C0 = self.camera_center(poses_list[0])
        Ck = self.camera_center(poses_list[self.n_fixed])
        diff = Ck - C0
        nrm = float(np.linalg.norm(diff))
        if nrm < 1e-9:
            return 0.0
        return self.scale_w * (nrm - self._anchor_target_val)

    # ------------------------------------------------------------------ 残差
    def _residual(self, x, poses0, points0, obs, opt_ids, n_opt):
        """残差向量：所有观测的重投影误差 (uv_obs - uv_pred) 拼起来，
        末尾（如启用锚）再追加尺度锚残差。

        x 的布局:  [ξ_0..ξ_{n_opt-1} (6 each)] [X_0..X_{m-1} (3 each)]
        """
        xi = x[: 6 * n_opt].reshape(n_opt, 6)
        X = x[6 * n_opt:].reshape(-1, 3)

        # 用增量更新位姿
        poses = []
        for i, T0 in enumerate(poses0):
            if i < self.n_fixed:
                poses.append(T0)
            else:
                poses.append(expSE3(xi[i - self.n_fixed]) @ T0)

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        n_row = 2 * len(obs) + (1 if self._anchor_enabled(len(poses)) else 0)
        r = np.empty(n_row)
        # 向量化：先把所有相关点变换到相机系
        for k, (i, j, uv) in enumerate(obs):
            T = poses[i]
            Pc = T[:3, :3] @ X[j] + T[:3, 3]
            Z = Pc[2]
            if abs(Z) < 1e-9:
                r[2 * k: 2 * k + 2] = 0.0
                continue
            u = fx * Pc[0] / Z + cx
            v = fy * Pc[1] / Z + cy
            r[2 * k] = uv[0] - u
            r[2 * k + 1] = uv[1] - v

        if self._anchor_enabled(len(poses)):
            r[-1] = self._anchor_val(poses)
        return r

    # ------------------------------------------------------------------ 雅可比
    def _jacobian(self, x, poses0, points0, obs, opt_ids, n_opt):
        """解析雅可比（稀疏）

        对每个观测 (i, j)：
            ∂r/∂P_c  = [[-fx/Z,  0,    fx·X/Z²],
                        [  0,  -fy/Z, fy·Y/Z²]]                      (2x3)

            ∂P_c/∂δξ_i = [ I | −P_c^ ]                                (3x6)
                —— 左乘扰动下，平移部分给单位阵，旋转部分给 −反对称矩阵
            ∂P_c/∂X_j  = R_i                                          (3x3)

        于是
            ∂r/∂δξ_i = ∂r/∂P_c · [ I | −P_c^ ]                        (2x6)
            ∂r/∂X_j  = ∂r/∂P_c · R_i                                  (2x3)

        ⚠️ 尺度锚的雅可比：锚残差只依赖"首个自由帧"的 6 个增量参数。
        它的解析梯度沿尺度方向接近常数、曲率≈0，为绝对稳妥，这里对该单行用
        **有限差分**计算（仅 6 列、12 次廉价评估），避免手推符号出错。
        """
        xi = x[: 6 * n_opt].reshape(n_opt, 6)
        X = x[6 * n_opt:].reshape(-1, 3)

        poses = []
        for i, T0 in enumerate(poses0):
            if i < self.n_fixed:
                poses.append(T0)
            else:
                poses.append(expSE3(xi[i - self.n_fixed]) @ T0)

        fx, fy = self.K[0, 0], self.K[1, 1]
        n_x = len(X)
        n_row = 2 * len(obs) + (1 if self._anchor_enabled(len(poses)) else 0)
        J = lil_matrix((n_row, 6 * n_opt + 3 * n_x))

        for k, (i, j, uv) in enumerate(obs):
            T = poses[i]
            R = T[:3, :3]
            Pc = R @ X[j] + T[:3, 3]
            Xc, Yc, Z = Pc
            if abs(Z) < 1e-9:
                continue

            # ∂r/∂P_c （注意符号：r = obs − pred）
            dr_dPc = np.array([[-fx / Z, 0.0, fx * Xc / (Z * Z)],
                               [0.0, -fy / Z, fy * Yc / (Z * Z)]])

            # 对地图点
            dr_dX = dr_dPc @ R                                    # (2,3)
            J[2 * k: 2 * k + 2, 6 * n_opt + 3 * j: 6 * n_opt + 3 * j + 3] = dr_dX

            # 对相机位姿（仅当该位姿参与优化）
            if i >= self.n_fixed:
                dPc_dxi = np.zeros((3, 6))
                dPc_dxi[:, :3] = np.eye(3)
                dPc_dxi[:, 3:] = -hat3(Pc)                        # = -[Pc]_×
                dr_dxi = dr_dPc @ dPc_dxi                         # (2,6)
                c0 = 6 * (i - self.n_fixed)
                J[2 * k: 2 * k + 2, c0: c0 + 6] = dr_dxi

        # ---- 尺度锚雅可比（单行，有限差分，稳）----
        if self._anchor_enabled(len(poses)):
            a_block = 0                       # 首个自由帧在 xi 中的块索引
            col0 = 6 * a_block
            eps = 1e-6
            r_base = self._anchor_val(poses)
            T_ref = poses0[self.n_fixed]      # 该帧的初值（用于重建扰动后的位姿）
            row = np.zeros(6 * n_opt + 3 * n_x)
            for c in range(6):
                xp = xi.copy()
                xp[a_block, c] += eps
                Tp = expSE3(xp[a_block]) @ T_ref
                poses2 = list(poses)
                poses2[self.n_fixed] = Tp
                rp = self._anchor_val(poses2)
                row[col0 + c] = (rp - r_base) / (2 * eps)
            J[-1, :] = row

        return csr_matrix(J)

    # ------------------------------------------------------------------ 求解
    def optimize(self, poses, points, obs, max_iter=30, verbose=False):
        """运行 BA

        参数
            poses  : (N,4,4) 或 list，初始 T_cw
            points : (M,3) 世界坐标地图点
            obs    : list of (pose_idx, point_idx, uv)
        返回
            poses_opt, points_opt, info
        """
        poses = [np.asarray(T, float).reshape(4, 4) for T in poses]
        points = np.asarray(points, float).reshape(-1, 3)
        n_p, n_x = len(poses), len(points)

        if len(obs) < 10 or n_x == 0:
            return np.array(poses), points, {"status": "skipped_too_few_obs"}

        # 过滤掉越界的观测
        obs = [(i, j, np.asarray(uv, float)) for i, j, uv in obs
               if 0 <= i < n_p and 0 <= j < n_x]

        n_opt = max(0, n_p - self.n_fixed)
        # 计算尺度锚目标（来自初值，即"真值尺度"的近似）
        self._anchor_target_val = self._anchor_target(poses) \
            if self._anchor_enabled(n_p) else None

        x0 = np.concatenate([np.zeros(6 * n_opt), points.ravel()])

        r0 = self._residual(x0, poses, points, obs, None, n_opt)
        cost0 = float(np.sum(r0 ** 2))

        # ⚠️ least_squares 的 max_nfev 是**函数评估次数**（不是迭代次数）！
        # 若直接传 max_iter 会严重欠拟合（实测 30 次评估就停，success=False，
        # 重投影误差只降一点点）。给足预算：每个 LM 迭代约需 2~4 次函数评估，
        # 而 BA 因参数尺度差异大、收敛偏慢，这里再乘一个系数确保真正收敛。
        max_nfev = max(max_iter * 40, 2000)
        sol = least_squares(
            self._residual, x0,
            jac=self._jacobian,
            args=(poses, points, obs, None, n_opt),
            method="trf",
            loss=self.robust if self.robust else "linear",
            f_scale=self.huber_f,
            max_nfev=max_nfev,
            verbose=2 if verbose else 0,
        )

        # 拆回位姿与点
        xi = sol.x[: 6 * n_opt].reshape(n_opt, 6)
        X_opt = sol.x[6 * n_opt:].reshape(-1, 3)
        poses_opt = []
        for i, T0 in enumerate(poses):
            if i < self.n_fixed:
                poses_opt.append(T0.copy())          # 第 0 帧严格固定
            else:
                poses_opt.append(expSE3(xi[i - self.n_fixed]) @ T0)

        cost1 = float(np.sum(sol.fun ** 2))

        info = {
            "status": "ok", "success": bool(sol.success),
            "cost_before": cost0, "cost_after": cost1,
            "rmse_before": float(np.sqrt(cost0 / max(1, len(r0)))),
            "rmse_after": float(np.sqrt(cost1 / max(1, len(sol.fun)))),
            "n_observations": len(obs), "n_poses": n_p, "n_points": n_x,
            "n_iter": int(sol.nfev),
            "n_fixed_poses": self.n_fixed,
            "scale_anchored": bool(self._anchor_enabled(n_p)),
            "anchor_target_baseline": self._anchor_target_val,
        }
        return np.array(poses_opt), X_opt, info


# ------------------------------------------------------------------ 便捷函数
def project_point_jac(K, T_cw, X_w):
    """同时返回投影结果与需要的中间量（供测试与推导对照）"""
    uv, Pc = project_point(K, T_cw, X_w)
    return uv, Pc


def ba_residual_jacobian(K, poses, points, obs):
    """一次性算出残差与**稠密**雅可比（用于教学演示与数值校验）"""
    ba = BundleAdjuster(K, n_fixed_poses=0)
    n_p, n_x = len(poses), len(points)
    x = np.concatenate([np.zeros(6 * n_p), points.ravel()])
    r = ba._residual(x, poses, points, obs, None, n_p)
    J = ba._jacobian(x, poses, points, obs, None, n_p)
    return r, np.asarray(J.todense())


def local_ba(K, poses, points, obs, n_fixed_poses=1, max_iter=30, verbose=False):
    """局部 BA 的便捷封装"""
    ba = BundleAdjuster(K, n_fixed_poses=n_fixed_poses)
    return ba.optimize(poses, points, obs, max_iter=max_iter, verbose=verbose)
