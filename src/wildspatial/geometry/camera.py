"""针孔相机模型

坐标系统一约定（全项目遵守）：
    世界点 P_w (3,) --T_cw--> 相机点 P_c (3,) --K--> 像素 (u,v)

    P_c = R_cw · P_w + t_cw
    s · [u, v, 1]^T = K · P_c

其中 K = [[fx, 0, cx],
          [0, fy, cy],
          [0,  0,  1]]

内参含义：
    fx, fy — 焦距（像素单位）= f_mm / pixel_size_mm，所以"像素"其实是角度的度量
    cx, cy — 主点（光轴与成像平面交点），理想是图像中心但实际有偏差
    有时还有 skew 项（非正交），这里设为 0

⚠️ 尺度歧义（单目 VO 的根本难题）：
    从投影方程看，P_c 和 s·P_c 投影到同一个像素。
    也就是说单目视觉**无法仅从图像恢复绝对尺度**——把整个场景和相机位姿同时放大 k 倍，
    观察到的图像完全一样。这就是为什么单目 VO 需要 IMU、已知物体尺寸或双目来定尺度。
"""

import numpy as np

__all__ = ["PinholeCamera", "normalize_points", "make_K", "project_jacobian"]


def make_K(fx, fy, cx, cy):
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]])


class PinholeCamera:
    def __init__(self, fx, fy, cx, cy, width=None, height=None, name=""):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = width
        self.height = height
        self.name = name

    # ---------- 基本量 ----------
    @property
    def K(self):
        return make_K(self.fx, self.fy, self.cx, self.cy)

    @property
    def Kinv(self):
        return np.array([[1.0 / self.fx, 0.0, -self.cx / self.fx],
                         [0.0, 1.0 / self.fy, -self.cy / self.fy],
                         [0.0, 0.0, 1.0]])

    @classmethod
    def from_K(cls, K, width=None, height=None, name=""):
        K = np.asarray(K, dtype=float).reshape(3, 3)
        return cls(K[0, 0], K[1, 1], K[0, 2], K[1, 2], width, height, name)

    def __repr__(self):
        wh = f" {self.width}x{self.height}" if self.width else ""
        return f"PinholeCamera(fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}{wh})"

    # ---------- 投影 ----------
    def project(self, P_c, return_depth=False):
        """相机坐标系 3D 点 → 像素坐标

        参数: P_c (N,3) 或 (3,)
        返回: uv (N,2)
        """
        P_c = np.asarray(P_c, dtype=float)
        single = (P_c.ndim == 1)
        P_c = P_c.reshape(-1, 3)
        z = P_c[:, 2]
        valid = np.abs(z) > 1e-12
        uv = np.full((P_c.shape[0], 2), np.nan)
        uv[valid, 0] = self.fx * P_c[valid, 0] / z[valid] + self.cx
        uv[valid, 1] = self.fy * P_c[valid, 1] / z[valid] + self.cy
        if return_depth:
            return (uv[0] if single else uv), (z[0] if single else z)
        return uv[0] if single else uv

    def project_world(self, P_w, T_cw, return_depth=False):
        """世界点 → 像素点。T_cw 为 4x4（世界→相机）"""
        P_w = np.asarray(P_w, dtype=float)
        single = (P_w.ndim == 1)
        P_w = P_w.reshape(-1, 3)
        P_h = np.concatenate([P_w, np.ones((P_w.shape[0], 1))], axis=1)
        P_c = (T_cw @ P_h.T).T[:, :3]
        uv, z = self.project(P_c, return_depth=True)
        uv = uv[0] if single else uv
        z = z[0] if single else z
        return (uv, z) if return_depth else uv

    # ---------- 反投影 ----------
    def unproject(self, uv, depth=1.0):
        """像素 + 深度 → 相机坐标系 3D 点。depth 默认 1 时得到**归一化平面**上的方向。"""
        uv = np.asarray(uv, dtype=float)
        single = (uv.ndim == 1)
        uv = uv.reshape(-1, 2)
        hom = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)   # (N,3)
        rays = (self.Kinv @ hom.T).T                                     # (N,3) 方向
        d = np.atleast_1d(np.asarray(depth, dtype=float))
        if d.size == 1:
            d = np.full((rays.shape[0],), float(d))
        pts = rays * d[:, None]
        return pts[0] if single else pts

    # ---------- 工具 ----------
    def is_visible(self, uv, margin=0.0):
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        if self.width is None:
            return np.ones(uv.shape[0], dtype=bool)
        return ((uv[:, 0] >= margin) & (uv[:, 0] < self.width - margin) &
                (uv[:, 1] >= margin) & (uv[:, 1] < self.height - margin))

    def fov(self):
        """水平和垂直视场角（度）"""
        if self.width is None:
            return None
        fx_deg = 2 * np.degrees(np.arctan(self.width / (2 * self.fx)))
        fy_deg = 2 * np.degrees(np.arctan(self.height / (2 * self.fy)))
        return fx_deg, fy_deg


def normalize_points(uv, K):
    """像素坐标 → 归一化平面坐标（去内参）。

        x = K^{-1} · [u,v,1]^T

    为什么重要：对极几何里的本质矩阵 E 定义在**归一化坐标**上
    （x2^T E x1 = 0），而基础矩阵 F 定义在像素坐标上（p2^T F p1 = 0）。
    八点法必须做归一化，否则数值条件数极差（见 epipolar.py）。
    """
    K = np.asarray(K, dtype=float).reshape(3, 3)
    Kinv = np.linalg.inv(K)
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    hom = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)
    xn = (Kinv @ hom.T).T
    return xn


def project_jacobian(P_c, fx, fy):
    """投影函数对相机坐标 P_c 的雅可比 (2x3)。

        u = fx·X/Z + cx      v = fy·Y/Z + cy
        ∂(u,v)/∂(X,Y,Z) = [[fx/Z, 0, -fx·X/Z²],
                           [0, fy/Z, -fy·Y/Z²]]

    BA（Bundle Adjustment）里每个残差的雅可比都从这里链式展开。
    """
    X, Y, Z = P_c
    return np.array([[fx / Z, 0.0, -fx * X / (Z * Z)],
                     [0.0, fy / Z, -fy * Y / (Z * Z)]])
