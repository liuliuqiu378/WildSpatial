"""特征提取

为什么从"特征"开始？
    对极几何需要**对应点**（同一个 3D 点在两幅图上的像素）。
    找对应点只有两条路：
      A. 稀疏特征：检测角点/斑点 → 描述子 → 匹配        （经典 SfM/SLAM，本 Module）
      B. 稠密光流/匹配：每个像素都算                    （LoFTR/RAFT 等深度学习方法，M4 之后）

    本 Module 走 A，因为要**亲手理解**"VGGT 到底替代了什么"。

三种主流特征（面试常考，工程常选）：
    SIFT   尺度/旋转不变，128 维浮点描述子，最稳但慢（有专利，但 OpenCV 4.4+ 已解禁）
    ORB    FAST 角点 + BRIEF 描述子（256 位二进制），极快，适合端侧，但对尺度/旋转鲁棒性较弱
    AKAZE  非线性尺度空间，二进制描述子，性能和速度折中

⚠️ 端侧部署视角（M8 会回来）：ORB 是 Jetson/瑞芯微上的首选，SIFT 在嵌入式上通常跑不动实时。
"""

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["FeatureSet", "extract_features", "draw_keypoints", "METHODS"]


METHODS = ("sift", "orb", "akaze")


@dataclass
class FeatureSet:
    """一次特征提取的结果"""
    keypoints: np.ndarray      # (N,2) 像素坐标 [u, v]
    descriptors: np.ndarray    # (N,D) 描述子；ORB/AKAZE 为 uint8（二进制）
    responses: np.ndarray      # (N,) 角点响应强度，越大越"显著"
    scales: np.ndarray         # (N,) 特征尺度
    angles: np.ndarray         # (N,) 主方向（弧度）

    def __len__(self):
        return len(self.keypoints)

    def top_k(self, k):
        """取响应最强的 k 个特征（控制计算量，端侧常用）"""
        if k is None or k >= len(self):
            return self
        idx = np.argsort(-self.responses)[:k]
        return FeatureSet(self.keypoints[idx], self.descriptors[idx],
                          self.responses[idx], self.scales[idx], self.angles[idx])


def _create_detector(method, **kw):
    method = method.lower()
    if method == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError(
                "当前 OpenCV 无 SIFT。请装 opencv-contrib-python（4.4+ 已无专利限制）")
        return cv2.SIFT_create(nfeatures=kw.get("max_features", 4000),
                               contrastThreshold=kw.get("contrast_threshold", 0.04),
                               edgeThreshold=kw.get("edge_threshold", 10))
    if method == "orb":
        return cv2.ORB_create(nfeatures=kw.get("max_features", 3000),
                              scaleFactor=kw.get("scale_factor", 1.2),
                              nlevels=kw.get("nlevels", 8),
                              fastThreshold=kw.get("fast_threshold", 20))
    if method == "akaze":
        return cv2.AKAZE_create()
    raise ValueError(f"未知特征方法 {method}，可选 {METHODS}")


def extract_features(image, method="sift", max_features=4000, mask=None, **kw):
    """提取特征点与描述子。

    参数
        image: (H,W) 灰度图 或 (H,W,3) BGR
        method: 'sift' | 'orb' | 'akaze'
    返回
        FeatureSet
    """
    if image is None:
        raise ValueError("image 为 None")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray)

    det = _create_detector(method, max_features=max_features, **kw)
    kps, desc = det.detectAndCompute(gray, mask)

    if not kps:
        return FeatureSet(np.zeros((0, 2)), np.zeros((0, 128)),
                          np.zeros(0), np.zeros(0), np.zeros(0))

    pts = np.array([kp.pt for kp in kps], dtype=np.float64)          # (x=u, y=v)
    resp = np.array([kp.response for kp in kps], dtype=np.float64)
    scale = np.array([kp.size for kp in kps], dtype=np.float64)
    ang = np.array([kp.angle for kp in kps], dtype=np.float64)

    return FeatureSet(keypoints=pts, descriptors=desc, responses=resp,
                      scales=scale, angles=ang)


def draw_keypoints(image, feats, color=(0, 255, 0), radius=2):
    """把特征点画到图上（可视化/调试用）"""
    img = image.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (u, v) in feats.keypoints.astype(int):
        cv2.circle(img, (int(u), int(v)), radius, color, -1)
    return img
