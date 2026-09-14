"""图像**合成退化**（Controlled Degradation）—— 用于可控的失效对照实验

=========================== 为什么需要它 ===========================
本项目真正关心的是"**恶劣物理环境**下系统怎么崩"（水下、地下、矿山、烟雾、低光照）。
但带**真值标签（ground truth）**的恶劣环境公开数据集非常稀缺（例如 TUM
`fr3/nostructure_notexture` 实测连 `rgb.txt` 和真值都没有，根本没法定量评估）。

解决办法：**在带真值的数据上施加可控退化**。
    干净序列（有真值）──施加退化──▶ 退化序列（**仍然有真值**）
这样就可以定量回答："模糊到什么程度，系统开始崩？崩在哪一步？"

⚠️ 纪律：任何用合成退化得出的结论，必须在文档与图注中**明确标注"退化为合成施加"**，
不能让读者误以为是真实采集的恶劣环境数据。

=========================== 退化类型与物理含义 ===========================
    motion_blur    运动模糊   ← 快门过长 / 平台高速运动 / 抖动
    gaussian_noise 高斯噪声   ← 高 ISO / 弱光下的传感器噪声
    low_light      低光照     ← 夜间 / 矿井 / 水下（同时伴随噪声抬升）
    downsample     分辨率下降 ← 远距离 / 有损压缩 / 低带宽回传
"""

import cv2
import numpy as np

__all__ = ["KINDS", "LEVELS", "degrade", "level_label"]

KINDS = {
    "motion_blur": "运动模糊（快门过长 / 平台抖动）",
    "gaussian_noise": "高斯噪声（高 ISO / 弱光传感器噪声）",
    "low_light": "低光照（暗光 + 噪声抬升）",
    "downsample": "分辨率下降（远距离 / 压缩 / 低带宽回传）",
}

# 统一的严重度档位（severity ∈ [0,1]）
LEVELS = [("轻", 0.33), ("中", 0.66), ("重", 1.0)]


def level_label(sev):
    """把 severity 映射到「轻/中/重」文字"""
    if sev <= 0.4:
        return "轻"
    if sev <= 0.75:
        return "中"
    return "重"


def degrade(img, kind="motion_blur", severity=0.5, rng=None):
    """对图像施加合成退化。

    参数
        img      : BGR 或灰度图（uint8）
        kind     : 见 KINDS；None / "clean" 表示不退化
        severity : 0~1，0 = 不退化，1 = 最强
    返回
        退化后的图像（与输入同尺寸同类型）
    """
    if img is None or kind is None or kind == "clean" or severity <= 0:
        return img
    rng = np.random.default_rng(0) if rng is None else rng
    s = float(np.clip(severity, 0.0, 1.0))

    if kind == "motion_blur":
        k = max(3, int(3 + s * 30))
        if k % 2 == 0:
            k += 1
        kernel = np.zeros((k, k))
        kernel[k // 2, :] = 1.0 / k          # 水平运动模糊核
        return cv2.filter2D(img, -1, kernel)

    if kind == "gaussian_noise":
        noise = rng.normal(0.0, s * 50.0, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if kind == "low_light":
        gain = float(np.clip(1.0 - 0.85 * s, 0.05, 1.0))
        dark = img.astype(np.float32) * gain
        noise = rng.normal(0.0, 3.0 + 8.0 * s, img.shape).astype(np.float32)
        return np.clip(dark + noise, 0, 255).astype(np.uint8)

    if kind == "downsample":
        scale = float(np.clip(1.0 - 0.7 * s, 0.15, 1.0))
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    return img
