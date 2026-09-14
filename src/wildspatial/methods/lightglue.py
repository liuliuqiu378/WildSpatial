"""LightGlue 学习式前端 VO 方法（M5 方法动物园「学习式匹配」对照栏）

为什么需要它（目的）
------------------
方法动物园现有：
    · handcrafted_vo  —— 经典几何（SIFT/ORB 前端 + 比值检验 + 几何后端）
    · colmap_sfm      —— 工业级 SfM（COLMAP，内置 SIFT）
    · vggt            —— 前馈 3D 基础模型（一次前向直接回归位姿）
缺一根「学习式匹配」的柱子：用 **SuperPoint（学习式检测+描述）+ LightGlue
（学习式匹配器）** 替换经典前端，但**后端几何（RANSAC/三角化/PnP/尺度/护栏）
完全复用 M1/vo.py 的现成实现**。

这样方法动物园变成**三个干净维度**的对比：
    经典几何（SIFT/ORB 比值检验）  vs  学习式匹配（SuperPoint+LightGlue）  vs  前馈模型（VGGT）
后端一致，差异只来自「特征/匹配质量」——这正是我们要量化的对象。

方法原理（专业）
--------------
SuperPoint 网络直接回归关键点位置与 256 维描述子（自监督训练，对光照/模糊鲁棒）；
LightGlue 用 Transformer 对两图关键点做互注意力匹配（自适应早停，难图更准）。
二者都比人工设计特征在退化下更稳。本方法把这对「学习式前端」插进同一个单目 VO
后端，输出与 handcrafted_vo 完全相同的相机光心轨迹，因此走同一条 ATE 评测链路，
公平可比。

权重来源
--------
SuperPoint/LightGlue 权重来自 cvg/LightGlue 的 GitHub release（v0.1_arxiv）：
    superpoint_v1.pth / superpoint_lightglue.pth
国内经 ghproxy 代理下载并置入 torch hub 缓存（~/.cache/torch/hub/checkpoints），
离线即可加载，无需运行时联网。
"""

import numpy as np

from .base import Method, MethodResult, register
from ..sfm import MonocularVO, VOConfig
from ..sfm.features import FeatureSet


@register
class LightGlueVO(Method):
    """SuperPoint + LightGlue 学习式前端，复用 M1 单目 VO 后端。

    kind = "learned"：代表「学习式匹配」这一方法维度。
    """
    name = "lightglue_vo"
    kind = "learned"
    requires_gpu = True

    # 类级缓存：整个 sweep（多退化条件）只加载一次权重
    _extractor = None
    _matcher = None
    _device = None

    def available(self):
        try:
            import torch  # noqa: F401
            import lightglue  # noqa: F401
            import kornia  # noqa: F401
            return True
        except Exception:
            return False

    @classmethod
    def _ensure_models(cls, device="cuda"):
        if cls._extractor is not None:
            return
        import torch
        from lightglue import SuperPoint, LightGlue
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        cls._device = dev
        # max_num_keypoints 控制计算量；2048 在 TUM 640x480 上足够稠密
        cls._extractor = SuperPoint(max_num_keypoints=2048).eval().to(dev)
        cls._matcher = LightGlue(features="superpoint").eval().to(dev)

    @staticmethod
    def _make_featset(kp, desc):
        n = len(kp)
        return FeatureSet(
            keypoints=kp.astype(np.float64),
            descriptors=desc.astype(np.float64),
            responses=np.zeros(n, dtype=np.float64),
            scales=np.zeros(n, dtype=np.float64),
            angles=np.zeros(n, dtype=np.float64),
        )

    def _frontend(self, ref, gray):
        """注入 VO 的前端：给定参考帧（或 None）与当前灰度图，返回 (FeatureSet, matches)。

        matches: (M,2) int32，第 0 列索引 ref.keypoints，第 1 列索引当前帧 keypoints。
        """
        import torch
        from lightglue.utils import rbd

        dev = self._device

        def to_t(x):
            t = torch.from_numpy(np.ascontiguousarray(x)).float() / 255.0
            return t[None, None].to(dev)  # (1,1,H,W) float [0,1]

        if ref is None:
            feats = self._extractor.extract(to_t(gray))
            kp = feats["keypoints"][0].cpu().numpy().astype(np.float64)
            desc = feats["descriptors"][0].cpu().numpy().astype(np.float64)
            return self._make_featset(kp, desc), np.zeros((0, 2), dtype=np.int32)

        # 重新提取参考帧（确定性，与 ref.keypoints 逐点相同）——保证 matches 索引一致
        feats0 = self._extractor.extract(to_t(ref.image))
        feats1 = self._extractor.extract(to_t(gray))
        pred = self._matcher({"image0": feats0, "image1": feats1})
        pred = rbd(pred)
        m = pred["matches"].cpu().numpy().astype(np.int64)
        m = m[(m[:, 0] >= 0) & (m[:, 1] >= 0)]

        kp0 = feats0["keypoints"][0].cpu().numpy().astype(np.float64)
        kp1 = feats1["keypoints"][0].cpu().numpy().astype(np.float64)
        desc0 = feats0["descriptors"][0].cpu().numpy().astype(np.float64)
        desc1 = feats1["descriptors"][0].cpu().numpy().astype(np.float64)

        # 用重新提取的结果覆盖 ref 的特征（与 matches[:,0] 对齐），确保后端一致
        ref.keypoints = kp0
        ref.descriptors = desc0
        return self._make_featset(kp1, desc1), m.astype(np.int32)

    def run(self, frames, K, max_features=2048, device="cuda", **kw) -> MethodResult:
        self._ensure_models(device)
        cfg = VOConfig()
        cfg.frontend = self._frontend
        cfg.max_features = max_features
        vo = MonocularVO(np.asarray(K, float), cfg)
        pos, idxs = [], []
        for i, img in enumerate(frames):
            vo.process(img)
            f = vo.frames[-1]
            if f.T_cw is not None:
                pos.append(np.linalg.inv(f.T_cw)[:3, 3])
                idxs.append(i)
        ok = len(pos) > 3
        return MethodResult(
            positions=np.array(pos, dtype=float),
            frame_indices=np.array(idxs, dtype=int),
            ok=ok,
            note="SuperPoint+LightGlue 学习式前端 + M1 几何后端（单目 VO）",
            extra={"n_map_points": int(vo.points_3d.shape[0])},
        )
