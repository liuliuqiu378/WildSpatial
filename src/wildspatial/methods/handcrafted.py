"""手搓 VO 方法（M1 自研管线，作为「自研基线」对照）"""

import numpy as np

from .base import Method, MethodResult, register
from ..sfm import MonocularVO, VOConfig


@register
class HandcraftedVO(Method):
    """把 M1 自研单目 VO 包成 Method 接口，作为「自研基线」。

    这是我们从零搭的 SIFT + 比值检验 + RANSAC + 三角化 + PnP + 运动护栏管线。
    在方法动物园里它代表"传统几何方法"的**自研实现**，用来和 COLMAP / VGGT 等
    成熟实现公平对比——我们自己写的到底离成熟工具差多少，一目了然。
    """
    name = "handcrafted_vo"
    kind = "classic"

    def available(self):
        return True

    def run(self, frames, K, **kw) -> MethodResult:
        cfg = VOConfig()
        if "max_features" in kw:
            cfg.max_features = kw["max_features"]
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
            note="自研 SIFT+VO（M1）：比值检验 + RANSAC + 三角化 + PnP + 运动护栏",
            extra={"n_map_points": int(vo.points_3d.shape[0])},
        )
