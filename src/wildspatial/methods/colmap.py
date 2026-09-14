"""COLMAP SfM 方法（成熟开源 SfM 工具，作为「传统几何方法」的工业级基线）

依赖：``pip install pycolmap``。装好即自动出现在方法动物园里，无需改代码。
"""

import os
import tempfile

import numpy as np

from .base import Method, MethodResult, register


@register
class ColmapSfM(Method):
    """用 COLMAP 做增量式 SfM，输出相机轨迹。

    COLMAP 是学术/工业界最成熟的 SfM 工具之一，代表"传统几何方法"的**工业级上限**。
    把它和我们的 handcrafted_vo 放在同一张退化压力测试表里，就能直观看出：
    自研管线离成熟工具到底差在哪、在哪种退化下两者一起崩。
    """
    name = "colmap_sfm"
    kind = "classic"

    def available(self):
        try:
            import pycolmap  # noqa: F401
            return True
        except Exception:
            return False

    def run(self, frames, K, workdir=None, **kw) -> MethodResult:
        import cv2
        import pycolmap

        tmp = workdir or tempfile.mkdtemp(prefix="colmap_")
        imgdir = os.path.join(tmp, "images")
        os.makedirs(imgdir, exist_ok=True)
        index_of = {}
        for i, img in enumerate(frames):
            name = f"{i:06d}.png"
            cv2.imwrite(os.path.join(imgdir, name), img)
            index_of[name] = i

        try:
            db = os.path.join(tmp, "db.db")
            # pycolmap 4.x 的 extract_features 不接受 camera_model/sift_max_num_features
            # 这类关键字（会报签名不兼容）；用默认参数即可，COLMAP 会在 BA 中自标定焦距。
            pycolmap.extract_features(db, imgdir)
            pycolmap.match_exhaustive(db)
            recdir = os.path.join(tmp, "sparse")
            os.makedirs(recdir, exist_ok=True)
            maps = pycolmap.incremental_mapping(db, imgdir, recdir)
            # incremental_mapping 返回 {recon_id: Reconstruction}
            recon = max(maps.items(), key=lambda kv: len(kv[1].points3D))[1] \
                if maps else None
            if recon is None or len(recon.images) == 0:
                return MethodResult(np.zeros((0, 3)), np.zeros(0, int), ok=False,
                                    note="COLMAP 未重建出任何图像")

            pos, idxs = [], []
            for img_id, im in recon.images.items():
                name = os.path.basename(im.name)
                if name in index_of:
                    # cam_from_world() 返回 Rigid3d（world→cam 变换）
                    M = im.cam_from_world().matrix()     # 4x4
                    R = M[:3, :3]
                    t = M[:3, 3]
                    C = -R.T @ t                         # 相机光心（世界系）
                    pos.append(C)
                    idxs.append(index_of[name])
            return MethodResult(
                positions=np.array(pos, dtype=float),
                frame_indices=np.array(idxs, dtype=int),
                ok=len(pos) > 3,
                note="COLMAP 增量式 SfM（工业级传统几何）",
                extra={"n_points3d": int(len(recon.points3D))},
            )
        except Exception as e:  # 任何运行错误都优雅降级
            import traceback
            traceback.print_exc()
            return MethodResult(np.zeros((0, 3)), np.zeros(0, int), ok=False,
                                note=f"COLMAP 运行失败: {str(e)[:200]}")
