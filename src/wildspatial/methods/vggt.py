"""VGGT 前馈 3D 基础模型（M4 核心：一次前向取代整条 SfM 管线）

依赖（较重）：``vggt`` + ``transformers`` + ``timm``（DINOv2 主干）；
flash-attn 可选，不装则用 PyTorch SDPA 退化。装好即自动出现在方法动物园里。

为什么把 VGGT 也当成「一个 Method」
-----------------------------------
VGGT 代表新一代"前馈 3D 基础模型"：给定一组图像，**一次前向传播**直接输出
相机位姿 + 深度 + 点云，彻底取代"特征→匹配→RANSAC→三角化→BA"整条几何管线。
把它和普通 Method 放进同一张退化表，正是本项目最核心的对比命题：
**在恶劣环境下，前馈模型相比传统几何方法，是更鲁棒还是更早崩？**

官方推理 API（与本项目包装对应的关键事实，避免踩坑）：
- ``load_and_preprocess_images(path_list)`` 接收**文件路径列表**，返回 ``(1, N, 3, H, W)`` 张量。
- 模型输出 ``predictions["pose_enc"]``，形状 ``(B, S, 9)``（平移/四元数/FoV）。
- ``pose_encoding_to_extri_intri(pose_enc, image_size_hw=(H, W))`` 把其转成
  ``extrinsics``（BxSx3x4，OpenCV 约定 world→cam 的 ``[R|t]``）。
- 相机光心（世界系）：``C = -Rᵀ t``。
"""

import os
import tempfile

import numpy as np

from .base import Method, MethodResult, register


@register
class VGGT(Method):
    name = "vggt"
    kind = "foundation"
    requires_gpu = True

    _model = None  # 类级缓存：三方法图谱里多条件复用时只加载一次 5GB 权重

    def available(self):
        try:
            import torch  # noqa: F401
            import vggt  # noqa: F401
            from vggt.models.vggt import VGGT as _M  # noqa: F401
            from vggt.utils.load_fn import load_and_preprocess_images  # noqa: F401
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: F401
            return torch.cuda.is_available()
        except Exception:
            return False

    def run(self, frames, K, device="cuda", **kw) -> MethodResult:
        import cv2
        import torch
        from PIL import Image

        from vggt.models.vggt import VGGT as VGGTModel
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        # VGGT 的 loader 接收文件路径，所以先把帧写进临时目录
        tmp = tempfile.mkdtemp(prefix="vggt_")
        paths = []
        for i, f in enumerate(frames):
            p = os.path.join(tmp, f"{i:06d}.png")
            cv2.imwrite(p, f)
            paths.append(p)

        try:
            # 类级缓存：多条件复用只加载一次权重
            if VGGT._model is None:
                # 优先用本地权重（可由 ModelScope 下载到 VGGT_WEIGHTS 指向的目录，
                # 避免国内访问 HF 超时）；否则回退 HF hub 自动下载。
                local = os.environ.get("VGGT_WEIGHTS") or os.path.expanduser(
                    "~/models/vggt-1b")
                if os.path.isdir(local) and os.path.exists(
                        os.path.join(local, "config.json")):
                    VGGT._model = VGGTModel.from_pretrained(local)
                else:
                    VGGT._model = VGGTModel.from_pretrained("facebook/VGGT-1B")
                VGGT._model.to(device).eval()
            model = VGGT._model

            images = load_and_preprocess_images(paths)
            images = images.to(device)

            with torch.no_grad():
                predictions = model(images)

            pose_enc = predictions["pose_enc"]          # (1, S, 9)
            H, W = images.shape[-2], images.shape[-1]
            extr, intr = pose_encoding_to_extri_intri(
                pose_enc, image_size_hw=(int(H), int(W)))
            # extr: (1, S, 3, 4)，world→cam 的 [R|t]
            R = extr[0, :, :3, :3].cpu().numpy()         # (S,3,3) world→cam
            t = extr[0, :, :, 3].cpu().numpy()           # (S,3)  world→cam
            C = np.einsum("sij,sj->si", R.transpose(0, 2, 1), t)
            C = -C                                        # 相机光心（世界系）

            # 完整位姿 T_cw（相机→世界），供下游反投影建图 / 占据栅格（闭环层需要）。
            # 因 extr 是 world→cam 变换 [R|t]，其逆 T_cw = [Rᵀ | -Rᵀ t]。
            Tcw = []
            for s in range(R.shape[0]):
                M = np.eye(4, dtype=float)
                M[:3, :3] = R[s].T
                M[:3, 3] = -R[s].T @ t[s]
                Tcw.append(M)

            # ---- 稠密输出（M4 深化：深度 / 点云质量，不止轨迹）----
            # return_dense=True 时才取，避免轨迹图谱占用额外显存。
            extra = {"intrinsics_available": intr is not None,
                     "n_images": int(len(C)),
                     "T_cw": Tcw}
            if kw.get("return_dense"):
                depth = predictions.get("depth")
                wp = predictions.get("world_points")
                dconf = predictions.get("depth_conf")
                wpconf = predictions.get("world_points_conf")
                if depth is not None:
                    # depth: (1, S, H, W, 1) -> (S, H, W)
                    extra["depth"] = depth[0, ..., 0].cpu().numpy().astype(np.float32)
                if wp is not None:
                    # world_points: (1, S, H, W, 3) -> (S, H, W, 3)
                    extra["world_points"] = wp[0].cpu().numpy().astype(np.float32)
                if dconf is not None:
                    extra["depth_conf"] = dconf[0].cpu().numpy().astype(np.float32)
                if wpconf is not None:
                    extra["world_points_conf"] = wpconf[0].cpu().numpy().astype(
                        np.float32)
                extra["dense_hw"] = [int(H), int(W)]

            return MethodResult(
                positions=C.astype(float),
                frame_indices=np.arange(len(C)),
                ok=len(C) > 3,
                note="VGGT 前馈 3D 基础模型（一次前向出相机轨迹 + 深度/点云）",
                extra=extra,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return MethodResult(
                np.zeros((0, 3)), np.zeros(0, int), ok=False,
                note=f"VGGT 运行失败: {str(e)[:200]}")
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
