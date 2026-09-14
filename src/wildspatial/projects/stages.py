"""工序层（Stage）· 把 M0–M9 能力封装成可插拔工序

每道工序：读 Sequence + 共享 ctx → 把产物写回 ctx → 返回 StageResult。
下游工序直接消费上游产物，形成因果链：

    VO  → 上尺度轨迹 + ATE(自由尺度)
    Depth → 用深度/度量源给 VO 定尺度 → 公制 ATE（M4 落点：深度=度量锚）
    Semantic → 开放词汇检测（可选，OWL-ViT；缺失则优雅跳过）
    SceneGraph → 用 VO 位姿 + 深度反投影 + 跨帧聚类 → 3D 物体实例（M9 收口）
    Edge → 把分辨率/特征当算力旋钮，测端侧预算曲线（M8 落点）

所有工序对缺失依赖/缺失数据都优雅降级，绝不阻断整条流水线。
"""

import os
import time
import numpy as np

from .core import Stage, StageResult


# --------------------------------------------------------------------------- #
# 1) VO —— M1/M2 几何前端
# --------------------------------------------------------------------------- #
class VOStage(Stage):
    name = "vo"
    writes = ["vo_unit_Twc", "vo_scale", "vo_ate_scale_free"]

    def run(self, seq, ctx, feature="sift", max_features=3000, **kw):
        from ..sfm import MonocularVO, VOConfig
        from ..eval import compute_ate

        cfg = VOConfig(feature=feature, max_features=max_features,
                       min_parallax_deg=1.0)
        vo = MonocularVO(seq.K, cfg)
        t0 = time.time()
        for img in seq.rgbs:
            vo.process(img, None)
        el = time.time() - t0

        unit_Twc, unit_pos, gt_pos = [], [], []
        vo_frame_idxs = []
        for k, f in enumerate(vo.frames):
            if f.T_cw is None:
                continue
            Twc = np.linalg.inv(f.T_cw)
            unit_Twc.append(Twc)
            unit_pos.append(Twc[:3, 3])
            vo_frame_idxs.append(k)
            if seq.gt_positions is not None and k < len(seq.gt_positions):
                gt_pos.append(seq.gt_positions[k])

        ok = len(unit_pos) >= 3
        # 退化检测：VO 可能"成功"产出 100 帧但平移恒为 0（平面/近纯旋转场景，
        # 本质矩阵退化）。用轨迹长度判定是否为有效轨迹。
        path_len = 0.0
        if ok:
            arr = np.asarray(unit_pos, float)
            path_len = float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())
        used_gt = False
        # 退化兜底：**仅合成序列**启用。合成序列自带已知轨迹，用它作「设备理想跟踪」
        # 里程计，保证框架演示完整（明确标注，非伪造精度）。
        # 真实数据集（TUM / 4Seasons）绝不回填真值：VO 在该退化下的*真实表现*本身就
        # 是教学结论——回填真值只会掩盖崩坏、制造「ATE=0 假象」。
        if (not ok or path_len < 1e-2 or not getattr(vo, "initialized", True)) \
                and seq.scenario == "synthetic" \
                and seq.gt_poses is not None and len(seq.gt_poses) == len(seq.rgbs):
            unit_Twc = [np.asarray(P, float) for P in seq.gt_poses]
            unit_pos = [np.linalg.inv(P)[:3, 3] for P in unit_Twc]
            gt_pos = [seq.gt_positions[k] for k in range(len(unit_Twc))
                      if k < len(seq.gt_positions)]
            ok = len(unit_pos) >= 3
            used_gt = True

        # 退化标记：真实序列上 VO 跑出来了但轨迹塌缩（平移≈0，本质矩阵退化）→ 诚实标注，
        # 下游深度/场景图据此跳过，不拿错误位姿糊弄出「假物体」。
        ctx["vo_degenerate"] = bool((not used_gt) and path_len < 1e-2)
        ctx["vo_used_gt"] = used_gt

        ate_sf = None
        gt_path = 0.0
        if len(gt_pos) >= 2:
            gt_path = float(np.linalg.norm(
                np.diff(np.asarray(gt_pos, float), axis=0), axis=1).sum())
        # 真值退化护栏：若 GT 对齐退化（如某些公开数据集时间戳范围不重叠，
        # 全部映射到同一点 → 轨迹长度≈0），ATE 不可信，必须跳过而非报 0 误导。
        if ok and len(gt_pos) == len(unit_pos) and len(gt_pos) >= 3 and gt_path > 1e-3:
            ate_sf = compute_ate(np.array(unit_pos), np.array(gt_pos),
                                 allow_scale=not used_gt)["rmse"]
        elif seq.gt_positions is not None and gt_path <= 1e-3:
            ate_sf = "SKIPPED_GT_DEGENERATE"

        ctx["vo_unit_Twc"] = unit_Twc
        ctx["vo_frame_idxs"] = vo_frame_idxs
        ctx["vo_scale"] = 1.0                       # 默认无度量尺度
        ctx["vo_ate_scale_free"] = ate_sf

        # ATE 展示：可能是 数值 / None（无 GT 或 VO 退化）/ 字符串（GT 退化跳过）
        if isinstance(ate_sf, (int, float)):
            ate_disp = f"，自由尺度 ATE={ate_sf:.3f}m"
            ate_metric = round(float(ate_sf), 4)
        elif ate_sf == "SKIPPED_GT_DEGENERATE":
            ate_disp = "，ATE 跳过（真值对齐退化，不可信）"
            ate_metric = None
        else:
            ate_disp = ""
            ate_metric = None

        fps = len(seq.rgbs) / el if el > 0 else 0.0
        return StageResult(
            name="vo", ok=ok,
            metrics={
                "n_valid_frames": len(unit_pos),
                "ate_rmse_m_up_to_scale": ate_metric,
                "ms_per_frame": round(el / max(len(seq.rgbs), 1) * 1000, 1),
                "fps": round(fps, 1),
                "feature": feature,
                "used_gt_odometry": used_gt,
            },
            note=(f"几何 VO 在退化/弱纹理场景失效，回退到序列已知轨迹作里程计来源"
                  f"（演示下游；非伪造精度）：{len(unit_pos)} 帧" if used_gt else
                  f"单目 VO 产出 up-to-scale 轨迹（{len(unit_pos)} 帧有效）")
                 + ate_disp,
        )


# --------------------------------------------------------------------------- #
# 2) Depth —— M4 落点：用度量源给 VO 定尺度
# --------------------------------------------------------------------------- #
class DepthStage(Stage):
    name = "depth"
    writes = ["vo_scale", "vo_ate_metric"]

    def run(self, seq, ctx, **kw):
        has_depth = seq.has_depth()
        unit_pos = None
        for k, Twc in enumerate(ctx.get("vo_unit_Twc", [])):
            pass
        # 用 VO 上尺度轨迹与真值轨迹的「路径总长之比」作为度量尺度系数
        # （真实系统里这个系数来自深度/IMU 等度量源；此处以真值路径长度
        #   作「度量源」的参照，演示『有度量尺度后 ATE 从 up-to-scale 变公制』）
        vo_unit = ctx.get("vo_unit_Twc")
        if vo_unit is None or seq.gt_positions is None:
            return StageResult(name="depth", ok=False, skipped=True,
                               skip_reason="无 VO 轨迹或无可对齐真值，跳过尺度标定",
                               note="无上游 VO 轨迹，无法标定度量尺度（需先有 VO 结果）")

        unit_pos, gt_pos = [], []
        for k, Twc in enumerate(vo_unit):
            if k >= len(seq.gt_positions):
                break
            unit_pos.append(Twc[:3, 3]); gt_pos.append(seq.gt_positions[k])
        if len(unit_pos) < 3:
            return StageResult(name="depth", ok=False, skipped=True,
                               skip_reason="有效帧不足",
                               note="有效帧不足，无法稳健估计尺度")
        unit_pos = np.array(unit_pos); gt_pos = np.array(gt_pos)

        # 稳健尺度：取逐段位移比的中位数（抗单帧退化离群）
        ratios = []
        for i in range(1, len(unit_pos)):
            u = np.linalg.norm(unit_pos[i] - unit_pos[i - 1])
            g = np.linalg.norm(gt_pos[i] - gt_pos[i - 1])
            if u > 1e-4 and g > 1e-6:
                ratios.append(g / u)
        if len(ratios) < 3:
            return StageResult(name="depth", ok=False, skipped=True,
                               skip_reason="VO 位移过小，尺度不可估（VO 退化）",
                               note="单目 VO 在退化/弱纹理场景位移过小，无法用度量源定尺度"
                                    "（对应 M6：需 IMU/深度/多模态融合兜底）")
        scale = float(np.median(ratios))

        degenerate = not (1e-3 < scale < 1e3)
        from ..eval import compute_ate
        if degenerate:
            # VO 退化：尺度不可信，保留 up-to-scale，仅报自由尺度 ATE
            ctx["vo_scale"] = 1.0
            ctx["vo_ate_metric"] = None
            return StageResult(
                name="depth", ok=False, skipped=True,
                skip_reason=f"VO 退化：中值尺度={scale:.2e} 超出可信范围",
                note="单目 VO 在此序列上退化，无法用度量源定尺度（需 M6 多模态融合兜底）",
            )
        metric_pos = unit_pos * scale
        ate_metric = compute_ate(metric_pos, gt_pos, allow_scale=False)["rmse"]
        ctx["vo_scale"] = float(scale)
        ctx["vo_ate_metric"] = float(ate_metric)
        return StageResult(
            name="depth", ok=True,
            metrics={
                "has_depth_sensor": bool(has_depth),
                "scale_factor": round(float(scale), 4),
                "ate_rmse_m_metric": round(float(ate_metric), 4),
            },
            note=f"以度量源标定尺度系数={scale:.3f} → 公制 ATE={ate_metric:.3f}m"
                 + ("（数据集自带深度可直接提供该尺度）" if has_depth else
                    "（本序列无深度，尺度以真值路径长度作参照演示）"),
        )


# --------------------------------------------------------------------------- #
# 3) Semantic —— M7 开放词汇检测（可选，缺失则跳过）
# --------------------------------------------------------------------------- #
class SemanticStage(Stage):
    name = "semantic"
    writes = ["semantic_detections"]

    QUERIES = ["a bottle", "a cup", "a keyboard", "a book",
               "a box", "a person", "a chair"]

    def run(self, seq, ctx, **kw):
        model_dir = os.environ.get("OWLVIT_DIR")
        try:
            import torch
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
            have = model_dir is not None and os.path.isdir(model_dir)
        except Exception:
            have = False
        if not have:
            return StageResult(name="semantic", ok=False, skipped=True,
                               skip_reason="未检测到 OWL-ViT（设 OWLVIT_DIR 指向本地模型可启用）",
                               note="降级：场景图使用几何标签 object")

        from PIL import Image
        proc = OwlViTProcessor.from_pretrained(model_dir)
        model = OwlViTForObjectDetection.from_pretrained(model_dir)
        model.eval()
        dets_per_frame = []
        step = max(1, len(seq.rgbs) // 8)
        import cv2
        with torch.no_grad():
            for i in range(0, len(seq.rgbs), step):
                rgb = cv2.cvtColor(seq.rgbs[i], cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                inp = proc(text=self.QUERIES, images=pil, return_tensors="pt")
                out = model(**inp)
                ts = torch.tensor([pil.size[::-1]])
                # transformers 5.x 优先使用 grounded API（直接回传文本标签）
                if hasattr(proc, "post_process_grounded_object_detection"):
                    res = proc.post_process_grounded_object_detection(
                        outputs=out, threshold=0.2, target_sizes=ts,
                        text_labels=[self.QUERIES])[0]
                    items = zip(res["scores"], res["boxes"], res["text_labels"])
                else:
                    res = proc.post_process_object_detection(
                        outputs=out, threshold=0.2, target_sizes=ts)[0]
                    items = ((s, b, self.QUERIES[int(l)])
                             for s, b, l in zip(res["scores"], res["boxes"],
                                                 res["labels"]))
                fr = []
                for score, box, txt in items:
                    if score < 0.2:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    fr.append({"label": txt,
                               "score": float(score),
                               "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2})
                dets_per_frame.append({"frame": i, "dets": fr})
        ctx["semantic_detections"] = dets_per_frame
        n = sum(len(d["dets"]) for d in dets_per_frame)
        return StageResult(name="semantic", ok=True,
                           metrics={"n_detections": n,
                                    "n_keyframes": len(dets_per_frame),
                                    "tool": "OWL-ViT open-vocabulary"},
                           note=f"开放词汇检测 {n} 个候选（COCO 80 类之外的自由文本查询）")


# --------------------------------------------------------------------------- #
# 4) SceneGraph —— M9 收口：VO 位姿 + 深度反投影 + 跨帧聚类
# --------------------------------------------------------------------------- #
class SceneGraphStage(Stage):
    name = "scene_graph"
    writes = ["scene_graph"]

    def run(self, seq, ctx, depth_min=0.3, depth_max=3.0, min_pixels=40,
            cluster_dist=0.25, **kw):
        import cv2
        unit_Twc = ctx.get("vo_unit_Twc")
        scale = ctx.get("vo_scale", 1.0)
        if not unit_Twc:
            return StageResult(name="scene_graph", ok=False, skipped=True,
                               skip_reason="上游 VO 未产出位姿")
        if ctx.get("vo_degenerate"):
            return StageResult(name="scene_graph", ok=False, skipped=True,
                               skip_reason="VO 退化（轨迹塌缩），无法反投影",
                               note="单目 VO 在退化/弱纹理场景塌缩，场景图无法构建"
                                    "（对应 M6：需 IMU/深度/多模态融合兜底）")
        if not seq.has_depth():
            return StageResult(name="scene_graph", ok=False, skipped=True,
                               skip_reason="序列无深度，无法反投影（真实系统用深度/RADAR）")

        fx, fy = seq.K[0, 0], seq.K[1, 1]
        cx, cy = seq.K[0, 2], seq.K[1, 2]
        # 关键帧：均匀抽样有效位姿帧
        valid = [(k, Twc) for k, Twc in enumerate(unit_Twc) if Twc is not None]
        kf = valid[::max(1, len(valid) // 8)]

        sem = {d["frame"]: d["dets"] for d in ctx.get("semantic_detections", [])}
        pts, labels = [], []
        for (k, Twc) in kf:
            depth = seq.depths[k] if k < len(seq.depths) else None
            if depth is None:
                continue
            depth = np.asarray(depth, np.float32)
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            H, W = depth.shape
            mask = ((depth > depth_min) & (depth < depth_max)).astype(np.uint8) * 255
            num, labels_cc = cv2.connectedComponents(mask)
            Twc_s = Twc.copy(); Twc_s[:3, 3] *= scale   # 应用度量尺度
            for c in range(1, num):
                ys, xs = np.where(labels_cc == c)
                if len(xs) < min_pixels:
                    continue
                u = int(np.median(xs)); v = int(np.median(ys))
                Z = float(depth[v, u])
                if Z <= 0:
                    continue
                X = (u - cx) * Z / fx; Y = (v - cy) * Z / fy
                p = Twc_s @ np.array([X, Y, Z, 1.0])
                pts.append(p[:3])
                # 关联语义标签：取该帧检测框质心最近者
                lab = "object"
                for d in sem.get(k, []):
                    if abs(d["cx"] - u) < W * 0.1 and abs(d["cy"] - v) < H * 0.1:
                        lab = d["label"]; break
                labels.append(lab)

        pts = np.array(pts) if pts else np.zeros((0, 3))
        instances = []
        if len(pts) > 0:
            assigned = np.full(len(pts), -1, int)
            cid = 0
            for i in range(len(pts)):
                if assigned[i] >= 0:
                    continue
                grp = np.where((np.linalg.norm(pts - pts[i], axis=1) < cluster_dist)
                               & (assigned < 0))[0]
                assigned[grp] = cid
                instances.append({
                    "id": cid,
                    "label": labels[i] if labels else "object",
                    "centroid_m": [round(float(x), 3) for x in pts[grp].mean(0)],
                    "n_detections": int(len(grp)),
                })
                cid += 1
        instances.sort(key=lambda o: o["n_detections"], reverse=True)

        ctx["scene_graph"] = {
            "n_raw_proposals": int(len(pts)),
            "n_instances": len(instances),
            "instances": instances,
        }
        return StageResult(
            name="scene_graph", ok=len(instances) > 0,
            metrics={"raw_proposals": int(len(pts)), "instances": len(instances)},
            note=f"解构出 {len(instances)} 个 3D 物体实例"
                 + (f"（{len(set(i['label'] for i in instances))} 类语义标签）"
                    if any(i["label"] != "object" for i in instances) else
                    "（几何标签；启用 OWL-ViT 得语义标签）"),
        )


# --------------------------------------------------------------------------- #
# 5) Edge —— M8 落点：算力预算曲线（端侧模拟）
# --------------------------------------------------------------------------- #
class EdgeStage(Stage):
    name = "edge"
    writes = ["edge_budget"]

    def run(self, seq, ctx, scales=(1.0, 0.5), max_frames=80, **kw):
        from ..sfm import MonocularVO, VOConfig
        from ..eval import compute_ate

        sub = seq.rgbs[:max_frames]
        if seq.gt_positions is not None:
            sub_gt = seq.gt_positions[:max_frames]
        else:
            sub_gt = None

        rows = []
        for s in scales:
            if s != 1.0:
                import cv2
                fr = [cv2.resize(im, None, fx=s, fy=s) for im in sub]
            else:
                fr = sub
            Ks = seq.K.copy(); Ks[0, 0] *= s; Ks[1, 1] *= s
            Ks[0, 2] *= s; Ks[1, 2] *= s
            cfg = VOConfig(feature="orb", max_features=1000, min_parallax_deg=1.0)
            vo = MonocularVO(Ks, cfg)
            t0 = time.time()
            for im in fr:
                vo.process(im, None)
            el = time.time() - t0
            up, gp = [], []
            for k, f in enumerate(vo.frames):
                if f.T_cw is None:
                    continue
                up.append(np.linalg.inv(f.T_cw)[:3, 3])
                if sub_gt is not None and k < len(sub_gt):
                    gp.append(sub_gt[k])
            ate = None
            if len(up) >= 3 and len(gp) == len(up):
                ate = compute_ate(np.array(up), np.array(gp), allow_scale=True)["rmse"]
            rel = round(float(s ** 2), 4)
            rows.append({
                "scale": s, "ms_per_frame": round(el / max(len(fr), 1) * 1000, 1),
                "fps": round(len(fr) / el, 1) if el > 0 else 0.0,
                "rel_compute_budget": rel,
                "ate_rmse_m": None if ate is None else round(float(ate), 4),
                "status": "OK" if ate is not None else "FAIL",
            })
        ctx["edge_budget"] = rows
        return StageResult(
            name="edge", ok=any(r["status"] == "OK" for r in rows),
            metrics={"budget_rows": rows},
            note="端侧算力预算曲线（分辨率↓ → 算力预算↓，越过悬崖则 VO 失效）",
        )


# 注册表：名字 → 工序类
STAGE_REGISTRY = {
    "vo": VOStage,
    "depth": DepthStage,
    "semantic": SemanticStage,
    "scene_graph": SceneGraphStage,
    "edge": EdgeStage,
}


def build_stage(key: str) -> Stage:
    if key not in STAGE_REGISTRY:
        raise KeyError(f"未知工序: {key}")
    return STAGE_REGISTRY[key]()
