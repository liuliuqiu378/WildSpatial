"""M9 · 综合闭环（语义增强版）：几何×语义真闭环，把世界解构成带标签的 3D 场景图
============================================================================

这是 `m9_scene_graph.py`（朴素版，class 字段为 unknown）的**升级闭环**：
把 `M7` 的 OWL-ViT 开放词汇检测真正接入 `M1/M2` 的几何，让每个 3D 物体候选
带上一句自然语言能描述的 **class 标签**——这正是 `docs/M9_*.md` §5「下一步 2」。

流水线（与朴素版相同的几何层 + 新增语义层）：
    ① 定位  M1 的 MonocularVO → 关键帧相机位姿 T_wc
    ② 语义  M7 的 OWL-ViT     → 关键帧 2D 开放词汇框（"a cup" / "a keyboard" …）
    ③ 解构  框内中位深度 + 反投影 → 相机系 3D → 世界系 3D（带 class）
    ④ 聚合  同类 + 3D 近距贪心聚类 → 持久物体实例 {class, xyz, n_detections}

离线性：OWL-ViT 权重来自本地缓存（默认 /tmp/owlvit，可由 OWLVIT_DIR 覆盖），
        不触网。VO / 深度来自 TUM fr1/desk。

用法：
    PYTHONPATH=src python scripts/m9_scene_graph_semantic.py --seq fr1/desk
产出 experiments/M9_closed_loop/figs/scene_graph_semantic.png
     + scene_graph_semantic.json + metrics_semantic.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from wildspatial.data.tum import TUMDataset
from wildspatial.sfm import MonocularVO, VOConfig
from wildspatial.viz import setup_plot_style

setup_plot_style()

TUM_K = {
    "fr1": (517.3, 516.5, 318.6, 255.3),
    "fr2": (520.9, 521.0, 325.1, 249.7),
    "fr3": (535.4, 539.2, 320.1, 247.6),
}
SEQUENCES = {
    "fr1/desk": "rgbd_dataset_freiburg1_desk",
    "fr1/room": "rgbd_dataset_freiburg1_room",
}
# 室内常见开放词汇查询（覆盖 fr1/desk 典型物体，也留几个"自由文本"式查询）
QUERIES = ["a bottle", "a cup", "a keyboard", "a book", "a laptop",
           "a monitor", "a mouse", "a chair", "a notebook", "a phone",
           "a computer", "a person"]
OWL_THRESH = 0.25     # 比 m7 演示的 0.1 更稳，减少误检
NMS_IOU = 0.5


def iou(b1, b2):
    x1, y1, x2, y2 = b1
    xa, ya, xb, yb = b2
    ix1, iy1 = max(x1, xa), max(y1, ya)
    ix2, iy2 = min(x2, xb), min(y2, yb)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a1 = max(0, x2 - x1) * max(0, y2 - y1)
    a2 = max(0, xb - xa) * max(0, yb - ya)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


@torch.no_grad()
def detect(proc, model, text_queries, image):
    inputs = proc(text=text_queries, images=image, return_tensors="pt")
    out = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    # transformers 5.x：grounded 后处理直接回传 text_labels，避免标签索引错配
    res = proc.post_process_grounded_object_detection(
        outputs=out, threshold=OWL_THRESH, target_sizes=target_sizes,
        text_labels=[text_queries])[0]
    raw = []
    for score, box, txt in zip(res["scores"], res["boxes"], res["text_labels"]):
        if score < OWL_THRESH:
            continue
        x1, y1, x2, y2 = box.tolist()
        raw.append((float(score), txt, [x1, y1, x2, y2]))
    # NMS：同一物体可能被多个 query 命中，保留 score 最高的框
    raw.sort(key=lambda x: -x[0])
    kept = []
    for s, q, b in raw:
        if all(iou(b, kb) < NMS_IOU for _, _, kb in kept):
            kept.append((s, q, b))
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="fr1/desk", choices=list(SEQUENCES))
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--depth-min", type=float, default=0.3)
    ap.add_argument("--depth-max", type=float, default=4.0)
    ap.add_argument("--cluster-dist", type=float, default=0.5,
                    help="同类 3D 质心聚类阈值 (m)")
    ap.add_argument("--vis-min-detections", type=int, default=2,
                    help="可视化时只显示被 ≥N 帧观测到的稳定实例（不影响 JSON 输出）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = os.path.join(ROOT, "data", "raw", SEQUENCES[args.seq])
    if not os.path.exists(root):
        print(f"[✗] 序列不存在: {root}"); return 1
    prefix = args.seq.split("/")[0]
    fx, fy, cx, cy = TUM_K[prefix]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    ds = TUMDataset(root)
    n = len(ds) if args.frames <= 0 else min(args.frames, len(ds))
    indices = list(range(0, n, max(1, args.stride)))

    # ① 几何层：VO 位姿
    cfg = VOConfig(feature="sift", max_features=3000, min_parallax_deg=1.0)
    vo = MonocularVO(K, cfg)
    t0 = time.time()
    for i in indices:
        vo.process(ds[i]["rgb"], None)
    vo_el = time.time() - t0
    fps = len(indices) / vo_el if vo_el > 0 else 0.0

    poses = []
    for k, i in enumerate(indices):
        T_cw = vo.frames[k].T_cw
        if T_cw is None:
            continue
        poses.append((i, np.linalg.inv(T_cw)))
    if not poses:
        print("[✗] VO 未初始化"); return 1
    kf = poses[::max(1, len(poses) // 5)]

    # ② 语义层：OWL-ViT（离线）
    MODEL_DIR = os.environ.get("OWLVIT_DIR", "/tmp/owlvit")
    print(f"[load] OWL-ViT from {MODEL_DIR}")
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    proc = OwlViTProcessor.from_pretrained(MODEL_DIR)
    model = OwlViTForObjectDetection.from_pretrained(MODEL_DIR)
    model.eval()

    detections = []   # (class, pos_world[3], frame_idx)
    kf_vis = []       # (rgb_bgr, list_of (label,box), frame_idx) 用于可视化
    class_counts = {}
    for (fi, Twc) in kf:
        f = ds[fi]
        rgb = f["rgb"]
        depth = f.get("depth")
        if depth is None:
            continue
        depth = depth.astype(np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        H, W = depth.shape
        pil = Image.fromarray(rgb)
        dets = detect(proc, model, QUERIES, pil)

        vis_list = []
        for s, q, (x1, y1, x2, y2) in dets:
            uu = int(np.clip((x1 + x2) / 2, 0, W - 1))
            vv = int(np.clip((y1 + y2) / 2, 0, H - 1))
            # 框内有效深度中位数更稳（切片取框内区域）
            x1i, y1i, x2i, y2i = (int(round(x1)), int(round(y1)),
                                  int(round(x2)), int(round(y2)))
            x1i, x2i = max(0, x1i), min(W - 1, x2i)
            y1i, y2i = max(0, y1i), min(H - 1, y2i)
            sub = depth[y1i:y2i + 1, x1i:x2i + 1]
            vals = sub[(sub > args.depth_min) & (sub < args.depth_max)]
            if vals.size == 0:
                continue
            Z = float(np.median(vals))
            if Z <= 0:
                continue
            X = (uu - cx) * Z / fx
            Y = (vv - cy) * Z / fy
            p_world = Twc @ np.array([X, Y, Z, 1.0])
            detections.append((q, p_world[:3], fi))
            class_counts[q] = class_counts.get(q, 0) + 1
            vis_list.append((q, s, (x1, y1, x2, y2)))
        kf_vis.append((rgb, vis_list, fi))

    # ④ 同类 3D 聚类成持久实例
    instances = []
    by_class = {}
    for q, p, fi in detections:
        by_class.setdefault(q, []).append(p)
    cid = 0
    for q, pts in by_class.items():
        pts = np.array(pts)
        assigned = np.full(len(pts), -1, dtype=int)
        for i in range(len(pts)):
            if assigned[i] >= 0:
                continue
            d = np.linalg.norm(pts - pts[i], axis=1)
            grp = np.where((d < args.cluster_dist) & (assigned < 0))[0]
            assigned[grp] = cid
            instances.append({
                "id": cid, "class": q,
                "centroid_m": [round(float(x), 3) for x in pts[grp].mean(0)],
                "n_detections": int(len(grp)),
            })
            cid += 1
    instances.sort(key=lambda o: o["n_detections"], reverse=True)

    # ---------- 可视化 ----------
    out_dir = args.out or os.path.join(ROOT, "experiments", "M9_closed_loop")
    figs = os.path.join(out_dir, "figs")
    os.makedirs(figs, exist_ok=True)

    # 左：关键帧 OWL-ViT 可视化（最多 5 张，竖向排列）
    n_kf = len(kf_vis)
    fig = plt.figure(figsize=(15, 4.6 * max(1, n_kf)))
    for ri, (rgb, vis_list, fi) in enumerate(kf_vis):
        ax = fig.add_subplot(n_kf, 1, ri + 1)
        vis = rgb.copy()
        for q, s, (x1, y1, x2, y2) in vis_list:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 120), 2)
            cv2.putText(vis, f"{q} {s:.2f}", (int(x1), int(y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 120), 1)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"关键帧 #{ri} (frame {fi}) · OWL-ViT 开放词汇检测（{len(vis_list)} 框）",
                     fontsize=10)
        ax.axis("off")
    fig.suptitle(f"M9 语义增强 · 关键帧 OWL-ViT 检测（{n_kf} 关键帧，查询 {len(QUERIES)} 类）",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "kf_owl_det.png"), dpi=110)
    plt.close(fig)

    # 右/主图：世界系 3D 物体散点，按 class 着色 + 标签
    fig = plt.figure(figsize=(14, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    # 可视化只展示被多帧稳定观测到的实例（不影响 JSON 全量记录）
    vis_instances = [o for o in instances if o["n_detections"] >= args.vis_min_detections]
    if vis_instances:
        cmap = plt.get_cmap("tab10")
        cls_uniq = sorted({o["class"] for o in vis_instances})
        clr = {c: cmap(i % 10) for i, c in enumerate(cls_uniq)}
        for o in vis_instances:
            x, _, z = o["centroid_m"]
            ax1.scatter([x], [z], s=300, marker="*", color=clr[o["class"]],
                        edgecolors="black", linewidths=0.6, zorder=5)
            ax1.annotate(f"{o['class']}#{o['id']}", (x, z), fontsize=8,
                         xytext=(6, 4), textcoords="offset points")
        ax1.set_xlabel("X (m)"); ax1.set_ylabel("Z (m)")
        ax1.set_title(f"① 世界系 3D 物体（带真实 class 标签，{len(vis_instances)} 个，"
                      f"n_det≥{args.vis_min_detections}）", fontsize=11)
        ax1.set_aspect("equal"); ax1.grid(alpha=0.3)
        # 图例
        from matplotlib.patches import Patch
        ax1.legend(handles=[Patch(color=clr[c], label=c) for c in cls_uniq],
                   loc="upper right", fontsize=8, title="class")

        top = vis_instances[:12]
        names = [f"{o['class']}#{o['id']}" for o in top][::-1]
        xs = [o["centroid_m"][0] for o in top][::-1]
        ys = [o["centroid_m"][1] for o in top][::-1]
        zs = [o["centroid_m"][2] for o in top][::-1]
        yy = np.arange(len(top)); h = 0.26
        ax2.barh(yy + h, xs, height=h, color="#4C72B0", label="X")
        ax2.barh(yy, ys, height=h, color="#DD8452", label="Y")
        ax2.barh(yy - h, zs, height=h, color="#55A868", label="Z")
        ax2.set_yticks(yy); ax2.set_yticklabels(names, fontsize=8)
        ax2.axvline(0, color="gray", lw=0.8)
        ax2.set_xlabel("世界系坐标 (m)")
        ax2.set_title(f"② 稳定实例坐标（{len(vis_instances)} 个，前 {len(top)}）",
                      fontsize=11)
        ax2.legend(loc="lower right", fontsize=9); ax2.grid(alpha=0.3, axis="x")
    else:
        ax1.text(0.5, 0.5, "无带标签实例", ha="center")
        ax2.text(0.5, 0.5, "无实例", ha="center")
    fig.suptitle(f"M9 语义增强 · 几何×语义真闭环：把世界解构成带标签的 3D 物体"
                 f"（{args.seq}，{fps:.1f} fps @ CPU）", fontsize=13)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12)
    fig.savefig(os.path.join(figs, "scene_graph_semantic.png"), dpi=130)
    plt.close(fig)

    scene_graph = {
        "sequence": args.seq,
        "frame_rate_fps": round(fps, 1),
        "objects": [
            {"id": o["id"], "class": o["class"], "position_m": o["centroid_m"],
             "detections": o["n_detections"]}
            for o in instances
        ],
    }
    metrics = {
        "sequence": args.seq, "vo_fps": round(fps, 1), "keyframes_used": n_kf,
        "owl_queries": QUERIES, "owl_threshold": OWL_THRESH,
        "raw_detections": len(detections), "class_counts": class_counts,
        "scene_graph_objects": len(instances),
        "instances_top": instances[:10],
    }
    with open(os.path.join(out_dir, "scene_graph_semantic.json"), "w") as f:
        json.dump(scene_graph, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "metrics_semantic.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[✓] M9 语义增强 | VO {fps:.1f} fps | 关键帧 {n_kf}")
    print(f"    原始检测 {len(detections)} | 聚类实例 {len(instances)} 个")
    print(f"    每类命中: {class_counts}")
    for o in instances[:8]:
        print(f"    #{o['id']:2d} {o['class']:<12} 位置=({o['centroid_m'][0]:+.2f},"
              f"{o['centroid_m'][1]:+.2f},{o['centroid_m'][2]:+.2f}) m  出现 {o['n_detections']} 次")
    print(f"    产出 → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
