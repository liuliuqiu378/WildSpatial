"""
landscape_demo.py — 领域全景教程的「真实工具 + 真实数据」实证脚本
=================================================================
对应 docs/00_LANDSCAPE.md §1.1（检测）/ §1.2（分割）。

用到的真实工具（均已在本机，无需联网下载）：
  - facebook/detr-resnet-50  （Transformers 检测器，COCO 预训练，HF 缓存已就绪）
  - OpenCV GrabCut           （经典交互式分割，OpenCV 内置，零权重）

用到的真实数据：
  - TUM RGB-D `fr1/desk` 的 rgb/ 与 depth/（含真值）

产出：
  experiments/landscape_showcase/figs/detection.png   检测框 + 类别
  experiments/landscape_showcase/figs/segmentation.png  GrabCut 实例分割
  experiments/landscape_showcase/metrics.json          量化指标
"""
import json, glob, os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ = os.path.join(ROOT, "data/raw/rgbd_dataset_freiburg1_desk")
OUT = os.path.join(ROOT, "experiments/landscape_showcase")
FIGS = os.path.join(OUT, "figs")
os.makedirs(FIGS, exist_ok=True)

# COCO 91 类（index 0 = 背景 N/A），与 DETR 输出类别索引对齐
COCO = ['N/A','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
'traffic light','fire hydrant','N/A','stop sign','parking meter','bench','bird','cat','dog',
'horse','sheep','cow','elephant','bear','zebra','giraffe','N/A','backpack','umbrella','N/A','N/A',
'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
'baseball glove','skateboard','surfboard','tennis racket','bottle','N/A','wine glass','cup','fork',
'knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
'donut','cake','chair','couch','potted plant','bed','N/A','dining table','N/A','toilet','N/A','tv',
'laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
'N/A','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']


def load_detr():
    print("[load] DETR facebook/detr-resnet-50 (offline from HF cache)...")
    proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model.eval()
    return proc, model


@torch.no_grad()
def detect(proc, model, img_bgr, thr=0.85):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    inp = proc(images=pil, return_tensors="pt")
    out = model(**inp)
    h, w = img_bgr.shape[:2]
    logits = out.logits[0]
    boxes = out.pred_boxes[0]
    prob = logits.softmax(-1)
    scores, idx = prob[..., :-1].max(-1)  # idx in 0..89 -> coco id = idx+1
    mask = scores > thr
    res = []
    for s, c, b in zip(scores[mask], idx[mask], boxes[mask]):
        cx, cy, bw, bh = b.tolist()
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        res.append((float(s), int(c) + 1, [x1, y1, x2, y2]))
    return res


def grabcut_from_box(img_bgr, box, iters=5):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_bgr.shape[1], x2), min(img_bgr.shape[0], y2)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return np.zeros(img_bgr.shape[:2], np.uint8)
    rect = (x1, y1, x2 - x1, y2 - y1)
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)


def main():
    rgb_files = sorted(glob.glob(os.path.join(SEQ, "rgb", "*.png")))
    # 取 4 帧做展示（首尾各取，中间均匀）
    pick = [0, len(rgb_files) // 3, 2 * len(rgb_files) // 3, len(rgb_files) - 1]
    pick = sorted(set(min(i, len(rgb_files) - 1) for i in pick))
    frames = [rgb_files[i] for i in pick]

    proc, model = load_detr()

    det_fig, det_axes = plt.subplots(2, 2, figsize=(14, 10))
    seg_fig, seg_axes = plt.subplots(2, 2, figsize=(14, 10))
    det_axes = det_axes.ravel()
    seg_axes = seg_axes.ravel()

    metrics = {"sequence": "TUM fr1/desk", "detector": "facebook/detr-resnet-50",
               "segmenter": "OpenCV GrabCut (seeded by DETR box)", "frames": []}
    all_classes = {}

    for ax_i, fpath in enumerate(frames):
        img = cv2.imread(fpath)
        dets = detect(proc, model, img, thr=0.85)
        # 记录指标
        fm = {"file": os.path.basename(fpath), "num_detections": len(dets),
              "classes": [COCO[c] for _, c, _ in dets]}
        metrics["frames"].append(fm)
        for _, c, _ in dets:
            all_classes[COCO[c]] = all_classes.get(COCO[c], 0) + 1

        # 检测图
        vis = img.copy()
        best_box = None
        best_score = -1
        for s, c, box in dets:
            x1, y1, x2, y2 = box
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
            cv2.putText(vis, f"{COCO[c]} {s:.2f}", (int(x1), int(y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
            if s > best_score:
                best_score, best_box = s, box
        det_axes[ax_i].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        det_axes[ax_i].set_title(f"{os.path.basename(fpath)}\n{len(dets)} dets: " +
                                 ", ".join(sorted(set(COCO[c] for _, c, _ in dets)))[:60])
        det_axes[ax_i].axis("off")

        # 分割图（取置信度最高的框做 GrabCut 种子）
        seg_vis = img.copy()
        if best_box is not None:
            fg = grabcut_from_box(img, best_box, iters=5)
            seg_vis[fg == 0] = (seg_vis[fg == 0] * 0.25).astype(np.uint8)  # 背景压暗
            # 叠加绿框
            x1, y1, x2, y2 = [int(v) for v in best_box]
            cv2.rectangle(seg_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            best_cls = [c for s, c, b in dets if b == best_box][0]
            fm["grabcut_target"] = COCO[best_cls]
        seg_axes[ax_i].imshow(cv2.cvtColor(seg_vis, cv2.COLOR_BGR2RGB))
        seg_axes[ax_i].set_title(f"GrabCut (seed={fm.get('grabcut_target','-')})")
        seg_axes[ax_i].axis("off")

    metrics["class_histogram"] = all_classes
    metrics["total_detections"] = sum(m["num_detections"] for m in metrics["frames"])

    det_fig.tight_layout()
    det_fig.savefig(os.path.join(FIGS, "detection.png"), dpi=110)
    seg_fig.tight_layout()
    seg_fig.savefig(os.path.join(FIGS, "segmentation.png"), dpi=110)
    plt.close(det_fig); plt.close(seg_fig)

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[done] detection.png / segmentation.png -> {FIGS}")
    print(f"[done] total detections={metrics['total_detections']}, classes={all_classes}")


if __name__ == "__main__":
    main()
