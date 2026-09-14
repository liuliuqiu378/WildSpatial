"""
d1_real_world_eval.py — D1 真实恶劣环境评测（语义层，真实数据 + 真值）
================================================================================
对应 PROGRESS §2.2 D1：用新下载的真实恶劣数据接入 M4/M5/M7，替代/对照部分合成退化。

本脚本聚焦**语义层（M7）在真实恶劣环境下的表现**，用项目既有工具链（OWL-ViT 开放词汇
检测器，离线缓存于 /tmp/owlvit）跑两类真实数据：

  Part A · 水下目标检测（GT 支撑的定量评测）
    - 数据：LibreYOLO `underwater-objects-5v7p8`（RF100 水下子集），valid 集 1520 张 +
            YOLO 格式真值框，5 类：echinus/holothurian/scallop/starfish/waterweeds。
    - 方法：OWL-ViT 开放词汇，用这 5 个类名做文本查询。
    - 评测：以真值框为基准，IoU≥0.5 匹配，算**逐类 precision / recall** + 图像级命中率。
    - 价值：这是项目首个「真实恶劣环境 + 真值」的定量语义结果，对照 M7 此前
            "TUM 干净室内 39 框" 的定性结论，量化"域偏移 + 细粒度 + 散射"对检测的伤害。

  Part B · 夜间城市驾驶检测（定性展示）
    - 数据：NightCity `sample/image`（低光照真实采集）。
    - 方法：OWL-ViT 用通用查询（car/person/bicycle/traffic light/bus/truck）。
    - 展示：低光下检测到的物体（定性，无真值）。

产出于 `experiments/D1_real_world/`：figs/（画廊）+ metrics.json + README.md。

运行（det 环境，transformers 4.48 原生支持 OWL-ViT）：
  /home/hmn-cjy/miniforge3/envs/det/bin/python scripts/d1_real_world_eval.py
"""
import json
import os
import glob
import random

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments/D1_real_world")
FIGS = os.path.join(OUT, "figs")
os.makedirs(FIGS, exist_ok=True)

# ---- 数据路径 ----
UW_ROOT = os.path.join(ROOT, "data/raw/ms/LibreYOLO__underwater-objects-5v7p8")
UW_VAL_IMG = os.path.join(UW_ROOT, "valid", "images")
UW_VAL_LBL = os.path.join(UW_ROOT, "valid", "labels")
NC_SAMPLE = os.path.join(ROOT, "data/raw/ms/OmniData__NightCity", "sample", "image")

# 水下 5 类（RF100 underwater-objects-5v7p8，见 data.yaml）
UW_CLASSES = ["echinus", "holothurian", "scallop", "starfish", "waterweeds"]
NC_QUERIES = ["a car", "a person", "a bicycle", "a traffic light", "a bus", "a truck"]

DETECT_THRESH = 0.1
IOU_THR = 0.5
UW_N = 300          # 水下评测抽样数（valid 共 1520，抽样保证统计可信且快）
NC_N = 12           # 夜间展示抽样数
SEED = 42


def yolo_to_xyxy(label_path, w, h):
    boxes, cls = [], []
    if not os.path.exists(label_path):
        return boxes, cls
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            cx, cy, bw, bh = (float(p) for p in parts[1:5])
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            boxes.append([x1, y1, x2, y2])
            cls.append(c)
    return boxes, cls


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / union


@torch.no_grad()
def detect(proc, model, queries, pil, device):
    inputs = proc(text=queries, images=pil, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    out = model(**inputs)
    target_sizes = torch.tensor([pil.size[::-1]])
    res = proc.post_process_object_detection(outputs=out, threshold=DETECT_THRESH,
                                             target_sizes=target_sizes)[0]
    dets = []
    for score, box, lab in zip(res["scores"], res["boxes"], res["labels"]):
        if score < DETECT_THRESH:
            continue
        x1, y1, x2, y2 = box.tolist()
        dets.append((float(score), int(lab), [x1, y1, x2, y2]))
    return dets


def main():
    random.seed(SEED)
    MODEL_DIR = os.environ.get("OWLVIT_DIR", "/tmp/owlvit")
    print(f"[load] OWL-ViT from {MODEL_DIR} (offline)...")
    proc = OwlViTProcessor.from_pretrained(MODEL_DIR)
    model = OwlViTForObjectDetection.from_pretrained(MODEL_DIR)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[device] {device}")

    # =================== Part A · 水下 GT 支撑评测 ===================
    uw_files = sorted(glob.glob(os.path.join(UW_VAL_IMG, "*.jpg")))
    uw_files = random.sample(uw_files, min(UW_N, len(uw_files)))
    # 统计容器
    tp = [0] * len(UW_CLASSES)
    fp = [0] * len(UW_CLASSES)
    gt_cnt = [0] * len(UW_CLASSES)
    img_with_gt = 0
    img_pred_fire = 0
    gallery = []  # (img_bgr, gt_boxes, gt_cls, preds, fname)

    for fp_img in uw_files:
        fname = os.path.basename(fp_img)
        lbl = os.path.join(UW_VAL_LBL, os.path.splitext(fname)[0] + ".txt")
        img_bgr = cv2.imread(fp_img)
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        gt_boxes, gt_cls = yolo_to_xyxy(lbl, w, h)
        for c in gt_cls:
            if 0 <= c < len(UW_CLASSES):
                gt_cnt[c] += 1
        if gt_cls:
            img_with_gt += 1
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        dets = detect(proc, model, UW_CLASSES, pil, device)
        if dets:
            img_pred_fire += 1
        # 匹配：每个 GT 找同类 IoU 最大且≥阈值的 pred
        matched_pred = set()
        for gi, (gb, gc) in enumerate(zip(gt_boxes, gt_cls)):
            if not (0 <= gc < len(UW_CLASSES)):
                continue
            best, bi = -1, -1
            for pi, (_, pc, pb) in enumerate(dets):
                if pi in matched_pred or pc != gc:
                    continue
                v = iou(gb, pb)
                if v > best:
                    best, bi = v, pi
            if bi >= 0 and best >= IOU_THR:
                tp[gc] += 1
                matched_pred.add(bi)
        for _, pc, _ in dets:
            if pc not in matched_pred and 0 <= pc < len(UW_CLASSES):
                fp[pc] += 1
        if len(gallery) < 12 and (gt_cls or dets):
            gallery.append((img_bgr, gt_boxes, gt_cls, dets, fname))

    per_class = {}
    for i, c in enumerate(UW_CLASSES):
        p = tp[i] / max(1, tp[i] + fp[i])
        r = tp[i] / max(1, gt_cnt[i])
        per_class[c] = {"gt": gt_cnt[i], "tp": tp[i], "fp": fp[i],
                        "precision": round(p, 3), "recall": round(r, 3)}
    mean_recall = round(sum(tp) / max(1, sum(gt_cnt)), 3)
    mean_precision = round(sum(tp) / max(1, sum(tp) + sum(fp)), 3)
    uw_metrics = {
        "dataset": "LibreYOLO underwater-objects-5v7p8 (RF100)",
        "tool": "google/owlvit-base-patch32 (open-vocabulary)",
        "n_images_eval": len(uw_files),
        "iou_thr": IOU_THR, "score_thr": DETECT_THRESH,
        "per_class": per_class,
        "mean_recall": mean_recall, "mean_precision": mean_precision,
        "image_level": {"images_with_gt": img_with_gt,
                        "images_pred_fired": img_pred_fire,
                        "pred_fire_rate": round(img_pred_fire / max(1, len(uw_files)), 3)},
    }
    print(f"[A] underwater mean recall={mean_recall} precision={mean_precision}")
    print(f"[A] per-class={per_class}")

    # 画廊：绿=GT，红=OWL-ViT 预测
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for ax, item in zip(axes.ravel(), gallery):
        img_bgr, gt_boxes, gt_cls, dets, fname = item
        vis = img_bgr.copy()
        for gb, gc in zip(gt_boxes, gt_cls):
            if 0 <= gc < len(UW_CLASSES):
                x1, y1, x2, y2 = [int(v) for v in gb]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        for s, pc, pb in dets:
            if 0 <= pc < len(UW_CLASSES):
                x1, y1, x2, y2 = [int(v) for v in pb]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 120), 2)
                cv2.putText(vis, f"{UW_CLASSES[pc]} {s:.2f}", (x1, max(0, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 120), 1)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(fname, fontsize=7)
        ax.axis("off")
    fig.suptitle("Underwater open-vocab detection: GREEN=ground-truth (YOLO), "
                 "PINK=OWL-ViT predictions\nReal scattering/backscatter domain shift vs clean TUM",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "underwater_det.png"), dpi=110)
    plt.close(fig)

    # =================== Part B · 夜间定性展示 ===================
    nc_files = sorted(glob.glob(os.path.join(NC_SAMPLE, "*.jpg")))
    nc_files = random.sample(nc_files, min(NC_N, len(nc_files)))
    nc_gallery = []
    for fp_img in nc_files:
        img_bgr = cv2.imread(fp_img)
        if img_bgr is None:
            continue
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        dets = detect(proc, model, NC_QUERIES, pil, device)
        nc_gallery.append((img_bgr, dets, os.path.basename(fp_img)))

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for ax, (img_bgr, dets, fname) in zip(axes.ravel(), nc_gallery):
        vis = img_bgr.copy()
        for s, pc, pb in dets:
            x1, y1, x2, y2 = [int(v) for v in pb]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 120), 2)
            cv2.putText(vis, f"{NC_QUERIES[pc]} {s:.2f}", (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 120), 1)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{fname} ({len(dets)} hits)", fontsize=7)
        ax.axis("off")
    fig.suptitle("NightCity low-light driving: OWL-ViT open-vocab detection (qualitative, no GT)\n"
                 "Backlit / glare / low SNR — the same regime M3 synthesizes as 'low_light'",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "night_det.png"), dpi=110)
    plt.close(fig)
    nc_total = sum(len(d[1]) for d in nc_gallery)
    print(f"[B] night images={len(nc_gallery)} total_hits={nc_total}")

    # =================== 写盘 ===================
    metrics = {"partA_underwater": uw_metrics,
               "partB_night": {"dataset": "NightCity sample", "n_images": len(nc_gallery),
                               "total_hits": nc_total, "queries": NC_QUERIES}}
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[done] -> {OUT}")


if __name__ == "__main__":
    main()
