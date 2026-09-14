"""
m7_semantic_demo.py — M7 语义层最小实证（开放词汇检测 = VLM 进场）
========================================================================
对应 docs/00_LANDSCAPE.md §2.2（开放词汇 3D / VLM）与 §3（VLM 融合增益）。

真实工具：google/owlvit-base-patch32（OWL-ViT，文本提示式开放词汇检测器，
          transformers 原生支持，区别于 DETR 的「封闭 80 类」）。
真实数据：TUM RGB-D `fr1/desk` 的 rgb/ 帧。

它要证明的（四段式 ④ 真实验证）：
  §3 增益#2「开放场景理解」+ §2.2「用自然语言查询场景」——
  检测器不再受训练类别限制，用一句自然语言（"a red cup" /
  "the object on the left" / "a book"）就能去图像里找，而 DETR 只能认 COCO 80 类。
"""
import json, glob, os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ = os.path.join(ROOT, "data/raw/rgbd_dataset_freiburg1_desk")
OUT = os.path.join(ROOT, "experiments/M7_semantic")
FIGS = os.path.join(OUT, "figs")
os.makedirs(FIGS, exist_ok=True)


def main():
    # 同一组 4 帧，与 §1.1 DETR 演示保持一致，便于对照
    rgb_files = sorted(glob.glob(os.path.join(SEQ, "rgb", "*.png")))
    pick = [0, len(rgb_files) // 3, 2 * len(rgb_files) // 3, len(rgb_files) - 1]
    pick = sorted(set(min(i, len(rgb_files) - 1) for i in pick))
    frames = [rgb_files[i] for i in pick]

    MODEL_DIR = os.environ.get("OWLVIT_DIR", "/tmp/owlvit")
    print(f"[load] OWL-ViT from {MODEL_DIR} (offline)...")
    proc = OwlViTProcessor.from_pretrained(MODEL_DIR)
    model = OwlViTForObjectDetection.from_pretrained(MODEL_DIR)
    model.eval()

    # 开放词汇查询：既有 COCO 常见词，也有「自由文本 / 关系 / 属性」式查询
    queries = ["a bottle", "a cup", "a keyboard", "a book",
               "a remote control", "a red object", "the object on the left"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.ravel()
    metrics = {"sequence": "TUM fr1/desk", "tool": "google/owlvit-base-patch32 (open-vocabulary)",
               "queries": queries, "frames": []}
    query_hit = {q: 0 for q in queries}

    @torch.no_grad()
    def detect(text_queries, image):
        inputs = proc(text=text_queries, images=image, return_tensors="pt")
        out = model(**inputs)
        # OWL-ViT: 返回单张图的 dict，labels 指向 text_queries 的索引
        target_sizes = torch.tensor([image.size[::-1]])
        res = proc.post_process_object_detection(outputs=out, threshold=0.1,
                                                 target_sizes=target_sizes)[0]
        dets = []
        for score, box, lab in zip(res["scores"], res["boxes"], res["labels"]):
            if score < 0.1:
                continue
            x1, y1, x2, y2 = box.tolist()
            dets.append((float(score), text_queries[int(lab)], [x1, y1, x2, y2]))
        return dets

    for ax_i, fpath in enumerate(frames):
        img_bgr = cv2.imread(fpath)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        dets = detect(queries, pil)

        vis = img_bgr.copy()
        for s, q, (x1, y1, x2, y2) in dets:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 120), 2)
            cv2.putText(vis, f"{q} {s:.2f}", (int(x1), int(y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 120), 1)
            query_hit[q] = query_hit.get(q, 0) + 1
        axes[ax_i].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[ax_i].set_title(f"{os.path.basename(fpath)} | {len(dets)} open-vocab hits", fontsize=10)
        axes[ax_i].axis("off")
        metrics["frames"].append({"file": os.path.basename(fpath),
                                  "num_hits": len(dets),
                                  "queries_fired": sorted({q for _, q, _ in dets})})

    fig.suptitle("OWL-ViT Open-Vocabulary Detection (natural-language queries, not limited to training classes)\n"
                 "vs §1.1 DETR: DETR only knows COCO-80; OWL-ViT accepts arbitrary text prompts",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "open_vocab_detection.png"), dpi=110)
    plt.close(fig)

    metrics["query_hit_counts"] = query_hit
    metrics["total_hits"] = sum(query_hit.values())
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[done] open_vocab_detection.png -> {FIGS}")
    print(f"[done] total open-vocab hits={metrics['total_hits']}, per-query={query_hit}")


if __name__ == "__main__":
    main()
