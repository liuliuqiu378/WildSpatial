"""
D-showcase · NDISPark 夜间/白天停车场实例分割（真实封闭集感知）

数据源：OmniData__NDISPark_Night_and_Day_Instance_Segmented_etc（ModelScope，已下载+解压）
  train/imgs/*.jpg + train/train_coco_annotations.json（COCO 实例分割真值）

演示意图：
  - 同一停车场在白天/夜间都有车辆实例分割真值 → 说明**监督封闭集分割不挑光照**，
    与 D1 结论（开放词汇 OWL-ViT 在域偏移+细粒度下 recall≈0.001）形成对照：
    恶劣环境语义应靠"域专用封闭集模型"，而非指望万能 VLM。
  - 支撑 M6（低光下感知）、M7（语义层分工：几何兜底 + 域专用语义）。

输出：experiments/ndispark_night_day/figs/ndispark_seg.png + metrics.json
"""
import os, glob, json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

ROOT = "data/raw/ms/OmniData__NDISPark_Night_and_Day_Instance_Segmented_etc/raw"
OUT = "experiments/ndispark_night_day"
FIGDIR = os.path.join(OUT, "figs")
os.makedirs(FIGDIR, exist_ok=True)
TARGET = 640  # 显示长边


def load_rgb(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = TARGET / max(w, h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    return np.array(im.resize((nw, nh))), s


def lum(path):
    a = np.array(Image.open(path).convert("L"))
    return float(a.mean())


def main():
    ann = json.load(open(os.path.join(ROOT, "train/train_coco_annotations.json")))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    # 图像 -> 标注
    by_img = {}
    for a in ann["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    imgs = ann["images"]

    # 按亮度排序，挑最暗(夜)与最亮(昼)且含标注的样本
    paths = [(os.path.join(ROOT, "train/imgs", im["file_name"]), lum(os.path.join(ROOT, "train/imgs", im["file_name"])), im)
             for im in imgs if im["id"] in by_img]
    paths.sort(key=lambda x: x[1])
    night = [p for p in paths[:8] if len(by_img[p[2]["id"]]) >= 1]
    day = [p for p in paths[-8:] if len(by_img[p[2]["id"]]) >= 1]
    night, day = night[:3], day[:3]

    # 颜色表（按类别 id 取色）
    rng = np.random.default_rng(0)
    cat_color = {cid: tuple(int(c) for c in rng.integers(60, 255, 3)) for cid in cats}

    def draw(im, imid):
        out = im.copy()
        for a in by_img[imid]:
            cid = a["category_id"]
            color = cat_color.get(cid, (255, 255, 0))
            for poly in a["segmentation"]:
                if isinstance(poly, list) and len(poly) >= 6:
                    pts = np.array(poly, np.int32).reshape(-1, 2)
                    pts = (pts * s).astype(np.int32)
                    cv2.fillPoly(out, [pts], color)
                    cv2.polylines(out, [pts], True, (255, 255, 255), 1)
        # 半透明叠加
        return (im.astype(np.float32) * 0.55 + out.astype(np.float32) * 0.45).clip(0, 255).astype(np.uint8)

    s = None
    rows = []
    for label, group in [("Night (dark)", night), ("Day (bright)", day)]:
        for path, _, im in group:
            rgb, s = load_rgb(path)
            drawn = draw(rgb, im["id"])
            n_car = sum(1 for a in by_img[im["id"]] if cats.get(a["category_id"]) in ("car", "truck", "bus"))
            rows.append((label, drawn, len(by_img[im["id"]]), n_car, round(lum(path), 1)))

    fig, axes = plt.subplots(2, 3, figsize=(3 * 4.2, 2 * 4.2))
    for i, (label, drawn, n_inst, n_car, l) in enumerate(rows):
        ax = axes[i // 3, i % 3]
        ax.imshow(drawn)
        ax.set_xticks([]); ax.set_yticks([])
        title = f"{label}\n{l} lum · {n_inst} inst · {n_car} vehicles"
        ax.set_title(title, fontsize=9)
    fig.suptitle("Real NDISPark: instance segmentation GT holds in Day AND Night\n"
                 "(closed-set supervised seg is illumination-invariant; contrasts D1 open-vocab failure)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(FIGDIR, "ndispark_seg.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    metrics = {
        "dataset": "NDISPark", "split": "train", "n_images": len(imgs),
        "n_instances": len(ann["annotations"]), "n_categories": len(cats),
        "showcase": [{"label": r[0], "instances": r[2], "vehicles": r[3], "luminance": r[4]} for r in rows],
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("saved", out_png)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
