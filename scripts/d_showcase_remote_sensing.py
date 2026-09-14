"""
D-showcase · 真实遥感变化检测（multi-modal + temporal）

数据源：Mriris__remote-sensing-change-detection（ModelScope，已下载）
  A/ 高分二号 事前光学（基准）   B/ 高分三号 事后 SAR
  C/ 哨兵二号 原始事后光学      D/ 哨兵二号 校正后事后光学
  E/ 黑白二值像素变化图         json/ Labelme 多边形真值（label=changed）

本脚本把"未消费的真实多模态时序数据"做成可看图，演示：
  - 光学(事前) ↔ 光学(事后) 的地面变化（城市建筑/地物）
  - 同场景 SAR（不受光照/云影响）作为"几何/结构"补充通道
  - 二值变化图 + 多边形标注 = 像素级真值
意义：呼应 M6 多模态融合（光学+SAR）、M9 解构世界（时序地物变化）、F5（多传感器）。

输出：experiments/remote_sensing_cd/figs/remote_sensing_cd.png + metrics.json
"""
import os, glob, json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly

ROOT = "data/raw/ms/Mriris__remote-sensing-change-detection"
OUT = "experiments/remote_sensing_cd"
FIGDIR = os.path.join(OUT, "figs")
os.makedirs(FIGDIR, exist_ok=True)
TARGET = 560  # 显示长边


def load_rgb(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = TARGET / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return np.array(im.resize((nw, nh))), scale, (nw, nh)


def load_mask(path, size):
    im = Image.open(path).convert("L").resize(size)
    return (np.array(im) > 127).astype(np.uint8)


def load_polys(json_path, scale):
    if not os.path.exists(json_path):
        return []
    d = json.load(open(json_path))
    polys = []
    for s in d.get("shapes", []):
        if s.get("label") == "changed" and s.get("shape_type") == "polygon":
            pts = [(p[0] * scale, p[1] * scale) for p in s["points"]]
            polys.append(pts)
    return polys


def draw_overlay(rgb, mask, polys):
    import cv2
    out = rgb.astype(np.float32).copy()
    # 变化区红色半透明叠加
    out[mask.astype(bool)] = out[mask.astype(bool)] * 0.45 + np.array([255, 0, 0], np.float32) * 0.55
    out = np.clip(out, 0, 255).astype(np.uint8)
    # 画 GT 多边形轮廓（绿色）
    for pts in polys:
        if len(pts) >= 3:
            arr = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [arr], isClosed=True, color=(0, 255, 0), thickness=1)
    return out, polys


def main():
    names = sorted(glob.glob(os.path.join(ROOT, "A", "*.tif")))
    names = [os.path.basename(p)[:-6] for p in names]  # strip _A.tif
    samples = names[:4]
    metrics = {"n_total_groups": len(names), "samples": []}

    rows, caps = [], []
    for nm in samples:
        a_rgb, sa, sz = load_rgb(os.path.join(ROOT, "A", f"{nm}_A.tif"))
        d_rgb, sd, _ = load_rgb(os.path.join(ROOT, "D", f"{nm}_D.tif"))
        b_rgb, sb, _ = load_rgb(os.path.join(ROOT, "B", f"{nm}_B.tif"))
        e_mask = load_mask(os.path.join(ROOT, "E", f"{nm}_E.png"), sz)
        polys = load_polys(os.path.join(ROOT, "json", f"{nm}_A.json"), sa)
        ov, _ = draw_overlay(a_rgb, e_mask, polys)

        rows.append((a_rgb, d_rgb, b_rgb, ov))
        frac = float(e_mask.mean())
        metrics["samples"].append({
            "name": nm, "size_HW": list(sz), "changed_pixel_frac": round(frac, 4),
            "changed_polygons": len(polys),
        })
        caps.append(f"{nm}\nchanged {frac*100:.1f}% · {len(polys)} polys")

    # 排版：samples 行 × [pre, post, SAR, overlay] 列
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(4 * 4.2, n * 4.0))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    titles = ["① Pre-event optical (Gaofen-2)", "② Post-event optical (Sentinel-2, corrected)",
              "③ Post-event SAR (Gaofen-3, light-invariant)", "④ Change overlay (red=changed, +GT polygons)"]
    for i, (row, cap) in enumerate(zip(rows, caps)):
        for j, (img, t) in enumerate(zip(row, titles)):
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(t, fontsize=9)
            if j == 0:
                ax.set_ylabel(cap, fontsize=8)
    fig.suptitle("Real remote-sensing change detection (multi-modal + temporal)\n"
                 "optical pre/post + SAR structure + pixel-level change GT", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(FIGDIR, "remote_sensing_cd.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("saved", out_png)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
