#!/usr/bin/env python
"""P1 四场景 · 真实数据集 showcase（把「模拟」换成「真实数据」的第一步）

为什么做这个（目的）
------------------
`docs/P1_field_projects.md` 讲了四个真实场景（园区扫地 / 酒店送物 / 自动驾驶 / 无人船），
但旗舰闭环里的**动态障碍是模拟的**、场景数据多来自 TUM/KITTI。本脚本把 ModelScope 上
新落地的**真实场景数据**可视化，让四场景从"讲故事"落到"看得见的真实素材"：

  · 园区扫地     → 扫地机器人第一视角（真实采集的垃圾/障碍样本）
  · 酒店送物     → 真实具身操作（LeRobot 格式，若已下）
  · 动态行人     → CVC-14 昼夜/红外行人（P1 动态避障的真实输入）
  · 无人船       → 水面漂浮物 + 船只检测（若已下）

产出
----
experiments/p1_datasets_showcase/figs/p1_scene_samples.png
experiments/p1_datasets_showcase/metrics.json
experiments/p1_datasets_showcase/README.md

诚实声明
--------
· 只展示**已实际下载**的数据；缺失的整块跳过并标注，绝不伪造。
· 样例为「数据集真实图像」，非本项目采集；仅用于说明"每个场景长什么样"。
"""

import os
import sys
import glob
import json
import argparse

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from wildspatial.viz import setup_plot_style
setup_plot_style()

RAW = os.path.join(ROOT, "data", "raw", "ms")


def imread_gray_safe(p):
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im


def collect_robotcleaner():
    d = os.path.join(RAW, "DatatangBeijing__190426ImagesofRobotCleanerPerspectiveCollectionData")
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg"))):
        im = imread_gray_safe(p)
        if im is not None:
            out.append((os.path.basename(p), im, 0))
    return out


def collect_cvc14():
    d = os.path.join(RAW, "OmniData__CVC-14")
    fs = sorted(glob.glob(os.path.join(d, "sample", "**", "*.jpg"), recursive=True))
    out = []
    for p in fs:
        im = imread_gray_safe(p)
        if im is not None:
            out.append((os.path.basename(p), im, 0))
    # 按亮度排序（暗→亮）便于展示昼夜/红外差异
    out.sort(key=lambda kv: kv[1].mean())
    return out


def collect_mot17():
    d = os.path.join(RAW, "OpenDataLab__MOT17")
    fs = sorted(glob.glob(os.path.join(d, "sample", "**", "*.jpg"), recursive=True))
    out = []
    for p in fs[:200]:
        im = imread_gray_safe(p)
        if im is not None:
            rel = os.path.relpath(p, os.path.join(d, "sample"))
            out.append((rel, im, 0))
    return out


def collect_floater():
    """水面漂浮垃圾：优先公开可下的 rf100-vl-floating-waste（带 COCO 真值框），
    兜底 Trash_floater / floating-det。返回 (name, img_with_boxes, n_boxes)。"""
    import json
    # 优先：带 COCO 标注的漂浮垃圾
    d = os.path.join(RAW, "isLinXu__rf100-vl-floating-waste")
    ann = os.path.join(d, "valid", "_annotations.coco.json")
    if os.path.exists(ann):
        coco = json.load(open(ann, encoding="utf-8"))
        img_by_id = {im["id"]: im for im in coco.get("images", [])}
        boxes_by_id = {}
        for a in coco.get("annotations", []):
            boxes_by_id.setdefault(a["image_id"], []).append(a["bbox"])  # xywh
        out = []
        for iid, meta in list(img_by_id.items())[:80]:
            p = os.path.join(d, "valid", meta["file_name"])
            im = imread_gray_safe(p)
            if im is None:
                continue
            nm = len(boxes_by_id.get(iid, []))
            for (x, y, w, h) in boxes_by_id.get(iid, []):
                cv2.rectangle(im, (int(x), int(y)), (int(x + w), int(y + h)),
                              (0, 0, 255), 2)
            out.append((meta["file_name"], im, nm))
        if out:
            return out
    # 兜底：无标注，仅图像
    out = []
    for cand in ("Echo0174__Trash_floater", "irhawks__floating-det", "isLinXu__rf100-vl-floating-waste"):
        dd = os.path.join(RAW, cand)
        fs = [p for p in glob.glob(os.path.join(dd, "**", "*"), recursive=True)
              if os.path.splitext(p)[1].lower() in (".jpg", ".png", ".jpeg")]
        for p in sorted(fs)[:80]:
            im = imread_gray_safe(p)
            if im is not None:
                out.append((os.path.relpath(p, dd), im, 0))
        if out:
            break
    return out


def collect_ship():
    """船只检测：Flier123/Ship 的红外船只（含 day/night × 天气标签）。

    解压后目录含 `_rain/_cloudy/_fog` 天气后缀——优先挑**恶劣天气**样本展示，
    呼应项目主线「恶劣环境感知」。
    """
    d = os.path.join(RAW, "Flier123__Ship")
    fs = [p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
          if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png")]
    if not fs:
        for cand in ("xiakeann__Ship_Detection", "Navigation.AI__Ship3D_KITTI"):
            dd = os.path.join(RAW, cand)
            fs = [p for p in glob.glob(os.path.join(dd, "**", "*"), recursive=True)
                  if os.path.splitext(p)[1].lower() in (".jpg", ".png", ".jpeg")]
            if fs:
                d = dd
                break
    # 恶劣天气优先（rain/fog/cloudy），并兼顾 day/night
    def key(p):
        low = p.lower()
        prio = 0 if ("_rain" in low or "_fog" in low) else (1 if "_cloudy" in low else 2)
        night = 0 if "/night/" in low else 1
        return (prio, night, p)
    out = []
    for p in sorted(fs, key=key)[:120]:
        im = imread_gray_safe(p)
        if im is not None:
            out.append((os.path.relpath(p, d), im, 0))
    return out


def make_panel(ax, title, items, n=6, note=""):
    """在 ax 上画一行 n 张缩略图（或单图占满）。items = (name, img, n_box)。"""
    ax.axis("off")
    if not items:
        ax.text(0.5, 0.5, "尚未下载 / 无图（压缩包需解压）", ha="center",
                va="center", fontsize=11, color="#999")
        ax.set_title(title, fontsize=11)
        return 0
    if len(items) == 1:
        im = items[0][1]
        h = 110
        w = int(im.shape[1] * h / im.shape[0])
        ax.imshow(cv2.cvtColor(cv2.resize(im, (w, h)), cv2.COLOR_BGR2RGB))
        ax.set_title(f"{title}\n({items[0][0][:32]})", fontsize=10)
        return 1
    sel = items[:n]
    thumbs = []
    for it in sel:
        im = it[1]
        h = 110
        w = int(im.shape[1] * h / im.shape[0])
        thumbs.append(cv2.cvtColor(cv2.resize(im, (w, h)), cv2.COLOR_BGR2RGB))
    ax.imshow(np.hstack([np.pad(t, ((0, 0), (0, 4), (0, 0)), constant_values=255)
                         for t in thumbs]))
    total_box = sum(it[2] for it in items)
    box_note = f"，真值框 {total_box}" if total_box else ""
    ax.set_title(f"{title}（{len(items)} 张{box_note}）{note}", fontsize=10)
    return len(sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "p1_datasets_showcase"))
    args = ap.parse_args()
    figs = os.path.join(args.out, "figs")
    os.makedirs(figs, exist_ok=True)

    scenes = [
        ("① 园区扫地 · 扫地机器人第一视角", collect_robotcleaner(),
         "真实采集的垃圾/障碍样本（bed/shoe/trash can/垃圾）"),
        ("② 动态行人 · CVC-14 昼夜红外行人", collect_cvc14(),
         "P1 动态避障的真实行人输入（按亮度排序）"),
        ("③ 动态行人跟踪 · MOT17", collect_mot17(),
         "真实多人轨迹（多帧序列）"),
        ("④ 无人船 · 水面漂浮物", collect_floater(),
         "垃圾打捞场景（水面目标）"),
        ("⑤ 无人船 · 红外船只（昼夜+恶劣天气）", collect_ship(),
         "三亚红外水面船只：day/night × rain/fog/cloudy"),
    ]

    fig, axes = plt.subplots(len(scenes), 1, figsize=(13, 2.2 * len(scenes)))
    counts = {}
    for ax, (title, items, note) in zip(axes, scenes):
        c = make_panel(ax, title, items, note=note)
        counts[title] = len(items)
    fig.suptitle("P1 四场景 · ModelScope 真实数据集样例（只展示已下载内容）",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    png = os.path.join(figs, "p1_scene_samples.png")
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"[✓] 场景样例图 → {png}")

    metrics = {"scenes": counts,
               "note": "只含已实际下载的数据集；缺失整块跳过，不伪造"}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # README
    lines = ["# P1 四场景 · 真实数据集样例\n",
             "> 把 `docs/P1_field_projects.md` 的四个真实场景，从「讲故事」落到「看得见的真实素材」。\n",
             "## 各场景数据落地情况\n",
             "| 场景 | 数据集 | 已下载张数 | 说明 |", "|---|---|---|---|"]
    notes = {
        "① 园区扫地 · 扫地机器人第一视角": ("DatatangBeijing/...RobotCleaner...", "扫地机器人真实视角的垃圾/障碍样本"),
        "② 动态行人 · CVC-14 昼夜红外行人": ("OmniData/CVC-14", "昼夜/红外行人，园区夜巡真实输入"),
        "③ 动态行人跟踪 · MOT17": ("OpenDataLab/MOT17", "真实多人多帧序列，可替换 P1 模拟行人"),
        "④ 无人船 · 水面漂浮物": ("isLinXu/rf100-vl-floating-waste", "无人船垃圾打捞场景（公开可下）"),
        "⑤ 无人船 · 红外船只（昼夜+恶劣天气）": ("Flier123/Ship", "三亚红外船只：昼夜 × 雨雾多云（公开可下）"),
    }
    for title, c in counts.items():
        ds, desc = notes[title]
        lines.append(f"| {title} | {ds} | {c} | {desc} |")
    lines += ["", "![P1 四场景真实数据样例](figs/p1_scene_samples.png)\n",
              "## 诚实声明\n",
              "- 只展示**已实际下载**的数据；未下载的整块标注「未下载」，绝不伪造。\n",
              "- 样例为**数据集真实图像**，非本项目采集，仅用于说明「每个场景长什么样」。\n",
              "- 下一步：把 MOT17 的真实行人轨迹接入 P1 的 `dynamic_obstacles`，替代模拟行人。\n",
              "## 复现\n", "```bash",
              "PYTHONPATH=src python scripts/p1_datasets_showcase.py",
              "```\n"]
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"[✓] 完成 → {args.out}")


if __name__ == "__main__":
    main()
