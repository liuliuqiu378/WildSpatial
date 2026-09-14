"""M9 · 综合闭环：把世界「解构」成 3D 场景图（Scene Graph）

=========================== 目的 ===========================
M1 给了相机位姿，M2/M4 给了深度，M7 给了语义。M9 是整条路线的**收口**：
把感知结果「解构」成机器能用的结构化表达——

    **场景图（Scene Graph）**：一组带 3D 坐标的「物体实例」
    {class, x, y, z (米), 出现帧数}

这就是「机器理解并解构世界」最朴素也最实在的产出：不是一张图，而是
「房间里有 N 个东西，分别在这几个坐标」。后续规划/抓取/导航都是在这张图上做。

本脚本 **不依赖任何大模型**（模型无关，立刻能跑）：用「深度前景分割 +
连通域」得到物体候选，再用 M1 的相机位姿把 2D 候选反投影到 3D 世界系，
跨帧聚类成物体实例。M7 的 OWL-ViT 语义标签可作为「class」字段即插即用。

=========================== 方法原理（专业）===========================
1. 几何层：在序列上跑 MonocularVO → 每帧相机位姿 T_wc（world-to-camera 的逆）。
2. 分割层：取若干关键帧的深度图，做前景掩膜（0.3~2.0m 内的有效深度），
   cv2.connectedComponents 得到连通域 = 物体候选，过滤小噪点。
3. 反投影：候选质心像素 (u,v) + 深度 Z → 相机系 3D → T_wc 变换到世界系。
4. 聚类：跨帧把所有 3D 质心按欧氏距离贪心聚类 → 物体实例；统计出现帧数。
5. 实时性：记录 VO 每帧耗时 → 帧率（fps），回答「能不能实时解构世界」。

用法：
    PYTHONPATH=src python scripts/m9_scene_graph.py --seq fr1/desk --frames 200 --stride 3
产出 experiments/M9_closed_loop/figs/scene_graph.png + scene_graph.json + metrics.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="fr1/desk", choices=list(SEQUENCES))
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--depth-min", type=float, default=0.3, help="前景深度下限 (m)")
    ap.add_argument("--depth-max", type=float, default=2.0, help="前景深度上限 (m)")
    ap.add_argument("--min-pixels", type=int, default=40, help="连通域最小像素数")
    ap.add_argument("--cluster-dist", type=float, default=0.25,
                    help="3D 质心聚类距离阈值 (m)")
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

    cfg = VOConfig(feature="sift", max_features=3000, min_parallax_deg=1.0)
    vo = MonocularVO(K, cfg)

    # 1) 几何层：VO 位姿
    t0 = time.time()
    for i in indices:
        vo.process(ds[i]["rgb"], None)
    vo_el = time.time() - t0

    poses = []  # (frame_idx, T_wc)
    for k, i in enumerate(indices):
        T_cw = vo.frames[k].T_cw
        if T_cw is None:
            continue
        poses.append((i, np.linalg.inv(T_cw)))
    if not poses:
        print("[✗] VO 未初始化，无法获取位姿"); return 1

    # 关键帧：均匀取若干
    kf = poses[::max(1, len(poses) // 5)]

    # 2)+3) 分割 + 反投影
    pts = []
    for (fi, Twc) in kf:
        f = ds[fi]
        depth = f.get("depth")
        if depth is None:
            continue
        depth = depth.astype(np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        H, W = depth.shape
        mask = ((depth > args.depth_min) & (depth < args.depth_max)).astype(np.uint8) * 255
        num, labels = cv2.connectedComponents(mask)
        for c in range(1, num):
            ys, xs = np.where(labels == c)
            if len(xs) < args.min_pixels:
                continue
            u = int(np.median(xs)); v = int(np.median(ys))
            Z = float(depth[v, u])
            if Z <= 0:
                continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            p_world = Twc @ np.array([X, Y, Z, 1.0])
            pts.append(p_world[:3])

    pts = np.array(pts) if pts else np.zeros((0, 3))

    # 4) 跨帧聚类成物体实例
    instances = []
    if len(pts) > 0:
        assigned = np.full(len(pts), -1, dtype=int)
        cid = 0
        for i in range(len(pts)):
            if assigned[i] >= 0:
                continue
            d = np.linalg.norm(pts - pts[i], axis=1)
            grp = np.where((d < args.cluster_dist) & (assigned < 0))[0]
            assigned[grp] = cid
            instances.append({
                "id": cid,
                "centroid_m": [round(float(x), 3) for x in pts[grp].mean(0)],
                "n_detections": int(len(grp)),
            })
            cid += 1
    instances.sort(key=lambda o: o["n_detections"], reverse=True)

    # 5) 实时性
    fps = len(indices) / vo_el if vo_el > 0 else 0.0

    metrics = {
        "sequence": args.seq,
        "vo_ms_per_frame": round(vo_el / max(len(indices), 1) * 1000, 1),
        "vo_fps": round(fps, 1),
        "keyframes_used": len(kf),
        "raw_proposals": int(len(pts)),
        "scene_graph_objects": len(instances),
        "instances_top": instances[:10],
    }

    out_dir = args.out or os.path.join(ROOT, "experiments", "M9_closed_loop")
    figs = os.path.join(out_dir, "figs")
    os.makedirs(figs, exist_ok=True)

    # 图：左=反投影候选(灰) 与 聚类质心(红) 的对比；右=每帧候选数 + 实例质心坐标表
    fig = plt.figure(figsize=(14, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    if len(pts) > 0:
        # 左：灰色小点=原始反投影候选；红色▲=跨帧聚类后的物体实例质心（大、带标签）
        ax1.scatter(pts[:, 0], pts[:, 2], s=22, c="lightsteelblue",
                    alpha=0.85, edgecolors="steelblue", linewidths=0.4,
                    label=f"反投影候选 ({len(pts)})")
        cx = [o["centroid_m"][0] for o in instances]
        cz = [o["centroid_m"][2] for o in instances]
        ax1.scatter(cx, cz, c="crimson", marker="*", s=260, zorder=5,
                    edgecolors="darkred", linewidths=0.8,
                    label=f"聚类物体实例 ({len(instances)})")
        for o in instances:
            x, _, z = o["centroid_m"]
            ax1.annotate(f"#{o['id']}", (x, z), fontsize=8.5,
                         xytext=(6, 4), textcoords="offset points", color="darkred")
        ax1.set_xlabel("X (m)"); ax1.set_ylabel("Z (m)")
        ax1.set_title(f"① 反投影候选 → 聚类物体实例（世界系俯视，{len(pts)}→{len(instances)}）",
                      fontsize=11)
        ax1.set_aspect("equal")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "无候选", ha="center")

    # 右：每个物体实例的 3D 坐标（横向条形 = X/Y/Z 三轴），比「全 1.0 计数」信息量大
    if instances:
        top = instances[:12]
        names = [f"#{o['id']}" for o in top][::-1]
        xs = [o["centroid_m"][0] for o in top][::-1]
        ys = [o["centroid_m"][1] for o in top][::-1]
        zs = [o["centroid_m"][2] for o in top][::-1]
        yy = np.arange(len(top))
        h = 0.26
        ax2.barh(yy + h, xs, height=h, color="#4C72B0", label="X")
        ax2.barh(yy, ys, height=h, color="#DD8452", label="Y")
        ax2.barh(yy - h, zs, height=h, color="#55A868", label="Z")
        ax2.set_yticks(yy); ax2.set_yticklabels(names, fontsize=9)
        ax2.axvline(0, color="gray", lw=0.8)
        ax2.set_xlabel("世界系坐标 (m)")
        ax2.set_title(f"② 解构出的物体实例坐标（{len(instances)} 个，展示前 {len(top)}）",
                      fontsize=11)
        ax2.legend(loc="lower right", fontsize=9)
        ax2.grid(alpha=0.3, axis="x")
        ax2.text(0.02, 0.98,
                 "每条 = 一个「机器可抓取/导航」的 3D 物体\nclass 字段预留接 M7 OWL-ViT 标签",
                 transform=ax2.transAxes, fontsize=8, va="top", color="dimgray",
                 bbox=dict(boxstyle="round", fc="white", ec="lightgray", alpha=0.85))
    else:
        ax2.text(0.5, 0.5, "无实例", ha="center")
    plt.suptitle(f"M9 场景图：把世界解构成 3D 物体（{args.seq}，{fps:.1f} fps @ CPU）",
                 fontsize=13)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12)
    fig.savefig(os.path.join(figs, "scene_graph.png"), dpi=130)
    plt.close(fig)

    scene_graph = {
        "sequence": args.seq,
        "frame_rate_fps": round(fps, 1),
        "objects": [
            {"id": o["id"], "class": "unknown (接 M7 OWL-ViT 标签)",
             "position_m": o["centroid_m"], "detections": o["n_detections"]}
            for o in instances
        ],
    }
    with open(os.path.join(out_dir, "scene_graph.json"), "w") as f:
        json.dump(scene_graph, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[✓] 序列 {args.seq} | VO {fps:.1f} fps")
    print(f"    关键帧 {len(kf)} | 物体候选 {len(pts)} | "
          f"解构出实例 {len(instances)} 个")
    for o in instances[:8]:
        print(f"    #{o['id']:2d}  位置=({o['centroid_m'][0]:+.2f},"
              f"{o['centroid_m'][1]:+.2f},{o['centroid_m'][2]:+.2f}) m  "
              f"出现 {o['n_detections']} 次")
    print(f"    产出 → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
