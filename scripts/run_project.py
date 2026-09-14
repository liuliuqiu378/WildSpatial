"""实战项目驱动器 · 把「知识(M0–M9)→实战(多场景项目)」跑通并产出交付物

==================== 用法 ====================
    # 室内机器人语义建图（TUM fr1/desk，本地已下）
    PYTHONPATH=src python scripts/run_project.py --project indoor_mapping

    # 自动驾驶感知栈（4Seasons 夜间，本地已下）
    PYTHONPATH=src python scripts/run_project.py --project autonomous_driving

    # 退化环境搜救（TUM fr3_nostructure）
    PYTHONPATH=src python scripts/run_project.py --project search_and_rescue

    # AR 设备端解构（合成序列，无需下载）
    PYTHONPATH=src python scripts/run_project.py --project ar_inspection

==================== 产出 ====================
    experiments/projects/<project>/metrics.json     各工序指标
    experiments/projects/<project>/figs/overview.png 项目总览图
    experiments/projects/<project>/README.md        知识→实战桥梁文档

设计：每个 Project = 场景源 + 有序 Stage。Stage 间通过共享 ctx 传递产物，
形成因果链（VO→深度定尺度→语义→场景图→端侧预算），逻辑自洽、可复现。
"""

import os
import sys
import json
import argparse
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from wildspatial.projects import (get_project, load_project_sequence,
                                  build_stage, render_project_fig)
from wildspatial.viz import setup_plot_style

setup_plot_style()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True,
                    choices=sorted({
                        "indoor_mapping", "autonomous_driving",
                        "search_and_rescue", "ar_inspection"}))
    ap.add_argument("--out", default=None, help="覆盖输出目录")
    args = ap.parse_args()

    proj = get_project(args.project)
    out_dir = args.out or os.path.join(ROOT, "experiments", "projects", args.project)
    figs = os.path.join(out_dir, "figs")
    os.makedirs(figs, exist_ok=True)

    print(f"[project] {proj.title}")
    print(f"  数据集: {proj.dataset_desc}")
    print(f"  工序链: {' → '.join(proj.stage_keys)}")

    # 1) 加载场景序列
    try:
        seq = load_project_sequence(args.project)
    except FileNotFoundError as e:
        print(f"[✗] 数据缺失: {e}")
        return 1
    print(f"[·] 载入 {len(seq)} 帧 | K=({seq.K[0,0]:.0f},{seq.K[1,1]:.0f}) | "
          f"场景={seq.scenario} | 有深度={'Y' if seq.has_depth() else 'N'} | "
          f"有真值={'Y' if seq.has_gt() else 'N'}")

    # 2) 跑工序链（共享 ctx 传送带）
    ctx = {}
    stage_results = []
    for key in proj.stage_keys:
        st = build_stage(key)
        t0 = time.time()
        r = st.run(seq, ctx)
        r.metrics["_elapsed_s"] = round(time.time() - t0, 2)
        stage_results.append(r.to_dict())
        tag = "跳过" if r.skipped else ("OK" if r.ok else "失败")
        print(f"  [{tag:>4}] {key:12s} | {r.note}")
        if r.skipped:
            continue

    # 3) 渲染交付物图
    png = None
    try:
        if proj.render is not None:
            png = proj.render(seq, ctx, os.path.join(figs, "overview.png"))
    except Exception as e:
        print(f"  [!] 图渲染异常: {e}")

    # 4) 汇总指标
    metrics = {
        "project": args.project,
        "title": proj.title,
        "dataset": proj.dataset_desc,
        "n_frames": len(seq),
        "scenario": seq.scenario,
        "has_depth": seq.has_depth(),
        "has_gt": seq.has_gt(),
        "stage_results": stage_results,
        "edge_budget": ctx.get("edge_budget"),
        "scene_graph": ctx.get("scene_graph"),
        "knowledge_bridge": proj.knowledge_bridge,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    # 5) 知识→实战 桥梁文档
    write_readme(out_dir, proj, seq, stage_results, ctx, png)
    print(f"[✓] 完成 → {out_dir}")


def write_readme(out_dir, proj, seq, stage_results, ctx, png):
    lines = []
    lines.append(f"# 实战项目：{proj.title}\n")
    lines.append(f"> 知识落点（M0–M9 → 实战）：{proj.knowledge_bridge}\n")
    lines.append(f"**数据集**：{proj.dataset_desc}  ")
    lines.append(f"**序列规模**：{len(seq)} 帧 | 场景标签 `{seq.scenario}` | "
                 f"深度={'有' if seq.has_depth() else '无'} | "
                 f"真值={'有' if seq.has_gt() else '无'}\n")
    lines.append("## 工序链（因果自洽）\n")
    lines.append("| 工序 | 模块 | 状态 | 关键指标 | 说明 |")
    lines.append("|---|---|---|---|---|")
    mod = {"vo": "M2", "depth": "M4", "semantic": "M7",
           "scene_graph": "M9", "edge": "M8"}
    for r in stage_results:
        m = r["metrics"]
        if r["name"] == "vo":
            kv = f"ATE(自由尺度)={m.get('ate_rmse_m_up_to_scale')}m, {m.get('fps')}fps"
        elif r["name"] == "depth":
            kv = f"尺度={m.get('scale_factor')}, ATE(公制)={m.get('ate_rmse_m_metric')}m"
        elif r["name"] == "semantic":
            kv = f"检测数={m.get('n_detections')}（{m.get('tool','几何')}）"
        elif r["name"] == "scene_graph":
            kv = f"物体实例={m.get('instances')}（原始候选 {m.get('raw_proposals')}）"
        elif r["name"] == "edge":
            rows = m.get("budget_rows", [])
            kv = "; ".join(f"{x['scale']}x→{x['rel_compute_budget']}预算/"
                           f"{x['ms_per_frame']}ms/{x['status']}" for x in rows)
        else:
            kv = ""
        status = "跳过" if r["skipped"] else ("OK" if r["ok"] else "失败")
        lines.append(f"| {r['name']} | {mod.get(r['name'],'—')} | {status} | "
                     f"{kv} | {r['note']} |")
    lines.append("")
    if png:
        lines.append(f"![项目总览](figs/{os.path.basename(png)})\n")
    # 端侧预算表（若有）
    eb = ctx.get("edge_budget")
    if eb:
        lines.append("## 端侧算力预算曲线（M8）\n")
        lines.append("| 分辨率 | 相对算力预算 | 单帧延迟 | fps | ATE(m) | 状态 |")
        lines.append("|---|---|---|---|---|---|")
        for x in eb:
            lines.append(f"| {x['scale']}x | {x['rel_compute_budget']} | "
                         f"{x['ms_per_frame']}ms | {x['fps']} | "
                         f"{x['ate_rmse_m']} | {x['status']} |")
        lines.append("")
    # 场景图清单（若有）
    sg = ctx.get("scene_graph")
    if sg and sg["instances"]:
        lines.append("## 解构出的 3D 场景图（M9）\n")
        lines.append("| ID | 标签 | 质心(m) | 跨帧出现次数 |")
        lines.append("|---|---|---|---|")
        for o in sg["instances"][:12]:
            c = o["centroid_m"]
            lines.append(f"| #{o['id']} | {o['label']} | "
                         f"({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}) | "
                         f"{o['n_detections']} |")
        lines.append("")
    lines.append("## 如何复现\n")
    lines.append("```bash")
    lines.append(f"PYTHONPATH=src python scripts/run_project.py --project {proj.key}")
    lines.append("```\n")
    lines.append("> 本框架与数据来源解耦：任何新公开数据集只要能归一成 "
                 "`Sequence(rgbs, K, depths, gt_positions)`，即可接入全部 M0–M9 工序。\n")

    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
