"""实战项目注册表 · 把「场景 + 工序 + 交付物」组合成完整项目

每个 Project 是一个逻辑自洽的端到端工程，知识落点明确：

  * indoor_mapping     室内机器人语义建图与导航   （TUM fr1/desk）
  * autonomous_driving 自动驾驶感知栈（恶劣天气） （4Seasons oldtown_night）
  * search_and_rescue  退化环境搜救（低纹理/夜间）（TUM fr3_nostructure）
  * ar_inspection      AR 设备端实时解构（无下载） （synthetic）

它们共用同一套 Stage（vo/depth/semantic/scene_graph/edge），只是场景源与
组合不同 —— 这正是「知识（M0–M9）→ 实战（多场景项目）」的桥梁。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import Project
from .loaders import get_sequence


# --------------------------------------------------------------------------- #
# 交付物渲染：把 ctx 里的多工序产物画成一张「项目总览图」
# --------------------------------------------------------------------------- #
def render_project_fig(seq, ctx, out_png):
    fig = plt.figure(figsize=(15, 5))
    # 1) 轨迹（估计 vs 真值）
    ax1 = fig.add_subplot(1, 3, 1)
    unit = ctx.get("vo_unit_Twc")
    if unit:
        P = np.array([T[:3, 3] for T in unit])
        scale = ctx.get("vo_scale", 1.0)
        Pm = P * scale
        ax1.plot(Pm[:, 0], Pm[:, 2], "-o", ms=3, color="#1f77b4", label="估计轨迹")
        if seq.gt_positions is not None:
            ax1.plot(seq.gt_positions[:, 0], seq.gt_positions[:, 2],
                     "--", color="#999", lw=1.2, label="真值")
        ax1.legend(fontsize=8); ax1.set_aspect("equal")
    else:
        ax1.text(0.5, 0.5, "VO 未初始化", ha="center")
    ax1.set_title("① 轨迹重建 (M2)", fontsize=10)
    ax1.set_xlabel("X(m)"); ax1.set_ylabel("Z(m)")

    # 2) 场景图：3D 物体实例俯视
    ax2 = fig.add_subplot(1, 3, 2)
    sg = ctx.get("scene_graph")
    if sg and sg["instances"]:
        for o in sg["instances"]:
            x, _, z = o["centroid_m"]
            ax2.scatter([x], [z], marker="X", s=70, color="#d62728")
            ax2.annotate(f"#{o['id']}", (x, z), fontsize=7,
                         xytext=(3, 3), textcoords="offset points")
        ax2.set_aspect("equal")
        ax2.set_title(f"② 3D 场景图 (M9): {sg['n_instances']} 物体", fontsize=10)
    else:
        ax2.text(0.5, 0.5, "无场景图", ha="center")
        ax2.set_title("② 3D 场景图 (M9)", fontsize=10)
    ax2.set_xlabel("X(m)"); ax2.set_ylabel("Z(m)")

    # 3) 端侧算力预算
    ax3 = fig.add_subplot(1, 3, 3)
    eb = ctx.get("edge_budget")
    if eb:
        xs = [f"{r['scale']:.2f}x" for r in eb]
        ys = [r["rel_compute_budget"] for r in eb]
        colors = ["#2ca02c" if r["status"] == "OK" else "#999" for r in eb]
        ax3.bar(xs, ys, color=colors)
        for i, r in enumerate(eb):
            ax3.text(i, r["rel_compute_budget"], f"{r['ms_per_frame']}ms",
                     ha="center", va="bottom", fontsize=7)
        ax3.set_title("③ 端侧算力预算 (M8)", fontsize=10)
        ax3.set_ylabel("相对算力预算")
    else:
        ax3.text(0.5, 0.5, "无端侧预算", ha="center")
        ax3.set_title("③ 端侧算力预算 (M8)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# 项目定义
# --------------------------------------------------------------------------- #
PROJECTS = {
    "indoor_mapping": Project(
        key="indoor_mapping",
        title="室内机器人语义建图与导航",
        scenario="tum",
        dataset_desc="TUM RGB-D fr1/desk（本地已下载，含深度+真值位姿）",
        stage_keys=["vo", "depth", "semantic", "scene_graph", "edge"],
        deliverable="3D 语义场景图 + 公制轨迹精度 + 端侧预算曲线",
        knowledge_bridge=(
            "把 M2(VO)→M4(深度定尺度)→M7(开放词汇语义)→M9(场景图)→M8(端侧预算) "
            "串成一条室内机器人『先建图、再理解、最后能上端侧』的链路：VO 给出 up-to-scale "
            "轨迹，深度传感器把尺度锚定到公制，开放词汇检测给物体贴自然语言标签，场景图把世界"
            "解构成『N 个物体在该坐标』，端侧预算曲线回答『能不能实时跑在机器人板子上』。"),
        render=render_project_fig,
    ),
    "autonomous_driving": Project(
        key="autonomous_driving",
        title="自动驾驶感知栈（恶劣天气）",
        scenario="4seasons",
        dataset_desc="4Seasons oldtown_night（本地已下，GNSS 真值，夜间退化）",
        stage_keys=["vo", "semantic", "scene_graph", "edge"],
        deliverable="恶劣天气下的位姿鲁棒性 + 道路物体分布场景图",
        knowledge_bridge=(
            "驾驶场景无深度传感器、且夜间退化严重，正对应 M5(失效图谱) 与 M6(融合救援) 的命题："
            "单目 VO 在夜间易退化，项目用 GNSS 真值量化『VO 在真实恶劣天气掉多少』，并演示 M6 "
            "思路（需 IMU/多传感器融合补几何）。场景图给出道路物体空间分布，端侧预算呼应车载"
            "算力约束。"),
        render=render_project_fig,
    ),
    "search_and_rescue": Project(
        key="search_and_rescue",
        title="退化环境搜救（合成退化模拟）",
        scenario="tum_degraded",
        dataset_desc="fr1/desk + M3 合成退化（低光+运动模糊，模拟搜救/地下退化；已明确标注）",
        stage_keys=["vo", "depth", "semantic", "scene_graph"],
        deliverable="退化场景下 VO 可行性边界 + 救援目标物体定位",
        knowledge_bridge=(
            "TUM fr3_nostructure 实测缺 rgb.txt/真值，无法定量评估；故用『带真值的 fr1/desk + M3 "
            "合成退化』可控地模拟搜救式退化环境。项目量化 M2 VO 在退化下的可行性边界（能否初始化、"
            "ATE 多少），并演示 M6 失效预警的必要性：一旦 VO 过『特征地板/分辨率悬崖』直接死，就"
            "必须靠多模态融合兜底。退化为合成施加，已在数据/图注中标注。"),
        render=render_project_fig,
    ),
    "ar_inspection": Project(
        key="ar_inspection",
        title="AR 设备端实时解构（无需下载）",
        scenario="synthetic",
        dataset_desc="程序合成序列（无需下载，演示无网环境也能跑通框架）",
        stage_keys=["vo", "depth", "scene_graph", "edge"],
        deliverable="合成巡检场景的 3D 物体解构 + 端侧实时预算",
        knowledge_bridge=(
            "用程序生成的『绕圈相机+彩色物体+已知位姿』序列，零下载验证整套整合层：VO 重建→深度"
            "定尺度→场景图解构→端侧预算。证明框架与数据来源解耦，任何新数据集只要能归一成 "
            "Sequence，就能立刻接入全部 M0–M9 工序。"),
        render=render_project_fig,
    ),
}


def get_project(key: str) -> Project:
    if key not in PROJECTS:
        raise KeyError(f"未知项目: {key}（可选 {list(PROJECTS)}）")
    return PROJECTS[key]


# 各项目默认加载参数（scenario + kwargs）
PROJECT_LOADERS = {
    "indoor_mapping": ("tum", {"seq": "fr1/desk", "max_frames": 200, "stride": 3}),
    "autonomous_driving": ("4seasons", {"split": "oldtown_night", "start": 400, "n": 180}),
    "search_and_rescue": ("tum_degraded",
                          {"seq": "fr1/desk", "max_frames": 200, "stride": 3,
                           "degrade": "low_light,motion_blur", "degrade_sev": 0.6}),
    "ar_inspection": ("synthetic", {"n_frames": 100}),
}


def load_project_sequence(key: str):
    scenario, kw = PROJECT_LOADERS[key]
    return get_sequence(scenario, **kw)
