# 00 · 文档总索引（系统导航）

> 本文件是 WildSpatial 教学文档的**总目录与阅读地图**。先看它，再决定读哪篇、按什么顺序读。
> 目标读者：想系统学「空间感知 / SLAM / 3D 视觉」、并最终能落地的**初学者 → 专业人员**。
> 本仓库明确服务两类起点：① 零基础；② **已懂机器学习/深度学习/一般 CS，想转 3D 视觉**（见 `F4`）。

---

## 一、文档分三层、两种难度

| 类型 | 文件 | 难度 | 适合谁 |
|---|---|---|---|
| **入门 / 总纲**（`00_` 开头） | `00_PRIMER` / `00_QUICKSTART` / `00_LANDSCAPE` / `00_ROADMAP` | 大白话、零公式 | 所有人，尤其零基础 |
| **数据台账** | `DATASETS` | **详解 + 真实样例图** | 想了解"每个数据集是什么、能做什么任务" |
| **基础篇**（`F0`–`F5`） | 见下表 | 初学者友好、一般场景 | **已懂 ML/DL 的转行者的主入口**；零基础补地基 |
| **模块文档**（`M0`–`M9`） | 见下表 | 分两派 ↓ | — |

模块文档有**两种写法并存**（项目强制「四段式」+ 初学者友好双轨）：

- **专家笔记版**（含公式推导、需一定基础）：`M0` `M1` `M3`
- **初学者版**（类比 + 零公式 + 钉真实图/数）：`M2` `M4` `M5` `M6` `M7` `M8` `M9`

> ⚠️ **关键顺序提醒**：M0/M1/M3 是地基但写成了专家笔记。**零基础/转行者请先读 `00_PRIMER` + `F0–F4` 基础篇 + M2/M4/M5/M6/M7/M8/M9 初学者版**，建立直觉后再回头啃 M0/M1/M3 的公式——不要一上来撞李群。

---

## 二、各模块七维覆盖一览

你关心的「系统 / 易懂 / 实例 / 效果 / 原理 / 作用 / 应用」是否齐备：

| 文档 | 原理 | 实例讲解 | 效果展示 | 作用介绍 | 实际应用 |
|---|---|---|---|---|---|
| `00_PRIMER` | ✅概念词典 | ✅生活类比 | 引真实图 | ✅M0–M9 映射 | ◐路线图 |
| `00_QUICKSTART` | ◐ | ✅跑命令+图 | ✅6 图 | ✅项目定位 | ◐ |
| `00_LANDSCAPE` | ✅四段 | ✅ | ✅ | ✅ | ✅端侧切入 |
| `F0 相机投影` | ✅ | ✅numpy 可跑 | ✅引 M0 图 | ✅ | ✅已补 |
| `F1 场景表示` | ✅全家桶 | ✅类比+选型 | ✅引 M2/M4 图 | ✅ | ✅已补 |
| `F2 深度基础` | ✅ | ✅公式自洽 | ✅引 M4 图 | ✅ | ✅已补 |
| `F3 正常流程` | ✅标准流水线 | ✅拼图类比 | ✅引 M1/M2 图 | ✅全栈目录 | ✅已补 |
| `F4 ML/DL 桥接` | ✅映射表 | ✅心智模型 | ✅引 M4/M7 图 | ✅ | ✅竞争力 |
| `F5 视觉+雷达融合` | ✅标定/投影/3 架构 | ✅鉴定师×卷尺类比 | ✅可跑投影 demo 图 | ✅量产范式 | ✅自动驾驶/AMR/水下 |
| `M0 几何地基` | ✅✅公式 | ✅6 图 | ✅ | ✅ | ⬜（基础，通用） |
| `M1 SfM/VO` | ✅✅ | ✅10+ 图 | ✅ | ✅ | ⬜（基础，通用） |
| `M2 深度与重建` | ✅ | ✅三级跳类比 | ✅recon.png | ✅ | ✅已补 |
| `M3 失效归因` | ✅四段 | ✅ | ✅3 图 | ✅ | ◐ |
| `M4 前馈 3D 模型` | ✅ | ✅一体化类比 | ✅depth 图 | ✅ | ✅已补 |
| `M5 压力测试` | ✅ | ✅热力图 | ✅ | ✅ | ✅已补 |
| `M7 语义层` | ✅ | ✅ | ✅ | ✅ | ✅已补 |
| `M8 端侧部署` | ✅ | ✅（本机缩预算仿真 + INT8 PTQ 真跑） | ✅（量级+真实验证） | ✅ | ✅已补（三维表 + PTQ 已实测） |
| `M6 多模态融合` | ✅ | ✅融合对照表 | ✅guard 图 | ✅ | ✅已补（低光救援区 96%） |
| `M9 综合闭环` | ✅ | ✅场景图 | ✅scene_graph 图 | ✅ | ✅已补（VO 9.8 fps @ CPU） |
| `P0 实战项目` | ✅ | ✅多场景数据集×M-module | ✅overview 图 | ✅ | ✅已补（4 项目跑通，逻辑自洽可复现） |
| `P1 工程叙事` | ✅ | ✅四场景 + 初学者常漏环节 | ✅map_plan/control_cmd | ✅ | ✅已补（感知→规划→控制 闭环实跑） |
| `P2 闭环仿真` | ✅ | ✅表示谱系 + 成熟度谱系 | ✅navfn_vs_p1 图 | ✅ | ✅已补（ROS2+Nav2 已装并跑通） |

---

## 三、三条推荐阅读路径

### 路径 A · 零基础（转行 / 在校生 / 没碰过 SLAM）
```
00_PRIMER（概念+经典理念，零公式）
   → F0（相机投影）→ F1（场景表示）→ F2（深度）→ F3（正常流程）→ F4（ML人桥接，若你有ML背景必看）
   → M2 → M4 → M5 → M7（初学者版，建立完整直觉）
   →（想深究原理）M0 → M1 → M3（专家笔记，此时公式已不吓人）
```

### 路径 B · 已懂 ML / DL / 一般 CS（本文档系列的主目标读者）
```
F4（你的知识地图，先建立信心）→ F0 → F1 → F2 → F3（4 篇补"一般场景+基础知识"地基）→ F5（视觉+雷达融合）
   → M4（前馈模型，你最熟悉的"网络出几何"）→ M2（重建）→ M7（语义）→ M8（端侧）
   → M0 → M1 → M3（专家笔记，按需补严谨）
   →（恶劣场景）M5（压力测试）→ M6（多模态融合）→ M9（综合闭环）
```

### 路径 C · 做项目 / 写简历 / 端侧落地
```
00_LANDSCAPE §4（你的差异化切入）
   → F3（正常流程打底）→ F5（视觉+激光雷达融合，量产主流范式）→ M5（失效图谱=选型依据）
   → M4（前馈模型）→ M7（语义+责任切分）
   → M8（端侧部署，初学者版已写，三维表已用本机缩预算实测）→ M9（综合闭环，已写）→ P0（实战项目整合层，已写）
```

---

## 四、真实实验图索引（效果展示都在哪）

| 实验图 | 路径 | 出现在 |
|---|---|---|
| 针孔投影 / 对极线 / 纯旋转退化等 6 图 | `experiments/M0_geometry_foundation/figs/` | M0, QUICKSTART, PRIMER, **F0** |
| 相机投影/表示对比/深度视差/三角化 4 图 | `experiments/foundations/figs/` | **F0, F1, F2, F3**（程序生成概念演示） |
| 轨迹 / 诊断 / ATE 漂移 + BA 对比 | `experiments/M1_vo_fr1_desk/figs/` | M1, QUICKSTART, **F3** |
| 特征/匹配/对极/重建/条件对比 5 图 | `experiments/M1_showcase/figs/` | M1 §6, **F3** |
| TSDF 重建网格 | `experiments/M2_recon/figs/recon.png` | M2, **F1/F3** |
| 失效图谱 / 样本图 / 状态分布 | `experiments/M3_degradation_sweep/figs/` | M3 |
| 深度随退化曲线 | `experiments/M4_depth_atlas/figs/depth_vs_degradation.png` | M4, LANDSCAPE, **F2** |
| 四方法 ATE 热力图 | `experiments/M5_method_atlas/figs/atlas_ate.png` | M5, LANDSCAPE, **PRIMER** |
| 开放词汇检测 | `experiments/M7_semantic/figs/open_vocab_detection.png` | M7, LANDSCAPE, **F4** |
| 退化监控 guard 图（干净/低光） | `experiments/M6_fusion_guard/figs/` | **M6** |
| 端侧算力预算权衡图 | `experiments/M8_edge/figs/tradeoff.png` | **M8** |
| INT8 PTQ 精度/延迟对比图 | `experiments/M8_edge/figs/ptq.png` | **M8** |
| 场景图解构（3D 物体散点） | `experiments/M9_closed_loop/figs/scene_graph.png` | **M9** |
| 场景图语义增强（带真实 class 标签） | `experiments/M9_closed_loop/figs/scene_graph_semantic.png` | **M9, M7** |
| 实战项目总览（轨迹/场景图/端侧预算） | `experiments/projects/*/figs/overview.png` | **P0** |
| P1 占据栅格 + A* 规划（感知→规划） | `experiments/P1_service_robot/figs/map_plan.png` | **P1** |
| P1 动态避障（速度障碍法 VO 减速） | `experiments/P1_service_robot/figs/dynamic_avoid.png` | **P1** |
| P1 差速轮控制指令序列（规划→控制） | `experiments/P1_service_robot/figs/control_cmd.png` | **P1** |
| P1 3D 场景图（语义版） | `experiments/P1_service_robot/figs/scene_graph.png` | **P1** |
| 公开数据集 / 输入样例拼图 | `experiments/projects/figs/dataset_samples.png` | **P0** |
| 视觉+深度/LiDAR 融合几何示意 | `experiments/projects/figs/vision_lidar_fusion.png` | **P0, F5** |
| 点云↔图像投影闭环（按距离着色） | `experiments/M10_lidar_fusion/figs/lidar_projection.png` | **F5** |
| **KITTI 真实 LiDAR + 相机融合**（真实传感器） | `experiments/M10_kitti_lidar/figs/kitti_lidar_fusion.png` | **F5** |
| D1 · 真实水下开放词汇检测（GT 对比） | `experiments/D1_real_world/figs/underwater_det.png` | **DATASETS, M7** |
| D1 · 真实夜间城市驾驶检测 | `experiments/D1_real_world/figs/night_det.png` | **DATASETS, M7** |
| DETR 封闭集 / GrabCut 分割 | `experiments/landscape_showcase/figs/` | LANDSCAPE |
| **数据集真实样例图集**（夜/室内深度/停车场/点云/驾驶） | `experiments/datasets_showcase/figs/` | **DATASETS** |
| SUN RGB-D 深度真值反投影 | `experiments/SUNRGBD_depth/figs/sunrgbd_scene.png` | **DATASETS** |
| **真实遥感变化检测（光学+SAR+变化真值）** | `experiments/remote_sensing_cd/figs/remote_sensing_cd.png` | **DATASETS, M6, M9** |
| **NDISPark 昼夜实例分割真值** | `experiments/ndispark_night_day/figs/ndispark_seg.png` | **DATASETS, M6, M7** |
| 点云退化图库 + Chamfer 曲线 | `experiments/ModelNet40C_corruption/figs/` | **DATASETS** |
| **P1 四场景真实数据样例**（扫地/行人/无人船） | `experiments/p1_datasets_showcase/figs/p1_scene_samples.png` | **DATASETS, P1** |
| **P2 工业级 Nav2 规划器 vs P1 手搓 A\*** | `experiments/P2_ros2/figs/navfn_vs_p1.png` | **P2** |
| **P2 TB3 无头闭环：LiDAR→膨胀→costmap** | `experiments/P2_tb3_nav2/figs/costmap_concept.png` | **P2** |
| **P2 感知实验台：仿真相机→真值深度** | `experiments/P2_gz_perception/figs/gazebo_rgbd.png` | **P2, M4** |

### 📷 一屏看懂本项目做了什么（精选图墙，按主线顺序）

*① 相机投影：3D 世界 → 2D 像素（F0）*
![相机投影](../experiments/M0_geometry_foundation/figs/fig1_projection.png)

*② 正常流程：一段视频 → 3D 轨迹与地图（F3/M1）*
![VO 轨迹](../experiments/M1_vo_fr1_desk/figs/trajectory.png)

*③ 深度 → 网格：真实重建出的 3D 模型（M2，84 万顶点）*
![TSDF 重建](../experiments/M2_recon/figs/recon.png)

*④ 方法动物园：谁在什么条件下崩（M5，四方法 × 13 退化 ATE 热力图）*
![ATE 热力图](../experiments/M5_method_atlas/figs/atlas_ate.png)

*⑤ 前馈 3D 基础模型：VGGT 深度在退化下几乎不退化（M4）*
![VGGT 深度](../experiments/M4_depth_atlas/figs/depth_vs_degradation.png)

*⑥ 语义层：开放词汇检测（M7 OWL-ViT，自然语言查询）*
![开放词汇检测](../experiments/M7_semantic/figs/open_vocab_detection.png)

*⑦ 解构世界：把场景变成带 3D 坐标的物体（M9 场景图）*
![场景图](../experiments/M9_closed_loop/figs/scene_graph.png)

*⑧ 端侧权衡：算力预算 vs 精度（M8）*
![M8 权衡](../experiments/M8_edge/figs/tradeoff.png)

*⑨ 视觉+激光雷达融合：KITTI 真实 LiDAR + 相机融合（F5）*
![视觉+雷达融合](../experiments/M10_kitti_lidar/figs/kitti_lidar_fusion.png)

*⑩ TUM 几何自洽：点云↔图像投影闭环（重投影残差 0.0000 px，F5）*
![TUM 投影闭环](../experiments/M10_lidar_fusion/figs/lidar_projection.png)

*⑪ 实战整合：多场景公开数据集串成可落地工程（P0）*
![数据集样例](../experiments/projects/figs/dataset_samples.png)

*⑫ 真实昼夜感知：NDISPark 停车场实例分割真值在夜间与白天都成立（M6/M7）*
![NDISPark 昼夜实例分割](../experiments/ndispark_night_day/figs/ndispark_seg.png)

---

## 五、覆盖状态确认（诚实标注，不假装系统已完整）

截至 2026-09-14，**M0–M9 + F0–F5 + P0 教学文档已全部就位**，无结构性缺口；且**真实数据闭环已跑通**（D1 真实语义评测、F5 真实 LiDAR、D3 数据台账）。下表为差异化路线的收口模块状态：

| 模块 | 状态 | 必含的 7 维度 | 现有素材 |
|---|---|---|---|
| **M6 多模态融合** | ✅ 初学者版 | 诊断驱动切换：何时、用谁（RGB-D/IMU/VGGT） | 退化监控脚本已跑（低光救援区 96%） |
| **M9 综合闭环** | ✅ 初学者版 | 感知→语义→规划→执行 demo（场景图解构） | VO+深度反投影脚本已跑（9.8 fps @ CPU，16 物体候选） |
| **F5 视觉+激光雷达** | ✅ 双证据 | TUM 数学自洽（0px 残差）+ **KITTI 真实雷达验证** | 真实 LiDAR 4 帧、每帧 ~1.8 万点、3.0–79.7 m |
| **D1 真实恶劣语义** | ✅ 已实跑 | 真实水下（YOLO 真值）+ 夜间定量评测 | OWL-ViT 水下 recall≈0.001、夜间≈7 命中/图 |
| **D3 数据扩充** | ✅ 已落地 | 11 个真实多模态数据集台账；**遥感/NDISPark 已可视化** | 水下/夜间/遥感/KITTI/SUN RGB-D/ModelNet40-C |

> ⚠️ **仍待补（非结构性 / gate 项）**：① `M8` 端侧 RKNN/ACL **真机交叉编译**（本机缩预算三维表已给出可移植结论，非卡点）；② **真实视频序列 VO 评测**（4Seasons/SubT-MRS/DUO，受本机网络限制，脚本已就绪，换机/本地路径即可消费）。详见 `PROGRESS.md §2.2`。
> ✅ **本次已补**：**视觉+激光雷达融合系统专章 `F5`**（标定 / 投影公式 / 前中后融合 / 3D 检测 + 可跑投影闭环 demo）——补齐此前"仅 `P0 §4` 一节"的缺口。

> ✅ **已补齐（本次）**：基础篇 `F0`–`F4`（相机投影 / 场景表示 / 深度基础 / 正常流程 / ML 人桥接），专门补足「一般场景 + 基础知识」地基，让初学者→专业人员有连续成长路径。
> ✅ `M6` 多模态融合、`M9` 综合闭环初学者版已写（均含可跑脚本与实测数字）。`M8` 端侧部署初学者版已写（三维表已用本机缩算力预算模拟实测，无需真机；RKNN/ACL 真机编译仍待真机但非卡点）。`P0` 实战项目整合层已写（`wildspatial/projects/` + `scripts/run_project.py`，4 个多场景项目端到端跑通、逻辑自洽、可复现）。

> 写作规范（项目强制·四段式）：每个知识点都要讲 **① 目的 → ② 方法原理（专业）→ ③ 直白讲解 → ④ 真实数据验证 + 效果图**，并补「🏭 实际应用」小节。基础篇（F 系列）额外强调「一般场景、零公式门槛、引用领域标准而非编造项目数字」。

---

*下一篇建议*：零基础从 `00_PRIMER.md` 开始；**已懂 ML/DL 从 `F4_ml_to_3d_bridge.md` 开始**；想直接看项目产出跳 `M5_stress_test.md`。
