# WildSpatial · 场景感知与重建实战教程

> **目标**：带一个**懂机器学习 / 深度学习 / 计算机视觉**的你，系统成长为能让机器**实时感知、重建、理解并解构世界**的场景感知 / 场景重建工程师。
> 这不是论文集，而是一条**从入门到专业、由易到难、全程跑得动**的成长路径。

📦 GitHub：[github.com/liuliuqiu378/WildSpatial](https://github.com/liuliuqiu378/WildSpatial)

---

## 一、这份教程要带你成为谁

你已有的 ML/DL/CV 底子，正是进入 3D 视觉的门票——但中间缺一层「一般场景 + 基础知识」的地基，以及一条把知识串成系统的路径。本教程补上它：

| 你现在的样子 | 学完这份教程后 |
|---|---|
| 会训 CNN/Transformer、懂反向传播、能调模型 | 会搭一整套**场景感知管线**：相机标定 → 深度 → 点云/网格 → 语义 → 端侧部署 |
| 知道"图像分类/检测"但分不清"点云/网格/SDF" | 能在**点云、网格、体素、NeRF、3DGS** 之间按任务选型 |
| 没碰过 SLAM/几何 | 能读懂并手搓**视觉里程计、SfM、BA**，知道"什么条件下它会崩" |
| 想做机器人/AR/自动驾驶 | 能落地一个**让机器实时理解并解构世界**的端到端系统 |

**为什么这事值得做**：机器人、AR、自动驾驶、工业检测，本质都在回答一个问题——"机器怎么知道自己在哪、周围世界长什么样、哪个东西能抓/能避/能走"。这正是场景感知与重建要解决的。

---

## 二、教程怎么满足你要的「四条」

| 你的要求 | 这份教程的做法 |
|---|---|
| **通俗易懂** | 双轨写作：**基础篇 `F0–F5`（零公式门槛、生活类比）** + **专家笔记（含公式）**；每个概念先给类比，再补严谨。零基础从 `00_PRIMER` 起，绝不先撞李群。 |
| **系统完整** | 从「相机投影」到「端侧部署」全链路打通；`F` 基础篇补地基，`M0–M9` 模块深挖，三条阅读路径覆盖零基础 / 转行者 / 做项目。 |
| **结合实战** | 每个知识点都**跑真实数据或代码**，图都是程序生成或真实实验产出（见第六章）；`scripts/` 里每条命令都能复现。 |
| **丰富工具算法 + 对比展示** | **第六章「工具与算法全景对比」** 用大量表格横向对比深度/位姿/表示/语义/部署各类方法与代表工具，并标注"本项目怎么用"。 |

---

## 三、怎么学（教学哲学）

每个知识点严格按 **四段式** 展开，缺一不可：

| 段 | 回答什么 | 要求 |
|---|---|---|
| ① **目的** | 它解决什么问题？为什么需要？ | 先给动机 |
| ② **方法原理（专业）** | 术语 + 公式 + 推导 | 严谨不回避，这是地基 |
| ③ **直白讲解** | 大白话 / 类比 | 建立直觉，不能只有术语 |
| ④ **真实数据验证 + 效果图** | 带真值公开数据集实跑 | 给量化指标 + 真实结果图 |

**关键原则：先手搓，再用现成的。** 不亲手写一遍 VO，就读不懂 VGGT 替代了什么、为什么会崩。**崩掉的那刻，才是真正理解的开始。**

---

## 四、三条学习路径（按你的起点选）

### 路径 A · 零基础（没碰过 3D 视觉）
```
00_PRIMER → F0→F1→F2→F3→F4（补一般场景地基）
   → M2 → M4 → M5 → M7（初学者版建直觉）
   → M0 → M1 → M3（专家笔记，此时公式已不吓人）
```

### 路径 B · 已懂 ML / DL / CV（★ 本教程主目标读者）
```
F4（你的知识地图，先建立信心）
   → F0 → F1 → F2 → F3（4 篇补「一般场景 + 基础知识」）
   → M4（前馈模型，你最熟的"网络出几何"）
   → M2（重建）→ M7（语义）→ M8（端侧）
   → M0 → M1 → M3（专家笔记，按需补严谨）
   → M5（压力测试）→ M6（多模态融合）→ M9（综合闭环）
```

### 路径 C · 做项目 / 写简历 / 端侧落地
```
00_LANDSCAPE §4 → F3（正常流程打底）→ M5（失效图谱=选型依据）
   → M4 → M7 → M8（端侧）→ M9（综合闭环，已写）
```

> 完整地图与七维覆盖见 [`docs/00_INDEX.md`](docs/00_INDEX.md)。

---

## 五、文档地图

| 文档 | 难度 | 作用 |
|---|---|---|
| [`docs/00_INDEX.md`](docs/00_INDEX.md) | 导航 | 系统总索引：七维覆盖 + 三条路径 + 真实图索引 |
| [`docs/00_PRIMER.md`](docs/00_PRIMER.md) | 入门 | 零基础概念词典 + 经典理念大白话 |
| [`docs/00_QUICKSTART.md`](docs/00_QUICKSTART.md) | 入门 | 项目在干啥、跑什么命令、看什么图 |
| [`docs/00_LANDSCAPE.md`](docs/00_LANDSCAPE.md) | 总纲 | 领域全景（流水线→VLM→世界模型）+ 端侧切入 |
| [`docs/00_ROADMAP.md`](docs/00_ROADMAP.md) | 计划 | 全局路线图 M0~M9 |
| [`docs/DATASETS.md`](docs/DATASETS.md) | 数据 | **多模态数据集详解**：每个数据集的场景/信息/任务 + **真实样例图** + 用途映射 |
| [`docs/F0_camera_projection.md`](docs/F0_camera_projection.md) | 基础 | **相机·投影·坐标帧**（地基中的地基） |
| [`docs/F1_scene_representations.md`](docs/F1_scene_representations.md) | 基础 | 场景表示全家桶（点云/网格/体素/SDF/NeRF/3DGS） |
| [`docs/F2_depth_basics.md`](docs/F2_depth_basics.md) | 基础 | 深度到底是什么（立体/单目歧义/RGB-D） |
| [`docs/F3_normal_pipeline.md`](docs/F3_normal_pipeline.md) | 基础 | 正常场景端到端流程（视频→3D 模型） |
| [`docs/F4_ml_to_3d_bridge.md`](docs/F4_ml_to_3d_bridge.md) | 基础 | **ML/DL 的人如何接入 3D 视觉**（转行者主入口） |
| [`docs/F5_vision_lidar_fusion.md`](docs/F5_vision_lidar_fusion.md) | 基础 | **视觉 + 激光雷达（点云）融合**：量产感知主流范式（标定/投影/前中后融合/3D 检测 + 可跑 demo） |
| [`docs/M2_depth_reconstruction.md`](docs/M2_depth_reconstruction.md) | 入门 | 深度→点云→网格（初学者版） |
| [`docs/M4_foundation_models.md`](docs/M4_foundation_models.md) | 入门 | 前馈 3D 基础模型（初学者版） |
| [`docs/M5_stress_test.md`](docs/M5_stress_test.md) | 入门 | 压力测试矩阵（初学者版，核心资产） |
| [`docs/M7_semantic_layer.md`](docs/M7_semantic_layer.md) | 入门 | 语义层（初学者版） |
| [`docs/M8_edge_deployment.md`](docs/M8_edge_deployment.md) | 入门 | 端侧部署（初学者版，三维表已用本机缩预算实测） |
| [`docs/M6_multimodal_fusion.md`](docs/M6_multimodal_fusion.md) | 入门 | 多模态融合（诊断驱动切换：何时用谁） |
| [`docs/M9_comprehensive_closed_loop.md`](docs/M9_comprehensive_closed_loop.md) | 入门 | 综合闭环（场景图解构世界） |
| [`docs/M0_geometry_foundation.md`](docs/M0_geometry_foundation.md) | 专家 | 几何地基（含 6 张教学图，公式推导） |
| [`docs/M1_sfm_from_scratch.md`](docs/M1_sfm_from_scratch.md) | 专家 | 手搓 SfM/单目 VO（10+ 真图 + BA/回环） |
| [`docs/M3_failure_attribution.md`](docs/M3_failure_attribution.md) | 专家 | 失效归因：什么条件崩、崩在哪一步 |
| [`docs/P0_projects.md`](docs/P0_projects.md) | 实战 | **实战项目整合层**：多场景公开数据集 × M-module 串成可落地工程 |
| [`docs/P1_field_projects.md`](docs/P1_field_projects.md) | 实战 | **工程叙事层**：四个真实场景（扫地/送物/自动驾驶/无人船）的完整工程叙事 + 三视角讲解 + 初学者常漏环节 |
| [`docs/P2_simulation.md`](docs/P2_simulation.md) | 实战 | **闭环仿真平台**：CARLA/Gazebo/Habitat 选型对比 + Gazebo+ROS2（免 sudo）接入方案（对照工业级 Nav2） |

> 📌 文档图用相对路径引用 `experiments/<module>/figs/`，已随代码提交，**在 GitHub 上直接显示**。
> ✅ **基础篇 `F0–F4` + 模块 `M0–M9` 教学文档已全部覆盖**；`M8` 端侧为初学者版（三维表已用本机缩算力预算模拟实测，无需真机），`M6/M9` 为初学者版（融合救援与场景图 demo 已实测跑通）。`P0/P1` 实战项目层已建：P0 是多场景抽象管线，P1 是面向面试官/客户的工程叙事 + 园区服务机器人感知→规划→控制闭环实跑。

### 🖼️ 成果一览（本项目真实实验产出，点进文档看细节）

| 一段视频 → 相机轨迹（M1） | 深度 → 3D 网格（M2） |
|---|---|
| ![VO 轨迹](experiments/M1_vo_fr1_desk/figs/trajectory.png) | ![TSDF 重建](experiments/M2_recon/figs/recon.png) |

| 方法 × 环境失效图谱（M5，核心资产） | 前馈模型深度鲁棒性（M4） |
|---|---|
| ![ATE 热力图](experiments/M5_method_atlas/figs/atlas_ate.png) | ![VGGT 深度](experiments/M4_depth_atlas/figs/depth_vs_degradation.png) |

| 开放词汇语义（M7，OWL-ViT） | 解构世界为 3D 物体（M9 场景图） |
|---|---|
| ![开放词汇检测](experiments/M7_semantic/figs/open_vocab_detection.png) | ![场景图](experiments/M9_closed_loop/figs/scene_graph.png) |

| 端侧算力权衡（M8） | 实战项目总览（P0） |
|---|---|
| ![M8 权衡](experiments/M8_edge/figs/tradeoff.png) | ![数据集样例](experiments/projects/figs/dataset_samples.png) |

---

## 六、工具与算法全景对比（★ 丰富工具 + 对比展示）

### 6.1 深度估计方法对比
| 方法 | 代表工具 / 模型 | 一句话原理 | 优点 | 缺点 | 本项目怎么用 |
|---|---|---|---|---|---|
| 双目立体 | OpenCV StereoBM / SGM | 两相机视差 $d=fB/Z$ | 纯几何、无需训练、有公制尺度 | 需双目标定、弱纹理失效 | 基线对照 |
| RGB-D 主动 | RealSense / Kinect / iPhone LiDAR | 主动发射红外测距 | 直接真深度、尺度准 | 受光照/透明/室外干扰 | M2 输入来源 |
| 单目学习式 | Depth Anything / DPT / MiDaS | CNN/Transformer 从 RGB 回归深度 | 单图即可、泛化好 | 只给相对深度、无量纲 | M4 对照 |
| 多视图几何 | 传统 SfM / MVS | 从运动/多视角恢复深度 | 无特殊硬件、可锚定公制 | 依赖特征与重叠、易崩 | M0 / M1 |
| 前馈 3D 模型 | **VGGT** / MASt3R / DUSt3R | 一次前向出深度+位姿+点云 | 极快、全栈、抗退化 | 尺度靠对齐、无语义、黑箱 | **M4 主角** |

### 6.2 位姿 / SfM / SLAM 方法对比
| 方法 | 代表 | 原理 | 优点 | 缺点 | 本项目 |
|---|---|---|---|---|---|
| 增量 SfM | **COLMAP** | 特征→匹配→位姿→BA | 成熟、精度高 | 慢、弱纹理易崩 | 基线 |
| 视觉里程计 | ORB-SLAM / 自研 VO | 前端跟踪+后端优化+回环 | 实时 | 退化敏感 | **M1** |
| 滤波紧耦合 | VINS / LIO-SAM | IMU+视觉/激光融合 | 高频、鲁棒 | 系统复杂 | M6（待补） |
| 前馈模型 | **VGGT** | 一次前向出全局几何 | 极快、抗退化 | 尺度/语义缺 | **M4** |

### 6.3 场景表示方法对比
| 表示 | 数学本质 | 何时用 | 本项目 |
|---|---|---|---|
| 点云 | 无序 3D 点集 | 机器人感知、激光 | M2 输入/输出 |
| 网格 | 顶点+面 | 渲染、交互、抓取 | **M2 TSDF 网格** |
| 体素 | 3D 占据栅格 | 喂 3D CNN | M0/M1 概念 |
| SDF / TSDF | 到表面距离场 | 融合、水密 | **M2 主角** |
| NeRF | 神经网络辐射场 | 照片级新视角 | 领域对照 |
| 3D Gaussian | 各向异性高斯泼溅 | 快且真 | 领域对照 |

### 6.4 语义 / 检测模型对比
| 类型 | 代表 | 特点 | 优点 | 缺点 | 本项目 |
|---|---|---|---|---|---|
| 封闭集检测 | **DETR**（COCO-80） | 固定词表 | 准确 | 类别受限 | landscape demo |
| 开放词汇 | **OWL-ViT** | 文本提示查询 | 自由自然语言 | 幻觉（见 M7 实证） | **M7 主角** |
| 分割 | **SAM** | promptable | 通用 | 无语义 | landscape demo |
| 视觉语言 | Qwen-VL / InternVL | 语言推理 | 理解指令 | 几何弱（F1 0.64） | M7 责任切分 |

### 6.5 端侧部署工具链对比
| 工具 | 平台 | 优点 | 注意 | 本项目 |
|---|---|---|---|---|
| **TensorRT** | NVIDIA / Jetson | 极快、生态好 | 仅 NVIDIA | **M8** |
| ONNX Runtime | 跨平台 | 通用 | 优化较弱 | M8 |
| **RKNN** | Rockchip RK3588 | 国产芯片 | 生态小 | M8（三维表已实测，真机编译待补） |
| **CANN/昇腾** | 华为 Ascend | 国产、算力大 | 工具链重 | M8（三维表已实测，真机编译待补） |

---

## 七、快速开始（实战）

```bash
# 0) 进入项目
cd /home/hmn-cjy/liuliuqiu/WildSpatial

# 1) 环境（conda）
conda activate wildspatial          # torch2.6+cu124 / opencv-contrib / open3d

# 2) 装依赖
pip install -r env/requirements.txt

# 3) M0 几何自检 + 生成 6 张教学图（约 3 秒）
PYTHONPATH=src python -m pytest tests/ -q
PYTHONPATH=src python scripts/m0_demo.py

# 4) M1 单目 VO 在 TUM fr1/desk 上跑轨迹/诊断图（约 20 秒）
PYTHONPATH=src python scripts/m1_run_vo.py --seq fr1/desk --frames 450 --stride 3

# 5) 生成 F 系列基础篇演示图（相机投影/表示/深度/三角化）
python scripts/foundations_make_figs.py
```

跑完你会看到：
- `experiments/M0_geometry_foundation/figs/`：投影、对极线、三角化、视差、纯旋转退化、PnP 共 6 张
- `experiments/M1_vo_fr1_desk/figs/`：轨迹对比、每帧诊断、ATE 漂移 + `metrics.json`
- `experiments/foundations/figs/`：F0–F3 程序生成的 4 张概念演示图

---

## 八、目录结构

```
WildSpatial/
├── PROGRESS.md          # 🔴 进度台账（AI 助手每次必读）
├── README.md            # 本文件
├── docs/                # 教学文档（四段式：目的+原理+讲解+真实验证·图）
│   ├── 00_INDEX.md / 00_PRIMER.md / 00_QUICKSTART.md / 00_LANDSCAPE.md / 00_ROADMAP.md
│   ├── DATASETS.md                                      # 多模态数据集总台账
│   ├── F0_camera_projection.md … F5_vision_lidar_fusion.md   # 基础篇（一般场景+基础知识，含视觉+雷达融合）
│   ├── M0_geometry_foundation.md / M1_sfm_from_scratch.md / M3_failure_attribution.md  # 专家笔记
│   └── M2/M4/M5/M7/M8/M6/M9_*.md                                    # 初学者版（M6 多模态 / M9 闭环）
├── src/wildspatial/     # 核心代码库（geometry / sfm / eval / viz，手搓）
├── scripts/             # 可执行入口（m0_demo / m1_run_vo / foundations_make_figs / fetch_datasets …）
├── tests/               # 数值自检（几何 bug 极难调，必须自检）
├── experiments/         # 每个实验：README + metrics.json + figs/（图随代码进 git）
├── data/                # raw / processed（不进 git）
└── env/                 # requirements.txt
```

---

## 九、技术栈

| 层 | 技术 |
|---|---|
| 几何底座 | NumPy / OpenCV / SciPy（**手搓**，理解原理） |
| 3D 基础模型 | VGGT / MASt3R / DUSt3R（M4 接入对比） |
| 语义层 | OWL-ViT / DETR / SAM / Qwen-VL（只做语义，不做几何） |
| 评测 | ATE / RPE / 深度误差（AbsRel/RMSE/δ1）/ 匹配召回率 |
| 部署 | TensorRT / ONNX / RKNN / 昇腾 CANN（M8） |

---

## 十、当前状态

| Module | 教学文档 | 关键数字 |
|---|---|---|
| M0 几何地基 | ✅ 专家笔记 | 25 项自检全绿 + 6 图 |
| M1 手搓 SfM/VO | ✅ 专家笔记 | TUM 基线 ATE 0.506 m；BA 重投影 1.54→0.20 px |
| M2 深度与重建 | ✅ 初学者版 | TSDF 网格 84.8 万顶点 |
| M3 失效归因 | ✅ 专家笔记 | 13 条件带真值失效分类学 |
| M4 前馈 3D 模型 | ✅ 初学者版 | VGGT δ1 0.962–0.986 |
| M5 压力测试 | ✅ 初学者版 | 13×4 ATE 热力图，VGGT 0.016–0.049 m |
| M7 语义层 | ✅ 初学者版 | OWL-ViT 39 框（含幻觉实证） |
| M8 端侧部署 | ✅ 初学者版（三维表已用本机缩预算实测） | 量化/编译链路已写，仿真实验+真图 |
| M6 多模态融合 | ✅ 初学者版 | 低光救援区 96%；VGGT 救援引用 M5(ATE 0.016–0.049) |
| M9 综合闭环 | ✅ 初学者版 | VO 9.8 fps(纯 CPU)；解构 16 个 3D 物体候选 |
| **P0 实战项目整合层** | ✅ 已建 | 4 项目跑通：TUM/4Seasons/合成退化/合成序列 × M2→M4→M7→M9→M8 |

> 完整路线图与每步踩坑记录在 [`PROGRESS.md`](PROGRESS.md)。

---

## 十一、诚实声明（重要）

- **哪些是项目实测**：M0 自检、M1 轨迹/ATE、M2 网格顶点数、M4 VGGT 深度指标、M5 ATE 热力图、M7 OWL-ViT 检测框——均为本项目在带真值数据（TUM RGB-D 等）上实跑的量化结果。
- **哪些是领域标准知识**：相机模型、双目公式、公开数据集（KITTI/ScanNet/NYUv2）、工具链对比——是可验证的通用事实，非本项目新测数字。
- **F0–F3 的 4 张图**为程序生成的概念演示（讲清原理用），非某数据集指标；M0/M1/M2/M4/M7 的图为真实实验产出。
- **M8 端侧部署**目前为初学者版 + 量化/编译链路已写；**三维表已用本机缩算力预算模拟实测**（延迟/精度直接测、功耗用相对算力预算代理，无需真机）；RKNN/ACL 真机编译仍待真机但非卡点。
- **M6 / M9** 已补为初学者版：M6 用「退化监控 + VGGT/深度救援」实测演示融合决策；M9 用「VO + 深度反投影」实测把场景解构成 3D 物体（9.8 fps @ CPU），并如实标注「跨帧数据关联」为生产化待补项。

---

## 十二、致读者与 AI 助手

- 不跳过"手搓"：跑不通才是收获。
- 每个结论要有**数字**支撑；没有数字的观点不写进教程。
- 优先做**可复现的小实验**，不盲目 fine-tune 大模型。
- 教学坚持**四段式** + 双轨（初学者版 / 专家笔记），让"通俗易懂"与"系统完整"兼得。
- 每完成一步，更新 `PROGRESS.md`。
