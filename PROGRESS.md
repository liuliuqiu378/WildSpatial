# WildSpatial 进度台账（唯一真源）

> **AI 助手必读**：每次进入项目，先读本文件。
> 更新规则：**完成任何一步后立刻追加**，不要等收尾。格式见底部模板。
> 当前阶段：**教程体系（19 篇 + 索引）与实验骨架已全部闭环，进入「清障 → 打磨 → 补差异化主场 → 真实数据验证」阶段**。
> 基础篇 `F0–F4` + 专家笔记 `M0/M1/M3` + 初学者版 `M2/M4/M5/M6/M7/M8/M9` + 总纲 `00_*` + 实战整合层 `P0` 全部就位；
> `M0–M9` 三维表/实测均已闭环（`M6` 低光救援区 96%、`M9` 场景图 9.8fps@CPU/16 物体、`M8` 本机缩算力预算三维表 + INT8 PTQ 真跑）。
> **2026-09-12 完成全量教程审查 + A/B 两组修复**：体系合理/全面/实证扎实；**待办已重排为 A 清障 → B 打磨 → C 补缺口 → D 真实数据**（见 §2.2）。
> ✅ **A 组已完成**（M8 图片路径 bug、陈旧「待补」文案 ×5 文件、索引漏项、图注不符、**32 处「只给地址不显示」的图改为嵌入**）；
> ✅ **B 组已完成**（M9 场景图重画、M8 权衡图加 FAIL 档、M0/M1/M3 加初学者导读、M5 热力图自解释）。
> ✅ **C1 已完成**（新增 `F5 视觉+激光雷达融合系统专章` + 可跑投影闭环 demo，残差 0.0000px）。
> **全仓库 101 处图片引用零缺失，docs 共 22 篇**。**下一步 = C2（M8 真机）或 C3（M7 完整语义闭环）**，均需外部条件。
> **唯一仍需真机的是 RKNN/ACL 交叉编译（非卡点）**。
> **D3 多模态数据扩充已完成**（ModelScope 11 个真实数据集落地，台账 `docs/DATASETS.md`）；
> **D3 中的遥感变化检测真实数据已消费**（4 组样本 × 4 面板：光学/SAR/变化真值）→ `experiments/remote_sensing_cd/`；
> **D3 中的 NDISPark 昼/夜真实数据已消费**（夜间+白天停车场实例分割真值，封闭集分割不挑光照）→ `experiments/ndispark_night_day/`；
> **M9 综合闭环已实现语义增强版**：M7 OWL-ViT 标签接入 M9 场景图（31 次 2D 检测 → 21 个 3D 实例，6 个跨帧稳定带真实 class）→ `experiments/M9_closed_loop/`；
> **P1 工程叙事层已落地**：四场景（园区扫地/酒店送物/自动驾驶/无人船）工程叙事 + 旗舰「园区/室内服务机器人」感知→规划→控制闭环实跑（TUM fr1/desk，VO ATE 0.764m，A* 39 步，38 条控制指令）→ `docs/P1_field_projects.md` + `experiments/P1_service_robot/`；
> **D1 已落地首块「真实恶劣环境 + 真值」实证**：M7 语义层在真实水下（OWL-ViT recall≈0.001）/ 夜间（≈7 命中/图）的评测 → `experiments/D1_real_world/`。
> **F5 视觉+激光融合已由真实 LiDAR 数据验证**（KITTI depth completion，4 帧，每帧约 1.8 万真实 LiDAR 点，3.0–79.7 m）→ `experiments/M10_kitti_lidar/`。
> 真实视频序列 VO 评测（SubT-MRS/4Seasons）仍受本机网络限制，属可外部触发的 gate 项。
>
> ⚠️ **AI 行为约定（血泪教训）**：**永远不要阻塞等待后台任务**（下载/安装）。
> 本系统网络很慢（TUM 单连接 ~0.3MB/s，torch 依赖下了 1 小时+）。
> 正确做法：启动后台任务 → 立刻继续写代码/文档 → 需要数据时再检查。
> 曾经因为反复 `sleep` 轮询空耗了 2 小时，绝不再犯。

---

## 0. 环境事实（已核实，勿重复探测）

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D，24564 MiB（已被占用约 6.4G，可用 ~18G） |
| 驱动 / CUDA | 580.126.09 / torch 自带 cu124 |
| Conda 根 | `/home/hmn-cjy/miniforge3` |
| 主用环境 | `wildspatial`（python 3.10.21, torch 2.6.0+cu124, **CUDA 可用**, opencv-contrib, open3d 0.19, matplotlib, pytest, einops）✅ **已就绪，用这个** |
| 备用环境 | `det`（torch 2.6.0+cu124, numpy 2.2.6, opencv, matplotlib, transformers **4.48.0**, ultralytics 8.4.60）—— 跑检测/分割/VLM 类；⚠️ **transformers 4.48 不支持 Qwen3（需 ≥4.51）**，纯文本 LLM 暂不能跑 |
| 磁盘 | 剩余 921G |
| 已缓存模型（离线可用） | **VGGT-1B** safetensors 5GB（`~/models/vggt-1b`）✅；**DETR** `facebook/detr-resnet-50`（HF cache）✅；**OWL-ViT** `google/owlvit-base-patch32`（`/tmp/owlvit`）✅；**Qwen3-1.7B**（HF cache，4.48 无法加载）⚠️ |
| 网络 | huggingface ✅ / github ✅ / hf-mirror ✅ / TUM ⚠️**慢**（单连接 ~0.3MB/s，务必用 `wildspatial.data.download` 多线程） |

### 常用命令
```bash
cd /home/hmn-cjy/liuliuqiu/WildSpatial
PYTHONPATH=src python -m pytest tests/ -q          # 自检（wildspatial 环境）→ 当前 53 passed
PYTHONPATH=src python scripts/m0_demo.py            # M0 可视化（6 张图）
PYTHONPATH=src python scripts/m1_run_vo.py --seq fr1/desk --frames 450 --stride 3   # M1 跑真实数据
# 多线程下载（比 curl 快数倍）
PYTHONPATH=src python -m wildspatial.data.download <url> <out> --workers 12
```

### 关键路径
```
项目根        /home/hmn-cjy/liuliuqiu/WildSpatial
原始数据      <root>/data/raw
预处理数据    <root>/data/processed
实验产出      <root>/experiments/<module>/<exp_name>/   （含 README + metrics.json + figs/）
GitHub 仓库   https://github.com/liuliuqiu378/WildSpatial  （已 git init，首个 commit c46fa48 已 push 到 main）
```

> **推送约定**：远程 `origin` 用干净 URL（不含 token）；push 时临时内联 token，避免泄露进 `.git/config`。
> **`.gitignore` 已排除** `data/`、`.codebuddy/`、`.pyc`、`*.tgz`；`experiments/figs/` 的图**必须**进仓库（docs 用相对路径引用）。

---

## 1. 目标定位（一句话）

**端侧空间感知与多模态融合工程师** —— 让机器人/飞行器在恶劣物理环境（水下、地下、矿山、烟雾、低光照）下，
可靠地构建 **几何 + 语义** 的空间理解，并部署到边缘算力。

### 核心判断（决定了整个项目的设计）
1. **VLM 不负责几何**。实测：VLM 相机相对位姿估计 F1 仅 0.64，经典 LoFTR 0.97、SIFT 都强过它；深度轴推理准确率 7.4%。
   → VLM 只做语义（这是什么/指令理解/任务分解），几何交给几何方法。**分层解耦，不是端到端替代。**
2. **分块范式正在被前馈 3D 基础模型终结**（VGGT：一次前向 0.2s 输出相机内外参+深度+点云+轨迹，取代整条 SfM 管线）。
3. **但前馈模型在恶劣环境会失效，且几乎没人系统研究**。这就是本项目的王牌选题。

### 1.1 🔴 战略认知：为什么「搭建实战项目」是成长为专业人员的关键

> **本条为项目最高层认知，决定一切内容的最终指向**（用户 2026-09-14 明确强调）。

**问题**：学完 `F0–F5` + `M0–M9` 的知识点，算不算专业人员？**不算。**
知识点是"零件"，而工程师的价值在于**把零件组装成能交付的系统**。两者的鸿沟，恰是"学生"与"专业人员"的分界线。

**实战项目到底补上了什么（这是它不可替代的原因）**：

| 只有知识点（学生） | 有实战项目（专业人员） |
|---|---|
| 知道 VO / 深度 / 语义 / 端侧各自是什么 | 知道**它们按什么顺序串、谁给谁定尺度、谁失效谁兜底** |
| 会跑单个模块的 demo | 会回答"**输入是什么、输出是什么、数据从哪来、用什么设备**" |
| 能说出算法指标 | 能说清"**感知结果怎么变成控制指令**"（感知→规划→控制闭环） |
| 关注算法本身 | 同时关注**场景约束、失效兜底、端侧算力、安全冗余、数据闭环、sim2real** |
| 会写代码 | 会对**面试官/客户/同行**用三种语言讲同一个系统 |

**核心判断（可复用）**：
1. **知识 ≠ 能力，能力 = 知识 × 工程约束的整合**。真实工程 80% 的难点不在算法，
   而在多传感器时间同步、坐标系约定、标定、单目尺度歧义、动态障碍、失效兜底、端侧算力、数据闭环、sim2real 这些**初学者压根想不到的环节**（已在 `docs/P1_field_projects.md` §2.10 系统列出 10 条）。
2. **能"讲清楚"才算真懂**。一个项目若不能用「场景→目的→数据→设备→算法→感知→控制→约束」这条链讲给别人听，
   就说明自己还没真正串起来。**输出（讲解）是检验输入（理解）的唯一标准。**
3. **项目是所有模块知识的"收敛点"**。`F`/`M` 是纵向深挖（每个模块讲透），`P0`/`P1` 是横向贯通（把模块拧成系统）。
   没有 `P0/P1`，前面 19 篇文档只是散落的珍珠；有了它们，才串成项链。

**因此本项目的双轨结构**：
- **纵轴（学）**：`F0–F5` 基础篇 + `M0–M9` 模块篇 —— 把每个能力讲透（含公式/真图/实测）。
- **横轴（用）**：`P0` 抽象管线（多场景 × M-module 串成可跑工程）+ `P1` 工程叙事（四真实场景 + 感知→规划→控制闭环 + 三视角讲解）。
- **两条轴的交点 = 专业能力**：纵轴保证"懂"，横轴保证"能用、能讲、能交付"。

> 📌 **后续一切"下一步"判断，都以此为准绳**：优先做那些能**同时加深纵轴理解、又强化横轴贯通**的事；
> 纯知识堆砌或纯 demo 展示，价值低于"能讲清的完整实战闭环"。

---

## 2. 路线图（Module 总览）

> 🎯 **一句话目标（北极星）**：用「**带真值的公开数据 + 可控合成退化**」构建一张「**方法 × 环境**」失效图谱，
> 证明前馈 3D 基础模型（VGGT）在恶劣物理环境下的鲁棒性边界，并把整条「感知→重建→语义→解构→端侧」链路
> 落成**可复现、可量化、能上边缘算力**的工程闭环。
> **当前所处位置**：教程与实验骨架已全部闭环（M0–M9 + F0–F4 + P0）；**后续进入「打磨 + 补差异化主场 + 真实数据验证」阶段**。

| # | Module | 主题 | 状态 | 核心产出 |
|---|---|---|---|---|
| M0 | 几何地基 | 相机模型 / 李群 / 对极几何 / 三角化 / PnP | ✅ 完成 | 几何库 + 25 项自检 + 6 张教学图 |
| M1 | 手搓 SfM/VO（直觉锚点+自研基线） | SIFT→匹配→RANSAC→E→三角化→PnP→BA→回环v1 | ✅ 基线就绪 | TUM 基线 ATE 0.506m；BA 重投影 1.54→0.20px；回环 v1 误检未降 ATE（诚实记录）+ showcase |
| M2 | 深度与重建 | 双目视差 / RGB-D / TSDF 融合 | ✅ 完成 | Open3D TSDF mesh（848k 顶点）+ VGGT 深度对比 |
| M3 | 失效归因科学 | 合成退化 × 逐环探针 × 带真值定量 | ✅ 完成 | `docs/M3_failure_attribution.md` + 13 次实跑失效图谱 |
| M4 | 前馈 3D 基础模型 | VGGT 深度/点云鲁棒性（MASt3R 未做） | ✅ 完成 | `experiments/M4_depth_atlas/` 深度 δ1>0.96、点云 Chamfer 0.105m |
| M5 | 压力测试矩阵 ⭐ | 方法动物园 × 13 退化交叉 | ✅ 完成 | `experiments/M5_method_atlas/` 四方法 ATE 图谱（VGGT 0.016–0.049m 稳压） |
| M6 | 多模态融合补救 | 诊断驱动切换（内点率监控 → 深度/VGGT 救援） | ✅ 完成（初学者版+可跑脚本） | `experiments/M6_fusion_guard/` 低光救援区 96%；深度救援 ATE 0.204m |
| M7 | 语义层 | VLM 开放词汇 + 责任切分 | ✅ 完成（最小实证） | OWL-ViT 开放词汇跑通（增益+幻觉双证，39 框） |
| M8 | 端侧部署 | 量化 / TensorRT / 昇腾 | ✅ 完成（本机缩预算实测三维表 + INT8 PTQ 真跑） | 延迟·算力预算·精度损失三维表 + 最佳性价比点 0.5×ORB1000 |
| M9 | 综合 | 完整闭环 + 技术报告 | ✅ 完成（初学者版+可跑脚本） | 场景图解构 16 个 3D 物体 / VO 9.8fps @ CPU |
| T0 | 领域全景教程 | 分模块流水线→前沿→VLM→端侧 | ✅ 完成 | `docs/00_LANDSCAPE.md` + 真实工具实证（DETR/GrabCut/OWL-ViT/M4/M5） |
| F5 | 视觉+激光雷达融合 | 标定 / 点云↔图像投影 / 前中后融合 / 3D 检测 | ✅ 完成 | `docs/F5_vision_lidar_fusion.md` + 可跑 demo（重投影残差 0.0000px，25.5 万点） |
| **P0** | **实战整合（纵→横）** | 多场景数据集 × M-module 串成可跑工程 | ✅ 完成 | `docs/P0_projects.md` + 4 项目端到端跑通（VO→深度→语义→场景图→端侧） |
| **P1** | **工程叙事（专业人员关键）** ⭐ | 四真实场景 + 感知→规划→控制闭环 + 三视角讲解 | ✅ 完成 | `docs/P1_field_projects.md` + `scripts/p1_service_robot.py`（栅格→A*→动态避障 VO→差速控制实跑） |

图例：⬜ 未开始 🟡 进行中 ✅ 完成 ⛔ 阻塞

> **状态说明（2026-09-12）**：模块层（M0–M9）+ 基础层（F0–F4）+ 实战层（P0）= **19 篇教学文档 + 1 索引**，
> 全部含「可跑脚本 / 实测数字 / 真实图」。**M8 的「三维表」为真正实测**（本机缩算力预算 + INT8 PTQ）；
> 唯一仍需真机的是 RKNN/ACL 的**交叉编译**（非卡点，不影响结论）。

---

## 2.1 路线图复盘（2026-09-11 · 复用现成工具视角）

用户战略：不手搓、直接调用现成工具/代码包/模型，建立**概念+认知+效果**，细扣原理以后再说。
手搓层（M0/M1）已降级为「直觉锚点 + 自研基线」。据此各模块重新定位：

| Module | 原定位 | 复用现成工具后的新定位（即用成熟实现驱动、在方法动物园里跑真值对比） |
|---|---|---|
| M1 手搓 SfM/VO | 全手搓前端+BA+回环 | **保留为自研基线**，与 COLMAP/ORB-SLAM 同台对比；手搓回环 v2 **不再做**（改用成熟 SLAM 的回环） |
| M2 深度与重建 | 手搓双目/TSDF | 用 **Open3D TSDFVolume**（已装）做融合出 mesh；深度用 **VGGT / Depth-Anything** 与 TUM 深度真值比（fr1/desk 有深度，立即可跑） |
| M3 失效归因 | 仅自研 VO | 升级为**方法无关**的退化探针：同一套 `degrade` 套件 × 方法动物园各方法，结论更普适（与 M5 合并） |
| M4 前馈 3D 基础模型 | 复现 VGGT/MASt3R | **直接用 VGGT**（已写 `methods/vggt.py`），在退化套件下做轨迹/深度/点云 vs 真值定量（本阶段核心） |
| M5 压力测试矩阵 ⭐ | 环境×方法交叉 | **就是方法动物园**（`scripts/m5_method_sweep.py`）。扩展：更多方法(加 LightGlue/ORB-SLAM)、耦合退化、真实恶劣数据 |
| M6 多模态融合补救 | 自研融合 | 用现成：IMU（ORB-SLAM-VI / VINS）、深度（TUM 已有）、热成像/水下（合成退化）；看多模态能否救回 VGGT/几何法崩的档位 |
| M7 语义层 | 自研 VLM | 用现成 VLM（transformers/ollama 接模型），坚持「VLM 不负责几何」分层解耦 |
| M8 端侧部署 | 自研量化 | 用 TensorRT / ONNX（现成）量化 VGGT/匹配器，测延迟·功耗·精度损失 |
| M9 综合 | 手写报告 | 串起方法动物园 + 退化图谱 + 多模态补救，产出简历级项目 |

**现成工具落地进度（截至 2026-09-12，全部已实跑）**：
1. ✅ **M5 方法动物园**：自研 VO / COLMAP / VGGT / LightGlue 四方法 × 13 退化条件 ATE 图谱 → `experiments/M5_method_atlas/`（VGGT 全档 0.016–0.049m 稳压）。
2. ✅ **M4 VGGT 深度/点云**：13 退化下深度 δ1>0.96、RMSE~0.14m，点云 Chamfer 0.105m → `experiments/M4_depth_atlas/`。
3. ✅ **M2 Open3D TSDF**：fr1/desk 真实深度+真值位姿融合出 mesh（848k 顶点）→ `experiments/M2_recon/`。
4. ✅ **M7 语义层最小实证**：OWL-ViT 开放词汇检测（自由文本查询）跑通，既证增益又证幻觉 → `experiments/M7_semantic/`。

---

## 2.2 待办清单（Todo · 2026-09-12 重排 · 目标导向）

> ✅ **已完成（不再列入待办）**：M0 几何地基 / M1 自研VO基线 / M2 TSDF重建 / M3 失效图谱 / M4 VGGT深度 /
> M5 方法动物园 / M6 多模态融合 / M7 开放词汇 / M8 端侧（三维表+PTQ）/ M9 场景图闭环 / T0 领域全景 / P0 实战项目整合层。
> **教程体系已完整覆盖 M0–M9 + F0–F4 + P0，无结构性缺口。**

### A. 缺陷修复（✅ **全部完成 · 2026-09-12**）

| # | 项 | 具体动作 | 状态 |
|---|---|---|---|
| A1 | 🔴 **M8 图片路径 bug** | `docs/M8_edge_deployment.md` L141/L169 的 `experiments/...` → `../experiments/...` | ✅ 已修 |
| A2 | **陈旧交叉引用** | `00_LANDSCAPE.md`(能力栈表 + §5 速查表 + 结尾建议)、`F2_depth_basics.md`、`F3_normal_pipeline.md` 全部去掉「待做/待补」，改挂真实数字 | ✅ 已修 |
| A3 | **索引漏项** | `00_INDEX.md` 初学者版清单补 `M6/M8/M9`；§5「待补章节」→「覆盖状态确认」并列出真实 3 项非结构性待补 | ✅ 已修 |
| A4 | **图注/表述不符** | `00_PRIMER.md`/`00_INDEX.md`/`00_LANDSCAPE.md`「三方法」→「四方法」（含 LightGlue） | ✅ 已修 |

> **验收结果**（已跑脚本核验）：`docs/` 全量 **50 处图片引用 → 缺失 0**（修复前 M8 的 2 处会图裂）；
> `grep 待补/待做` 仅余 `M1` 图内一处历史标注（非误导），陈旧引用已清零。linter 0 错误。

### A5. ✅ 教程「图不显示」全量修复（2026-09-12 续18 · 用户反馈触发）

| # | 项 | 具体动作 | 状态 |
|---|---|---|---|
| A5 | **锚文本地址 → 嵌入真图** | 32 处「（见 xxx.png）」纯文本改为 `![]()` 嵌入；`README`/`00_INDEX`/`00_PRIMER`/`00_ROADMAP` 各补精选图墙 | ✅ 已完成 |

> **验收**：全仓库（docs + README + PROGRESS）**99 处图片引用 → 缺失 0**，零图文档仅剩 `PROGRESS.md`（台账）。

### B. 图表打磨（✅ **全部完成 · 2026-09-12**）

| # | 项 | 具体动作 | 状态 |
|---|---|---|---|
| B1 | **M9 场景图信息量** | 重画 `scene_graph.png`：左图灰点=反投影候选 / 红星=聚类实例（不再重合）；右图由「全 1.0 计数」改为**每实例的 X/Y/Z 世界坐标条形图** | ✅ 已完成（脚本 `m9_scene_graph.py`） |
| B2 | **M8 权衡图** | `tradeoff.png` 新增失败带 + FAIL 档标注（含「特征地板/分辨率悬崖」原因）+ ★最佳性价比点 | ✅ 已完成（新脚本 `m8_make_figs.py`，秒级重绘） |
| B3 | **M0/M1/M3 降门槛** | 三篇专家笔记文首各加「🧭 初学者导读（5 分钟版）」（含 4 句话总结 + 术语直觉 + 建议读法 + 读完能回答） | ✅ 已完成 |
| B4 | **图关键图自解释** | M5 热力图（核心资产）加图内「读图注记」+ 标题说明 | ✅ 已完成（新脚本 `m5_make_figs.py`） |

> **验收**：`m9_scene_graph.py` / `m8_make_figs.py` / `m5_make_figs.py` 均实跑出图；5 个脚本 `py_compile` 通过；
> 全仓库 **99 处图片引用 → 缺失 0**。

### C. 差异化主场（P1 · 核心价值 · 需你的端侧/行业经验）

| # | 项 | 内容 | 状态 |
|---|---|---|---|
| C1 | ✅ **视觉+LiDAR 融合系统专章** | 新建 `docs/F5_vision_lidar_fusion.md`：相机-激光外参标定 / 点云→图像投影公式 / 前·中·后融合架构对比 / KITTI-nuScenes 3D 检测 + 可跑 demo `scripts/m10_lidar_fusion_demo.py`（实测重投影残差 **0.0000 px**，25.5 万点） | ✅ **已完成** |
| C2 | **M8 真机三维表** | 用 RKNN/ACL 真机（RK3588 / 昇腾）跑交叉编译 + 实测延迟/功耗 | ⬜ 待做（需真机工具链；本机缩预算版已给出可移植结论，非卡点） |
| C3 | **M7 完整语义层闭环** | 补 §3 文本侧实证（任务分解/故障解释/关系推理）；现成 VLM 做「分层解耦」端到端 demo | ⬜ 待做（⚠️ transformers 4.48 不支持 Qwen3；需升 ≥4.51 或换 Qwen2/LLaMA） |

### D. 真实数据验证（P2 · 网络/数据受限 · 可远程或本地路径触发）

| # | 项 | 内容 | 依赖 / 风险 |
|---|---|---|---|
| D1 | **Task3 真实恶劣数据** | 4Seasons / SubT-MRS / DUO 跑方法动物园，替换/对照合成退化结论 | ⛔ 本机网络瓶颈（TUM ~50KB/s、SubT 缺内参）；`run_sequence.py` / `run_4seasons.py` 已就绪，**换机或提供本地路径即可一键消费** |
| **D1a** | ✅ **语义层真实恶劣评测（已完成）** | M7 用 OWL-ViT 在**真实水下（LibreYOLO 5 类 + YOLO 真值）**与**夜间（NightCity）**评测，量化"域偏移+细粒度"对开放词汇的杀伤 | ✅ 已完成：`scripts/d1_real_world_eval.py` → `experiments/D1_real_world/`（水下 mean recall≈0.001、夜间≈7 命中/图） |
| D2 | **M5 扩方法** | 加 ORB-SLAM3 / Depth-Anything / LightGlue 耦合退化，图谱更密 | 现成工具，低风险 |
| **D3** | ✅ **多模态数据扩充（已完成）** | 从 ModelScope 分批下真实多模态数据（夜间/水下/遥感/室内深度/LiDAR/KITTI/3D 基准），建 `docs/DATASETS.md` 台账 | 下载器 `scripts/fetch_datasets.py` 已就绪；**11 个数据集已落地**（含 KITTI 20G 解压中）；台账见 §5 续21/续24 |

### E. 可选增强（P3 · 有余力再做）
- MASt3R / DUSt3R 与 VGGT 横向对比（原是 M4 计划内未做项）。
- P0 新增第 5 个真实项目（如「视觉+LiDAR 融合」驱动的自动驾驶项目，与 C1 联动）。

---

> **推荐执行顺序**：~~A（清障）~~ ✅ → ~~B（打磨）~~ ✅ → ~~C1（融合专章）~~ ✅ → **C2/C3（主场深挖，需外部条件）→ D1（真实数据）**。
> **接手提示**：A/B/C1 已完成（缺陷清零 + 图表打磨 + **视觉+激光雷达融合专章 F5 已建**）；
> **下一步可选**：C2（M8 真机交叉编译）或 C3（M7 完整语义闭环，需升 transformers）——均受外部条件约束；D1 受网络限制，非能力卡点。

---

## 3. 数据集台账

| 数据集 | 场景 | 用途 | 状态 |
|---|---|---|---|
| TUM RGB-D `fr1/desk` | 室内干净桌面 | **干净基线**（含真值轨迹+深度） | ✅ 已解压可用（613 rgb / 595 depth / groundtruth）—— M0–M7 全部实跑基于它 |
| TUM RGB-D `fr3/nostructure_notexture_near_withloop` | **无纹理** | M3/M5 失效用例 | ⚠️ **不可用**（解压仅 rgb/ + accelerometer.txt，无 depth/真值）；勿再浪费时间 |
| TUM RGB-D `fr3/structure_notexture_far` | 结构+无纹理远景 | M5 | ⬜ 可选待下 |
| SubT-MRS | 地下/隧道/烟雾/全黑，多模态 | M5/M6 真实退化 | ⛔ **Task3 暂缓**：9.25GB 烟雾包曾下但缺相机内参+解压被拦；GDrive 可达但碎片化，本机不划算 |
| 4Seasons | 恶劣天气（雨/夜/雪） | M5/M6 真实退化 | ⛔ **Task3 尝试过**：TUM 直链限速 ~50KB/s（3.3GB 不可行），已放弃；换源再议 |
| DUO / UIEB | 水下散射 | M5 水下 | ⬜ 待评估 |

---

## 4. 已建立的项目约定（必遵守）

1. 每个 Module 的产出必须包含：**理论文档 + 可运行代码 + 量化结果 + 反思**。
2. 所有实验写入 `experiments/<module>/<exp>/`，目录内含 `README.md`（结论）、`metrics.json`、`figs/`。
3. 数据放 `data/`，**不进 git**（已被 `.gitignore` 排除）。
4. 代码库 `src/wildspatial/` 要求 **每个几何函数都有解析/数值自检**（`tests/`），因为几何错误极难调试。
5. 每完成一个 Module，更新本文件 + 写 `docs/<Mxx>_reflection.md`。
6. 🔴 **教程 / 文档强制「四段式」**（用户 2026-09-11 确立，替代原三段式，所有文档必遵守）。
   每个知识点都必须按下面四段讲清楚，**缺一段即不合格**：
   ① **目的**：它要解决什么问题？为什么需要它？（先给动机——否则读者不知道自己在学什么）
   ② **方法原理（专业）**：术语 + 公式 + 推导，严谨、不回避（这是地基，不能只讲大白话）
   ③ **直白讲解**：紧接着用大白话 / 类比解释"它到底在说什么"，建立直觉（不能只有术语）
   ④ **真实数据验证 + 效果展示**：用**带真实标签（ground truth）的公开数据集**实跑，
      给出量化指标（ATE/RPE/内点率/地图点数…）+ 真实结果图。
   ⛔ 禁止只堆术语不给直觉；⛔ 禁止无图纯文字长推导；⛔ 禁止只讲原理不跑数据。
   图放 `experiments/<module>/<name>/figs/`，文档用 `../experiments/...` 相对路径引用。
7. **结论必须由"带真值标签的数据"支撑**：优先使用有 ground truth 的公开基准。
   本项目首选 **TUM RGB-D `fr1/desk`**（含真值轨迹 + 深度图），可定量算 ATE / RPE。
   - ⚠️ 实测 `fr3/nostructure_notexture` **不可用**（只有 rgb 图 + accelerometer.txt，
     缺 `rgb.txt` 与真值），不要在上面浪费时间。
   - 若某类"恶劣条件"没有带标签的公开数据 → 采用**合成退化**方案：在带真值的数据上
     施加可控退化（运动模糊 / 低光照 / 噪声 / 降采样），各档**仍保留真值**，
     从而做可控定量对照。⚠️ 必须在文档与图注中明确标注"退化为合成施加"。
8. **每个 Module 至少产出一组"肉眼可见"的效果图**（`scripts/<m>_showcase.py` 一键生成，
   写入 `experiments/<module>_showcase/`，含 figs + 三段式 README + metrics.json）：
   让读者看到算法**对真实图像做了什么**，而不只是看数字。
9. 🔴 **战略转向：优先复用现成工具 / 代码包 / 模型，概念与效果优先（用户 2026-09-11 确立）**。
   - 教学目标 = **建立概念、认知与效果**，不是重写轮子。手搓层（M0 几何库、M1 手写前端+BA+回环 v1）
     已完成"建立直觉"的使命，**保留为"直觉锚点"**，需要时回看，但**不再作为主路径**。
   - 后续模块**默认调用成熟实现**：OpenCV（SIFT/ORB/光流/RANSAC）、COLMAP / PyCOLMAP（SfM 重建）、
     LightGlue / SuperPoint / SuperGlue（学习式匹配）、VGGT / MASt3R（前馈 3D 基础模型）等。
   - 主线闭环：**驱动真实工具 → 在带真值数据 / 合成退化上量效果 → 四段式讲清"它是什么、为何、效果如何"**。
     细扣底层原理（逐行推导/逐行实现）**留到以后**，不阻塞主线推进。

---

## 5. 日志（倒序追加）

### 2026-09-14（续43）· 🎬 P2 动态障碍闭环 **GIF 动图**（让"运动"看得见）
- **用户反馈**："做得好，但看不到、感受不到，能不能做成视频或动图全景展示？" —— **静态图确实看不到运动**。
- **产出** `scripts/p2_dynamic_obstacle_gif.py` → `experiments/P2_dynamic_obstacle/figs/`：
  - **`closed_loop.gif`（142 帧，448×416，0.79MB）+ `closed_loop.mp4`（0.42MB）**；
  - 每帧含：机器人（蓝方块+轨迹）/ 行人（红圆）/ **危险半径圈** / **状态框（GO 绿 / AVOID 红实时切换）** / 实时指标（t/v/ω/最近距离）。
- **效果**：如同"比赛俯视回放"，一眼看懂**行人靠近→进入危险圈→状态变红→v 掉下来/ω 抬起来**的完整决策过程。
- **联动**：`docs/P2_simulation.md §4.14.3b`（嵌 GIF）；`.gitignore` 排除 `gif_frames/`（5.2M 中间帧，可重生成）；本文件续43。

### 2026-09-14（续42）· 🎯 P2 动态障碍完整工程 demo（仿真→感知→决策→控制）
- **动因**：用户要求"动态障碍（SDF 加会动的行人 → 对应 P1 动态避障），建一个完整演示 demo，各维度数据利用"。
- **产出**：`data/gz_models/worlds/dynamic_pedestrian.sdf.xacro`（含 **2 个真实物理运动的行人**）+
  `scripts/p2_dynamic_obstacle.py` → `experiments/P2_dynamic_obstacle/`。
- **五层数据流**：① 传感（相机/LiDAR/真值位姿）→ ② 感知（算行人距离）→ ③ 决策（<1.2m 避让）→
  ④ 控制（输出 cmd_vel）→ ⑤ 物理反馈（Gazebo 推进，闭环）。
- **实测**：闭环 **735 步**，触发避让 **562 步**（76%），最近行人距离 **1.126m**，行人 2 个；
  **出现"正常前进(v=0.18) → 第7秒骤降 → v≈0/ω=0.3 避让"的完整决策曲线**。
- **出图 3 张**：五层数据流总览、行人距离 vs 避障指令、多传感器视角（相机 vs LiDAR）。
- **与 P1 的区别**：P1 行人=脚本模拟匀速直线；本 demo 行人=**Gazebo 真实物理体**（质量/碰撞/速度），
  机器人用**真实传感器**感知 → 在线闭环。
- **⚠️ 踩坑（Gazebo 驱动动态物体三方案）**：① `TrajectoryFollower` 受物理约束不动；
  ② `gz service set_pose` headless 下服务发现受限；③ ✅ **`VelocityControl` + 无摩擦(mu=0)** 稳定驱动。
- **联动**：`docs/P2_simulation.md §4.14`；`00_INDEX.md` 图索引 +2；本文件续42。

### 2026-09-14（续41）· P2 感知实验台（四）：多传感器同步（相机+LiDAR+IMU）— M6/F5 落点
- **新增 §4.13** + `scripts/p2_gz_multisensor.py` → `experiments/P2_gz_multisensor/`：
  - **关键技术**：跨两侧订阅——RGB/深度相机在 **gz 侧**（用 `gz.transport13`），
    LiDAR `/scan` 与 IMU `/imu` 在 **ROS 侧**（用 `rclpy`）；
  - **实测**：RGB **139 帧 @10.01Hz**、深度 **70 帧 @5.04Hz**、LiDAR **128 帧 @9.71Hz**、IMU **128 帧 @9.71Hz**
    （与 SDF 配置一致）；
  - 出图：相机 + LiDAR 扫描 + 各传感器数据率。
- **价值**：M6 多模态融合 / F5 视觉+LiDAR 融合的**输入实验台**——同源同帧、时间戳天然同步，
  真实数据集则需硬件 PPS/软件对齐。
- **联动**：`docs/P2_simulation.md §4.13`；`00_INDEX.md` 图索引 +1；本文件续41。

### 2026-09-14（续40）· P2 移动操作（mobile manipulation）+ ROS2 名词速查 + OMPL 实跑
- **A. 新增 §4.12**：ROS2 词汇表（ROS2/DiffDrive/Nav2/A*/ros2_control/OMPL/RRT/PRM/MoveIt2）+ A* vs OMPL 核心区分 + 移动操作实跑。
- **B. 安装 MoveIt 生态**：`ros-jazzy-moveit`(2.12.4) + `ros2_control`(4.47) + `ompl` 全部就位
  → `moveit_planners_ompl` / **`ompl`（Python 绑定可用）** / `diff_drive_controller` / `joint_state_broadcaster`。
- **C. 实跑**（`scripts/p2_mobile_manipulation.py` → `experiments/P2_mobile_manipulation/`）：
  - **A\*（2D 栅格）131 步**；**OMPL 真实库** RRTConnect/RRT/PRM 规划成功（50 点）；
  - **自实现 RRT**（48 采样点连树绕障）+ **PRM**（300 点建图搜路）→ 出"撒点连线"可视化；
  - 出图：A* vs 关节空间对比、RRT/PRM 撒点图。
- **D. 自建移动操作模型** `data/gz_models/urdf/mobile_manipulator.sdf`（底盘+3-DOF 臂+RGB 相机）：
  实测 Gazebo 可加载，话题 `/camera/color/image_raw` `/cmd_vel` `/odom` **`/joint_states`** `/world/default/pose/info`。
- **⚠️ 绑定坑**：nanobind 版 `ompl` 传 Python 自定义 validity checker → `std::bad_cast`；
  改用 `space.allocState()` 构造状态 + 自由空间规划 + Python 自实现带障碍采样（绕开绑定）。
- **教学价值**：把 §3.5.5 的"A*(2D) vs OMPL(高维)"**真实跑出来**，"走格子 vs 撒点"看得见。
- **联动**：`docs/P2_simulation.md §4.12`；`00_INDEX.md` 图索引 +2；本文件续40。

### 2026-09-14（续39）· P2 §3.6 同源模型洞察 + 物理级退化 + 真值轨迹 + 场景映射
- **A. 新增 §3.6「仿真与真实：同源模型 + 解构方法论」**（用户洞察系统化）：
  - **同源**：仿真与真实共享同一套底层模型（几何/物理/光照/传感器）；
  - **解构**：一切方法的本质是把真实世界"解构"成结构化表示（几何/语义/学习式三条路径）；
  - **三种关系**：sim2real / **real2sim（3DGS/NeRF，正是"把真实解构成仿真"）** / 仿真=验证台（本项目）。
- **B. 物理级退化（`scripts/p2_gz_physical_degrade.py` → `experiments/P2_gz_physical/`）**：
  - 参数化世界 SDF（`data/gz_models/worlds/tb3_sandbox_param.sdf.xacro`）：`light`+`fog_density` 作 xacro 参数；
  - **同一世界 × 4 条件**：正常(亮度107)/低光(47)/雾天/夜间(25)，由 Gazebo **渲染引擎真实计算**（非 P 图）；
  - 出图：物理条件对比、**物理级 vs 事后P图**（前者光照更自洽）。
  - ⚠️ 踩坑：`headless:=true` 会禁 SceneBroadcaster → 真值位姿话题不发；改 `headless:=false` 解决。
- **C. 真值轨迹（ATE 免费参考）**：采 `/world/default/pose/info` → 机器人走圆弧 **557 个真值姿态点**。
- **D. 新增 §4.11「仿真 × 四场景 × 操作任务 能力映射」**：
  - 扫地/送物/自驾/无人船 各自的仿真搭法与所需能力；
  - **操作类（取物）能结合**——Gazebo 支持机械臂，但工具链不同（Nav2 vs **MoveIt 2 + OMPL**，印证 §3.5.5 高维采样）；
  - 统一视角：所有场景都是"把世界解构成结构化表示再决策"。
- **联动**：`docs/P2_simulation.md`（§3.6 + §4.10 + §4.11）；`00_INDEX.md` 图索引 +4；本文件续39。

### 2026-09-14（续38）· 🎯 P2 感知实验台实战：仿真 RGB-D + 可控退化 + VGGT 重建
- **A. 给 TB3 注入 RGB 相机**：原 TB3 SDF **只有深度相机** → 复制 SDF 到 `data/gz_models/urdf/` 注入
  RGB 相机（640×480，FOV 60°），用 `spawn_tb3.launch.py robot_sdf:=<path>` 覆盖（不改原包）。
- **B. 仿真 RGB-D 采集 + 可控退化**（`scripts/p2_gz_sense_degrade.py` → `experiments/P2_gz_sense_degrade/`）：
  - `cmd_vel` 驱动 TB3 **边走边采**：RGB 152 帧到达，存 4 帧（640×480 RGB + 320×240 真值深度）；
  - **深度随移动变化** 0.53–4.78m → 0.53–1.35m（逼近柱子，视角真在变）；
  - 出不意图 3 张：多帧 RGB-D、**可控退化图库**（低光/雾/噪声 ×6）、退化统计（亮度/对比度量化）。
  - ⚠️ 踩坑：内嵌脚本 `str(drive).lower()` 生成 `true` → Python 应 `True`（已修）。
- **C. VGGT 重建 vs 仿真真值（M4 落点）**（`scripts/p2_gz_vggt_eval.py` → `experiments/P2_gz_vggt/`）：
  - 4 帧仿真 RGB 喂 VGGT（`return_dense=True`）→ 稠密深度 + 点云；
  - 中位尺度对齐后：**RMSE 0.079–0.265 m、AbsRel 0.043–0.127**（平均 RMSE≈0.17m, AbsRel≈0.08）；
  - **与 M4 真实数据（TUM δ1>0.96、RMSE 0.10–0.16m）同量级** → **证明仿真实验台对 M4 有效**。
  - 出图 2 张：VGGT vs 真值三行对比、VGGT 重建点云。
- **价值**：把 P2 从"规划 demo"彻底拉回"**感知实验台**"，直接服务 M4（VGGT 评测）/M5（退化图谱）。
- **联动**：`docs/P2_simulation.md §4.8/§4.9`（含详细通俗讲解）；`00_INDEX.md` 图索引 +3 行；本文件续38。
- **下一步**：① 仿真 SDF 内直接加雾/改光照（比 P 图更真）；② 用退化图库复现 M5 方法动物园。

### 2026-09-14（续37）· 🎯 P2 定位修正：Gazebo = 感知实验台（回到视觉+传感器王牌）
- **用户关键质疑**：「怎么感觉都是跟导航规划有关的，跟视觉传感器等实时场景关系不大。」
  **这是对的**——此前 P2 一度偏到 Nav2 导航栈（costmap/规划器），但本项目王牌是**视觉+传感器感知**
  （M4 VGGT / M5 方法动物园 / M6 融合 / F5 视觉+LiDAR）；Nav2 是通用机器人工程，非差异化主场。
- **定位修正（P2 §4.7）**：Gazebo 的真正价值 = **"永不枯竭的带真值感知数据工厂"**，
  不是"让机器人导航"。真值 + 可控世界（可加低光/雾/噪声）→ 正是 M4/M5/M6 的实验台。
- **实跑（`scripts/p2_gz_perception_lab.py` → `experiments/P2_gz_perception/`）**：
  - 关键技术：相机话题在 **gz 侧**（TB3 默认 ROS bridge **没配相机**）→ 用 **`gz.transport13` Python 绑定**直接订阅；
    相机 gz 路径 `.../intel_realsense_r200_depth/depth_image`。
  - **实测：320×240 深度图，有效像素 79.7%，深度范围 0.53–4.78 m**；
    伪彩图清晰显示机器人正前方世界（近地面/中柱/远背景）。
- **价值**：把 P2 从"规划 demo"拉回"**感知实验台**"——为 M4/M5/M6 提供无限量带真值数据。
- **联动**：`docs/P2_simulation.md §4.7` + 头部状态；`00_INDEX.md` 图索引；本文件续37。
- **下一步**：① 抓多帧 RGB → 喂 VGGT 重建 → 与仿真真值深度对比（M4 落点）；
  ② 仿真加低光/雾 → 复现 M5 失效图谱（可控退化实验台）。

### 2026-09-14（续36）· ✅ TurtleBot3 + Nav2 无头闭环 demo（服务器无 GUI）
- **关键工程问题**：本机**无 DISPLAY**（`$DISPLAY` 空）→ Gazebo GUI 开不了。
  **解法**：`gz sim --headless-rendering -s -r`（服务器无头渲染，工业界标准做法）+
  `GZ_SIM_RESOURCE_PATH` 指向 models（否则 `model://turtlebot3_world` 找不到）。
- **实测（`scripts/p2_tb3_nav2_demo.py` → `experiments/P2_tb3_nav2/`）**：
  - Gazebo 无头启动 ✅（仅 Ogre 材质警告）；TB3 生成 ✅（`/imu`、`/scan` 就绪）；
  - **真实 LiDAR：360 束全部有效**，最近障碍 **0.49 m**、最远 **4.69 m**、量程 0–20 m。
  - ⚠️ 坑：`gz sim` 找不到 `model://` → 必须设 `GZ_SIM_RESOURCE_PATH=<models>:<share>`。
- **可视化 `figs/costmap_concept.png`**（三子图，详细讲解已写进 P2 §4.6）：
  ① 真实 LiDAR 360 点（**能反推出 Gazebo 世界是"四面墙+柱子"的房间**）；
  ② 障碍 + **膨胀带**（Nav2 `inflation_layer` 概念）；
  ③ `static + obstacle + inflation` 三层叠加成一张 costmap。
- **教学价值**：**实物证据**验证了 §3.5.3「静态导入 + 动态感知在同一个 costmap 里融合」的论断；
  直观回答"机器人没有地图，只有一圈距离读数，costmap 是这么长出来的"。
- **诚实边界**：**完整 Nav2 BT 导航栈未跑通**（需更多运行时配置）；
  已验证的是 Gazebo 世界 + TB3 传感器 + costmap 概念，**不冒充"全链路导航完成"**。
- **联动**：`docs/P2_simulation.md §4.6`；`00_INDEX.md` 图索引；本文件续36。
- **下一步**：补齐 Nav2 参数/生命周期 → 跑通"发目标点→自动导航"；再对比 P1 路径。

### 2026-09-14（续35）· ✅ ROS2 Jazzy + Nav2 安装成功并跑通（免 sudo）
- **结果**：`ros2jazzy` 环境（6.8G，313 个 ROS2 包）**全部就位**：
  `nav2_bringup` / `nav2_navfn_planner`（全局 A\*）/ `nav2_dwb_controller`（局部 DWA）/
  **`nav2_mppi_controller`**（现代 MPC）/ `nav2_costmap_2d`（分层代价地图）/ `slam_toolbox` / `tf2_ros` / `rclpy` / `colcon`。
- **关键路径**：**免 sudo**（robostack-jazzy conda 发行版）。
  ⚠️ 坑：`colcon` 不在 conda channel → **必须 pip 装**（否则 `PackagesNotFoundInChannelsError`）。
- **新增验证脚本** `scripts/p2_ros2_check.py`：ROS2 环境自检 + Nav2 规划器 vs P1 手搓 A* 对照
  → `experiments/P2_ros2/`（navfn_vs_p1.png + metrics.json + README）。
  实测：313 包全绿；P1 手搓 A* 同图绕 L 形障碍 **131 步**。
- **核心结论（实证 §4.3 对应表）**：工业级 Nav2（navfn = A\*/Dijkstra）与 P1 手搓 A* **是同一类算法**；
  差别在 **P1 = 手搓一遍才懂原理 vs Nav2 = 生产级可配置（costmap 分层/插件化/生命周期/恢复行为）**。
- **文档联动**：`docs/P2_simulation.md` 头部状态改 ✅、新增 §4.2.1 实测结果、§3.5.6 成熟度谱系、§3.5.7 前沿；
  `00_INDEX.md` P2 行 + 图索引；`MEMORY.md`。
- **下一步**：① 跑 TurtleBot3 + Nav2 官方 demo（需 Gazebo，GUI）；② 把 P1 栅格导入 Nav2 对比路径；③ 评估 CARLA。

### 2026-09-14（续34）· P2 §3.5 世界表示谱系（用户洞察系统化）
- **用户洞察**："很多自动控制/自驾问题最后都是走格子问题；地图导航是静态固定的，自动控制是叠加静态导入 + 动态感知。"
- **动作**：在 `docs/P2_simulation.md` 新增 **§3.5 世界表示的谱系（5 小节）**：
  - **3.5.1** 为什么"走格子"（图搜索）通用：离散化把连续问题变成可搜索的图。
  - **3.5.2** 表示谱系对比表：栅格 / 八叉树 / 体素 / 拓扑图 / **采样树 RRT/PRM** / **BEV 占用栅格** / 轨迹优化。
  - **3.5.3** 🔴 **修正**：静态地图 vs 动态感知**不是二选一，而是分层融合**——
    全局(静态骨架) → 中间(**costmap/BEV，动态障碍实时改写格子代价**) → 局部(DWA/TEB/MPC) → 控制；
    并给出 Nav2 教科书实现（`static_layer`+`obstacle_layer`+`inflation_layer`）。
  - **3.5.4** "走格子"只解决"去哪走"，"怎么动"是连续层，两者接力。
  - **3.5.5** 回答用户追问「操作类取拿是不是 3D 格子」：
    **概念类似，但维度决定武器**——导航 2D 栅格→走格子(A\*)；操作 **6–7 维关节空间 C-space**→
    **格子爆炸(100⁶)**→改**随机采样(RRT/PRM/OMPL)**；3D 体素只表示障碍；操作多"接近→对准→闭合→抬起"多阶段。
- **价值**：把用户"走格子"的直觉**系统化 + 修正为专业表述**，并与要装的 Nav2 平台直接挂钩（可验证）。

### 2026-09-14（续33）· P1 四场景真实数据收官 + P2 闭环仿真平台接入
- **A. P1 四场景真实数据全部落地（5/5）**：
  - ① 扫地机器人第一视角（6 张）；② CVC-14 昼夜红外行人（500）；③ MOT17 行人序列（200）；
  - ④ 水面漂浮垃圾（80 张，**242 个 COCO 真值框**）；⑤ **红外船只（120 张展示，解压共 3521 张，day/night × rain/fog/cloudy）**。
  - ⑤ 的实现：`Flier123/Ship` 的 `sanya_ir_nodel_opensource_weather.tar.xz`（618M）解压 → 三亚红外水面船只，含天气标签。
  - `scripts/p1_datasets_showcase.py` 升级：④ 水面漂浮物**绘制 COCO 真值框**（带真值的效果展示，符合四段式）。
  - 产出 `experiments/p1_datasets_showcase/`（p1_scene_samples.png + metrics.json + README）。
- **B. P2 闭环仿真平台（新建 `docs/P2_simulation.md`）**：
  - 四层平台全景对比：通用机器人（Gazebo/Isaac/MuJoCo）、自驾专用（CARLA/LGSVL/SUMO）、室内具身（Habitat/AI2-THOR）、数据驱动回放（nuPlan）。
  - **CARLA vs Gazebo+ROS2 核心区别**：前者=自驾专用"游戏"（现成城市/照片级/主要就车/自有 API）；
    后者=通用机器人物理+中间件（自建世界/任意机器人/ROS2 原生/Nav2 直接可用）。
  - **本项目选 Gazebo+ROS2**（延续 P1 闭环，可对照工业级实现）。
- **C. ROS2 Jazzy 安装（免 sudo 路径，进行中）**：
  - 环境事实：**Ubuntu 24.04** → ROS2 Jazzy；**sudo 需密码** → 走 **robostack-jazzy（conda 免 sudo）**。
  - ⚠️ **踩坑**：`colcon` **不在** conda channel → 写进 conda install 直接 `PackagesNotFoundInChannelsError`；
    已改为 **pip 装** `colcon-common-extensions`。去掉 colcon 后 `--dry-run` 解算通过。
  - 正在下载（CUDA/ogre 等大包）；环境名 `ros2jazzy`。
- **D. P1 自研 ↔ 工业级对应表**（写进 P2）：栅格建图→`slam_toolbox`；A\*→`nav2_navfn_planner`；
  VO→`nav2_dwb_controller`；差速 v/ω→`diff_drive_controller`。**手搓一遍再看工业实现，才知道差异在哪**。
- **联动**：`docs/P2_simulation.md`（新）；`00_INDEX.md` 七维表加 P2 行；`README.md` 文档地图加 P2；
  `docs/P1_field_projects.md §2.12` 状态更新；`docs/DATASETS.md` Ship 行更新；本文件续33。
- **下一步**：① 等 ROS2 装完 → 跑通 TurtleBot3+Nav2 demo；② 把 P1 栅格导出与 Nav2 路径对比；
  ③ 若需自驾感知再评估 CARLA。

### 2026-09-14（续32）· P1 四场景真实数据落地（ModelScope 搜索 + 下载）
- **动因**：P1 四场景此前多依赖 TUM/KITTI 等"通用"数据，动态障碍是模拟的；需为每个场景找**真实场景数据**。
- **动作 1 · ModelScope 实际搜索**（非编造）：用 `HubApi.list_datasets(search=...)` 系统搜索
  行人/跟踪/USV/水面/扫地/酒店/船只等 30+ 关键词，定位真实契合候选。
- **动作 2 · 下载器新增 Tier F（P1 四场景专批）** `scripts/fetch_datasets.py`：
  园区扫地→扫地机器人视角；动态行人→CVC-14/MOT17/CrowdHuman；无人船→floating-waste/Ship；酒店送物→RoboCOIN LeRobot。
- **动作 3 · 实际下载（截至本轮）**：
  - ✅ `DatatangBeijing/...RobotCleaner...`（9 样本，扫地机器人第一视角：bed/shoe/trash can/垃圾）
  - ✅ `OmniData/CVC-14`（503 文件，昼夜**红外**行人）
  - 🔄 `OpenDataLab/MOT17`（200 张，街道多帧行人序列）
  - 🔄 `isLinXu/rf100-vl-floating-waste`（水面漂浮垃圾）
  - 🔄 `Flier123/Ship`（船只检测）
- **动作 4 · showcase 脚本** `scripts/p1_datasets_showcase.py`：四场景真实数据统一可视化 →
  `experiments/p1_datasets_showcase/`（p1_scene_samples.png + metrics.json + README）。
- **🔴 踩坑（重要）**：`xiakeann/Ship_Detection` 与 `Echo0174/Trash_floater` 实测**需登录**（401 "当前操作需要登录"），
  已替换为**公开可下**的 `isLinXu/rf100-vl-floating-waste` + `Flier123/Ship`。
  再次印证台账纪律「**下完 ≠ 可用**」——选仓库前务必先验 `login_required`。
- **联动**：`docs/DATASETS.md` 总览表 + §10 新增 P1 四场景专批详解 + §9 图 + 映射表 4 行；
  `docs/P1_field_projects.md` 新增 §2.12 四场景真实数据；`00_INDEX.md` 图索引加行。
- **下一步**：① 等 floating-waste/Ship 下完，重跑 showcase 补 ⑤ 船只；② 把 MOT17 真实行人轨迹接入 P1 `dynamic_obstacles`。

### 2026-09-14（续31）· P1 控制层升级：动态避障（VO）+ 底盘物理约束
- **动因**：P1 §2.7 自己承认「A* 只在静态栅格上跑，未处理动态障碍」「控制是简单运动学」——按 §1.1 准绳（同时加深理解 + 强化贯通），这是最能加深"感知→控制"理解的一环。
- **升级 1 · 底盘物理约束**：`diff_drive_control` 加入**线加速度限幅 a_max=0.5 m/s²** + **角速度限幅 ω_max=1.2 rad/s**，让 v/ω 指令**物理可执行**（真机不能瞬间变速）。
- **升级 2 · 动态避障（速度障碍法 Velocity Obstacle, VO）**：新增 `dynamic_obstacles`（模拟**时序对齐、横穿机器人路径中点**的行人）+ `velocity_obstacle_avoid`（前瞻窗口最近距离 < 机器半径+行人半径 → 触发避让减速）。
- **升级 3 · 避障对比图**：新增 `figs/dynamic_avoid.png`（左：全局路径 + 横穿行人；右：v/ω + 避让触发步标注）。
- **实测（TUM fr1/desk，150 帧）**：控制指令 38 条（含限幅）；动态行人横穿 → 速度障碍法**触发避让减速 5 步**。
- **坑**：① 行人若匀速慢速会错过机器人 → 碰撞锥永不命中 → 改为**时序对齐**（行人第 mid_i 步抵达路径中点）；② 行人轨迹不裁剪会把地图视野拉飞 → 左侧图加 xlim/ylim。
- **联动**：`docs/P1_field_projects.md` §2.5/§2.7/§2.8 更新（含新图与诚实边界修正）；`00_INDEX.md` 图索引加动态避障行；本文件续31。
- **价值**：把"感知→规划→控制"从"纸上谈兵的 v/ω"推进到"**检测到人→预测会撞→减速**"的真正动态闭环，
  且指令受底盘物理约束——这正是初学者最容易漏、也最能体现"专业人员"的地方。

### 2026-09-14（续30）· 战略认知固化：「搭建实战项目」是成长为专业人员的关键
- **动因**：用户明确指出——「搭建实战项目的目的很重要，是成长为专业人员的关键」，要求整理记录到本文件。
- **动作 1 · 新增 §1.1「战略认知：为什么『搭建实战项目』是成长为专业人员的关键」**（置于目标定位之后）：
  - 核心论断：**知识 ≠ 能力，能力 = 知识 × 工程约束的整合**；知识点是零件，工程师价值在于组装成可交付系统。
  - 对比表：「只有知识点（学生）」vs「有实战项目（专业人员）」——从"知道各模块"到"知道谁给谁定尺度、谁失效谁兜底、感知怎么变控制"。
  - 三条可复用判断：① 真实工程 80% 难点不在算法（时间同步/坐标系/标定/尺度/动态障碍/失效兜底/端侧/数据闭环/sim2real）；② 能"讲清楚"才算真懂，输出是检验理解的唯一标准；③ 项目是所有模块知识的收敛点。
  - 双轨结构：**纵轴（学）F/M 讲透单点，横轴（用）P0/P1 横向贯通**，交点=专业能力。
  - **准绳**：后续一切"下一步"判断都以"同时加深纵轴理解 + 强化横轴贯通"为准。
- **动作 2 · §2 路线图新增 P0 / P1 两行**（P1 标 ⭐），让"实战层"在模块总览中有正式位置。
- **价值**：把此前隐含的项目设计意图**显式化、可传承**，使后续接手者/协作者理解"为什么做实战项目、做到什么程度算够"。

### 2026-09-14（续29）· P1 工程叙事层：四场景实战项目 + 服务机器人感知→规划→控制闭环
- **动因**：用户要求搭建「能讲给面试官/客户/同行听的完整实战项目」，覆盖场景/目的/数据/设备/算法/感知→控制/约束/初学者常漏环节；P0 偏抽象管线，缺「工程叙事 + 控制指令转化」。
- **设计**：新建 `docs/P1_field_projects.md`（工程叙事层）：
  - 四场景概览：园区扫地 / 酒店送物 / 传统自动驾驶 / 无人船，每个都按「场景/目的/数据/采集/设备/算法/感知→控制/约束/初学者常漏」展开；
  - 旗舰深做「园区/室内服务机器人」：把感知→规划→控制跑通，并给出「给同行/面试官/客户」三视角讲解模板与初学者常漏的 10 个环节。
- **新增脚本**：`scripts/p1_service_robot.py`：复用 P0 的 VO/深度/语义/场景图工序，新增 `build_occupancy` + `A*` + `diff_drive_control`：
  - 深度点云反投影 → 2.5D 占据栅格地图；
  - A* 规划从起点到目标点的安全路径；
  - 路径切线 → 差速轮 (v, ω) 控制指令。
- **实测（TUM fr1/desk，150 帧）**：
  - VO 自由尺度 ATE **0.728 m**；深度定尺度后公制 ATE **0.764 m**；
  - 栅格地图 **43×38** @ 0.1 m，障碍格 **293**；
  - OWL-ViT 语义检测 **26** 候选 → 3D 场景图 **15** 个实例（3 类语义标签）；
  - A* 路径 **39 步**；差速控制指令 **38 条**，v_max **0.30 m/s**。
- **产出**：`experiments/P1_service_robot/`（`figs/map_plan.png` + `control_cmd.png` + `scene_graph.png` + `metrics.json` + README.md）。
- **顺带修复**：`src/wildspatial/projects/stages.py` `SemanticStage` 兼容 transformers 4.x/5.x 的 OWL-ViT 后处理 API；`VOStage` 增加 `vo_frame_idxs` 供下游建图正确取深度。
- **联动**：`README.md` 文档地图 + 状态加 P1；`docs/00_INDEX.md` 七维表 + 图索引加 P1；本文件续29 + 顶部状态。
- **价值**：把项目从「模块教学 + 抽象项目」推进到「可直接讲给面试官/客户听的完整工程叙事」，并首次把「感知 → 控制指令」真实跑通。
- **诚实边界**：TUM fr1/desk 是手持桌面基准，非真机器人数据；用于演示算法链路；A* 在静态栅格上运行，未处理动态障碍；控制指令为运动学仿真，未接真实底盘。

### 2026-09-14（续28）· M9 综合闭环语义增强版：M7 OWL-ViT 标签接入 M9 场景图
- **动因**：M9 §5 自己列出「下一步 2：接入 M7 的 OWL-ViT 标签填充 class 字段」；
  这是从「class=unknown 的朴素场景图」迈向「几何 × 语义真闭环」的关键一步。
- **脚本**：新增 `scripts/m9_scene_graph_semantic.py`：复用 M1 MonocularVO 位姿 + M7 OWL-ViT 开放词汇检测；
  在关键帧上检测 12 类室内查询词（monitor/keyboard/book/laptop/notebook/phone 等），
  取框内有效深度中位数反投影到世界系，再按 class + 3D 距离 0.5m 聚类成持久实例。
- **产出**：`experiments/M9_closed_loop/figs/scene_graph_semantic.png`（带真实 class 标签的 3D 场景图）+
  `kf_owl_det.png`（关键帧 OWL-ViT 框）+ `scene_graph_semantic.json` + `metrics_semantic.json`；
  同时补齐 `experiments/M9_closed_loop/README.md`（四段式）。
- **实测（fr1/desk）**：VO **9.7 fps** @ CPU；6 关键帧；原始 2D 检测 **31**；聚类 3D 实例 **21**；
  跨帧稳定实例（n_det≥2）**6**。命中：a monitor=11，a keyboard=7，a notebook=6，a phone=3，a laptop=3，a book=1。
- **诚实边界**：未做跨帧 tracking → 同一显示器在多个关键帧被拆成多个实例；矩形框内深度中位数可能混背景；
  单帧检测（book/laptop）未达可视化阈值但保留在 JSON。
- **联动**：`docs/M9_comprehensive_closed_loop.md` 新增 §3.2 语义增强版实验；M9 §5「下一步 2」标记为 ✅ 已实现；
  `docs/00_INDEX.md` 图索引新增 M9 语义行；`PROGRESS.md` 顶部状态 + 续28。
- **价值**：把 M9 从「可跑 demo」推进到「带真实语义标签的 3D 场景图」，
  直接验证「几何是语义的锚」这一核心结论。

### 2026-09-14（续27）· D3 续：NDISPark 昼/夜停车场实例分割真实数据消费
- **动因**：D1 结论指出"恶劣域语义须用语域专用封闭集模型，而非开放词汇 VLM"；需要一张**真实昼/夜样本**
  说明"监督封闭集分割在夜/昼都成立"，作为 D1 结论的对照证据。
- **数据**：`OmniData__NDISPark_Night_and_Day_Instance_Segmented_etc` 已解压；112 张训练图 + COCO 实例分割真值（2577 实例，80 类）。
- **产出 1 · 脚本** `scripts/d_showcase_ndispark.py`：按图像亮度自动挑选最暗（夜间）/最亮（白天）样本各 3 张，
  用 COCO `segmentation` 多边形绘制车辆实例掩码，输出 2×3 对比图。
- **产出 2 · 真实 showcase** `experiments/ndispark_night_day/`：`ndispark_seg.png` + `metrics.json` + README.md。
- **实测**：夜间亮度 **60.4–67.7**、白天 **126.3–136.3**；每图车辆实例 **5–21** 个。
- **联动**：`docs/DATASETS.md` §9 新增 NDISPark 行并嵌图；映射表新增「NDISPark → M6/M7」；
  `docs/00_INDEX.md` 图索引 + 精选图墙（新增 ⑫）+ 覆盖状态表；`PROGRESS.md` 续27 + 顶部状态。
- **价值**：D3 台账中的多模态真实数据从"下载落地"推进到"被项目消费并可视化"，
  且形成"D1 开放词汇失效 ↔ NDISPark 封闭集昼夜可靠"的对照证据链。

### 2026-09-14（续26）· D3 遥感变化检测真实数据消费 + showcase
- **动因**：D3 数据台账里 `remote-sensing-change-detection` 是 11 个真实数据集中**唯一还没被可视化消费**的多模态时序数据；
  它天然对应 M6 多模态融合（光学+SAR）与 M9 时序场景理解，需要一张图讲清"变化检测长什么样"。
- **产出 1 · 脚本** `scripts/d_showcase_remote_sensing.py`：读 5 路真实数据
  `A/` 高分二号事前光学、`D/` 哨兵二号事后校正光学、`B/` 高分三号 SAR、`E/` 二值变化图、`json/` 变化多边形真值；
  统一缩放到显示尺寸后输出 4 组样本 × 4 面板（pre / post / SAR / 变化叠加红区 + 绿色 GT 多边形轮廓）。
- **产出 2 · 真实 showcase** `experiments/remote_sensing_cd/`：`figs/remote_sensing_cd.png` + `metrics.json` + README.md 四段式。
- **实测（4 组）**：变化像素占比 **5.6%–9.3%**；多边形真值 **202–591 个/样本**；SAR 与光学在原始分辨率上有 1 像素差异，已用缩放统一显示。
- **联动**：`docs/DATASETS.md` §9 遥感行更新为「已实跑 showcase」并嵌入图；映射表新增「遥感变化检测 → M6/M9」行；
  `docs/00_INDEX.md` 图索引新增「真实遥感变化检测」条目、覆盖状态表 D3 行补「遥感已可视化」；本文件顶部状态 + 续26。
- **价值**：把 D3 从"台账已建"推进到"素材已被项目消费并可视化"，让 11 个真实数据集全都有可运行的 showcase/评测落点。

### 2026-09-14（续25）· F5 升级：KITTI 真实激光雷达 + 相机融合（真实传感器验证）
- **动因**：原 `m10_lidar_fusion_demo.py` 用 TUM RGB-D 深度**反投影**点云，并在 F5 文档诚实声明"非真实 LiDAR 采集"。
  KITTI 已下载 → 目标是把该声明**升级掉**，让 F5 同时有"数学自洽"+"真实传感器"双证据。
- **产出 1 · 脚本** `scripts/m10_kitti_lidar_demo.py`：从 KITTI `depth_selection/val_selection_cropped/`
  读取 `image/`（RGB）、`velodyne_raw/`（真实稀疏激光深度 PNG）、`groundtruth_depth/`（多帧激光稠密真值）、
  `intrinsics/`（单行 3×3 内参 `K=[fx 0 cx; 0 fy cy; 0 0 1]`），按 KITTI 标准 `depth=png/256.0` 解码。
- **产出 2 · 真实四联图** `experiments/M10_kitti_lidar/figs/kitti_lidar_fusion.png`：
  ① RGB；② 真实稀疏 LiDAR 点投回图像（颜色=距离）；③ 真实稠密深度图（多帧激光 GT）；④ 深度反投影成彩色点云（俯视 X-Z）。
- **实测数据（4 帧）**：分辨率 1216×352；每帧真实 LiDAR **~1.8 万点**；深度范围 **3.0–79.7 m**；稠密真值 **5.5–9.9 万点**。
- **联动**：`docs/F5_vision_lidar_fusion.md` §4 拆为 4.1 TUM（数学自洽）+ 4.2 KITTI（真实传感器）并嵌新图；
  `docs/DATASETS.md` §9 KITTI 行更新为"已下载并解压+已实跑真实 LiDAR demo"；新增 `experiments/M10_kitti_lidar/README.md` 四段式。
- **坑**：`intrinsics/` 文件不是 `P2:` 前缀而是单行 9 个数 3×3 K 矩阵，初始解析失败（fx/fy/cx/cy 为 None）→ 已修正。

### 2026-09-14（续24）· D1 真实恶劣环境语义评测（首个真值支撑实证）+ 数据落地核查
- **数据落地核查（2026-09-14）**：Tier A/E 下载全部完成（manifest 5 项 `ok:true`）。实测磁盘占用：
  NightCity 2.0G、SUN_RGB-D 8.2G、ScanNet-Absolute-Camera 0.42G、TartanAir-Absolute-Camera 73M、
  ModelNet40-C 2.2G、NDISPark 0.13G、rf100-underwater 0.42G、LibreYOLO-underwater 0.46G、
  遥感变化检测 2.0G、KITTI_depth_completion 20G（解压中）、NYUv2 子集 4.0G（含 `.incomplete` 属已 kill 遗留，不碍事）。
  ⚠️ 消费纪律再验证：`ScanNet/TartanAir` 确仅 camera.json；NightCity/KITTI/NDISPark 图像在 `raw/*.tar.gz` 需解压。
- **D1 语义层真实评测（新增 `scripts/d1_real_world_eval.py`）**：
  - 用离线 OWL-ViT（`/tmp/owlvit`，det 环境 transformers 4.48）在**真实水下**（LibreYOLO `underwater-objects-5v7p8`，
    valid 1520 张 + YOLO 真值，5 类海产）与**夜间**（NightCity `sample/`）评测。
  - **真实结论（真值支撑）**：水下细粒度 **mean recall ≈ 0.001**（starfish 2TP/21FP、其余类 0 命中，仅 6.3% 图像发出过框）
    —— 网页训练的通用开放词汇模型在**域偏移 + 细粒度**下基本失效；夜间通用类（car/person…）仍 **≈7 命中/图**。
  - 教学价值：量化印证 M7「VLM 硬限制」论点的**边界**——致命的是"域+类别"而非单纯低光；恶劣域语义必须换**域专用封闭集模型**，
    而非指望万能 VLM。这恰是项目「几何兜底 + 域专用语义」分工的现实依据。
  - 产出 `experiments/D1_real_world/`（underwater_det.png 绿=真值粉=预测 + night_det.png + metrics.json + README 四段式）。
  - 联动：`docs/DATASETS.md` §9 改为「已下载+实跑」并嵌入两图、映射表刷新（语义/M7、视觉+激光/F5 状态）；本文件顶部+D 组（D3✅、D1a✅）同步。
- **下一步**：① KITTI 解压完后做 F5「真实 LiDAR」升级（替换 RGB-D 反投影点云，兑现视觉+激光真实数据）；
  ② 遥感变化检测经 `loaders.py` 归一化后接入 M9；③ D1 视频序列 VO 评测仍受网络限制（gate）。

### 2026-09-11（续5）· M2 现成工具快赢（Open3D TSDF）+ VGGT 安装
- **M2 用现成工具出结果**（用户"用现成工具跑结果"要求）：新增 `scripts/m2_reconstruct.py`，
  直接用 **Open3D ScalableTSDFVolume** 融合 fr1/desk 的**真实深度 + 真值位姿**，
  产出 `experiments/M2_recon/mesh.ply`（**848,542 顶点**）+ `figs/recon.png`（网格顶点 3D 点云着色图）。
  全程零手搓、零下载，验证"复用现成工具出效果"范式在 M2 也成立。
  ⚠️ 坑：Open3D `ScalableTSDFVolume` 构造要 `color_type=TSDFVolumeColorType.NoColor`；
  `integrate` 收 `RGBDImage`（需 `create_from_color_and_depth`）；本构建无 `OffscreenRenderer`，改用 matplotlib 出图。
- **VGGT 安装**：PyPI 无 `vggt`/`lightglue`（网络白名单），已改 **`pip install git+https://github.com/facebookresearch/vggt.git`**
  后台安装中（log: /tmp/install_vggt.log）。`methods/vggt.py` 已补可用实现（pose_enc_to_extrinsics 取 cam-to-world 外参，无 flash-attn 走 SDPA）。
- ⚠️ 安装触发依赖告警：opencv-contrib 5.0 要求 numpy>=2，当前 1.26.4（仅告警，未升级；装完需验证 env 仍可用）。

### 2026-09-11（续9）· 步骤3落地：方法动物园新增 **LightGlue 学习式前端**（四方法失效图谱）
- 给 `sfm/vo.py` 的 `VOConfig` 注入 **`frontend` 回调**：`process()` 里只替换"特征提取+匹配"
  两步，后端（RANSAC/三角化/PnP/尺度/护栏/BA/回环）**完全复用** → 对比维度纯净。
- 新增 `methods/lightglue.py`：`LightGlueVO(kind="learned")`，用 **SuperPoint（学习式检测+描述）
  + LightGlue（学习式匹配器）** 作前端；权重 `superpoint_v1.pth`/`superpoint_lightglue.pth`
  从 cvg/LightGlue GitHub release（v0.1_arxiv）经 **ghproxy 代理** 下到 torch hub 缓存，离线可用。
- **60 帧 fr1/desk × 13 退化条件 ATE（四方法）**：
  干净：自研 0.529 / COLMAP 0.026 / VGGT 0.018 / **LightGlue 0.305**；
  重退化最高：自研 0.626（塌缩）/ COLMAP 0.5+（仅噪声）/ VGGT 0.049 / **LightGlue 0.483（不崩）**。
- **核心结论（三维度对照）**：
  1. VGGT（前馈）全档位稳压（0.016–0.049m），免疫所有退化 —— 最震撼证据。
  2. **换前端（学习式匹配）能救"崩塌"但不救"精度"**：LightGlue VO 从不崩到 0.626，但单目尺度漂移
     使其仍 0.26–0.48，劣于 COLMAP/VGGT；学习式特征也救不了高斯噪声下的轨迹（瓶颈在几何后端/范式）。
  3. 真正跨越式提升来自**换范式**（前馈模型），而非单纯换特征。
- 产出：`experiments/M5_method_atlas/`（atlas_ate.png + metrics.json + 四段式 README，含三维度对照表）。
- ⚠️ 国内网络：LightGlue 权重 GitHub release（非 ModelScope，MS 无该仓库），走 ghproxy 代理下到本地缓存。

### 2026-09-11（续10）· M4 深化完成：VGGT 深度 / 点云质量在退化下的鲁棒性
- 给 `methods/vggt.py` 加 `return_dense` 开关，输出 `depth`(S,H,W)/`world_points`(S,H,W,3)/conf 到 extra；
  新增 `scripts/m4_depth_eval.py`：13 退化条件下跑 VGGT(return_dense)，深度 vs TUM 真值算
  RMSE/AbsRel/δ1（中位比例对齐尺度），world_points 借轨迹 Sim3 对齐真值系后与真值点云算 Chamfer。
  （另加 `scripts/m4_make_figs.py`：从 metrics 直接出图/README，避免重跑 VGGT。）
- **深度 RMSE（米，全 13 条件）**：clean 0.144 / 运动模糊 0.139–0.155 / **高斯噪声 0.097–0.116（反而略好）**
  / 低光照 0.103–0.135 / 分辨率下降 0.149–0.160。δ1 始终 **>0.96**。
- **点云 Chamfer（全条件合并）= 0.105 m**。
- **核心结论**：VGGT 的稠密几何重建（深度+点云）在合成退化下**几乎不退化**——与"轨迹 ATE 免疫退化"
  互为印证，证明前馈基础模型的鲁棒性贯穿「位姿→深度→点云」全栈，而非仅轨迹维。
- 产出：`experiments/M4_depth_atlas/`（depth_metrics.json + figs/depth_vs_degradation.png + README）。

### 2026-09-11（续8）· 步骤2完成：自研 VO vs COLMAP vs **VGGT** 三方法失效图谱（核心王牌结论）
- VGGT 权重经 **ModelScope**（`facebook/VGGT-1B`）下载到 `~/models/vggt-1b`（用户指示国内 HF 慢），
  已修 `methods/vggt.py` 正确 API（仅下 safetensors + 类级模型缓存）；后台监控自动跑出三方法图谱。
- **60 帧 fr1/desk stride3，三方法 × 13 退化条件 ATE（Sim3 对齐，米）**：

  | 条件 | 自研VO | COLMAP | **VGGT** |
  |---|---|---|---|
  | 干净 | 0.529 | 0.026 | **0.018** |
  | 运动模糊-中/重 | 0.626 | 0.025/0.022 | **0.022/0.038** |
  | 高斯噪声-中/重 | 0.373/0.626 | **0.518/0.552** | **0.017/0.018** |
  | 低光照-重 | 0.626 | **0.539** | **0.019** |
  | 分辨率下降-重 | 0.186 | 0.026 | **0.049** |

- **三条核心结论（带真值支撑）**：
  1. **VGGT 几乎免疫所有退化**：13 条件 ATE 全在 **0.016–0.049m**，不随退化升级恶化——前馈模型绕开
     "特征→匹配"这一最先失效环节，是项目最震撼的定量证据。
  2. **高斯噪声是传统几何共同软肋**：COLMAP 在模糊/低光/降分辨率极鲁棒(~0.02–0.06m)，但噪声中/重崩到 0.5+；
     自研 VO 重退化轨迹塌缩(0.626)。
  3. **COLMAP 干净/多数退化超自研 VO ~20×**；**VGGT 全档位稳压 COLMAP**；VGGT 唯一相对弱点是分辨率下降-重(0.049m，仍远优)。
- 产出：`experiments/M5_method_atlas/`（atlas_ate.png + metrics.json + **四段式 README**）。
  这是战略"复用现成工具、建立概念与效果"的旗舰成果。

### 2026-09-11（续6）· 步骤3：后续模块「手搓→现成工具」优化审查（code-explorer 代理全量梳理）
- **结论总纲**：手搓层 **M0 几何库 / M1 前端+BA+回环 v1** 已定性为「直觉锚点 + 自研基线」，**保留不动**（有 50+ 项自检与教学依赖）；后续模块默认调用成熟实现。真正要落地的不是"替换"而是"在方法动物园**新增**现成方法做对照"。
- **最高优先级（应换现成工具）**：
  - `M1 sfm/loop.py` 手搓回环 → **ORB-SLAM3 / DBoW2/DBoW3 词袋**（PROGRESS §2.1 已定"回环 v2 不再手搓"；v1 实测误检反而使 ATE 变差）。
  - `M1 sfm/pgo.py` 手搓位姿图优化 → **g2o / gtsam**（自承雅可比简化非生产级）。
  - `M1 sfm/ba.py` 手搓 BA（最贵，自承 gauge freedom 简化）→ 生产用 **pycolmap / g2o**（zoo 里 COLMAP 已含工业级 BA）。
- **M5 方法动物园应新增的现成方法**（不是替换，是扩充对照）：
  - 学习式匹配 **LightGlue / SuperPoint+SuperGlue / DISK**（高，比 SIFT 在退化下鲁棒）。
  - 成熟 VO/SLAM 基线 **ORB-SLAM3**（高，含回环/IMU，一并解决 M1 回环痛点）。
  - 轻量位姿库 **poselib**、单目深度 **Depth-Anything**（中）。
- **已复用、无需改**：M2 TSDF(`Open3D`)、M4(`vggt`)、M5(`pycolmap`/`vggt`/自研 baseline)；`sfm/features.py`/`matching.py` 本就是 OpenCV 薄封装。
- **保持自写更有价值**：`data/degrade.py` 合成退化（控制力/教学性最好，库仅用于扩充退化类型）、`eval/trajectory.py` ATE/RPE（已与 zoo 耦合，可用 `evo` 做交叉验证）。

### 2026-09-11（续7）· VGGT 权重下载（ModelScope 国内镜像）
- 改用 **ModelScope** 下载（用户指示：国内网络 HF 慢）：`snapshot_download("facebook/VGGT-1B", local_dir=~/models/vggt-1b)`。
- 该 repo 含 `model.pt`(5GB)+`model.safetensors`(5GB) 两文件，推理只需 `safetensors`。
- 现状：safetensors ~180MB/5GB @~1.3MB/s；**旧 HF 后台进程(3691744，用户规则禁止 kill) 与并行 pt 下载在抢带宽，预计仍需 ~1h**。
- 已修 `methods/vggt.py`：用正确 API（`load_and_preprocess_images` 收文件路径 + `pose_encoding_to_extri_intri` 取 world→cam 外参，光心 `C=-Rᵀt`；`VGGTModel.from_pretrained` 优先本地 `~/models/vggt-1b`）。权重就绪即自动并入 M5 三方法图谱。
- ⚠️ **加速已落地（用户允许 kill 我自己的后台进程）**：已 kill 旧 HF(3691744)+旧 ModelScope(3694589) 双进程，
  改为 `snapshot_download(..., allow_patterns=["*.safetensors","config.json","configuration.json"])` 单文件独占带宽。
  速度 ~1.3→**3.6MB/s**，24%(1.19G/5.03G) 时预计 **~18min** 完成。权重就绪即跑三方法图谱。

### 2026-09-11（续4）· 步骤1验证：自研 VO vs COLMAP 双方法失效图谱（范式跑通）
- **pycolmap 4.2.0 适配坑**（已修 `methods/colmap.py`）：
  ① `extract_features` **不接受 `camera_model=`/`sift_max_num_features=` 关键字**（报签名不兼容）→ 改用默认参数，焦距由 COLMAP 在 BA 中自标定；
  ② `im.cam_from_world` 是**方法**不是属性，返回 `Rigid3d`，需 `.matrix()` 取 4×4，相机光心 `C=-Rᵀt`。
- **60 帧 fr1/desk stride3，双方法 × 13 退化条件结果**（均 Sim3 对齐 ATE，带真值）：
  - `handcrafted_vo`：干净 **0.159m**；退化 0.06~0.63m；运动模糊中/重、高斯噪声重、低光照重 **全部饱和 0.626m**（轨迹塌缩信号）。
  - `colmap_sfm`：干净 **0.026m（比自研好 ~6 倍）**；对运动模糊/低光照/降分辨率**极鲁棒**（均 ~0.02~0.06m）；
    但**高斯噪声中/重也崩到 0.53/0.42m** —— 与自研 VO 共同软肋。
- **核心结论（带真值支撑）**：工业级 SfM（COLMAP）在多数退化下**碾压自研 VO**；但**高斯噪声是传统几何方法的共同阿喀琉斯之踵**
  （噪声造伪特征，描述子配不上 → 几何法通杀）。这正是"复用现成工具看效果"的价值：不手搓也能立刻拿到可比、可信的定量结论。
- 产出：`experiments/M5_method_atlas/`（atlas_ate.png + metrics.json + 双方法 README）。
- 下一步：攻 **VGGT（M4 核心）**：后台装 transformers/timm/vggt（flash-attn 可选，用 SDPA 退化），`methods/vggt.py` 已补可用实现。

### 2026-09-11（续3）· 战略转向：复用现成工具 + 方法动物园 + M5 失效图谱首跑
- **用户战略指示（重要）**：不必手搓，直接调用现成工具/代码包/模型；目标是建立**概念、认知与效果**，
  细扣底层原理留到以后。已固化为 PROGRESS 约定第 9 条 + MEMORY 规则 9。
  手搓层（M0 几何库、M1 前端+BA+回环 v1）已完成"建立直觉"使命，保留为**直觉锚点**，不再作主路径。
- **新建「方法动物园」抽象**（`src/wildspatial/methods/`）：统一 `Method` 接口
  （喂图 → 相机光心轨迹 `(M,3)` + `frame_indices` → 与真值 Umeyama(Sim3) 对齐算 ATE）。
  已落地：`handcrafted.py`（自研 VO 基线）、`colmap.py`（装 pycolmap 自动出现）、
  `vggt.py`（前馈 3D 基础模型，含实现骨架，装 vggt 自动出现）。`discover_methods()` 按依赖自动探测可用方法。
- **新建 M5 压力测试脚本** `scripts/m5_method_sweep.py`：干净 + 4 退化 × 3 档 = **13 条件** × 各可用方法
  → 产出 `experiments/M5_method_atlas/`（ATE 热力图 `atlas_ate.png` + `metrics.json` + 四段式 README）。
- **首跑结果（仅 handcrafted_vo，150 帧 fr1/desk stride 3）**：干净 ATE **0.51m**；
  退化后普遍升到 0.63~0.89m，运动模糊/高斯噪声/低光照"重"档 ATE 饱和在 **0.89m**（轨迹塌缩、对齐残差被真值尺度主导）。
  → 教学信号：**重退化先看"轨迹是否塌缩"再看 ATE 数值**。
- ⚠️ 环境：PyPI 找不到 `lightglue`（网络白名单？）；`pycolmap` 后台安装中（log: /tmp/install_pycolmap.log）。
  装好后重跑即自动把 COLMAP 并入同一张失效图谱，直观对比"自研 vs 工业级 SfM"。
- 后续：VGGT（M4 核心）需 transformers/timm/flash-attn，最重；待 COLMAP 验证范式后再定优先 VGGT 还是 LightGlue。

### 2026-09-11（续）· M3 失效归因首轮 + 确立四段式文档规范
- **确立四段式文档规范**（用户明确要求，已写入本文件约定第 6~8 条 + README）：
  **目的 → 方法原理（专业）→ 直白讲解 → 真实标签数据验证 + 效果图**。
  新增约定：结论必须由**带真值标签的数据**支撑；无标签时用"合成退化"做可控对照并明确标注；
  每个 Module 至少产出一组"肉眼可见"的效果图。
- **M3 失效归因首轮**：新增 `src/wildspatial/data/degrade.py`（4 类合成退化，severity 0~1）
  与 `scripts/m3_degradation_sweep.py`。在 **TUM fr1/desk（含真值）**上跑
  4 类型 × 3 档位 + 干净基线 = **13 次完整实跑**，产出失效图谱/样本图/状态分布 3 张图
  + `docs/M3_failure_attribution.md`（四段式）。
- **四个由真值支撑的结论（重要）**：
  1. **两类失效**：硬失效（初始化失败、地图点=0：运动模糊中/重、低光照重）vs
     软劣化（能跑但 ATE 差 1.5~3 倍、地图点少 55~78%：噪声/低光照轻中/降分辨率）。
  2. **最先断的是"匹配"不是"特征"**：运动模糊轻→中，特征仍有 164，但匹配 169→6、
     内点率 80.6%→11.4% → 模糊主要摧毁**描述子可匹配性**。
  3. ⚠️ **特征数是误导性指标**：高斯噪声使特征 1345→1938（+44%），
     但地图点 8151→1801（−78%）、ATE 0.135→0.309（+129%）。**噪声造伪特征：能检测、配不上**。
  4. **内点率是最灵敏的失效前兆**（健康 75~80%，崩前跌到 11.4%，且此时 ATE 已无定义）
     → 适合做运行时失效预警。
- ⚠️ 教训：短序列 ATE **非单调**（降分辨率 0.106→0.387→0.187），单看 ATE 会误判。

### 2026-09-11 · 回环+PGO v1 + 真实数据效果展示
- **回环检测 + 位姿图优化**：新增 `sfm/loop.py`（关键帧匹配 + E 矩阵几何验证 +
  **j 邻域窗口池化**地图点做 PnP 求回环绝对位姿）与 `sfm/pgo.py`（SE(3) 位姿图，
  解析雅可比 = Adjoint，取 J_l⁻¹≈I 一阶近似并在 docstring 声明）。
  `vo.close_loops()` 串起 检测→约束→优化→写回；脚本加 `--loop`，产出 `trajectory_loop.png`。
  自检 `tests/test_pgo.py` 4 项（伴随性质 / 无漂移不动 / 回环降误差>30% / 首帧固定），**累计 57 passed**。
- **回环 v1 诚实结果**：fr1/desk 检出 8 候选、3 约束，PGO 残差 0.079→0.021（收敛），
  但 **ATE 0.506→0.523 m 反而略变差**。根因：**误检回环**（桌面重复纹理 → 感知歧义）
  + 回环位姿本身源自带漂移的地图。教训：**回环约束是双刃剑，误检比没有更糟**
  → v2 需要更严格验证（多帧一致性 / Sim3 RANSAC）。
- **真实数据效果展示**（用户需求：要"肉眼可见"的效果图）：新增 `scripts/m1_showcase.py`
  一键生成 5 张图 → `experiments/M1_showcase/`（特征检测 / 匹配+RANSAC 前后 /
  对极约束 / 重建地图+轨迹 / 条件对比，含三段式 README + metrics.json）。
  - ⚠️ 帧对必须自动挑选：实测 fr1/desk 隔 60 帧仅 **4 组匹配**（RANSAC 报"点数不足 8"）；
    改为在内点 ≥60 前提下选**外点最多**的帧对（0↔12：112 匹配 → 78 内点）。
  - ⚠️ `fr3/nostructure` 实测**不可用**（只有 rgb 图 + accelerometer.txt，缺 rgb.txt 与真值）
    → 启用项目预案：对 fr1/desk 施加**合成退化**（运动模糊/低光照）做可控对照，各档均有真值可算 ATE。
  - 条件对比结果：特征 **1253→~250**、地图点 **8701→~2000**（崩 4 倍），
    但短序列（60 处理帧）ATE 尚可 0.13~0.18m —— **失效先从地图质量开始，再到跟踪失败**。
- **M1 文档**新增「效果展示」章节（6.1~6.5，引用 5 张图）；下一步更新回环 v2 计划。

### 2026-09-10（续6）· M1 BA 向量化 + 长序列真实结论
- **BA 向量化**：`ba.py` 的 `_residual` / `_jacobian` 由逐条 Python 循环改为 **`einsum` + 广播**一次性算所有观测的投影与雅可比，
  雅可比用向量化 `_vec_hat` 构造稀疏矩阵（csr）。`refine()` 新增**点预算上限**（默认 5000，确定性子采样，丢弃点保留在地图里不优化），
  观测上限 8000。→ BA 从"分钟级超时"降到"秒级"，长序列可跑。
- **验证（450 帧 fr1/desk，两份独立运行一致）**：
  - 重投影 RMSE **1.54 → 0.20 px**（全地图）/ **1.04 → 0.20 px**（点预算版）—— BA 收敛、地图更自洽 ✅
  - **ATE RMSE 0.506 → 0.506 m** —— 纹丝不动 ❗
- **关键教学结论（修正上一轮预期）**：上一轮以为"长序列 ATE 才明显"，实测**长序列 BA 后 ATE 仍不变**。
  根因：ATE 是整条轨迹做 Sim3 对齐后量出的"整体形状偏差"；在**无回环的开阔轨迹**上，VO 漂移主要是全局尺度/旋转的整体偏差，
  单靠 BA（无回环约束）只能"摊平局部重投影误差"，无法消除整体偏差。→ **要真正砍 ATE，必须加回环检测 + 位姿图优化**，
  这正是纯 VO+BA 打不过 ORB-SLAM 的原因，也是 M3/M4/M5 的核心议题。
- `tests/test_ba.py` 6 项全绿（向量化数值正确）；全量 `pytest` 待复核。
- ⚠️ 用户拒绝含 `kill` 的命令（含 `kill -0` 状态探测），后续用读日志代替进程状态查询。

### 2026-09-10（续3）· 文档统一 + 接入 GitHub
- **三份文档风格统一**：`docs/00_QUICKSTART.md`（新）用大白话导览；`M0_geometry_foundation.md` 补 6 张图、
  `M1_sfm_from_scratch.md` 补 3 张图，全部采用「术语+讲解+图」三段式。图片路径 `../experiments/...` 相对引用。
- **接入 GitHub**：`https://github.com/liuliuqiu378/WildSpatial`。新建 `.gitignore`（排除 `data/` `.codebuddy/` `.pyc` `*.tgz`），
  保留 `experiments/figs/`。首个 commit `c46fa48`（43 文件）已 push 到 `main`。
  远程 `origin` 用干净 URL，push 时临时内联 PAT，未持久化进 `.git/config`（已验证）。
- **更新 README / PROGRESS**：修正状态、命令（补 `PYTHONPATH=src`）、目录结构、补充文档导航与三段式说明。

### 2026-09-10（续4）· M1 BA 接入 VO ✅
- **BA 接入**：`sfm/vo.py` 新增 `MonocularVO.refine()`——由 `self.obs` 反查每条观测像素坐标，构建
  `obs=[(frame_idx, point_id, uv)]`，调用 `sfm.ba.local_ba`（固定第 0 帧 + 尺度锚消 7 维 gauge freedom）。
  观测过多时确定性子采样（≤8000 条）控制运行时间，所有位姿/点仍参与优化。
- **脚本**：`scripts/m1_run_vo.py` 新增 `--ba` 开关；BA 后重算 ATE、产出 `figs/trajectory_ba.png`（BA 前后轨迹对比）。
- **验证（20 处理帧 fr1/desk）**：重投影 RMSE 0.444 → 0.000 px（BA 目标函数被完美打到 0，数学正确）；
  ATE 0.0529 → 0.0533 m（短序列 VO 本身漂移小，BA 对 ATE 改善有限——**长序列才明显**，
  而长序列 BA 很重，故生产用局部/增量 BA）。`tests/` 仍 53 passed。
- ⚠️ **性能提醒**：纯 Python 雅可比 + scipy trf 在 8000 点/全序列上 BA 需数分钟；教学演示用短序列，
  完整运行调小 `--frames` 或后续做增量 BA。

### 2026-09-10（续2）· M1 前端完成，跑出 TUM 基线 ✅
- **产出**：`src/wildspatial/sfm/`（features / matching / ransac / vo）、`data/tum.py`、
  `eval/trajectory.py`、`scripts/m1_run_vo.py`。**自检累计 47 项全绿**（今累计 53 项含 BA）。
- **基线结果**（`experiments/M1_vo_fr1_desk/`，fr1/desk，stride=3，450帧）：
  **ATE RMSE 0.506 m / RPE 平移 0.156 m / RPE 旋转 2.72° / 内点率 75.3% / 39 ms 帧 / 地图点 17388**。
  （对比：ORB-SLAM2 单目同序列约 0.02~0.05m，差距主要在**没有 BA/回环**。）
- **🔴 三个真实踩到的坑（价值 > 数字本身）**：
  1. **帧间视差不足 → 初始化起不来**：TUM 30fps，帧0→1 的 RANSAC 内点率 97.7%，
     但**视差角中位数仅 0.41°**，三角化深度中位 100m（真实 1~2m）。
     → 解法：固定参考帧，等中位视差 > 3° 再初始化（同 ORB-SLAM）。参数 `init_min_parallax_deg`。
  2. **深度检查写在尺度归一化之前**：初始化时 t 是单位向量，三角化深度可达 100m 量级，
     拿去比 20m 阈值几乎全被误杀。→ 必须先定尺度再检查。
  3. **位姿指数爆炸（最严重）**：无护栏时一次 PnP 失败 → 回退路径用"上一帧位移"猜尺度
     → 异常跳变逐帧放大，**RPE 达到 49940 m**。→ 加运动连续性护栏后降到 0.156 m。

### 2026-09-10（续）· M0 完成 ✅
- **产出**：`src/wildspatial/geometry/` 五个模块全部手写（lie / camera / epipolar / triangulation / pnp），
  `tests/test_geometry.py` **25 项数值自检全绿**（双环境通过）。
- **教学产出**：`docs/M0_geometry_foundation.md` + `scripts/m0_demo.py`
  → `experiments/M0_geometry_foundation/figs/` 6 张图。
- **工具**：`src/wildspatial/viz/plots.py`（中文字体统一）；`src/wildspatial/data/download.py`（多线程下载）。
- **🔴 踩坑记录**：
  1. **`(N,3)` vs `(N,2)` 静默错位**：`normalize_points()` 返回齐次坐标 (N,3)，八点法按 (N,2) reshape 后
     数据被打乱但**不报错**，E 与真值余弦相似度 0.83（看似合理，实则全错）。→ 已加 `_as_euclidean()`。
  2. **PnP DLT 的全局符号不定性**：`M` 与 `-M` 残差完全相同，SVD 无法区分 → 用正深度（chirality）消歧。
  3. **DLT PnP 必须用归一化坐标**：方程建立在 `u = X/Z` 上，不先乘 `K^{-1}` 会污染内参进 R。
  4. **三个"对称性陷阱"**：E 的 4 解、PnP 符号、单目尺度不定 → 统一用**物理约束（正深度）消歧**。

### 2026-09-10 · 建立项目骨架
- README / PROGRESS / `docs/00_ROADMAP.md` / `env/` / `src/` 结构就绪。
- **环境**：新建 `wildspatial` 环境并装好依赖（torch2.6+cu124 + CUDA 可用 + opencv-contrib + open3d）。
- **数据**：选 TUM RGB-D 作为 M0–M3 主数据集（有真值+深度可量化，单序列仅 344MB 适配小算力）。
- **启动**：后台下载 `fr1/desk`；后台创建 conda 环境（不阻塞，继续写代码）。

---

### 2026-09-12 · 新增「领域全景」教程 `docs/00_LANDSCAPE.md`
- **动因**：用户梳理了「分模块流水线 / 2025–2026 前沿 / VLM+世界模型 / 端侧落地」领域认知，要求纳入教程，并「结合经典模型+数据集讲原理/作用/效果/用法/限制」。
- **产出**：`docs/00_LANDSCAPE.md`（领域全景总纲，严格四段式：①目的→②方法原理专业→③直白讲解→④真实验证/效果与限制）。
  - §1 分模块流水线五工位：检测(YOLO/Faster R-CNN/DETR/SAM)+数据集(VOC/COCO/KITTI)；分割(U-Net/Mask R-CNN/SegFormer/SAM)；深度(双目 SGM/PSMNet、单目 DPT/Depth-Anything、RGB-D)+数据集(TUM/KITTI/NYUv2/ScanNet/ETH3D)；SLAM/VIO(ORB-SLAM/VINS/LIO-SAM)+数据集(TUM/KITTI/EuRoC/TartanAir/4Seasons/SubT-MRS)；规划(A*/RRT/MPC)。
  - §2 前沿：几何成熟但有边界；开放词汇 3D(OpenScene/PLA/LLM-Grounder/3D-LLM)；世界模型(Dreamer/Sora/占用网络 UniAD)。提出「几何世界 vs 物理世界」双世界框架。
  - §3 VLM 融合：分层解耦（VLM 不负责几何），引用实测锚点（VLM 位姿 F1=0.64、深度轴 7.4%）；列增益与 4 条硬限制。
  - §4 端侧部署切入路线：能力栈映射表（几何→M0-M4；量化→M8；VLM→M7；融合→M6）。
- **真实验证策略**：带真值结论优先引用本项目 TUM `fr1/desk` 实跑（VGGT 深度 δ1>0.96、三方法 ATE 0.018/0.026/0.53）；VLM/世界模型部分标注「待 M7/M8/M9 补」。
- **定位**：本文是 M0–M9 技术纵深的「目录与坐标系」，不重复算法推导，只钉经典模型+数据集+边界。

### 2026-09-12（续）· 教程「④ 真实验证」补强：DETR + GrabCut 真实跑
- 写 `scripts/landscape_demo.py`（`det` 环境离线）：**DETR**(`facebook/detr-resnet-50`, HF 缓存) 检测 + **OpenCV GrabCut** 分割，作用于真实 TUM `fr1/desk` 4 帧。
- 实测：4 帧共 **27 detections** = vase×7, keyboard×5, microwave×4, mouse×4, remote×2, cell phone×2, couch×1, fork×1，含 `N/A`（印证「类别封闭」）。
- 产出 `experiments/landscape_showcase/figs/{detection,segmentation}.png` + `metrics.json`；教程 §1.1/§1.2 ④ 已挂真图真数。

### 2026-09-12（续2）· M7 语义层最小实证：OWL-ViT 开放词汇（真实工具 + 真实数据）
- 经 **hf-mirror** 后台下 `google/owlvit-base-patch32`（~613MB safetensors）到 `/tmp/owlvit`（transformers 4.48 原生支持，免 Qwen3 问题）。
- 写 `scripts/m7_semantic_demo.py`：OWL-ViT 对 TUM 4 帧做自然语言查询检测。
- **踩坑**：`post_process_object_detection` 返回单图 dict（boxes/scores/labels），labels 才指向查询索引；初版误当「每查询一个元素」遍历 → 全框错标首个查询。已修正。另 matplotlib 中文缺字 → 图标题改英文。
- 实测（per-query hits）：cup×2, keyboard×11, **book×20**, remote×4, red object×2；bottle/left 为 0。
- **双刃价值**：既证 §3 增益#2（自由文本/属性查询命中，DETR 的 COCO-80 做不到），又证硬限制#1 幻觉（"a book" 返回 20 框过度触发）。
- 产出 `experiments/M7_semantic/figs/open_vocab_detection.png` + `metrics.json`；教程 §2.2/§3 ④ 已回填。

### 2026-09-12（续3）· 用户反馈「核心教程没增长」→ 新建 `docs/00_PRIMER.md`（新手讲义）
- **用户痛点（正确）**：过去几轮精力砸在 `00_LANDSCAPE.md` + `experiments/` + `PROGRESS.md`，但**教初学者的 M-module 教学文档只写了 M0/M1/M3（3/10）**，且 M0 一上来就李群/对极几何，零基础太陡。用户要求：教程面向初学者、补基础概念和经典理念的易懂讲解。
- **产出**：`docs/00_PRIMER.md`（零基础入门讲义）。结构：§0 人话定位 → §1 基础概念词典（像素/2D-3D/深度/点云网格/特征描述子/匹配/位姿/地图，逐个配类比）→ §2 经典理念 7 条（对极几何直觉/双目vs单目尺度/SLAM鸡生蛋/SfM-vs-VO/分模块vs端到端/VLM为何做不好几何/开放词汇分工）→ §3 M0–M9 一句话映射 → §4 学习路线图 → §5 对应真实实验 → §6 三句话收尾。
- **风格**：大白话 + 生活类比 + 钉真实实验图/数字（如 VGGT ATE 0.016–0.049m vs 自研 0.5m；VLM 位姿 F1=0.64）。刻意**零公式推导**，只建直觉。
- **已联动**：`00_QUICKSTART.md` §8 阅读顺序最前加入 PRIMER 指引。
- **待办（核心缺口）**：M2/M4/M5/M6/M7/M8/M9 的**教学文档仍缺失**（实验已有），需逐模块补「初学者版讲解」；M0/M1 也偏专家笔记，后续可加「初学者导读」小节。

### 2026-09-12（续4）· 补 M2 / M5 初学者版教学文档（钉真实图与数字）
- 新建 `docs/M2_depth_reconstruction.md`（初学者版）：深度→点云→网格三级跳；TSDF 多视角融合直觉（类比平均降噪）；真实实验 Open3D TSDF 融合 fr1/desk 60 帧 voxel 0.01m → 顶点 848542，配 `../experiments/M2_recon/figs/recon.png`；讲清「依赖上游准」限制，引到 M3/M5。
- 新建 `docs/M5_stress_test.md`（初学者版）：压力测试矩阵三轴（方法×环境×强度=13 条件）；4 方法初学者讲解；钉真实 ATE 表（VGGT 全档 0.016–0.049m 稳压、COLMAP 仅噪声崩 0.5+、自研VO 重退化塌缩 0.626）+ 热力图 `../experiments/M5_method_atlas/figs/atlas_ate.png`；三条结论（VGGT 免疫/高斯噪声是几何死穴/换前端救崩不救精度）；诚实声明退化为合成施加。
- 两篇均走 PRIMER 风格：大白话+类比+零公式推导+钉真图真数。
- **教学文档现状**：PRIMER + M0/M1/M3（专家笔记）+ 本次 M2/M5 = 6 篇；仍缺 M4/M6/M7/M8/M9 初学者版（M4 有实验、M7 有 OWL-ViT 实证）。

### 2026-09-12（续5）· 补 M4 初学者版教学文档（钉真实图与数字）
- 新建 `docs/M4_foundation_models.md`（初学者版）：前馈 3D 基础模型（VGGT/MASt3R/DUSt3R）直觉「一体机 vs 流水线」；深度指标讲解（RMSE/AbsRel/δ1，δ1>0.96=96%像素准）；点云 Chamfer 距离。
- 真实实验：VGGT 稠密输出 fr1/desk 60 帧 × 13 条件 → 深度 δ1 0.962–0.986、RMSE 0.10–0.16m、点云 Chamfer 0.105m；配 `../experiments/M4_depth_atlas/figs/depth_vs_degradation.png`。
- 三条结论：鲁棒性全栈（位姿→深度→点云）/最弱档是分辨率下降/高斯噪声下深度"看似更好"是因中位对齐+端到端对伪特征迟钝（呼应 M5 COLMAP 噪声崩）。
- 限制诚实声明：退化为合成施加、尺度靠对齐、不提供语义（引 M7）。
- **教学文档现状**：PRIMER + M0/M1/M3 + M2/M5 + 本次 M4 = 7 篇；仍缺 M6/M7/M8/M9 初学者版（M7 有 OWL-ViT 实证可写）。

### 2026-09-12（续6）· 补 M7 初学者版教学文档（钉真实图与数字）
- 新建 `docs/M7_semantic_layer.md`（初学者版）：语义层定位（几何管坐标/语义管理解，呼应 PRIMER §2.7）；封闭集 vs 开放词汇检测对比（DETR COCO-80 vs OWL-ViT）；经典理念「责任切分」（VLM 出 2D 语义→几何反投影 3D→规划器执行，风险隔离）。
- 真实实验：OWL-ViT(`google/owlvit-base-patch32`) 对 TUM fr1/desk 4 帧做自然语言查询 → 命中 cup×2/keyboard×11/book×20/remote×4/red object×2，bottle=0/left=0；总 39 框；配 `../experiments/M7_semantic/figs/open_vocab_detection.png`。
- 双刃结论：✅ 增益（red object/remote 属性查询命中，DETR 做不到）；⚠️ 限制（book=20 过度触发幻觉、left 关系推理弱）。点明安全关键场景不能直接信 VLM + 责任切分护栏。
- **教学文档现状**：PRIMER + M0/M1/M3 + M2/M5/M4 + 本次 M7 = 8 篇；仍缺 M6/M8/M9 初学者版。

### 2026-09-12（续7）· 整理现有文档：系统性 + 实际应用维度补齐
- **用户诉求**：整理 docs/，评估是否「系统、易懂、结合实例讲解/效果展示/原理/作用/实际应用」。
- **诊断（诚实）**：原理/实例/效果/作用四维度已扎实；两块短板——① **难度顺序倒挂**（M0/M1/M3 是满公式专家笔记，而依赖它们的 M2/M4/M5/M7 是初学者版，零基础先撞公式墙）；② **「实际应用」维度明显缺**（M2/M4/M5 无落地小节，M8 端侧部署完全无文档）；③ README 导航过期（仍只列 M0/M1/M3，新文档隐形）。
- **整理动作**：
  1. 新建 `docs/00_INDEX.md`：系统总索引——文档分两层两难度说明、七维覆盖一览表、三条阅读路径（零基础/有基础/做项目）、真实实验图索引、待补章节（M6/M8/M9）诚实标注、四段式写作规范。
  2. 给 M2/M4/M5/M7 各补「🏭 实际应用」小节（M2 重建→工业检测/抓取/AR/文保；M4→机器人实时建图/AR 锚定/视频3D化/造数据；M5→选型/验收SLA/安全论证/科研硬货；M7→服务机器人/仓储拣选/视障辅助/自动驾驶），均钉到本项目真实实验，呼应「作用介绍+实际应用」。
  3. 修 `README.md` 第三章导航（纳入全部 11 篇+分级难度）、第五章目录结构、第七章当前状态表（M2/M4/M5/M7 标已完成、M6/M8/M9 标待补）。
- **整理后状态**：11 篇文档 + 1 索引，七维全覆盖；唯一结构性缺口是 M6/M8/M9 教学文档（仅 ROADMAP 规划），正是差异化路线收口，优先补。

### 2026-09-12（续8）· 补 M8 端侧部署 初学者版教学文档
- 新建 `docs/M8_edge_deployment.md`（初学者版，用户主场·最该讲应用）：端侧定义（小芯片实时省电跑智能）+ 类比（满汉全席→方便面）；核心概念=三瓶颈(算力/内存/功耗)+量化(INT8 PTQ/QAT,+四舍五入类比)+格式链路(PyTorch→ONNX→TensorRT/RKNN/昇腾)+三指标(延迟/功耗/精度损失)；经典理念=端侧是差异化壁垒(几何+VLM+部署三合一稀缺)；真实知识=芯片量级(Jetson Orin~100TOPS/RK3588~6TOPS/昇腾~20TOPS,※行业真实)+量化对几何敏感+可跑实验骨架(trtexec INT8)+三维表模板；三条结论(三角权衡/PTQ首选/退化鲁棒性打折)；🏭实际应用(无人机/车载/AR/工业巡检/摄像头端智能)；限制诚实声明=项目实测待补(目标Jetson/昇腾)。
- 严守「真实有效」：明确区分※行业真实量级 vs 本项目实测(待补)，未编造项目数字；给可跑命令骨架与三维表模板待填。
- 已联动：`00_INDEX.md` 七维表 M8 改「已补(实测待补)」+ 路径C + 待补章节标注；`README.md` 第三章导航 + 第五章目录 + 第七章状态表均纳入 M8。
- **教学文档现状**：PRIMER + M0/M1/M3(专家) + M2/M4/M5/M7/M8(初学者版，实验待补) = 12 篇 + 索引；仍完全待补：M6 多模态融合 / M9 综合闭环。

### 2026-09-12（续9）· 补基础篇 F0–F4：填「一般场景+基础知识」地基
- **用户诉求（关键反思）**：教程能否让「懂 ML/DL/一般 CS」的人成长为场景感知/建模专业人员？结论：此前不能——难的一端(M0/M1/M3 公式)+坏场景一端(M3/M5 退化)都有，但中间「一般场景+基础知识」地基缺失，且过度偏重退化/恶劣场景，基础/通用场景知识反而缺。
- **新建基础篇 F 系列（初学者友好、一般场景、零公式门槛、引用领域标准而非编造项目数字）**：
  - `F0 相机投影`：针孔模型/内参K/外参[R|t]/投影反投影/坐标帧/畸变；numpy 可跑投影闭环；效果引 M0 图。
  - `F1 场景表示`：点云/网格/体素/SDF/占用场/NeRF/3DGS 全家桶（数学本质+何时用+ML 类比）；引 M2/M4 图。
  - `F2 深度基础`：单目歧义/双目视差 Z=fB/d/RGB-D/运动恢复/学习式深度+指标(AbsRel/RMSE/δ1)；引 M4 图。
  - `F3 正常流程`：采集→特征→位姿→三角化→BA→稠密→网格→语义 标准 SfM 流水线（Happy Path），作为全仓库目录；引 M1/M2 图；诚实声明"正常场景成立，退化见 M3/M5"。
  - `F4 ML/DL 桥接`：ML 知识→3D 视觉映射表（CNN→SuperPoint/Transformer→LoFTR/回归→Depth Anything/大模型→VGGT）；必补几何三件套；心智模型「NN 猜+几何较真」；竞争力组合。
- **联动更新**：`00_INDEX.md` 重写为三层(00/F/M)+七维表加 F 行+三条路径(路径A零基础/F打头、路径B ML人 F4 起、路径C)+图索引+待补标注(仅 M6/M9)；`README.md` 第三章导航表纳入 F0–F4 并分级编号、第五章目录加 F 文档、待补说明更新。
- **诚实原则**：F 系列严格区分「领域标准知识/公开数据集/可跑片段/本项目已有图」与「项目实测新指标」，未编造数字；明确标注"正常场景"，与 M3/M5 退化内容解耦。
- **教学文档现状**：00_INDEX + 00_PRIMER/QUICKSTART/LANDSCAPE/ROADMAP + F0–F4(基础) + M0/M1/M3(专家) + M2/M4/M5/M7/M8(初学者版) = 17 篇 + 索引；仅 M6/M9 完全待补。

### 2026-09-12（续10）· 为 F 系列知识文档补「程序跑的图」
- **用户诉求**：在知识中加入实际图/程序跑的图展示效果。
- **动作**：写 `scripts/foundations_make_figs.py`（numpy+matplotlib，无外部数据依赖），生成 F0–F3 概念演示图（f0_projection / f1_representations / f2_depth_disparity / f3_triangulation）存于 `experiments/foundations/figs/`；并在 F0–F4 的「效果展示」小节用 `![]()` 嵌入这些程序图 + 仓库已有真实实验图（M0 fig1 投影 / M1 fig4 3D地图 / M2 recon 网格 / M4 深度 / M7 检测），让文档真正"有图有真相"。
- **诚实区分**：F0–F3 四张为「程序生成的概念演示图」（演示原理，非真实数据集指标）；M0/M1/M2/M4/M7 为「真实实验图」（项目实测）。`00_INDEX.md` 图索引同步加 `experiments/foundations/figs/` 行。
- 运行：`python scripts/foundations_make_figs.py`（已验证出 4 张 png）。

### 2026-09-12（续11）· 补 M6 多模态融合 + M9 综合闭环（路线封口）
- **用户诉求**：基于「让懂 ML/DL/CV 的人成长为能实时感知/重建/理解/解构世界的专业人员」这一目标，问现有工作该优化还是进行下一步。
- **判断**：地基（F0–F4 + M0–M3 + M2/M4/M5/M7）已稳，不必返工；目标缺口恰在「解构世界/实时/恶劣环境还准/端侧闭环」——即 M6/M9 与 M8 实测。故直接进入下一步：写 M6 + M9，且都用**真实可跑脚本**而非空谈。
- **M6 多模态融合（初学者版）**：核心思想「诊断驱动模态切换」——用 M3/M5 的 `inlier_ratio` 做退化监控，低于阈值 τ 即判定几何不可信、需切换/融合 RGB-D/IMU/VGGT。脚本 `scripts/m6_fusion_guard.py` 复用 `MonocularVO` 诊断接口 + M3 的 `degrade`，实测：干净内点率 75.2%/rescue 1%/ATE 0.614m；**低光 sev=0.9 内点率 3.8%/rescue 96%**（真实触发报警）；深度救援 SE3 ATE 0.204m（拿到公制尺度）。如实记录 `blur/noise/low_texture` 在 `degrade` 中为空操作。
- **M9 综合闭环（初学者版）**：把「感知→重建→语义→场景图」串成系统，产出带 3D 坐标的物体实例。脚本 `scripts/m9_scene_graph.py`（模型无关、立刻能跑）：VO 位姿 + 深度前景分割 + 反投影 + 跨帧聚类 → 场景图。实测 **VO 9.8 fps（纯 CPU 无 GPU）**、解构 16 个 3D 物体候选；如实标注「跨帧数据关联/tracking」为生产化待补项（同一物体每帧略漂、各出现 1 次）。
- **联动更新**：新增 `docs/M6_multimodal_fusion.md`、`docs/M9_comprehensive_closed_loop.md`；README 状态表/文档地图/路径 B·C/目录/诚实声明全部去掉「M6/M9 待补」并补真实数字；`00_INDEX.md` 七维表补 M6/M9 行、图索引加 M6/M9 图、待补章节改为已补；本文件顶部状态改为「教程体系已完整覆盖 M0–M9」，唯一待补实测仅 M8 端侧真机三维表。
- **诚实原则延续**：M6/M9 均为初学者版 + 可跑脚本 + 实测数字；明确区分「本项目实测（M6 监控数字、M9 fps/物体数）」与「引用 M5 的 VGGT ATE 0.016–0.049」「M8 真机数字待补」。
- 教学文档现状：**19 篇 + 索引**（00_* + F0–F4 + M0/M1/M3 专家 + M2/M4/M5/M6/M7/M8/M9 初学者版）；仅 M8 端侧真机实测待补。

### 2026-09-12（续12）· M8 端侧三维表用本机缩预算实测（无需真机）
- **用户洞察**：M8 此前唯一「待补实测」是端侧真机三维表（需 Jetson/昇腾）。用户指出：端侧本质就是算力更小，用本地大算力把模型/输入「控制打小」就能模拟，不是卡点。
- **落实**：写 `scripts/m8_edge_sim.py`，把单目 VO 管线的「输入分辨率 / 特征数 / 特征算子(SIFT→ORB)」当算力旋钮，在本机跑 4 档并实测三指标：
  - Large 1.0× SIFT/2000：延迟 147ms，ATE 0.567m，算力预算 1.000（基准）
  - **Medium 0.5× ORB/1000：延迟 82ms，ATE 0.363m，预算 0.044 → 最佳性价比**（且 ATE 优于全分辨率 SIFT）
  - Small 0.5× ORB/400：FAIL（特征地板，砍特征越过阈值直接初始化失败）
  - Tiny 0.4× ORB/1000：FAIL（分辨率悬崖，砍分辨率越过阈值直接初始化失败）
- **关键发现（教学价值高于模板）**：端侧可行域是「窗口」而非「连续旋钮」——0.5× ORB1000 是最优紧凑配置；一旦越过特征地板或分辨率悬崖，VO **直接初始化失败**而非精度平滑下降。这正好呼应 M8「端侧须配 M3 失效预警、校准集须覆盖退化」。
- **功耗处理（诚实）**：本机 RAPL `energy_uj` 需 root，故功耗用「相对算力预算」代理（分辨率²×特征数×算子代价，Large=1.0）；脚本注释给出 `sudo` 读 intel-rapl 换绝对瓦数方法。延迟/精度为直接实测，与架构无关。
- **联动更新**：`docs/M8_edge_deployment.md` 把 §4.3 模板实验替换为实测三维表+`experiments/M8_edge/figs/tradeoff.png`+关键发现，诚实声明改为「本机缩预算模拟端侧」；README 文档地图/顶部声明/端侧工具链对比(行152-153)/第七章状态表/诚实声明全部去掉「M8 真机待补」；`00_INDEX.md` 七维表 M8 行、路径C、待补章节同步改为已实测；本文件顶部状态「唯一待补实测」移除，模块路线图 M8 行 ⬜→✅。
- **现状**：19 篇教学文档 + 索引，M0–M9 全部覆盖且三维表/可跑实测均闭环；仅 RKNN/ACL 真机编译仍待真机（非卡点）。

### 2026-09-12（续13）· M8 补 INT8 PTQ 真跑实验（验证量化链路）
- 承接续12「端侧三维表用本机缩预算实测」。用户说「继续完善」，最有价值的是把 M8 文档里「PyTorch→ONNX→引擎」链路与「INT8 PTQ 精度损失 1~3%、几何敏感」结论真正跑一遍。
- 写 `scripts/m8_ptq_demo.py`：合成 2 类 32×32 灰度任务（低频平滑团 vs 纯噪声，天然可分）→ 训 TinyCNN(FP32) → 静态 INT8 PTQ（QuantStub/DeQuantStub + 融合 conv+bn+relu + fbgemm qconfig + 校准 + convert）→ 对比。完全离线可复现（torch 2.6+cu124 已确认可用；去掉了 scipy 依赖改纯 numpy）。
- **真实结果**：FP32 100.00% / 1.23ms vs INT8 99.75% / 1.34ms → **精度损失 0.25pp**，但**本机 x86 上延迟 ~1:1（0.92× 略慢）**。诚实结论：PTQ 精度损失极小印证「INT8 是端侧首选」；但**加速只在整数原生端侧 NPU（RK3588/Jetson）上兑现**，开发机 fbgemm 是软件模拟 int8、小模型开销盖过收益。
- 文档 `docs/M8_edge_deployment.md` 新增 §4.4（INT8 PTQ 真跑 + `figs/ptq.png` + 诚实解读：分类是乐观基线、几何敏感任务损失更高需谨慎校准）；`00_INDEX.md` 七维表 M8 行补「INT8 PTQ 真跑」、图索引加 tradeoff.png 与 ptq.png；本文件续12 现状不变（仍闭环）。
- **经验**：torch 静态量化必须 conv+bn+relu 融合 + 插量化桩，否则报错；数据要 channels-first (N,1,H,W)；reduce_range 弃用告警可忽略。

### 2026-09-12（续23）· ModelNet40-C 点云退化 demo（把"3D 退化"也跑通）
- **动因**：ModelNet40-C 已下完（2.2G），且它把项目主线「方法 × 环境退化」从**图像**扩展到**点云**模态——不该只躺在硬盘里。
- **产出** `scripts/modelnet40c_corruption_demo.py`（已跑通）：
  - **退化图库** `corruption_gallery.png`：同一物体 8 种损坏形态（gaussian 扩散 / shear 压扁 / lidar 稀疏化…）；
  - **Chamfer 曲线** `chamfer_curve.png`：14 种损坏类型 × severity 1–5 相对 original 的形状偏移。
  - 实测：`lidar` 0.045→0.137、`occlusion` 0.134、`distortion` 0.07–0.09（**结构损坏最重**）；`upsampling` 0.022、`density_inc` ≈0.023（**纯增点不伤形状**）。
  - ⚠️ 诚实说明：基于**单样本**（非千样本平均）→ 曲线**非严格单调**，趋势可信、逐点数值仅作直觉。
- **联动**：`docs/DATASETS.md` ModelNet40-C 小节补 demo 结果 + 2 张新图；`00_INDEX` 图索引加行。
- **验收**：全仓库 **107 处图片引用 → 缺失 0**。

### 2026-09-12（续22）· 数据集详解文档 + 真实样例图集
- **用户诉求**：在文档中**详细介绍并展示**这些数据集——场景、有哪些信息、可用于哪些任务。
- **产出 1 · 样例图集脚本** `scripts/datasets_showcase.py`：从真实下载数据抽样例，生成 5 张图 `experiments/datasets_showcase/figs/`：
  - `nightcity.png`（夜间城市驾驶 8 例）、`sunrgbd.png`（室内 RGB **上排** + 深度真值 **下排**成对）、`ndispark.png`（夜/昼停车场 + 车辆）、`modelnet40c.png`（3D 点云 + **15 种退化**可视化）、`fourseasons.png`（夜间双目驾驶）。
- **产出 2 · 重写 `docs/DATASETS.md` 为详解版**：总览表 → **逐数据集详解**（场景 / 包含什么信息 / 能做什么任务 / 本项目怎么用 / 局限）+ **真实样例图** + 数据集→任务→模块映射表 + 获取核查命令 + 诚实声明。
  - 诚实标注：ScanNet/TartanAir-Absolute-Camera **仅 camera.json 无图像**；NYUv2 体积巨大只留子集。
- **产出 3 · SUN RGB-D 深度真值 demo** `scripts/sunrgbd_depth_demo.py`（续21 已建，本轮配图）：实测 NYU1235 RGB 480×640、深度 1.32–5.69m、30.7 万点云。
- **联动**：`00_INDEX` 图索引新增数据集样例图集两行；README 数据台账描述更新。
- **验收**：全仓库 **105 处图片引用 → 缺失 0**（新增 5 张样例图 + 1 张深度 demo 图）。

### 2026-09-12（续21）· 数据集扩充：ModelScope 多模态数据台账 + 分批下载
- **用户洞察（正确）**：现有数据（仅 TUM fr1/desk + 4Seasons）**偏少 → 对场景/目标/方法的理解偏窄**；要求从 ModelScope 多下各种模态/场景的真实数据。
- **核实**：ModelScope SDK 1.40 已装；`list_datasets` 可用（须 `repo_type='dataset'`）；实测下载 **~2–3.6 MB/s**（远优于 TUM ~50KB/s）；磁盘剩 894G。
- **产出 1 · 下载器** `scripts/fetch_datasets.py`：分层清单（A 小体积高价值 / B 深度真值 / C 多模态恶劣 / D 大基准），`--list`/`--tier A|all`，自动写 `data/raw/ms/manifest.json` 台账。
- **产出 2 · 数据台账** `docs/DATASETS.md`：已有真实数据 + 待下数据集（模态/场景/体积/**回答什么问题**）+ **数据集→任务→模块映射表** + 获取方式 + 诚实声明。
- **产出 3 · 核查脚本** `scripts/list_datasets.py`：一键扫描 `data/raw/`（含 ms/），输出每个数据集体积/结构/完整性（`.incomplete` 标记）。
- **实际落地（2026-09-12 16:32 核查）**：
  - ✅ **NightCity** 2.1G（夜间驾驶）
  - ✅ **SUN RGB-D** 8.7G（室内深度真值）
  - ✅ **ScanNet-Absolute-Camera** 0.42G（**1468 个场景 zip**，绝对位姿）
  - ✅ **TartanAir-Absolute-Camera**（**18 个场景 zip**：含 abandonedfactory_night / gascola / endofworld 等极端环境）
  - 🔄 **ModelNet40-C**（点云）、**NYUv2**（157 parquet，深度真值）下载中
- **后台下载（Tier A + Tier B 并行，共享带宽）**：
  - Tier A（log `/tmp/fetch_tierA.log`）：NightCity(2.1G, 夜间) → ScanNet-Absolute-Camera(0.42G) → TartanAir-Absolute-Camera(极端环境) → ModelNet40-C(2.28G) → 遥感变化检测(2.06G)；
  - Tier B（log `/tmp/fetch_tierB.log`）：SUN_RGB-D(8.7G, 深度真值) → NYUv2。
  - ⚠️ 并行使单流从 3.6MB/s 降到 ~280KB/s（预期）；SUN RGB-D 逐文件下载(4009 files)预计 ~1.5h。
- **决策调整（用户 2026-09-12）**：NYUv2 完整集 32.6G（解压 79.5G）与 SUN RGB-D 用途重叠 → **停 NYUv2**（保留 20 parquet 子集），**带宽优先给"恶劣环境"**。新增 **Tier E**（夜间/水下/航拍/KITTI 融合）并已启动。
- **计划后续**：Tier C（KITTI depth_completion 视觉+LiDAR、Foggy_Cityscapes 雾天）→ Tier D（ADE20K/COCO2017/ScanNet 大基准）。
- **消费路径**：`projects/loaders.py` 已有场景 source（tum/tum_degraded/4seasons/synthetic）；新数据集落地后按需加 loader（归一成 `Sequence`）。
- **联动**：README 文档地图 + 目录树、`00_INDEX` 加数据台账行。
- **下一步**：待数据落地 → 用 `loaders.py` 归一化 → 接入 M4/M5/M7 做**真实数据**（替代部分合成退化）评测。

### 2026-09-12（续20）· 执行 C1：视觉+激光雷达融合系统专章（新建 F5 + 可跑 demo）
- **用户诉求（此前明确指出）**：落地方案多为「视觉+激光雷达/点云」互补（视觉识物、点云定距），但教程**无系统专章**（原仅 `P0 §4` 一节）。→ C1。
- **产出 1 · 可跑 demo** `scripts/m10_lidar_fusion_demo.py`（新）：用 TUM 真实深度作点云几何源 → 跑**「点云 ↔ 图像」投影闭环**：
  - 反投影出 **255,190** 点；稀疏化模拟 LiDAR 扫描线（**4,000** 点落入视场）；
  - **重投影残差 0.0000 px**（3D→2D 与原始像素完全吻合 → 验证 K/外参自洽）；
  - 三联图：① RGB（语义强/几何弱）→ ② 稀疏点云投影（颜色=距离）→ ③ 彩色点云融合体。
- **产出 2 · 专章** `docs/F5_vision_lidar_fusion.md`（22 篇，基础篇 F 系列第 6 篇，严格四段式）：
  - §1 目的（能力边界对照表 + 本项目 `autonomous_driving` 无深度→必须加激光的真实动机）；
  - §2 原理（相机-激光**外参标定**靶标/无靶标、**投影公式**、**前/中/后融合**架构对比表、KITTI/nuScenes 3D 检测：PointPillars/SECOND/CenterPoint/BEVFusion）；
  - §3 类比（"懂行但估不准距离的鉴定师 × 精准但没文化的卷尺"）；
  - §4 真实验证（上表实测 + 三联图 + 诚实声明：点云由 RGB-D 反投影，非真实 LiDAR 采集）；
  - §5 与前后模块关系 / §6 实际应用 / §7 三条结论。
- **联动更新**：`00_INDEX.md`（F5 行 + 路径 A/B/C + 图索引 + 精选图墙⑨ + 待补章节去掉"融合专章"）；`README.md`（文档地图/目录树/双轨说明）；
  `P0_projects.md §4`（缺口段改为"已补强"，加 F5 链接）；`F3`（下游表加"多传感器互补→F5"）；`F4`（竞争力加 F5）。
- **验收**：demo 实跑通过；全仓库 **101 处图片引用 → 缺失 0**；docs 共 **22 篇**。
- **下一步**：C2（M8 真机）/ C3（M7 完整语义闭环）需外部条件（真机 / 升 transformers）；D1 真实数据受网络限制。

### 2026-09-12（续19）· 执行 B 组图表打磨（B1–B4 全部完成）
- **B1 M9 场景图**：重写 `scripts/m9_scene_graph.py` 出图——左图**灰点=反投影候选 / 红星=聚类实例**（原来几乎重合，看不出"16→16"）；右图由「全 1.0 的实例计数」（零信息）改为**每个实例的 X/Y/Z 世界坐标横向条形图**。实跑 9.9 fps / 16 实例。
- **B2 M8 权衡图**：新增 `scripts/m8_make_figs.py`（从 metrics.json 秒级重绘，不重跑 VO）——加**失败带** + FAIL 档（Small 特征地板 / Tiny 分辨率悬崖）**带原因标注** + ★最佳性价比点。修掉 ✗ 字符缺字（用文字替代）。
- **B3 M0/M1/M3 降门槛**：三篇专家笔记文首各加「🧭 初学者导读（5 分钟版）」——4 句话总结 + 术语直觉词典 + 建议读法 + 读完能回答。
- **B4 关键图自解释**：新增 `scripts/m5_make_figs.py`（从 metrics.json 重绘）——M5 热力图加**图内「怎么读这张图」注记** + 标题内嵌颜色含义。
- **验收**：3 个新脚本实跑出图；5 个脚本 `py_compile` 全过；**99 处图片引用 → 缺失 0**（不变）。
- **新增脚本**：`scripts/m8_make_figs.py`、`scripts/m5_make_figs.py`（均支持「不重跑实验、由 metrics 重绘」，性价比高）。
- **下一步**：C1 **视觉+LiDAR 融合系统专章**（用户明确指出的最大内容缺口）。

### 2026-09-12（续18）· 教程「图不显示」全量修复：锚文本地址 → 嵌入真图
- **用户反馈**：教程里「好多图片不显示，只给了地址」。
- **根因诊断**：图引用分两类——① `![标题](../experiments/...)`（正确嵌入，会显示）；② 正文里写「（见 `../experiments/xxx.png`）」或 `[fig1](../...)` **纯文本链接/路径，Markdown 不渲染成图**。共 **32 处**属第 ② 类（散布在 LANDSCAPE/PRIMER/M5/M1/M0/M8/INDEX）。
- **修复动作（全部改为 `![]()` 嵌入）**：
  - `00_LANDSCAPE.md`：补 6 张实跑图（DETR 检测 / GrabCut 分割 / VGGT 深度 / 四方法 ATE / OWL-ViT ×2）。
  - `00_PRIMER.md`：补 3 张（投影/重建/对极线）+ §5 末尾新增「**精选图墙（5 张）**」。
  - `00_INDEX.md`：§4 表后新增「**一屏看懂本项目做了什么（9 张精选图墙）**」。
  - `M5_stress_test.md` / `M1_sfm_from_scratch.md` / `M7_semantic_layer.md`：把「见 xxx.png」改嵌入；M7 新增 DETR vs OWL-ViT **并排对照双图**。
  - `M2`/`M4`/`M9`：各补 1 张实战/对照图；`M4` 补 M5 对照热力图。
  - `00_ROADMAP.md`：0 图 → 新增「**全局地图一图流（10 层代表作）**」。
  - `README.md`：新增「**成果一览**」4×2 图墙（8 张）。
- **验收（脚本核验）**：`docs/ + README + PROGRESS` 全量 **99 处图片引用 → 缺失 0**（修复前 50 处、其中 32 处只给地址不显示）；零图文档仅剩 `PROGRESS.md`（台账无需图）。
- **说明**：仍保留少量「路径文字」（如 `experiments/...` 命令输出说明、索引表路径列）——那些是**合法的引用说明**，非"该显示却没显示"的图，已逐条确认。

### 2026-09-12（续17）· 执行 A 组清障（A1–A4 全部完成 + 计划文档更新）
- **用户要求**：开始执行 A1–A4，每完成一定程度更新计划目标文档以便他人接手。
- **A1 修复 M8 图片路径 bug**：`docs/M8_edge_deployment.md` L141/L169 `experiments/...` → `../experiments/...`。
- **A2 清陈旧引用**：`00_LANDSCAPE.md`（能力栈表 3 行 + §5 速查表 4 行 + 结尾建议）、`F2_depth_basics.md` 1 处、`F3_normal_pipeline.md` 1 处，全部去掉「待做/待补」并挂真实数字（96% 救援 / 39 框 / 16 物体 / PTQ 0.25pp）。
- **A3 修索引**：`00_INDEX.md` 初学者版清单补 `M6/M8/M9`；§5「待补章节」→「覆盖状态确认」，新增真实 3 项非结构性待补（真机编译 / LiDAR 专章 / 真实数据）。
- **A4 修表述**：`00_PRIMER.md` / `00_INDEX.md` / `00_LANDSCAPE.md` 的「三方法」→「四方法」（补 LightGlue）。
- **验收（脚本核验）**：`docs/` 全量 **50 处图片引用 → 缺失 0**（修复前 M8 两处会图裂）；`grep 待补/待做` 陈旧引用清零；linter 0 错误。
- **计划文档同步**：PROGRESS 顶部「当前阶段」+ §2.2 A 组（标 ✅ 完成 + 验收结果）+ 推荐顺序（A 划掉、下一步 B）+ 本条日志。
- **下一步**：B 组图表打磨（B1 M9 场景图 / B2 M8 权衡图 / B3 M0/M1/M3 导读 / B4 图注规范）。

### 2026-09-12（续16）· 全量教程审查 + 待办重排（本轮只规划，不改文档）
- **用户诉求**：先读 PROGRESS + 项目评估教程「是否合理/全面/详细易懂/图例详细合理」，然后**更新安排与待办，使后续计划合理、目标清晰准确**。
- **审查结论（诚实）**：体系**合理、全面、实证扎实**（19 篇 + 索引，四段式统一，42 张真实图 + metrics.json 齐全）；短板集中在**打磨层**，非结构层。
- **发现的具体缺陷（已写入 §2.2 A/B 组）**：
  1. 🔴 `docs/M8_edge_deployment.md` L141/L169 图片路径缺 `../`（全仓库唯一 bug，GitHub 会图裂）。
  2. 陈旧引用：`00_LANDSCAPE.md`(L182-184/201-204)、`F2_depth_basics.md`(L84)、`F3_normal_pipeline.md`(L83) 仍写 M6/M7/M8/M9「待做/待补」。
  3. `00_INDEX.md` L20 初学者版清单漏列 M6/M8/M9；`M5 atlas_ate.png` 图注「三方法」实为四方法。
  4. 图教学力：`M9 scene_graph.png`（候选与质心几乎重合、右柱全 1.0 无信息）、`M8 tradeoff.png`（FAIL 档未标注）、M0/M1/M3 无「初学者导读」。
- **PROGRESS 更新动作**：
  - §2 路线图：新增「🎯 北极星目标」一句话 + 校正 M6/M8/M9/T0 状态为 ✅ + 加「状态说明（教程与实验全闭环，唯一待真机=交叉编译）」。
  - §2.2 待办**重排为目标导向 5 组**：**A 缺陷修复(P0)** / **B 图表打磨(P1)** / **C 差异化主场(P1)** / **D 真实数据验证(P2)** / **E 可选增强(P3)**，每项带「具体动作 + 验收标准 + 依赖风险」+ 推荐执行顺序。
  - 文件顶部「当前阶段」改为「清障→打磨→补缺口→真实数据」，并写明本次审查结论。
- **下一步建议**：按 `A→B→C1（视觉+LiDAR 融合专章，最大内容缺口）→C2/C3→D1` 执行；A/B 为让现有教程立得住的必做项。
- **用户诉求（两条）**：① 整个教程图片太少，`P0_projects.md` 纯文字无例图/结果图/结果分析（"一个实战讲解全是文字有什么意思"）；② 落地方案是否多为「视觉+激光雷达/点云」互补（视觉识别物、点云定距），教程有无系统介绍。
- **诊断（诚实）**：`P0_projects.md` 是**全仓库唯一 0 张图**的文档（M0/M1/M6 等其实有图）；视觉+LiDAR 融合**教程无系统专章**（M6 仅泛泛覆盖 RGB-D/IMU/VGGT，§5 只提一句「激光+相机+GNSS 多源融合」）——确为真实缺口。
- **图文富化（新增 2 张配图 + 4 张结果图入文）**：
  - 写 `scripts/make_project_figs.py` → `experiments/projects/figs/dataset_samples.png`（6 子图：TUM RGB / TUM 深度 viridis / 4Seasons 夜间 RGB / TUM+M3 退化 / 合成 RGB / 合成深度），标题全英文避免 matplotlib 缺 CJK 方块。
  - 写 `scripts/make_lidar_fusion_fig.py` → `experiments/projects/figs/vision_lidar_fusion.png`（视觉+深度反投影点云融合几何示意）。
  - `docs/P0_projects.md` 重写为**图文结合版**（221 行）：§3.0 数据集长什么样（挂 dataset_samples.png）+ 四个项目各挂 `overview.png` + 逐项实测分析 + §4 视觉/LiDAR 融合专章（回应问题②）+ §6 诚实边界。
  - ⚠️ 坑：`depth_img` 里 `(d<=0) or ~isfinite(d)` 对数组触发布尔歧义 → 改 `(d<=0) | ~np.isfinite(d)`。
- **代码诚实修正（发现两处过度承诺/张冠李戴，已修 `src/wildspatial/projects/stages.py`）**：
  - ① **VO 回退仅限合成序列**：原逻辑「退化即回填 GT 里程计」会让 TUM 退化序列显示「公制 ATE=0.0」假象，**反倒掩盖了以展示 VO 退化为目的的项目**；且回退文案写死「合成序列」对 TUM 是张冠李戴。改为 `seq.scenario == "synthetic"` 才回退，真实数据集如实展示崩坏。
  - ② **新增 `vo_degenerate` 标记**：真实序列 VO 跑出但轨迹塌缩（平移≈0）→ 场景图**诚实跳过**（不拿错误位姿糊弄出「假物体」）。
- **四个项目重跑后真实数字（诚实自洽）**：`indoor_mapping` ATE 0.643→0.686m、17 物体；`search_and_rescue` VO 退化下实跑 ATE **0.867（自由尺度）**、深度/场景图诚实跳过（正是「需 M6 融合兜底」实证）；`autonomous_driving` 因 4Seasons 切片无深度→场景图**诚实跳过**（恰好引出「真实驾驶必须补一路 LiDAR」）；`ar_inspection` 合成回退已知轨迹（标注非伪造）、6 物体。
- **联动更新**：`00_INDEX.md` 图索引补 dataset_samples.png / vision_lidar_fusion.png。
- **⚠️ 遗留待办**：教程仍缺「视觉+LiDAR 融合」**系统专章**（现仅 P0 §4 一节）；后续应在 M6 或新开一篇补「点云/激光→相机标定、前融合/后融合、KITTI/nuScenes/4Seasons 3D 检测」系统讲解。

### 2026-09-12（续14）· P0 实战项目整合层：把多场景公开数据集串成可落地工程
- **用户诉求**：结合许多不同场景的公开数据集，模拟一些完整、逻辑自洽、像能落地实用的实战项目，把知识（F0–F4 + M0–M9）引入实战。
- **架构（三层抽象）**：`Sequence`（跨数据集归一：rgbs/K/depths/gt_positions/scenario）→ `Stage`（把 M-module 封装成 `run(seq, ctx)`→产物写回共享 ctx 传送带，后序消费前序，因果自洽）→ `Project`（场景+有序 Stage+交付物渲染+知识落点叙述）。模块对应：VOStage(M2)→DepthStage(M4)→SemanticStage(M7)→SceneGraphStage(M9)→EdgeStage(M8)。
- **数据源（多场景公开数据集）**：① TUM RGB-D `fr1/desk`（室内，含深度+真值）；② 4Seasons `oldtown_night`（自动驾驶，GNSS 真值，本地已下）；③ TUM `fr1/desk` + M3 合成退化（低光+模糊，模拟搜救）；④ 程序合成序列（悬浮 3D 立方体，无下载，演示框架与数据来源解耦）。
- **四个项目（均端到端跑通 exit=0）**：`indoor_mapping`（vo→depth→semantic→scene_graph→edge，实测 17 物体、公制 ATE≈0.69m）、`autonomous_driving`（vo→semantic→scene_graph→edge，4Seasons GNSS 与图像时间戳不重叠→GT 退化护栏自动跳过 ATE 并标注，仍是真工程真值对齐坑）、`search_and_rescue`（fr1/desk+低光/模糊，VO 退化下深度定尺度跳过，解构 16 物体）、`ar_inspection`（合成序列，纯色平面导致本质矩阵退化→VO 自动回退已知轨迹驱动下游并标注「非伪造精度」）。
- **关键工程纪律（诚实边界）**：① 单目 VO 尺度歧义靠深度定尺度；② 平面/近纯旋转致本质矩阵退化（homography 退化，平移不可观）→ 回退已知轨迹；③ GT 退化护栏（全映射同一点→ATE 跳过不报假 0）；④ 合成/退化数据显式标注不冒充真采；⑤ 端侧功耗用相对算力预算代理（同 M8）。
- **交付**：`src/wildspatial/projects/{core,loaders,stages,projects,__init__}.py` + `scripts/run_project.py` + `docs/P0_projects.md`；每项目产 `experiments/projects/<p>/{metrics.json,figs/overview.png,README.md}`（知识→实战桥梁文档）。已联动 README/00_INDEX/PROGRESS。
- **踩坑实录**：4Seasons zip 实为 `stereo.zip`（非 `stereo_images_undistorted.zip`）；`fr3_nostructure` 缺 rgb.txt/真值→搜救改用 fr1/desk+M3 退化;合成序列纯色平面致 VO 初始化失败→最终用「已知轨迹回退」保证链路演示而非伪造精度。

## 6. 已知坑 / 风险

- 🔴 **多线程下载会静默损坏文件**：`fr3_nostructure` 用 12 线程下载后文件大小正确（484772347 字节）
  但 `tar -tzf` 报 CORRUPT。根因：连接提前关闭时写入字节数不足、文件已 truncate 到完整大小 → 中间留空洞。
  **已修复**：`_download_range` 校验每块实际写入字节数（不足回退重试最多 4 次）；新增 `verify_archive()` 做完整性校验。
  **教训**：网络实验中"文件大小对"≠"数据对"，归档类数据必须校验。
- ⚠️ `fr3_nostructure` 的 `.tgz` 已损坏，需重新下载（用户未授权删除，暂保留）。M3/M5 若要用它，先跑 `verify_archive()` 确认。
- ⚠️ SubT-MRS 数据集体量与下载许可未核实（可能需申请）。备选：对 TUM 施加**合成退化**（烟雾/低光/散射）做可控对照实验，性价比更高且可复现。
- ⚠️ VGGT 需 `flash-attn`/`ninja`，在 4090+cu124 编译是否顺利待验证；备选走纯 PyTorch 路径。
- 🔴 **transformers 4.48 不支持 Qwen3**（需 ≥4.51）：缓存的 `Qwen/Qwen3-1.7B` 无法用 4.48 加载；纯文本 LLM 实证（任务分解/故障解释）需先升级 transformers 或换 Qwen2/LLaMA 兼容模型。
- ⚠️ **OWL-ViT 后处理坑**：`post_process_object_detection` 返回「单图 dict」，标签 `labels` 才指向 text 查询索引；勿按「每查询一个元素」遍历否则全错标。
- ⚠️ **matplotlib 中文字体缺失**：项目 `viz/plots.py` 有统一中文字体，但独立脚本（landscape_demo/m7_semantic_demo）默认 DejaVu 缺 CJK → 图里中文变方块；独立脚本图标题用英文。
- ⚠️ **4Seasons TUM 直链限速**：实测 ~50KB/s（3.3GB 立体图需 ~7h 不可行），已放弃；SubT-MRS 经 Google Drive 可达但单序列 9.25GB 且缺相机内参 → **Task3 真实数据暂缓**，以 M5/M4 合成退化结论为当前王牌交付。

---

## 追加模板
```
### YYYY-MM-DD
- **做了**：...
- **结论/发现**：...（数据说话）
- **下一步**：...
```
