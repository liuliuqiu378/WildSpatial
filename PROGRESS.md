# WildSpatial 进度台账（唯一真源）

> **AI 助手必读**：每次进入项目，先读本文件。
> 更新规则：**完成任何一步后立刻追加**，不要等收尾。格式见底部模板。
> 当前阶段：`M1 - 手搓 SfM/VO`（前端完成 + BA 已接入 VO；长序列 ATE 完整评测进行中）
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
| 备用环境 | `det`（torch 2.6.0+cu124, numpy 2.2.6, opencv, matplotlib, transformers, ultralytics）—— M0 用它跑过，可作备份 |
| 磁盘 | 剩余 921G |
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

---

## 2. 路线图（Module 总览）

| # | Module | 主题 | 状态 | 核心产出 |
|---|---|---|---|---|
| M0 | 几何地基 | 相机模型 / 李群 / 对极几何 / 三角化 / PnP | ✅ 完成 | 几何库 + 25 项自检 + 6 张教学图 |
| M1 | 手搓 SfM/VO | SIFT→匹配→RANSAC→E→三角化→PnP→BA→回环+PGO | 🟡 前端+BA+回环v1+效果展示 | TUM 基线 ATE 0.506m；BA 重投影 1.54→0.20px；回环 v1 因误检未降 ATE（诚实记录） |
| M2 | 深度与重建 | 双目视差 / RGB-D / TSDF 融合 | ⬜ | mesh 产出 |
| M3 | 失效归因科学 | 合成退化 × 逐环探针 × 带真值定量 | 🟡 首轮完成 | `docs/M3_failure_attribution.md` + 13 次实跑失效图谱 |
| M4 | 前馈 3D 基础模型 | VGGT / MASt3R 复现与对比 | ⬜ | 一体化 vs 分块 实证 |
| M5 | 压力测试矩阵 ⭐ | 环境×方法 交叉评测 | ⬜ | **核心资产：失效图谱** |
| M6 | 多模态融合补救 | IMU / 深度 / 热成像 互补 | ⬜ | 融合方案 + 提升量化 |
| M7 | 语义层 | VLM 接入 + 责任切分（TAMP-Nav 式） | ⬜ | 语义闭环 demo |
| M8 | 端侧部署 | 量化 / TensorRT / Jetson | ⬜ | 延迟·功耗·精度损失 |
| M9 | 综合 | 完整闭环 + 技术报告 | ⬜ | 简历级项目 |

图例：⬜ 未开始 🟡 进行中 ✅ 完成 ⛔ 阻塞

---

## 3. 数据集台账

| 数据集 | 场景 | 用途 | 状态 |
|---|---|---|---|
| TUM RGB-D `fr1/desk` | 室内干净桌面 | **干净基线**（含真值轨迹+深度） | ✅ 已解压可用（613 rgb / 595 depth / groundtruth） |
| TUM RGB-D `fr3/nostructure_notexture_near_withloop` | **无纹理** | M3/M5 失效用例 | ⚠️ tgz 已下载，但解压后**只有 rgb/ 与 accelerometer.txt**（无 depth/groundtruth），可用性存疑 |
| TUM RGB-D `fr3/structure_notexture_far` | 结构+无纹理远景 | M5 | ⬜ 待下 |
| SubT-MRS | 地下/隧道/烟雾/全黑，多模态 | M5/M6 主战场 | ⬜ 待评估 |
| DUO / UIEB | 水下 | M5 水下散射 | ⬜ 待评估 |

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

---

## 5. 日志（倒序追加）

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

## 6. 已知坑 / 风险

- 🔴 **多线程下载会静默损坏文件**：`fr3_nostructure` 用 12 线程下载后文件大小正确（484772347 字节）
  但 `tar -tzf` 报 CORRUPT。根因：连接提前关闭时写入字节数不足、文件已 truncate 到完整大小 → 中间留空洞。
  **已修复**：`_download_range` 校验每块实际写入字节数（不足回退重试最多 4 次）；新增 `verify_archive()` 做完整性校验。
  **教训**：网络实验中"文件大小对"≠"数据对"，归档类数据必须校验。
- ⚠️ `fr3_nostructure` 的 `.tgz` 已损坏，需重新下载（用户未授权删除，暂保留）。M3/M5 若要用它，先跑 `verify_archive()` 确认。
- ⚠️ SubT-MRS 数据集体量与下载许可未核实（可能需申请）。备选：对 TUM 施加**合成退化**（烟雾/低光/散射）做可控对照实验，性价比更高且可复现。
- ⚠️ VGGT 需 `flash-attn`/`ninja`，在 4090+cu124 编译是否顺利待验证；备选走纯 PyTorch 路径。

---

## 追加模板
```
### YYYY-MM-DD
- **做了**：...
- **结论/发现**：...（数据说话）
- **下一步**：...
```
