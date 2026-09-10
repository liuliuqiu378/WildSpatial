# WildSpatial 进度台账（唯一真源）

> **AI 助手必读**：每次进入项目，先读本文件。
> 更新规则：**完成任何一步后立刻追加**，不要等收尾。格式见底部模板。
> 当前阶段：`M1 - 手搓 SfM/VO`（进行中；M0 已完成）
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
PYTHONPATH=src python -m pytest tests/ -q          # 自检（wildspatial 环境）
PYTHONPATH=src python scripts/m0_demo.py            # M0 可视化
# 多线程下载（比 curl 快数倍）
PYTHONPATH=src python -m wildspatial.data.download <url> <out> --workers 12
```

### 关键路径
```
项目根        /home/hmn-cjy/liuliuqiu/WildSpatial
原始数据      <root>/data/raw
预处理数据    <root>/data/processed
实验产出      <root>/experiments/<module>/<exp_name>/
```

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
| M1 | 手搓 SfM/VO | SIFT→匹配→RANSAC→E→三角化→PnP→(BA) | 🟡 前端完成，**BA 待做** | TUM 基线 ATE 0.506m ✅ |
| M2 | 深度与重建 | 双目视差 / RGB-D / TSDF 融合 | ⬜ | mesh 产出 |
| M3 | 失效归因科学 | 退化建模 + 失效分类学 | ⬜ | **第一份归因报告** |
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
3. 数据放 `data/`，**不进 git**。
4. 代码库 `src/wildspatial/` 要求 **每个几何函数都有解析/数值自检**（`tests/`），因为几何错误极难调试。
5. 每完成一个 Module，更新本文件 + 写 `docs/<Mxx>_reflection.md`。

---

## 5. 日志（倒序追加）

### 2026-09-10
- **建立项目骨架**：README / PROGRESS / docs/00_ROADMAP / env / src 结构。
- **环境决策**：新建 `wildspatial` 环境（后台创建中），M0/M1 阶段先用 `det` 环境跑（依赖已够）。
- **数据决策**：选 TUM RGB-D 作为 M0–M3 主数据集。理由：① 有真值轨迹与深度，可量化；② 自带
  `nostructure_notexture` 序列，天然是"无纹理失效"实验台；③ 单序列仅 344MB，适配小团队算力。
- **启动**：后台下载 `fr1/desk`；后台创建 conda 环境。

### 2026-09-10（续）· M0 完成 ✅
- **产出**：`src/wildspatial/geometry/` 五个模块全部手写（lie / camera / epipolar / triangulation / pnp），
  `tests/test_geometry.py` **25 项数值自检全绿**（`wildspatial` 与 `det` 双环境均通过）。
- **教学产出**：`docs/M0_geometry_foundation.md`（概念+推导+坑+反思）+ `scripts/m0_demo.py`
  → `experiments/M0_geometry_foundation/figs/` 6 张图（投影、极线、噪声退化、视差角、纯旋转退化、PnP精化）。
- **工具**：`src/wildspatial/viz/plots.py`（中文字体统一处理，Noto Sans CJK）；
  `src/wildspatial/data/download.py`（多线程分块下载，TUM 单连接太慢）。
- **🔴 踩坑记录（重要，后面会反复遇到）**：
  1. **`(N,3)` vs `(N,2)` 静默错位**：`normalize_points()` 返回齐次坐标 (N,3)，八点法按 (N,2) reshape 后
     数据被打乱但**不报错**，解出的 E 与真值余弦相似度 0.83 —— 看起来合理，实则全错。
     → 已加 `_as_euclidean()` 统一处理。**教训：几何代码必须数值自检。**
  2. **PnP DLT 的全局符号不定性**：`M` 与 `-M` 残差完全相同，SVD 无法区分。
     → 用正深度（chirality）消歧，否则平移整体反号（测试直接抓到）。
  3. **DLT PnP 必须用归一化坐标**：方程建立在 `u = X/Z` 上，不先乘 `K^{-1}` 就会把内参污染进 R。
  4. **三个"对称性陷阱"**（已写入文档）：E 的 4 解、PnP 符号、单目尺度不定 —— 统一解法都是**物理约束消歧**。
- **环境**：`wildspatial` 已装好（torch 2.6.0+cu124 + CUDA 可用 + open3d 0.19 + opencv-contrib）。**后续统一用它**。
- **下一步（M1）**：SIFT 特征提取 → 描述子匹配（比值检验）→ RANSAC 剔外点 → 八点法/五点法估 E
  → 三角化建图 → PnP 定位 → 手写 BA；在 `fr1/desk` 上跑出轨迹并与真值算 ATE/RPE。

### 2026-09-10（续2）· M1 前端完成，跑出 TUM 基线 ✅
- **产出**：`src/wildspatial/sfm/`（features / matching / ransac / vo）、`data/tum.py`、
  `eval/trajectory.py`、`scripts/m1_run_vo.py`。**自检累计 47 项全绿**。
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
- **下一步（M1 收尾）**：① **局部 BA**（手写残差 + 雅可比，scipy LM）→ 预计 ATE 大幅下降；
  ② `--use-depth` 跑 RGB-D 定尺度版对比 SE3/Sim3；③ 关键帧机制；④ 进入 M3。

---

## 6. 已知坑 / 风险

- 🔴 **多线程下载会静默损坏文件**（2026-09-10 实测）：
  `fr3_nostructure` 用 12 线程下载后**文件大小完全正确（484772347 字节）但 `tar -tzf` 报 CORRUPT**。
  根因：连接提前关闭时写入字节数不足，而文件已 truncate 到完整大小 → 中间留空洞。
  **已修复**：`_download_range` 现在校验每块实际写入字节数，不足则回退进度重试（最多 4 次）；
  并新增 `verify_archive()` 对 tgz/zip 做完整性校验。
  **教训**：网络实验中"文件大小对"≠"数据对"，归档类数据必须校验。
- ⚠️ 待办：`fr3_nostructure` 的 `.tgz` 已损坏，需重新下载（用户未授权删除，暂保留）。
  M3/M5 若要用它，先跑 `verify_archive()` 确认。
- ⚠️ SubT-MRS 数据集体量与下载许可未核实（可能需申请）。备选：对 TUM 施加**合成退化**
  （烟雾/低光/散射）做可控对照实验，性价比更高且可复现。
- ⚠️ VGGT 需 `flash-attn`/`ninja`，在 4090+cu124 编译是否顺利待验证；备选走纯 PyTorch 路径。

- ⚠️ SubT-MRS 数据集体量与下载许可未核实（可能需申请）—— M5 前再评估，备选：用退化合成（见 `docs/03`）对 TUM 施加烟雾/低光/散射，做**可控对照实验**，性价比更高且可复现。
- ⚠️ VGGT 需 `flash-attn`/`ninja`，在 4090+cu124 编译是否顺利待验证；备选走纯 PyTorch 路径（慢但能跑）。

---

## 追加模板
```
### YYYY-MM-DD
- **做了**：...
- **结论/发现**：...（数据说话）
- **下一步**：...
```
