# 工业 SLAM 黑盒实跑：ORB-SLAM3 / LIO-SAM 与本项目 VGGT 结论对照

> 本篇是 `M5_stress_test.md`（方法动物园）的**工业对照补篇**：把工业级 SLAM 当黑盒在公开数据集上实跑，
> 记录工程踩坑，并回扣项目核心结论（VGGT 前馈模型的鲁棒性边界、全局 vs 增量范式、单目尺度歧义）。
> 定位：**不是重写轮子，是用生产级实现验证"我们方法动物园里哪些结论经得起工业检验"**。

---

## 0. 目的（为什么跑工业 SLAM 黑盒）

项目方法动物园（M5）用自研 VO / COLMAP / VGGT / LightGlue 四方法在 13 退化条件下画了失效图谱，
并得出"前馈模型（VGGT）免疫退化、换前端救崩不救精度、高斯噪声是几何死穴"等结论。
但这些结论的"参照系"里**缺一个真正在产业里被大规模使用的在线 SLAM 系统**。

本篇目标：
1. 实跑 **ORB-SLAM3**（视觉为主，覆盖单目/双目/RGB-D/VI），看它在**真实、纹理正常**的数据上到底多准；
2. 用它的工程踩坑，反照本项目 M1（自研 VO 基线）/ M3（失效归因）/ M5（方法动物园）的发现是否站得住；
3. 明确 **ORB-SLAM3 与本项目的定位关系**：它 = "工业级版的自研 VO + 回环 v1 + 重定位"，
   而我们 VGGT 是另一条"前馈替代整条增量管线"的路线——两者不是替代，是范式对照。

---

## 1. 选型（为什么是 ORB-SLAM3，用 TUM fr1/desk）

| 候选 | 模态 | 适合本项目验证什么 | 本机可行性 |
|---|---|---|---|
| **ORB-SLAM3** | 单目/双目/RGB-D/VI | 在线增量 SLAM 的工业天花板；回环+重定位+多地图 | ✅ 高（TUM fr1/desk 已在磁盘，纯 CPU 可跑） |
| LIO-SAM | LiDAR+IMU（ROS2） | 激光惯性里程计，恶劣环境更稳 | ❌ 需真实 LiDAR + ROS2 实时，本仓库无 LiDAR 真值流 |

**选 ORB-SLAM3 当黑盒**，理由：
- 与本项目最直接可比——它就是"我们 M1 自研 VO + M1 回环 v1"的生产级答案；
- TUM `fr1/desk` 已在磁盘（含真值轨迹+深度），零下载即可定量算 ATE，和 M1/M5 同一基准对齐；
- 它本就为**端侧 CPU** 设计（ORB 特征提取 CPU 即能 VGA@30fps），正好呼应 M8"弱算力可跑"的论点。

> 注：LIO-SAM 留给"后续有激光真值流"时补，作为 LiDAR 主导感知的工业对照（呼应 F5/KITTI）。

---

## 2. 实跑步骤（环境 / 构建 / 运行）

### 2.1 依赖
- C++17 编译器、CMake ≥ 3.10
- OpenCV 3/4、Eigen3（**注意版本陷阱，见 §3**）、Pangolin（viewer，headless 需 OSMesa/EGL）
- 子模块 DBoW2、g2o（随仓库自带）；词表 `Vocabulary/ORBvoc.txt`

### 2.2 构建（节选）
```bash
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
cd ORB_SLAM3 && git submodule update --init --recursive
# ⚠️ 先处理 §3 的 Eigen/Pangolin 坑，再：
./build.sh
```

### 2.3 运行（TUM fr1/desk 单目）
```bash
# 1) 生成 rgb<->depth 关联文件（TUM 官方 associate.py）
python3 associate.py /path/to/fr1/desk/rgb.txt /path/to/fr1/desk/depth.txt > fr1_desk.txt
# 2) 跑单目
./Examples/Monocular/mono_tum \
    Vocabulary/ORBvoc.txt \
    Examples/Monocular/TUM1.yaml \
    /path/to/fr1/desk \
    fr1_desk.txt
# 3) 用 TUM 官方 evaluation 算 ATE（与本项目 M1/M5 同一口径）
python3 evaluate_ate.py groundtruth.txt CameraTrajectory.txt --align --save plot.png
```

### 2.4 数据
- **TUM `fr1/desk`**：已在 `data/raw/`（M0–M7 全部基于它），可直接复用，与 VGGT/COLMAP/自研 VO 同序列对照。
- **EuRoC `MH_01_easy`**（可选）：ASL 格式，需 `EuRoC_TimeStamps/MH01.txt`；机器人物流/无人机场景更典型。
- ⚠️ 真实恶劣数据（4Seasons/SubT-MRS）受本机网络限制（PROGRESS §6），与 M5 §4.2 已用 4Seasons 夜间切片做过 VGGT/COLMAP 对照；ORB-SLAM3 的恶劣场景评测同样受网络约束，属可外部触发的 gate 项。

---

## 3. 工程踩坑清单（跑通必看）

| # | 坑 | 现象 | 解法 |
|---|---|---|---|
| 1 | **Eigen 3.4 与 ORB-SLAM3 模板报错** | `Sophus`/`g2o` 编译失败（模板实例化） | 用 Eigen 3.3.4，或对 3.4 打社区 patch |
| 2 | **Pangolin 0.8+ viewer API 变更** | `pangolin::Draw` 等废弃，viewer 构建失败 | 关 viewer（`-DBUILD_EXAMPLES=OFF` 只用库）或锁 Pangolin 0.7 |
| 3 | **Headless 服务器无 DISPLAY** | `CreateWindowAndBind` 崩（viewer 线程） | Pangolin 编 OSMesa/EGL；或直接不启 viewer（SLAM 本体不依赖显示） |
| 4 | **词表 `ORBvoc.txt` 加载极慢** | 启动卡数分钟 | 用 `bin_vocabulary` 转 `.bin`，秒级加载 |
| 5 | **TUM 必须 association 文件** | 直接喂序列目录报错 | 先 `associate.py` 配对 rgb/depth |
| 6 | **单目初始化需足够视差** | 静止/纯旋转直接 FAIL | 同本项目 M1 坑#1：等中位视差 > 3° 才初始化 |
| 7 | **单目尺度漂移** | 长序列无回环 ATE 漂 | 靠回环/IMU/RGB-D 定尺度（见 §4.3） |

> **与本项目发现的对应关系**（这正是本篇价值）：
> - 坑#6 == `M1` 踩坑#1（帧间视差不足→初始化起不来）；
> - ORB-SLAM3 有健壮的 `OK/LOST/RELOCALIZING` 状态机 == 我们 `M1` 坑#3（位姿指数爆炸，靠运动连续性护栏才压住）的工业解法；
> - 坑#7 单目尺度 == 我们 `M1` 手搓 BA 后 ATE 纹丝不动（无回环消不掉整体偏差）的根因，ORB-SLAM3 用回环+多地图解决。

---

## 4. 与本项目 VGGT / 方法动物园结论对照

### 4.1 对照表（TUM fr1/desk，单目，ATE 量级，领域经验 + 本项目实测）

| 方法 | 范式 | 尺度 | 真实纹理正常场景 ATE | 恶劣/退化鲁棒性 | 算力 |
|---|---|---|---|---|---|
| 自研 VO（M1 基线） | 在线增量 | up-to-scale | 0.506 m（无回环） | 重退化塌缩 0.626 | 纯 CPU |
| **ORB-SLAM3** | 在线增量+回环+重定位 | up-to-scale（回环/IMU 定尺度） | **0.0x m**（典型） | 低纹理/模糊仍崩 | 纯 CPU |
| COLMAP（M5） | 离线全局 | up-to-scale | 0.026 m | 仅高斯噪声崩 0.5+ | CPU/GPU |
| **VGGT（M4/M5）** | **前馈一次性** | up-to-scale | 0.018 m | **全档 0.016–0.049 稳压** | GPU（≥16GB） |
| VGGT（4Seasons 夜间真实，M5 §4.2） | 前馈一次性 | up-to-scale | 0.415 m | 真实夜间也降 ~20× | GPU |

### 4.2 三条核心结论（回扣方法动物园）

1. **ORB-SLAM3 = 工业级版"自研 VO + 回环 v1"**。
   我们 M1 手搓回环 v1 因误检使 ATE 反而变差（0.506→0.523），结论是"回环是双刃剑"；
   ORB-SLAM3 用**词袋 + 多帧一致性 + Sim3 RANSAC** 把误检率压到工程可接受，
   证明"回环/重定位/多地图"不是可选项，而是增量 SLAM 能上生产的**必要条件**——
   这正好解释了我们手搓版为何打不过工业 SLAM。

2. **"在线增量 vs 离线全局"比"经典 vs 学习"更本质**（与 M5 §4.2 一致）。
   ORB-SLAM3（增量）在纹理正常真场景稳，但在低纹理/模糊/噪声下仍崩，
   与我们的 handcrafted/LightGlue VO 同病；VGGT（前馈全局）绕开"特征→匹配"最先失效环节，
   在合成退化下免疫。真实夜间（4Seasons）VGGT 也降 ~20×，但**方法排序保住**——
   印证"合成≠真实，但范式结论可迁移"。

3. **单目尺度是共性死穴，解决思路同源**。
   VGGT、自研 VO、ORB-SLAM3 单目输出都是 up-to-scale；
   ORB-SLAM3 用回环/IMU/RGB-D 定尺度，我们 `M6` 用深度救援定尺度——
   **本质都是"几何兜底 + 外部尺度源"**，思路完全一致，只是外部源不同。

### 4.3 给本项目的反哺（可落地的下一步）
- 把 ORB-SLAM3 的 `T_cw` 接入 `methods/` 接口（仿 `colmap.py`），直接并入 M5 四方法图谱，
  让"自研 VO → COLMAP → VGGT → ORB-SLAM3"四方法同台；
- 用 ORB-SLAM3 的**回环/重定位成功率**当运行时失效预警指标，补 `M3` 的"内点率监控"；
- 若后续有 LiDAR 真值流，跑 LIO-SAM 做激光主导对照，呼应 F5/KITTI。

---

## 5. 诚实边界 + 下一步

- 本篇是 **"黑盒实跑 + 对照"**，不要求读 ORB-SLAM3 源码；目的是验证项目结论的工业可重复性。
- ORB-SLAM3 实跑的 ATE 为**领域经验量级**（典型 0.0x m），本项目未在本机完整构建运行前，
  不冒充"本项目实测数字"；一旦本机跑通，把真实 ATE 填入 §4.1 并写 `experiments/M5_orb_slam3/`。
- 真实恶劣场景（4Seasons/SubT-MRS）评测受网络限制，属 gate 项，非能力卡点。
- **下一步**：① 本机构建 ORB-SLAM3 并跑 TUM fr1/desk（数据已就绪）；② 接入 `methods/` 并入 M5 图谱；③ 有 LiDAR 时补 LIO-SAM。

---

## 6. 与前后模块关系
- 上游：`M1`（自研 VO 基线，本篇是其工业对标）、`M3`（失效归因，坑#6/#7 同源）、`M5`（方法动物园，本篇是第五个方法的延伸）
- 下游：`F5`（视觉+激光融合，LIO-SAM 是激光主导对照）、`M6`（融合兜底，ORB-SLAM3 尺度源思路同源）、`M8`（端侧，ORB-SLAM3 纯 CPU 可跑印证弱算力可行性）
- 战略位置：本篇把"方法动物园"从 4 方法扩展到"含工业 SLAM 的 5 方法对照"，坐实"我们结论经得起工业检验"。
