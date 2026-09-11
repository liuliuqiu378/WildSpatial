# WildSpatial

> 从几何地基到端侧空间感知 —— 一个**边教边学、自闭环**的实战项目。
> 主线问题：**移动中的机器人/飞行器，如何在恶劣物理环境下可靠地构建"几何 + 语义"的空间理解，并塞进边缘设备？**

📦 GitHub：[github.com/liuliuqiu378/WildSpatial](https://github.com/liuliuqiu378/WildSpatial)

---

## 一、这个项目为什么存在

我判断空间感知是值得押注的方向，但切入方式必须避开两个坑：

- **坑 1**：以为"学 VLM 就能做空间智能"。实测数据打脸 —— VLM 做相机相对位姿估计 F1 仅 0.64，
  而二十年前的 SIFT 特征匹配是 0.97，LoFTR 0.97。**VLM 打不过 SIFT。**
  VLM 的真正价值在语义层（"这是门/电梯口"、"理解指令"、"任务分解"）。
- **坑 2**：以为必须啃完 SLAM/控制理论才能入场。工业趋势恰恰是**分层解耦** ——
  TAMP-Nav 的做法是：VLM 只在图上"指一个像素点"表示我要往哪走，剩下的几何交给深度图+内参转 3D 坐标，再由 SLAM 执行。
  **VLM 管"去哪"，SLAM 管"怎么去"。**

所以本项目的立场是：**几何方法打底 + VLM 做语义 + 世界模型做预测 + 端侧工程兜底**。

而差异化机会在一个几乎无人竞争的位置：
> 现有 3D 基础模型（VGGT 等）在干净场景很强，但在**水下散射 / 矿山粉尘 / 地下无纹理 / 烟雾 / 低光照**下会失效。
> 学术界不屑做，工业界急需。

---

## 二、怎么学（教学哲学）

每个 Module 严格按 **5 步闭环**推进，缺一不可：

```
概念(是什么) → 理论(为什么，含公式推导) → 实现(手搓一遍) → 实验(真实数据+量化) → 反思(它会在哪崩)
```

**关键原则：先手搓，再用现成的。**
不了解传统 SfM 管线（特征→匹配→位姿→三角化→BA），就读不懂 VGGT 到底替代了什么，更读不懂它为什么会崩。
所以 M1 会让你手写一个会"跑崩"的 VO —— **崩掉的那刻，才是真正理解的开始**。

**教程写作约定（本项目强制 · 四段式）**：所有教学文档，每个知识点都必须讲清楚四件事：

| 段 | 回答什么 | 要求 |
|---|---|---|
| ① **目的** | 它要解决什么问题？为什么需要它？ | 先给动机，否则读者不知道在学什么 |
| ② **方法原理（专业）** | 术语 + 公式 + 推导 | 严谨、不回避，这是地基 |
| ③ **直白讲解** | 用大白话/类比"它到底在说什么" | 建立直觉，不能只有术语 |
| ④ **真实数据验证 + 效果图** | 在**带真值标签的公开数据集**上实跑 | 给量化指标 + 真实结果图 |

验证首选 **TUM RGB-D `fr1/desk`**（含真值轨迹 + 深度图，可定量算 ATE/RPE）。
⛔ 禁止只堆术语、⛔ 禁止无图纯文字长推导、⛔ 禁止只讲原理不跑数据。

---

## 三、怎么读这个仓库（文档导航）

不要按文件名字母序读，按"先建立直觉、再扣细节"的顺序：

| 顺序 | 文档 | 作用 |
|---|---|---|
| 1 | [`docs/00_QUICKSTART.md`](docs/00_QUICKSTART.md) | **从这儿开始**：大白话讲项目在干啥、跑什么命令、看什么图 |
| 2 | [`docs/00_ROADMAP.md`](docs/00_ROADMAP.md) | 全局路线图：M0~M9 每个阶段的主题与产出 |
| 3 | [`docs/M0_geometry_foundation.md`](docs/M0_geometry_foundation.md) | M0 几何地基（含 6 张教学图） |
| 4 | [`docs/M1_sfm_from_scratch.md`](docs/M1_sfm_from_scratch.md) | M1 手搓 SfM/单目 VO（含真实数据效果图 5 张 + 结果图） |
| 5 | [`docs/M3_failure_attribution.md`](docs/M3_failure_attribution.md) | M3 失效归因：**什么条件会让系统崩、崩在哪一步**（13 次带真值实跑） |
| — | [`PROGRESS.md`](PROGRESS.md) | 🔴 **进度唯一真源**：AI 助手每次必读，做完即更新 |

> 📌 文档里的结果图用相对路径引用 `experiments/<module>/figs/`，已随代码提交，**在 GitHub 上能直接显示**。

---

## 四、快速开始

```bash
# 0) 进入项目
cd /home/hmn-cjy/liuliuqiu/WildSpatial

# 1) 环境（conda）
conda activate wildspatial          # 推荐：torch2.6+cu124 / opencv-contrib / open3d

# 2) 装依赖
pip install -r env/requirements.txt

# 3) 跑 M0 几何自检 + 生成 6 张教学图（约 3 秒）
PYTHONPATH=src python -m pytest tests/ -q
PYTHONPATH=src python scripts/m0_demo.py

# 4) 跑 M1 单目 VO 在 TUM fr1/desk 上，出轨迹/诊断图（约 20 秒）
PYTHONPATH=src python scripts/m1_run_vo.py --seq fr1/desk --frames 450 --stride 3
```

跑完你会看到：
- `experiments/M0_geometry_foundation/figs/` 下 6 张图（投影、对极线、三角化、视差、纯旋转退化、PnP 精化）
- `experiments/M1_vo_fr1_desk/figs/` 下 3 张图（轨迹对比、每帧诊断、ATE 漂移）+ `metrics.json`

---

## 五、目录结构

```
WildSpatial/
├── PROGRESS.md          # 🔴 进度台账，AI 助手每次必读
├── README.md            # 本文件
├── docs/                # 教学文档（四段式：目的+原理+讲解+真实数据验证·图）
│   ├── 00_QUICKSTART.md
│   ├── 00_ROADMAP.md
│   ├── M0_geometry_foundation.md
│   ├── M1_sfm_from_scratch.md
│   └── M3_failure_attribution.md
├── src/wildspatial/     # 核心代码库
│   ├── geometry/        # M0：lie / camera / epipolar / triangulation / pnp（手搓）
│   ├── sfm/             # M1：features / matching / ransac / vo / ba
│   ├── eval/            # 轨迹对齐 + ATE / RPE
│   └── viz/             # 画图工具（中文字体等）
├── scripts/             # 可执行入口（m0_demo / m1_run_vo）
├── tests/               # 数值自检（几何 bug 极难调，必须自检）
├── experiments/         # 每个实验：README + metrics.json + figs/（图随代码进 git）
├── data/                # raw / processed（不进 git，体积大）
└── env/                 # requirements.txt
```

---

## 六、技术栈

| 层 | 技术 |
|---|---|
| 几何底座 | NumPy / OpenCV / SciPy（**手搓**，理解原理） |
| 3D 基础模型 | VGGT / MASt3R / DUSt3R（M4 接入对比） |
| 语义层 | Qwen-VL / InternVL（只做语义，不做几何） |
| 评测 | ATE / RPE / 深度误差 / 匹配召回率 |
| 部署 | TensorRT / ONNX / Jetson（M8） |

---

## 七、当前状态

| Module | 状态 | 关键数字 |
|---|---|---|
| M0 几何地基 | ✅ 完成 | 25 项自检全绿 + 6 张教学图 |
| M1 手搓 SfM/VO | 🟡 前端完成；BA 模块已写好待接入验证 | TUM 基线 ATE 0.506 m |
| M2 深度与重建 | ⬜ 未开始 | — |
| M3 失效归因 | ⬜ 未开始 | — |

> 完整路线图与每步踩坑记录在 [`PROGRESS.md`](PROGRESS.md)。

---

## 八、致未来的我（和 AI 助手）

- 不要跳过"手搓"环节。跑不通才是收获。
- 每个结论都要有**数字**支撑，没有数字的观点不写进报告。
- 优先做**可复现的小实验**，而不是大模型的 fine-tune（算力不占优）。
- 教学文档坚持**四段式**：目的 → 方法原理（专业）→ 直白讲解 → 真实标签数据验证 + 效果图。
- 结论必须用**带真值标签的数据集**实跑验证；没有标签时，用"合成退化"在带真值数据上做可控对照并明确标注。
- 每完成一步，更新 `PROGRESS.md`。
