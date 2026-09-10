# WildSpatial

> 从几何地基到端侧空间感知 —— 一个**边教边学、自闭环**的实战项目。
> 主线问题：**移动中的机器人/飞行器，如何在恶劣物理环境下可靠地构建"几何 + 语义"的空间理解，并塞进边缘设备？**

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

而我的差异化机会在于一个几乎无人竞争的位置：
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

---

## 三、快速开始

```bash
# 0) 进入项目
cd /home/hmn-cjy/liuliuqiu/WildSpatial

# 1) 环境（二选一）
conda activate wildspatial     # 新建的（推荐，M4 之后必须用）
conda activate det             # 已有的，M0/M1 够用

# 2) 装依赖
pip install -r env/requirements.txt

# 3) 先看路线图
cat docs/00_ROADMAP.md

# 4) 跑 M0 自检
python -m pytest tests/ -q
python scripts/m0_demo.py
```

> 📌 **进度在哪、下一步做什么**：见 [`PROGRESS.md`](PROGRESS.md)。**这是唯一真源，每次做完都更新。**

---

## 四、目录结构

```
WildSpatial/
├── PROGRESS.md          # 🔴 进度台账，AI 助手每次必读
├── docs/                # 教学文档：概念 + 理论 + 反思
│   ├── 00_ROADMAP.md
│   ├── M0_geometry_foundation.md
│   ├── M1_sfm_from_scratch.md
│   ├── M3_failure_taxonomy.md
│   └── ...
├── src/wildspatial/     # 核心代码库（几何 / 退化 / 模型 / 评测 / 归因）
├── scripts/             # 可执行入口（下载、跑实验、出图）
├── tests/               # 几何自检（几何 bug 极难调，必须自检）
├── experiments/         # 每个实验一个目录：README + metrics.json + figs/
├── notebooks/           # 交互式教学
├── data/                # raw / processed（不进 git）
└── env/                 # requirements.txt / setup.sh
```

---

## 五、技术栈

| 层 | 技术 |
|---|---|
| 几何底座 | NumPy / OpenCV / SciPy（**手搓**，理解原理） |
| 3D 基础模型 | VGGT / MASt3R / DUSt3R |
| 语义层 | Qwen-VL / InternVL（只做语义，不做几何） |
| 评测 | ATE / RPE / 深度误差 / Chamfer / 匹配召回率 |
| 部署 | TensorRT / ONNX / Jetson |

---

## 六、当前状态

🟡 **M0 几何地基** —— 详见 [`PROGRESS.md`](PROGRESS.md)

---

## 七、致未来的我（和 AI 助手）

- 不要跳过"手搓"环节。跑不通才是收获。
- 每个结论都要有**数字**支撑，没有数字的观点不写进报告。
- 优先做**可复现的小实验**，而不是大模型的 fine-tune（算力不占优）。
- 每完成一步，更新 `PROGRESS.md`。
