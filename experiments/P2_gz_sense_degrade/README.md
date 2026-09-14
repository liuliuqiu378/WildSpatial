# P2 · 感知实验台（一）：仿真 RGB-D 采集 + 可控退化

> 回到**视觉/传感器主线**：在 Gazebo 里边走边采 RGB-D，再施加可控退化。

## ① 目的

为感知方法（M4 深度 / M5 失效图谱 / M6 融合）提供**带真值、可控、无限量**的数据。

## ② 关键工程问题

1. **TB3 原模型只有深度相机**（`intel_realsense_r200_depth`）→ 本脚本**注入了一个 RGB 相机**（640×480，原 SDF 副本在 `data/gz_models/urdf/`），用 `robot_sdf:=` 参数覆盖。

2. 用 `gz.transport13` 直接订阅 gz 话题（比配 ROS bridge 简单）。

3. 通过 `cmd_vel` 驱动 TB3 缓慢移动，**边动边采**（视角变化）。

## ③ 实测结果

- 采集帧数：**4**
- RGB 分辨率：**[480, 640, 3]**、深度：**[240, 320]**
- 深度范围：**0.53 – 4.78 m**

![多帧 RGB + 真值深度](figs/gz_rgbd_frames.png)

> 上行 = 机器人视角 RGB；下行 = 对应**真值深度**（伪彩）。
> 注意机器人**边移动边采**，视角在变化——这正是一段『仿真录制的数据集』。

![可控退化图库](figs/degradation_gallery.png)

> 同一帧 × 6 种退化：低光、雾天、高斯噪声——**这正是 M5『方法 × 环境』失效图谱的原料**。

![退化统计](figs/degradation_stats.png)

> 量化退化对『感知难度』的影响：亮度↓（低光）、对比度↓（雾/噪声）。

## ④ 直白讲解

**真实数据 + 合成退化**：退化是事后『P 图』，光照/材质是假的；
**仿真**：物理引擎在算，光照/材质/几何都是真的——退化更接近真实物理。
所以 Gazebo 是『**比合成退化更真、比真实数据更可控**』的中间地带。

## ⑤ 诚实边界

- RGB 相机为**注入**（真实 TB3 waffle 常配 RealSense RGB-D，属合理补全，已标注）。

- 退化为**仿真 RGB 上合成施加**（非仿真引擎内的雾）；下一步可直接在 SDF 里加雾/改光照。

## 如何复现

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ros2jazzy
PYTHONPATH=src python scripts/p2_gz_sense_degrade.py
```
