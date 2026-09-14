# P2 · 感知实验台（四）：多传感器同步（相机+LiDAR+IMU）

> M6 多模态融合 / F5 视觉+LiDAR 融合的**输入实验台**。

## ① 目的

M6/F5 都需要**多传感器同步数据**。真实数据集时间戳需对齐、真值需标定；
仿真**同源同帧**直接输出，时间戳天然同步。

## ② 实测结果

| 传感器 | 帧数 | 频率 |
|---|---|---|
| RGB 相机 | 139 | 10.01 Hz |
| 深度相机 | 70 | 5.04 Hz |
| LiDAR | 128 | 9.71 Hz |
| IMU | 128 | 9.71 Hz |

![多传感器融合](figs/sensor_fusion.png)

> ① RGB（外观/语义）；② LiDAR（精确距离）；③ 各传感器数据率。

## ③ 直白讲解

**相机**告诉你『前面是什么』（语义强、距离弱）；
**LiDAR** 告诉你『前面多远』（距离准、无颜色）。
两者**同时拿到**，就是 F5 视觉+激光融合的起点——**仿真是最省事的实验台**。

## ④ 诚实边界

- 三路传感器由 Gazebo 同一物理步输出，时间戳天然同步（真实需硬件 PPS/软件对齐）。

- 相机与 LiDAR 有外参（安装位置差），本脚本未做点云→图像投影（见 F5 与 `m10_kitti_lidar_demo.py`）。

## 如何复现

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ros2jazzy
PYTHONPATH=src python scripts/p2_gz_multisensor.py
```
