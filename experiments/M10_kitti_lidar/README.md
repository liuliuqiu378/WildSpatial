# M10 / F5 · KITTI 真实激光雷达 + 相机融合

> 对应 `docs/F5_vision_lidar_fusion.md` §4：视觉+激光融合的真实数据验证。
> 这是项目第一个用**真实 LiDAR 数据**跑通的视觉+激光融合 demo。

## 目的
原 `m10_lidar_fusion_demo.py` 用 TUM RGB-D 深度**反投影**点云演示融合几何，已在 F5 文档中诚实声明：
"本 demo 的激光点云由 TUM 真实深度反投影而来（非真实 LiDAR 采集）"。
本实验的目标就是**把那个诚实声明升级掉**：换用 **KITTI depth completion** 的**真实激光雷达**数据，
让 F5 的"相机↔激光融合"既保留数学自洽（TUM，0px 残差），又落在真实传感器数据上。

## 数据与工具
- 数据集：`KITTI depth completion` 的 `val_selection_cropped`（已解压）。
  - `image/` 真实车载相机 RGB（左目 image_02）
  - `velodyne_raw/` **真实稀疏激光雷达深度**（投影到图像，16-bit PNG）
  - `groundtruth_depth/` **真实稠密深度**（多帧激光累积真值）
  - `intrinsics/` 单行 3×3 内参矩阵 `K = [fx 0 cx; 0 fy cy; 0 0 1]`
- 深度解码：`depth_m = png / 256.0`（KITTI 标准；0 为无效）
- 工具链：纯 OpenCV + numpy + matplotlib，无需训练/模型。

## 方法原理
1. 用内参 `K` 把每个深度像素反投影到相机 3D 系：
   `X = (u-cx) * Z / fx`, `Y = (v-cy) * Z / fy`, `Z = depth`。
2. 对 **真实稀疏激光 `velodyne_raw`**：按 `u,v` 位置把 LiDAR 点投回图像，颜色=距离。
   这直接显示"真实 LiDAR 扫描线长什么样"——稀疏、落在真实路面上。
3. 对 **真实稠密深度 `groundtruth_depth`**：做整张深度图可视化 + 反投影成彩色 3D 点云，
   展示"相机看外观、激光看几何"的融合体。

## 真实结果
| 指标 | 实测值（4 帧平均 / 范围） | 说明 |
|---|---|---|
| 图像分辨率 | **1216 × 352** | KITTI 典型车载分辨率 |
| 真实 LiDAR 有效点 | **~18,500 点/帧** | 来自 `velodyne_raw`，已投影到图像 |
| LiDAR 深度范围 | **3.0 ~ 79.7 m** | 城市/公路驾驶的真实距离尺度 |
| 稠密真值点 | **5.5 ~ 9.9 万点/帧** | 多帧激光累积，作为"理想 3D 几何" |

![KITTI 真实 LiDAR + 相机融合：RGB / 真实稀疏激光 / 真实稠密深度 / 彩色点云](figs/kitti_lidar_fusion.png)

### 你看到什么
- **① Camera RGB**：看得懂场景但没距离。
- **② Real sparse LiDAR → image**：**真实 LiDAR 点**稀疏地落在路面、车辆、护栏上，按距离着色。
  这正是量产车拿到的原始形态——不是稠密图，而是稀疏扫描线。
- **③ Real dense depth**：多帧激光累积出的稠密深度真值，颜色=距离。
- **④ Colored point cloud (X–Z top view)**：给真实 3D 点赋上 RGB，得到"外观+几何"融合体。

## 复现
```bash
conda activate wildspatial
PYTHONPATH=src python scripts/m10_kitti_lidar_demo.py
```
数据路径：
```
data/raw/ms/OmniData__KITTI_depth_completion/raw/KITTI_depth_completion/depth_selection/val_selection_cropped/
```

## 与 TUM demo 的关系
- TUM demo：用 RGB-D 反投影点云，证明**投影公式 + 内参自洽**（重投影残差 0.0000 px）。
- KITTI demo：用**真实 LiDAR** 数据，证明同样的几何在**真实传感器**上成立。
- 两者合在一起，F5 文档的"真实验证"才完整：**数学对，且真实数据也对**。

## 诚实声明
- 稠密深度 `groundtruth_depth` 是 KITTI 官方用多帧激光累积生成的**真值**，不是单帧原始点云；
  这正好说明为什么工程上需要"多帧累积/稠密重建"（单帧 LiDAR 是稀疏的）。
- `velodyne_raw` 是 KITTI 提供的**真实单帧激光投影**，未做合成或渲染。
