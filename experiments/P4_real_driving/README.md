# P4 · 真实驾驶动态建图（真实 RGB + 真实 LiDAR · 双视角回放）

> **定位**：把「真实多传感器数据当实时输入」完整跑一遍
> 【真实 RGB 视觉定位（VGGT）→ 真实 LiDAR 尺度定标 → 逐帧 BEV 增量建图 → 第一视角 + 俯视全景双视角回放】。
> 数据：**KITTI depth_completion 真实街道驾驶**（已在磁盘，零下载），**非仿真、非合成退化**。

## ① 目的（场景与需求）

- **场景**：真实城市街道驾驶（两侧停车、行道树、建筑），车流真实、噪声真实。
- **真实问题**：只有 RGB 看不懂距离、只有 LiDAR 稀疏且无语义；要把「看得懂」（相机）和「量得准」（激光）
  拼成一张**随车增长的俯视全景地图**，并同步回放"车怎么开完这段路"。
- **需求**：① 视觉定位出轨迹 ② 轨迹要有度量尺度 ③ 逐帧真实 LiDAR 反投影累积成 BEV ④ 第一视角 + 俯视全景同屏回放全程。

## ② 方法原理（专业）

- **视觉定位**：项目方法动物园 **VGGT** 一次前向处理 30 帧真实 RGB → 相机轨迹 `T_cw`（up-to-scale）。
- **尺度定标**：VGGT 输出深度与真实 LiDAR 深度（`velodyne_raw`，depth = png/256 m）逐帧取 median 比
  → 全局尺度 **72.433** → VGGT 平移乘回度量级。
- **BEV 建图**：每帧真实 LiDAR 深度按内参反投影成相机系 3D 点 → 乘已定标 `T_cw` 变换到世界系
  → 逐帧增量写入 0.4 m 栅格 BEV（按高度着色：蓝=地面，暖=车/树/建筑）。
- **双视角回放**：左=第一视角 RGB 叠真实 LiDAR 扫描（近暖远蓝）；右=BEV 已建地图 + 绿色已走轨迹 + 红色车体（沿 heading，forward 朝上）。

## ③ 直白讲解

车往前开：眼睛（相机）负责"认路"——VGGT 看连续画面就知道车怎么拐；
激光雷达负责"量距离"——每帧打出一束稀疏但真实的点。
把每帧量到的点按"眼睛报出的位置"拼起来，就长出一张俯视地图。
视频里左边是司机看到的画面（叠着激光打到的点），右边是这张地图随车一点点长出来、车标沿轨迹往前走。

## ④ 真实数据验证 + 效果

- 数据：KITTI depth_completion · val_selection_cropped · drive `2011_09_26_drive_0023_sync`，30 帧连续（帧距 0.6 s ≈ 18 s 行车）。
- 尺度定标因子 **72.433**（真实 LiDAR / VGGT 相对深度 median 比）。
- 轨迹长度 **125.16 m**（≈7 m/s，城市车速合理）；每帧真实 LiDAR **1.7–2.0 万点**，累计 **56.5 万点**。
- BEV 覆盖约 **150 m × 190 m**（0.4 m 栅格），街道走廊、两侧停车、行道树清晰可辨。

![双视角回放](figs/driving_dualview.gif)

![BEV 终图](figs/bev_final_map.png)

- 视频：`driving_dualview.mp4`（1946×400 @6 fps，5 s）；关键帧 `figs/frame_*.png`（6 张）；终图 `figs/bev_final_map.png`。

## 诚实边界

- 轨迹来自 **VGGT 视觉定位**，非 GPS/IMU 真值；该基准**无逐帧真值位姿 → 不报 ATE**（本 demo 定位是"真实数据上的系统构建"，不是"带真值评测"）。
- 尺度由**真实 LiDAR 深度**锚定（median 比），非 IMU/轮速。
- val_selection 为抽帧序列（帧距 0.6 s），非满帧率 10 Hz；LiDAR 为 `velodyne_raw` 真实稀疏扫描（已投影到 image_02 平面）。
- 「雷达」= 激光雷达 LiDAR（该数据集无毫米波 radar）。

## 复现

```bash
PYTHONPATH=src python scripts/p4_real_driving.py --drive 2011_09_26_drive_0023_sync --max-frames 30
```

（需 GPU + 本地 VGGT-1B 权重 `~/models/vggt-1b`；数据已在 `data/raw/ms/OmniData__KITTI_depth_completion/`）
