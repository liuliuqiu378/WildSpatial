# P2 · 机器人第一视角送货演示 —— Gazebo 真实相机版

> 本实验是 `p2_delivery_fpv.py`（纯合成 3D 渲染）向 **Gazebo 真实 RGB 相机** 的升级验证。
> 在把当前用户加入 `render` 组、解决 `/dev/dri/renderD128` 权限后，Gazebo `--headless-rendering`
> 可输出真实 3D 画面。

## 任务

TB3 从广场南侧 `(0, -3.8)` 出发，沿固定航点前往北侧送货点 `(0, 3.8)`；
途中依次处理：

| 场景 | 触发条件 | 机器人反应 |
|---|---|---|
| 巡航 | 开阔路段 | 以 0.28 m/s 沿航点前进 |
| ① 静态障碍·绕行 | 接近广场中央柱子（<1.0 m） | 减速 + 转向绕行 |
| ② 动态行人·停车让行 | 行人进入 1.3 m 安全区 | 停车等待 |
| ⑥ 送达点·到达 | 距离目标 <1.0 m | 减速停靠 |

## 运行方式

**必须在 `render` 组内运行 Gazebo**，否则 EGL headless 渲染无权限，相机会灰屏。

如果已执行过 `sudo usermod -aG render $USER` 但**尚未重新登录**：

```bash
sg render -c 'python scripts/p2_delivery_fpv_gazebo.py'
```

如果已经重新登录（或在新 shell 里 `id` 显示 `render` 组）：

```bash
python scripts/p2_delivery_fpv_gazebo.py
```

## 实现要点

1. **真实 RGB 相机**：订阅 Gazebo `/camera/color/image_raw`，画面来自 TB3 搭载的 RealSense 相机模型。
2. **`/clock` 仿真时间积分**：本机 headless rendering 负载高，Gazebo 仿真时间比墙上时间慢，
   因此机器人/行人都按 `/clock` 推进积分，保证 HUD/小地图与画面同步。
3. **画家算法替代**：这里没有自研渲染器，**所有 3D 画面由 Gazebo/Ogre2 真实计算**。
4. **行人位置解析估计**：按世界 xacro 中 `VelocityControl` 的参数（速度 0.35 m/s、边界 ±3.0 m）
   在仿真时间下计算，避免依赖 `/world/default/pose/info` 的更新稳定性。

## 产出

```
experiments/P2_delivery_fpv_gazebo/
    figs/delivery_fpv_gazebo.gif    # FPV 动图（带 HUD）
    figs/delivery_fpv_gazebo.mp4    # 同内容 MP4
    figs/keyframes.png              # 关键帧
    metrics.json                    # 逐帧元数据
```

## 诚实边界

- 机器人控制使用**固定航点表 + 比例控制**，尚未接入 Nav2/SLAM。
- 行人位置由脚本按已知运动模型估计，真实系统应来自 M7 检测跟踪。
- 本 demo 展示的是 **Gazebo 仿真相机 FPV**，不是真实物理机器人相机。
- 纯合成版 `p2_delivery_fpv.py` 仍保留，支持更复杂的 6 类路况与精确构图。
