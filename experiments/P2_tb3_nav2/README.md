# P2 · TurtleBot3 + Nav2 无头闭环 demo

> 在**无图形界面**的服务器上，用 Gazebo 无头渲染跑通 TB3 机器人仿真，
> 并抓取真实 LiDAR 演示「静态地图 + 动态障碍 → costmap」的形成。

## ① 目的

把 P2 §4.4 的闭环（Gazebo→传感器→SLAM→Nav2→差速控制）**真跑一遍**，
并用无头环境下的可视化证明「成本地图（costmap）是三层叠加」。

## ② 关键工程问题：无 DISPLAY 怎么跑仿真？

本机**没有图形界面**（`$DISPLAY` 为空）。工业界在服务器集群上跑仿真的标准做法是：

```bash
gz sim --headless-rendering -s -r <world>   # 无头渲染（不弹窗口）
export GZ_SIM_RESOURCE_PATH=<models>:<share>  # 让 Gazebo 找到 model://
```
本文正是这么做的。

## ③ 实测过程

- Gazebo 无头启动：**✅ 成功**
- TB3 生成（/scan 等话题）：**✅ 成功**
- 真实 LiDAR 抓取：**✅**（360 束有效，最近障碍 0.49 m，量程 9.999999747378752e-06–20.0 m）
- 完整 Nav2 BT 导航栈：skipped（完整 Nav2 BT 栈需更多运行时配置，见 README）

## ④ 可视化：costmap 是三层叠加

![costmap 三层叠加](figs/costmap_concept.png)

> **左**：机器人视角的真实 LiDAR 点（红色=障碍）；
> **中**：障碍 + 膨胀带（对应 Nav2 `inflation_layer`）；
> **右**：`static + obstacle + inflation` 三层叠加成**同一张 costmap**。

### 直白讲解

把 costmap 想成一张『风险地图』：

1. **静态层**：离线地图先画好『哪里有墙』（先验）；

2. **动态层**：每秒用激光扫到的『新东西』（行人/箱子）实时写上去；

3. **膨胀层**：在障碍周围画一圈『危险区』，让机器人**保持安全距离**而不是贴边过。

三者**叠加成一张图**，交给规划器（A\*/DWA）——**这就是『静态导入 + 动态感知』的融合点**。

## ⑤ 诚实边界

- 无 GUI → 用**无头渲染 + 话题抓取**出图，不是 Gazebo 窗口截图。

- 完整 Nav2 导航（BT navigator 全链路）：skipped（完整 Nav2 BT 栈需更多运行时配置，见 README）

## 如何复现

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ros2jazzy
PYTHONPATH=src python scripts/p2_tb3_nav2_demo.py
```
