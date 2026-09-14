# P2 · 移动操作（mobile manipulation）：A* vs OMPL/OMPL

> 演示 §3.5.5 的核心区分：**移动底盘走格子（A*），机械臂撒点（OMPL）**。

## ① 目的

讲清『两种规划』的本质区别，并演示 mobile manipulation = 底盘 + 机械臂。

## ② ROS2 名词速查

| 名词 | 是什么 | 类比 |
|---|---|---|
| **ROS2** | 机器人中间件（节点用 topic/service/action 通信） | 微信群+快递系统 |
| **DiffDrive** | Gazebo 插件：把 cmd_vel(v,ω) 变左右轮转速 | 油门+方向盘 |
| **Nav2** | 导航栈（建图/定位/规划/避障工具包） | 导航 App |
| **A*** | 全局路径规划（2D 栅格最短路径） | 规划整条路线 |
| **ros2_control** | 统一硬件控制框架（换硬件只换驱动） | 驱动层 |
| **OMPL** | 运动规划库（RRT/PRM 采样） | 关节空间撒点连线 |
| **RRT/PRM** | 采样式规划算法（高维唯一可行） | 闭眼扔豆子连成路 |
| **MoveIt 2** | ROS2 机械臂规划框架（内部用 OMPL） | 机械臂动作大脑 |

## ③ 实测结果

- **A***（2D 栅格）：131 步（遍历格子，最优）
- **OMPL RRTConnect**（真实库）：50 点，0.007s
- **自实现 RRT**（带障碍，可视化）：48 采样点，路径 24 点
- **自实现 PRM**（带障碍）：路径 11 点

![规划对比](figs/planner_compare.png)

> ① 底盘：2D 栅格 A*；② 机械臂：关节空间采样规划；③ 本质区别说明。

![RRT/PRM 撒点](figs/rrt_prm.png)

> RRT 在关节空间**随机撒点连成树**；PRM **撒点建图再搜路**——『撒点连线』的直观。

## ④ 直白讲解

**A*** = 走地图格子（2D 存得下，求最优）；
**OMPL/RRT** = 闭眼扔豆子连成路（6-7D 格子爆炸，只求可行）。
**移动操作** = 先开过去（A*）再伸手拿（OMPL）。

## ⑤ 诚实边界

- OMPL 用**自建 3-DOF 臂**演示（真实机械臂 6-7 DOF，原理相同）；
- Gazebo 里的移动操作模型（`data/gz_models/urdf/mobile_manipulator.sdf`）已可加载，
  完整 MoveIt 2 配置（SRDF/controller）为下一步。

## 如何复现

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ros2jazzy
python scripts/p2_mobile_manipulation.py
```
