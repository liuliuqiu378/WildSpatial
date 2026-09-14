# 实战项目：退化环境搜救（合成退化模拟）

> 知识落点（M0–M9 → 实战）：TUM fr3_nostructure 实测缺 rgb.txt/真值，无法定量评估；故用『带真值的 fr1/desk + M3 合成退化』可控地模拟搜救式退化环境。项目量化 M2 VO 在退化下的可行性边界（能否初始化、ATE 多少），并演示 M6 失效预警的必要性：一旦 VO 过『特征地板/分辨率悬崖』直接死，就必须靠多模态融合兜底。退化为合成施加，已在数据/图注中标注。

**数据集**：fr1/desk + M3 合成退化（低光+运动模糊，模拟搜救/地下退化；已明确标注）  
**序列规模**：199 帧 | 场景标签 `degraded` | 深度=有 | 真值=有

## 工序链（因果自洽）

| 工序 | 模块 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|
| vo | M2 | OK | ATE(自由尺度)=0.8674m, 31.8fps | 单目 VO 产出 up-to-scale 轨迹（199 帧有效），自由尺度 ATE=0.867m |
| depth | M4 | 跳过 | 尺度=None, ATE(公制)=Nonem | 单目 VO 在退化/弱纹理场景位移过小，无法用度量源定尺度（对应 M6：需 IMU/深度/多模态融合兜底） |
| semantic | M7 | 跳过 | 检测数=None（几何） | 降级：场景图使用几何标签 object |
| scene_graph | M9 | 跳过 | 物体实例=None（原始候选 None） | 单目 VO 在退化/弱纹理场景塌缩，场景图无法构建（对应 M6：需 IMU/深度/多模态融合兜底） |

![项目总览](figs/overview.png)

## 如何复现

```bash
PYTHONPATH=src python scripts/run_project.py --project search_and_rescue
```

> 本框架与数据来源解耦：任何新公开数据集只要能归一成 `Sequence(rgbs, K, depths, gt_positions)`，即可接入全部 M0–M9 工序。
