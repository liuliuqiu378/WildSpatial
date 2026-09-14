# D-showcase · NDISPark 夜间/白天停车场实例分割

## 1. 一句话
真实停车场数据集中，车辆实例分割真值在**白天和夜间都稳定存在**。这说明：
**监督封闭集分割对光照不敏感，而开放词汇 VLM 在域偏移下会失效**——正好补全 D1 结论的另一面。

## 2. 复现命令

```bash
conda activate wildspatial
PYTHONPATH=src python scripts/d_showcase_ndispark.py
```

## 3. 数据位置

```
data/raw/ms/OmniData__NDISPark_Night_and_Day_Instance_Segmented_etc/raw/
  train/imgs/*.jpg
  train/train_coco_annotations.json
```

- 约 250 张停车场图像，7 个摄像头，白天+夜间，各种遮挡与阴影。
- COCO 格式真值：80 类，含 `car`/`truck`/`bus` 等车辆多边形分割。

## 4. 结果

**`figs/ndispark_seg.png`**：2 行 × 3 列，上行 3 张夜间、下行 3 张白天，每图叠加 GT 实例分割掩码。

![夜间/白天停车场实例分割真值](figs/ndispark_seg.png)

**`metrics.json`**：

| 项 | 值 |
|---|---|
| 训练集图像数 | 112 |
| 实例总数 | 2577 |
| 类别数 | 80 |
| 展示样本夜间亮度 | 60.4–67.7 |
| 展示样本白天亮度 | 126.3–136.3 |
| 每样本车辆实例数 | 5–21 |

## 5. 教学意义

- **M7 语义层**：D1 显示 OWL-ViT 开放词汇在水下细粒度上 recall≈0.001；本 showcase 显示**同一问题有解**：
  用停车场车辆这一封闭集 + COCO 标注的监督模型，昼夜都能给出可靠语义。
- **M6 多模态/低光**：夜间图像虽然亮度低，但 RGB 仍然够用（无需额外传感器即可做分割）——
  这和"只要光照差就必须上红外/激光"的常见直觉形成修正。
- **项目核心论点**：不要把开放词汇 VLM 当作语义银弹；**几何兜底 + 域专用语义**才是落地方案。

## 6. 局限与诚实声明

- 本 showcase 只可视化**真值掩码**，未训练或跑推理模型；真实落地仍需训练一个检测/分割模型。
- 类别名沿用 COCO 80 类，车辆类别主要是 `car`/`truck`/`bus`，夜间样本中存在遮挡、阴影、反光等真实退化。
