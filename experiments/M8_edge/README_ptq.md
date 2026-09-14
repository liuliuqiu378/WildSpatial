# M8 · INT8 PTQ 真跑实验

- 任务：2-class synthetic (low-freq vs high-freq), TinyCNN
- FP32  acc=100.00%  latency=1.23 ms/batch
- INT8  acc=99.75%  latency=1.34 ms/batch
- **精度损失=0.25pp，推理加速=0.92x**（引擎 fbgemm，静态 PTQ）

> 说明：本实验是**分类**任务（INT8 最友好的情形），损失很小——印证「INT8 PTQ 是端侧首选」。位姿/深度等**几何敏感**任务损失会更高，故量化校准集必须覆盖退化场景（呼应 M8 §2.2/§7）。
