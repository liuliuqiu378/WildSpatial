"""方法动物园（Method Zoo）—— 统一封装「现成工具 / 代码包 / 模型」

为什么需要它（目的）
------------------
本项目已经从零手搓过 M0 几何库与 M1 前端（SIFT→匹配→RANSAC→三角化→PnP→BA→回环），
建立了"直觉地基"。但**继续手搓不是目的**——我们要的是在**带真值的数据 / 合成退化**上，
**定量对比不同方法的效果**，从而建立"方法 × 环境"的系统性认知（这正是 M5 失效图谱）。

因此这里把"一个空间感知方法"抽象成统一接口，让 OpenCV / COLMAP / LightGlue /
VGGT 等成熟实现都能即插即用、在**同一套退化压力测试**下公平对比。

方法原理（专业）
--------------
- 每个 ``Method`` 接收一组图像帧 ``frames`` 与内参 ``K``，输出 ``MethodResult``：
  相机光心轨迹 ``positions (M,3)`` + 对应的输入帧序号 ``frame_indices (M,)``
  （某些方法会丢弃无法定位的帧，故用 frame_indices 把结果与真值对齐）。
- 评测统一走 ``eval.compute_ate``：先把估计轨迹与真值做 Umeyama（Sim3）对齐，
  再算位置 RMSE（米）。**所有方法在同一对齐协议下比较**，结论才公平。
- ``discover_methods()`` 自动探测当前环境装了哪些依赖，只返回「可用」的方法——
  没装 COLMAP/VGGT 时它们自动隐身，不阻塞主流程。

直白讲解
--------
把各种"建图/定位算法"都看成同一种黑盒：喂图进去，吐出一条相机走过的路径。
不管里面是手写 SIFT 还是深度学习 VGGT，对外我们只关心"它走的路和真值差多少"。
这样就能像做实验一样，换方法、换退化档位，看哪根线掉下去。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class MethodResult:
    """一个方法的输出。

    positions      : (M,3) 相机光心位置（方法自身世界系下）
    frame_indices  : (M,)  每条位置对应的「输入帧序号」（用于和真值对齐）
    ok             : 是否成功产出有效轨迹
    note           : 人类可读说明（写进 metrics / 图例）
    extra          : 任意附加诊断（内点率、地图点数…）
    """
    positions: np.ndarray
    frame_indices: np.ndarray
    ok: bool = True
    note: str = ""
    extra: dict = field(default_factory=dict)

    def aligned_ate(self, gt_positions_all, allow_scale=True):
        """用真值算 ATE（自动按 frame_indices 对齐 + Sim3 对齐）。"""
        from ..eval import compute_ate
        gt = np.asarray(gt_positions_all)[np.asarray(self.frame_indices)]
        return compute_ate(self.positions, gt, allow_scale=allow_scale)


class Method(ABC):
    """所有方法的统一接口。"""
    name: str = "base"
    kind: str = "unknown"          # classic（几何/传统） / learned（学习式匹配） / foundation（前馈 3D 基础模型）
    requires_gpu: bool = False

    @abstractmethod
    def run(self, frames, K, **kw) -> MethodResult:
        """输入帧列表（BGR uint8）与内参 K，输出轨迹。"""
        ...

    def available(self) -> bool:
        """依赖是否齐备。子类可覆盖（如检查 pycolmap / transformers 是否装了）。"""
        return True


_REGISTRY: List[type] = []


def register(cls):
    """类装饰器：把方法类登记进动物园。"""
    _REGISTRY.append(cls)
    return cls


def discover_methods() -> List[Method]:
    """返回当前环境「可用」的方法实例（自动跳过缺依赖的）。"""
    out: List[Method] = []
    for cls in _REGISTRY:
        try:
            inst = cls()
            if inst.available():
                out.append(inst)
        except Exception:
            continue
    return out


def get_method(name: str) -> Method:
    for cls in _REGISTRY:
        try:
            inst = cls()
            if inst.name == name and inst.available():
                return inst
        except Exception:
            continue
    raise KeyError(f"方法未找到或不可用: {name}")
