"""实战项目整合层 · 核心抽象

为什么需要它（目的）
------------------
F0–F4 基础篇 + M0–M9 模块篇已经把**单个能力**讲透（VO、深度、法线、语义、端侧…），
也各自提供了可跑脚本。但真实工程**不是单点能力**，而是：

    「一个公开数据集 → 一串模块按场景串起来 → 一个能交付的实战产出」

本模块把这件事结构化：
  * ``Sequence``    —— 把 TUM / 4Seasons / EuRoC / 合成数据 **归一成同一个对象**
                       （rgb 帧、内参 K、深度、真值位姿、场景标签）。
  * ``Stage``       —— 把 M0–M9 的**某一个能力**封装成统一接口的「工序」
                       （输入 Sequence + 共享上下文 ctx，输出 StageResult 写回 ctx）。
  * ``Project``     —— 一个「实战项目」= 场景 + 有序工序 + 交付物渲染 + 知识落点叙述。
  * ``run_project`` —— 把项目跑通，产出 metrics / 图 / README（知识→实战的桥梁文档）。

设计原则
--------
1. **数据集无关**：上层 Project 不关心数据来自 TUM 还是 4Seasons，只看 Sequence 字段。
2. **工序可插拔 + 优雅降级**：某个 Stage 依赖缺失（如 OWL-ViT 没下）就标注 skip，不阻断。
3. **逻辑自洽**：后序工序消费前序 ctx 里的产物（如 SceneGraph 用 VO 位姿 + Depth 反投影），
   形成一条真实可解释的因果链，而不是各算各的。

直白讲解
--------
把「做工程」想成工厂流水线：原料（Sequence）进厂 → 多道工序（Stage）依次加工，
每道工序把半成品放在传送带（ctx）上 → 最后一道工序打包成成品（交付物）。
不同工厂（Project）只是换了原料和工序组合。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np


@dataclass
class Sequence:
    """一个被归一化的序列（跨数据集的核心数据结构）。

    rgbs         : List[(H,W,3)] BGR uint8，按帧序
    K            : (3,3) 内参
    depths       : List[(H,W) float32 米] 或 None（逐帧，可能部分帧为 None）
    gt_poses     : (N,4,4) T_wc（世界→相机逆），与 rgbs 等长；无真值则 None
    gt_positions : (N,3) 相机光心世界坐标，与 rgbs 等长；无真值则 None
    scenario     : "indoor" / "driving" / "degraded" / "synthetic" …（语义标签）
    frame_names  : List[str]
    meta         : 任意附加信息（数据集名、序列名、下载链接…）
    """
    name: str
    scenario: str
    rgbs: List[np.ndarray]
    K: np.ndarray
    depths: Optional[List[Optional[np.ndarray]]] = None
    gt_poses: Optional[np.ndarray] = None
    gt_positions: Optional[np.ndarray] = None
    frame_names: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self):
        return len(self.rgbs)

    def has_depth(self) -> bool:
        return bool(self.depths) and any(d is not None for d in self.depths)

    def has_gt(self) -> bool:
        return self.gt_positions is not None and len(self.gt_positions) == len(self.rgbs)


@dataclass
class StageResult:
    """一道工序的产物摘要（写进最终 metrics / README）。"""
    name: str
    ok: bool
    skipped: bool = False
    skip_reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self):
        return {
            "name": self.name,
            "ok": self.ok,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "metrics": self.metrics,
            "note": self.note,
        }


class Stage(ABC):
    """所有工序的统一接口。

    run(seq, ctx, **kw) 读取 seq 与前序 ctx，把本工序产物写入 ctx[<key>]，
    返回 StageResult。ctx 是跨工序共享的「传送带」。
    """
    name: str = "base"
    # 本工序往 ctx 写入的键（供下游 / README 引用）
    writes: List[str] = []

    @abstractmethod
    def run(self, seq: Sequence, ctx: Dict[str, Any], **kw) -> StageResult:
        ...


@dataclass
class Project:
    """一个「实战项目」的定义（纯数据 + 渲染函数引用）。"""
    key: str
    title: str
    scenario: str
    dataset_desc: str          # 用到的公开数据集（诚实说明是否需要下载）
    stage_keys: List[str]      # 有序工序
    deliverable: str           # 项目最终交付物（一句话）
    knowledge_bridge: str      # 知识→实战的落点叙述
    # 渲染最终交付物图（可选），签名 (seq, ctx, out_dir) -> 图路径或 None
    render: Any = None
