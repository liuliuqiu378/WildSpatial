"""实战项目整合层（WildSpatial Projects）

把多场景公开数据集（TUM / 4Seasons / EuRoC / 合成）归一成统一 ``Sequence``，
把 M0–M9 能力封装成可插拔 ``Stage``，再组合成逻辑自洽的「实战项目」。
典型用法见 ``scripts/run_project.py``。
"""

from .core import (Sequence, Stage, StageResult, Project)
from .loaders import get_sequence
from .stages import build_stage, STAGE_REGISTRY
from .projects import (PROJECTS, get_project, load_project_sequence,
                       render_project_fig)

__all__ = ["Sequence", "Stage", "StageResult", "Project", "get_sequence",
           "build_stage", "STAGE_REGISTRY", "PROJECTS", "get_project",
           "load_project_sequence", "render_project_fig"]
