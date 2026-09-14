"""方法动物园：把各种「现成工具 / 代码包 / 模型」统一封装、即插即用。

导入本模块即完成所有方法的注册（见各子模块顶部的 @register）。
常用入口：
    from wildspatial.methods import discover_methods, get_method
    methods = discover_methods()          # 仅返回当前环境可用的
"""

from .base import (Method, MethodResult, register, discover_methods,
                   get_method)
from . import handcrafted  # 自研 VO 基线（始终可用）
from . import colmap       # 成熟 SfM（装了 pycolmap 才出现）
from . import vggt         # 前馈 3D 基础模型（装了 vggt 才出现）
from . import lightglue    # 学习式匹配前端 VO（装了 lightglue 才出现）

__all__ = ["Method", "MethodResult", "register", "discover_methods",
           "get_method"]
