"""统一绘图风格（中文字体 + 保存）

用法：
    from wildspatial.viz import setup_plot_style, savefig
    setup_plot_style()
    ...
    savefig(fig, "experiments/M0_geometry_foundation/figs/fig1.png")
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
    "AR PL UMing CN", "AR PL UKai CN",
    "SimHei", "Microsoft YaHei",
]

_initialized = False


def setup_plot_style():
    """设置中文字体与全局风格。重复调用安全。"""
    global _initialized
    if _initialized:
        return
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [c for c in _CANDIDATES if c in available]
    if chosen:
        plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    else:
        # 兜底：直接扫描字体文件
        for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
                  "/usr/share/fonts/truetype/arphic/uming.ttc"]:
            if os.path.exists(p):
                font_manager.fontManager.addfont(p)
        plt.rcParams["font.sans-serif"] = _CANDIDATES + ["DejaVu Sans"]

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 130
    plt.rcParams["font.size"] = 10.5
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    _initialized = True


def savefig(fig, path, dpi=130):
    """保存图片（自动建目录）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
