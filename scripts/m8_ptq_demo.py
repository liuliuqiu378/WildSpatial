"""M8 · INT8 PTQ（训练后量化）真跑实验（本机可跑，无需真机）

把 M8 文档里「PyTorch → ONNX → 引擎」链路中最关键的 **INT8 PTQ** 这一招
真正跑一遍，用真实数字验证两个结论：
  1) INT8 PTQ 精度损失通常很小（分类任务 1~3%）—— 这就是为什么它是端侧首选；
  2) 几何敏感任务损失会更高 —— 本实验是「分类」乐观情形，作为对照基线；
     位姿/深度这类几何任务需更谨慎（呼应 M8 §2.2 / §7）。

做法（完全离线、可复现）：
  · 合成一个 2 类 32×32 灰度数据集（低频平滑 vs 高频噪声，极易分）；
  · 训一个 2 卷积 + 1 全连接的小 CNN（FP32）；
  · 静态 INT8 PTQ：插入 QuantStub/DeQuantStub → 融合 conv+bn+relu
    → 设 qconfig(fbgemm/x86) → prepare → 校准 → convert；
  · 对比 FP32 vs INT8 的 准确率 / 单 batch 推理延迟（加速比）。

用法：
    conda activate wildspatial
    cd /home/hmn-cjy/liuliuqiu/WildSpatial
    PYTHONPATH=src python scripts/m8_ptq_demo.py --epochs 4 --seed 0

产出（experiments/M8_edge/）：
    figs/ptq.png        FP32 vs INT8 准确率 & 延迟对比
    metrics_ptq.json    完整数字
    README_ptq.md       结论
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.ao.quantization as tq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from wildspatial.viz import setup_plot_style
setup_plot_style()


# ---------------- 合成数据 ----------------
def make_dataset(n_train=2000, n_test=400, seed=0, size=32):
    rng = np.random.default_rng(seed)
    def gen(cls, n):
        imgs, labels = [], []
        for _ in range(n):
            if cls == 0:                       # 低频：平滑高斯团
                x = np.linspace(-1, 1, size)
                X, Y = np.meshgrid(x, x)
                cx, cy = rng.uniform(-0.5, 0.5, 2)
                s = rng.uniform(0.2, 0.5)
                img = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s ** 2))
            else:                              # 高频：纯随机噪声图（与低频平滑图天然可分）
                img = rng.uniform(-1, 1, (size, size)).astype(np.float32)
                img = (img - img.mean()) / (img.std() + 1e-6)
            imgs.append(img.astype(np.float32))
            labels.append(cls)
        return np.array(imgs)[:, None], np.array(labels)
    xt, yt = gen(0, n_train // 2); xa, ya = gen(1, n_train // 2)
    Xtr = np.concatenate([xt, xa]); Ytr = np.concatenate([yt, ya])
    xt, yt = gen(0, n_test // 2); xa, ya = gen(1, n_test // 2)
    Xte = np.concatenate([xt, xa]); Yte = np.concatenate([yt, ya])
    return (torch.tensor(Xtr), torch.tensor(Ytr)), (torch.tensor(Xte), torch.tensor(Yte))


# ---------------- 小 CNN（带量化桩）----------------
class TinyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.quant = tq.QuantStub()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(16 * 8 * 8, num_classes)
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool(self.relu2(self.bn2(self.conv2(x))))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)
        return x


def train_fp32(model, data, epochs=4, lr=1e-3, batch=64):
    (Xtr, Ytr), (Xte, Yte) = data
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    n = len(Xtr)
    model.train()
    for ep in range(epochs):
        idx = torch.randperm(n)
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            opt.zero_grad()
            loss = crit(model(Xtr[b]), Ytr[b])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == Yte).float().mean().item()
    return acc


def eval_acc(model, Xte, Yte, batch=200):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), batch):
            preds.append(model(Xte[i:i + batch]).argmax(1))
    preds = torch.cat(preds)
    return (preds == Yte).float().mean().item()


def bench_latency(model, Xte, batch=200, warmup=3, reps=10):
    model.eval()
    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(Xte[:batch])
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            _ = model(Xte[:batch])
    return (time.perf_counter() - t0) / reps * 1000.0  # ms / batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = make_dataset(seed=args.seed)
    (_, _), (Xte, Yte) = data

    # ---- FP32 ----
    model = TinyCNN()
    acc_fp32 = train_fp32(model, data, epochs=args.epochs)
    lat_fp32 = bench_latency(model, Xte)
    print(f"[FP32 ] acc={acc_fp32*100:.2f}%  latency={lat_fp32:.2f} ms/batch")

    # ---- INT8 静态 PTQ ----
    torch.backends.quantized.engine = "fbgemm"
    qmodel = TinyCNN()
    qmodel.load_state_dict(model.state_dict())
    qmodel.eval()
    # 1) 融合 conv+bn+relu
    tq.fuse_modules(qmodel, [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"]],
                    inplace=True)
    # 2) 设 qconfig（x86 / fbgemm 默认 Observer）
    qmodel.qconfig = tq.get_default_qconfig("fbgemm")
    tq.prepare(qmodel, inplace=True)
    # 3) 校准（用少量训练/测试数据走一遍，收集激活分布）
    calib, _ = data
    with torch.no_grad():
        for i in range(0, min(400, len(calib[0])), 50):
            qmodel(calib[0][i:i + 50])
    # 4) 转换
    tq.convert(qmodel, inplace=True)
    acc_int8 = eval_acc(qmodel, Xte, Yte)
    lat_int8 = bench_latency(qmodel, Xte)
    print(f"[INT8 ] acc={acc_int8*100:.2f}%  latency={lat_int8:.2f} ms/batch")
    loss_pct = (acc_fp32 - acc_int8) * 100.0
    speedup = lat_fp32 / lat_int8 if lat_int8 > 0 else float("nan")
    print(f"[对比 ] 精度损失={loss_pct:.2f}pp  推理加速={speedup:.2f}x")

    # ---------------- 输出 ----------------
    out_dir = args.out or os.path.join(ROOT, "experiments", "M8_edge")
    figs = os.path.join(out_dir, "figs")
    os.makedirs(figs, exist_ok=True)
    res = {
        "task": "2-class synthetic (low-freq vs high-freq), TinyCNN",
        "fp32_acc": round(acc_fp32, 4), "int8_acc": round(acc_int8, 4),
        "acc_loss_pp": round(loss_pct, 3),
        "fp32_latency_ms_batch": round(lat_fp32, 2),
        "int8_latency_ms_batch": round(lat_int8, 2),
        "speedup_x": round(speedup, 2),
        "engine": "fbgemm", "quant": "INT8 static PTQ",
    }
    with open(os.path.join(out_dir, "metrics_ptq.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    # 图
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.bar(["FP32", "INT8"], [acc_fp32 * 100, acc_int8 * 100], color=["navy", "darkorange"])
    for i, v in enumerate([acc_fp32 * 100, acc_int8 * 100]):
        a1.text(i, v + 1, f"{v:.2f}%", ha="center", fontsize=10)
    a1.set_ylim(0, 108); a1.set_ylabel("准确率 (%)"); a1.set_title("INT8 PTQ 精度（分类：乐观基线）", fontsize=12)
    a2.bar(["FP32", "INT8"], [lat_fp32, lat_int8], color=["navy", "darkorange"])
    for i, v in enumerate([lat_fp32, lat_int8]):
        a2.text(i, v + 1, f"{v:.1f}ms", ha="center", fontsize=10)
    a2.set_ylabel("延迟 (ms/batch)"); a2.set_title(f"推理延迟（加速 {speedup:.2f}x）", fontsize=12)
    plt.suptitle("M8 · INT8 PTQ 真跑：精度仅掉 %.2fpp，推理加速 %.2fx" % (loss_pct, speedup), fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "ptq.png"), dpi=130)
    plt.close()

    with open(os.path.join(out_dir, "README_ptq.md"), "w") as f:
        f.write("# M8 · INT8 PTQ 真跑实验\n\n")
        f.write(f"- 任务：{res['task']}\n")
        f.write(f"- FP32  acc={acc_fp32*100:.2f}%  latency={lat_fp32:.2f} ms/batch\n")
        f.write(f"- INT8  acc={acc_int8*100:.2f}%  latency={lat_int8:.2f} ms/batch\n")
        f.write(f"- **精度损失={loss_pct:.2f}pp，推理加速={speedup:.2f}x**（引擎 {res['engine']}，静态 PTQ）\n")
        f.write("\n> 说明：本实验是**分类**任务（INT8 最友好的情形），损失很小——印证"
                "「INT8 PTQ 是端侧首选」。位姿/深度等**几何敏感**任务损失会更高，"
                "故量化校准集必须覆盖退化场景（呼应 M8 §2.2/§7）。\n")

    print(f"\n[✓] 产出: {out_dir}  (metrics_ptq.json / figs/ptq.png / README_ptq.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
