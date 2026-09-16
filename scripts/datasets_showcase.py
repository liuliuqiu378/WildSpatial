"""数据集样例图集 · 从真实下载的数据里抽样例，拼成"一图看清每个数据集长什么样"

用途：为 docs/DATASETS.md 提供**真实样例图**（不是文字描述）。
产出：experiments/datasets_showcase/figs/<dataset>.png

用法：
    conda activate wildspatial
    PYTHONPATH=src python scripts/datasets_showcase.py
"""
import os
import sys
import glob
import zipfile
import tarfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from wildspatial.viz import setup_plot_style  # noqa: E402

setup_plot_style()
MS = os.path.join(ROOT, "data", "raw", "ms")
OUT = os.path.join(ROOT, "experiments", "datasets_showcase", "figs")


def _imread(path):
    import cv2
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def _show_grid(imgs, titles, out, suptitle, ncol=4, cmaps=None):
    """imgs: list of (img, cmap) 或 纯 img"""
    def _unpack(x):
        if isinstance(x, (tuple, list)) and len(x) == 2:
            return x[0], x[1]
        return x, None
    n = len(imgs)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow))
    axes = np.array(axes).reshape(-1)
    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            im, cm = _unpack(imgs[i])
            ax.imshow(im, cmap=cm)
            ax.set_title(titles[i], fontsize=8)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] {os.path.basename(out)}  ({n} 样例)")


def showcase_sample_dir(name, path, out_name, suptitle, n=8, recursive=False):
    pat = "**/*" if recursive else "*"
    files = sorted(glob.glob(os.path.join(path, pat), recursive=recursive))
    files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))][:n]
    if not files:
        print(f"[·] {name}: 无样例图（跳过）"); return
    imgs = [(_imread(f), None) for f in files]
    titles = [os.path.basename(f)[:22] for f in files]
    _show_grid(imgs, titles, os.path.join(OUT, out_name), suptitle)


def showcase_zip_images(zip_path, out_name, suptitle, n=8, pattern=""):
    if not os.path.exists(zip_path):
        print(f"[·] {out_name}: zip 不存在（跳过）"); return
    import cv2
    with zipfile.ZipFile(zip_path) as z:
        names = [nm for nm in z.namelist()
                 if nm.lower().endswith((".jpg", ".jpeg", ".png")) and pattern in nm]
        names = sorted(names)[:n]
        if not names:
            print(f"[·] {out_name}: zip 内无匹配图像（跳过）"); return
        imgs = []
        for nm in names:
            im = cv2.imdecode(np.frombuffer(z.read(nm), np.uint8), cv2.IMREAD_COLOR)
            imgs.append(im)
        titles = [os.path.basename(nm)[:22] for nm in names]
    _show_grid(imgs, titles, os.path.join(OUT, out_name), suptitle)


def showcase_tar_zip_images(tar_path, out_name, suptitle, n=8):
    """NightCity: tar 内套 zip"""
    if not os.path.exists(tar_path):
        print(f"[·] {out_name}: tar 不存在（跳过）"); return
    import io
    import cv2
    with tarfile.open(tar_path) as t:
        inner = [m for m in t.getmembers() if m.name.lower().endswith(".zip")]
        if not inner:
            print(f"[·] {out_name}: tar 内无 zip（跳过）"); return
        # 取第一个 zip 里的图（images zip）
        imgz = None
        for m in inner:
            if "image" in m.name.lower():
                imgz = m; break
        imgz = imgz or inner[0]
        data = t.extractfile(imgz).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = sorted([nm for nm in z.namelist() if nm.lower().endswith((".jpg", ".png"))])[:n]
        imgs = [cv2.imdecode(np.frombuffer(z.read(nm), np.uint8), cv2.IMREAD_COLOR) for nm in names]
        titles = [os.path.basename(nm)[:22] for nm in names]
    _show_grid(imgs, titles, os.path.join(OUT, out_name), suptitle)


def showcase_sunrgbd(out_name):
    """SUN RGB-D: 从主 zip 里抽 fullres RGB + depth 成对展示"""
    zip_path = os.path.join(MS, "OmniData__SUN_RGB-D", "raw", "SUNRGBD.zip")
    if not os.path.exists(zip_path):
        print("[·] SUN RGB-D: zip 不存在（跳过）"); return
    import cv2
    with zipfile.ZipFile(zip_path) as z:
        fullres = [nm for nm in z.namelist()
                   if nm.endswith((".jpg", ".png")) and "/fullres/" in nm][:4]
        depths = [nm for nm in z.namelist()
                  if nm.endswith(".png") and ("/depth_bfx/" in nm or "/depth/" in nm)][:4]
        imgs, titles = [], []
        for nm in fullres:
            im = cv2.imdecode(np.frombuffer(z.read(nm), np.uint8), cv2.IMREAD_COLOR)
            imgs.append((im, None)); titles.append("RGB " + nm.split("/")[-2][:14])
        for nm in depths:
            d = cv2.imdecode(np.frombuffer(z.read(nm), np.uint8), cv2.IMREAD_UNCHANGED)
            if d is not None:
                dm = np.ma.masked_where(d == 0, d.astype(np.float32) / 8000.0)
                imgs.append((dm, "viridis")); titles.append("Depth " + nm.split("/")[-2][:14])
    _show_grid(imgs, titles, os.path.join(OUT, out_name),
               "SUN RGB-D：室内 RGB（上排）+ 深度真值（下排，米）", ncol=4)


def showcase_modelnet(out_name):
    """ModelNet40-C: 从 .npy 里抽几个点云散点图（含退化类型）"""
    d = os.path.join(MS, "OmniData__ModelNet40-C", "raw", "modelnet40_c")
    npys = sorted(glob.glob(os.path.join(d, "data_*.npy")))
    npys = [p for p in npys if not p.endswith(".incomplete")][:6]
    if not npys:
        print("[·] ModelNet40-C: 无 npy（跳过）"); return
    fig = plt.figure(figsize=(3.0 * len(npys), 3.2))
    for i, p in enumerate(npys):
        try:
            arr = np.load(p, allow_pickle=True)
            pts = arr[0] if arr.ndim == 3 else arr
            pts = np.asarray(pts)[:, :3] if np.asarray(pts).ndim == 2 else np.asarray(pts).reshape(-1, 3)
        except Exception as e:
            print(f"[·] {os.path.basename(p)} 读取失败 {str(e)[:40]}"); continue
        ax = fig.add_subplot(1, len(npys), i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5)
        ax.set_title(os.path.basename(p).replace("data_", "").replace(".npy", ""), fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.suptitle("ModelNet40-C：3D 点云 + 15 种真实性退化（噪声/剪切/畸变…）", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out_name), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] {out_name}")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("生成数据集样例图集…")

    # NightCity（夜间驾驶）
    showcase_tar_zip_images(
        os.path.join(MS, "OmniData__NightCity", "raw", "NightCity.tar.gz"),
        "nightcity.png", "NightCity：夜间城市驾驶（低光照真实采集）")

    # SUN RGB-D（室内 RGB-D 深度真值）
    showcase_sunrgbd("sunrgbd.png")

    # NDISPark（夜/昼实例分割）
    showcase_zip_images(
        os.path.join(MS, "OmniData__NDISPark_Night_and_Day_Instance_Segmented_etc", "raw", "ndis_park.zip"),
        "ndispark.png", "NDISPark：夜间/白天停车场（实例分割）", n=8, pattern="imgs/")

    # 水下检测（图像在 train/valid/test 子目录，需递归）
    showcase_sample_dir(
        "underwater", os.path.join(MS, "isLinXu__rf100-vl-underwater-objects"),
        "underwater.png", "rf100 水下目标检测（散射/浑浊/偏色）", recursive=True)

    # ModelNet40-C（点云 + 退化）
    showcase_modelnet("modelnet40c.png")

    # 4Seasons（已有，夜间驾驶）
    showcase_zip_images(
        os.path.join(ROOT, "data", "raw", "4seasons", "oldtown_night", "stereo.zip"),
        "fourseasons.png", "4Seasons oldtown_night：夜间驾驶（双目）", n=8, pattern="cam0/")

    print(f"\n[✓] 全部 → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
