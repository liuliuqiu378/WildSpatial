"""M2 · 用现成工具 Open3D 做 TSDF 融合重建（无需手搓、无需下载）

==================== 目的 ====================
M2 原定"手搓双目/TSDF"。按"复用现成工具"战略，这里直接用 **Open3D 的
ScalableTSDFVolume**（工业级融合实现），喂入 TUM fr1/desk 的**真实深度图**
和**真值相机位姿**，一键重建出 3D 网格。重点是"看效果 + 建立概念"，
而不是自己实现融合数学。

==================== 用法 ====================
    conda activate wildspatial
    cd /home/hmn-cjy/liuliuqiu/WildSpatial
    PYTHONPATH=src python scripts/m2_reconstruct.py --seq fr1/desk --frames 60 --stride 3

产出（experiments/M2_recon/）：
    mesh.ply              融合出的三角网格
    figs/recon.png        离屏渲染的重建图（肉眼可见效果）
    README.md             四段式说明
"""

import os
import sys
import argparse

import numpy as np
import cv2
import open3d as o3d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from wildspatial.data.tum import TUMDataset

TUM_K = {
    "fr1": (517.3, 516.5, 318.6, 255.3),
    "fr2": (520.9, 521.0, 325.1, 249.7),
    "fr3": (535.4, 539.2, 320.1, 247.6),
}
SEQUENCES = {
    "fr1/desk": "rgbd_dataset_freiburg1_desk",
    "fr1/room": "rgbd_dataset_freiburg1_room",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="fr1/desk")
    ap.add_argument("--root", default=None)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--voxel", type=float, default=0.01, help="TSDF 体素边长（米）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = args.root or os.path.join(ROOT, "data", "raw", SEQUENCES[args.seq])
    if not os.path.exists(root):
        print(f"[✗] 序列不存在: {root}")
        return 1

    prefix = args.seq.split("/")[0]
    fx, fy, cx, cy = TUM_K.get(prefix, TUM_K["fr1"])
    ds = TUMDataset(root)
    sel = list(range(0, len(ds), args.stride))[:args.frames]
    print(f"[·] {len(sel)} 帧（stride={args.stride}），有深度: {ds.with_depth}")

    # Open3D 内参（TUM fr1 为 640x480）
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(640, 480, fx, fy, cx, cy)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel, sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)

    n_used = 0
    for i in sel:
        f = ds[i]
        if f["depth"] is None:
            continue
        rgb = cv2.cvtColor(f["rgb"], cv2.COLOR_BGR2RGB)
        depth = f["depth"].astype(np.float32)            # 已除以 5000 → 米
        # Open3D 需要 RGBDImage（即便只融合深度，也需构造该对象）
        color_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=5.0,
            convert_rgb_to_intensity=False)
        # integrate 的 extrinsic = 相机→世界（TUM 真值 T_wc 正是此约定）
        volume.integrate(
            rgbd,
            intrinsic,
            np.asarray(f["T_wc"], dtype=np.float64),
        )
        n_used += 1
    print(f"[·] 已融合 {n_used} 帧深度")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    out = args.out or os.path.join(ROOT, "experiments", "M2_recon")
    figdir = os.path.join(out, "figs")
    os.makedirs(figdir, exist_ok=True)
    o3d.io.write_triangle_mesh(os.path.join(out, "mesh.ply"), mesh)

    # 可视化：用 matplotlib 对网格顶点做 3D 点云着色图（headless 可靠）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wildspatial.viz import setup_plot_style
    setup_plot_style()

    verts = np.asarray(mesh.vertices)
    if len(verts) > 30000:                 # 下采样以便绘图
        rng = np.random.default_rng(0)
        idx = rng.choice(len(verts), 30000, replace=False)
        verts = verts[idx]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    z = verts[:, 2]
    sc = ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2],
                    c=z, s=1.0, cmap="viridis")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"M2 TSDF 重建点云（{len(verts)} 采样点 · Open3D）")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Z (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "recon.png"), dpi=130)
    plt.close(fig)
    print(f"    图: {os.path.join(figdir, 'recon.png')}")

    nv = len(mesh.vertices)
    print(f"[✓] 网格顶点数: {nv} → {os.path.join(out, 'mesh.ply')}")
    print(f"    图: {os.path.join(figdir, 'recon.png')}")

    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write("# M2 深度重建（Open3D TSDF 融合，现成工具）\n\n")
        fh.write(f"**序列**：{args.seq}（{n_used} 帧融合，体素 {args.voxel}m）\n\n")
        fh.write("## 目的\n用现成工具 Open3D 的 TSDF 融合，把真实深度图 + 真值位姿一键重建成 3D 网格，"
                 "建立「深度→几何」的直观认知，而非手搓融合数学。\n\n")
        fh.write("## 方法原理\nTSDF（截断符号距离场）把每帧深度反投影成空间点的有向距离，"
                 "多视角融合取平均得到平滑表面，再 Marching Cubes 提取网格。\n\n")
        fh.write("## 直白讲解\n就像把很多张带深度的照片「糊」进同一个 3D 体素格子，"
                 "离表面近的地方填满、远的地方空着，最后把「刚好填满的边界」描成网格。\n\n")
        fh.write(f"## 结果\n融合 {n_used} 帧，生成网格顶点 **{nv}**，见 `mesh.ply` 与 `figs/recon.png`。\n")


if __name__ == "__main__":
    main()
