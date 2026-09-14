"""Generate program-run conceptual figures for the F-series foundational docs.

All figures are produced by numpy + matplotlib (no external data needed), so they
are genuinely "program-run" demonstrations of the concepts taught in F0-F3.
English titles avoid the matplotlib CJK font gap in standalone scripts.

Output: experiments/foundations/figs/{f0..f3}_*.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "experiments/foundations/figs"
os.makedirs(OUT, exist_ok=True)


def fig_f0():
    """Pinhole projection + coordinate-frame intuition."""
    K = np.array([[525.0, 0.0, 320.0],
                  [0.0, 525.0, 240.0],
                  [0.0, 0.0, 1.0]])
    Z1, Z2, s = 2.0, 6.0, 0.5
    corners = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0], [0, 0, 0]])
    pts_near = corners + np.array([0, 0, Z1])
    pts_far = corners + np.array([0, 0, Z2])

    def proj(P):
        u = K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2]
        v = K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]
        return u, v

    un, vn = proj(pts_near)
    uf, vf = proj(pts_far)

    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.quiver(0, 0, 0, 1, 0, 0, color="r", length=1)
    ax1.quiver(0, 0, 0, 0, 1, 0, color="g", length=1)
    ax1.quiver(0, 0, 0, 0, 0, 1, color="b", length=1)
    ax1.scatter(pts_near[:, 0], pts_near[:, 1], pts_near[:, 2],
                c="orange", s=40, label=f"near (Z={Z1}m)")
    ax1.scatter(pts_far[:, 0], pts_far[:, 1], pts_far[:, 2],
                c="purple", s=40, label=f"far (Z={Z2}m)")
    ax1.set_xlabel("X (right)"); ax1.set_ylabel("Y (down)"); ax1.set_zlabel("Z (forward)")
    ax1.set_title("F0 (a) 3D scene: camera at origin\n(world & camera frame coincide)")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.invert_yaxis()
    ax2.plot(un, vn, "o-", color="orange", label=f"near square (Z={Z1}m, wider)")
    ax2.plot(uf, vf, "o-", color="purple", label=f"far square (Z={Z2}m, narrower)")
    ax2.set_xlabel("pixel u"); ax2.set_ylabel("pixel v")
    ax2.set_title("F0 (b) Pinhole projection:\nnear objects spread wider (scale ~ 1/Z)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/f0_projection.png", dpi=120)
    plt.close(fig)


def fig_f1():
    """Three representations of the SAME sphere: point cloud / mesh / voxel."""
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones(40), np.cos(v))
    xs, ys, zs = X.ravel(), Y.ravel(), Z.ravel()

    N = 24
    edges = np.linspace(-1.1, 1.1, N)
    occ = np.zeros((N, N, N), dtype=bool)
    ix = np.clip(((xs + 1.1) / 2.2 * N).astype(int), 0, N - 1)
    iy = np.clip(((ys + 1.1) / 2.2 * N).astype(int), 0, N - 1)
    iz = np.clip(((zs + 1.1) / 2.2 * N).astype(int), 0, N - 1)
    occ[ix, iy, iz] = True

    fig = plt.figure(figsize=(13, 4))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(xs, ys, zs, s=5, c=zs, cmap="viridis")
    ax1.set_title("F1 (a) Point Cloud\n(unordered 3D points)")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9)
    ax2.set_title("F1 (b) Mesh\n(vertices + faces, renderable)")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    vx, vy, vz = np.where(occ)
    rng = np.random.RandomState(0)
    sel = rng.choice(len(vx), size=min(700, len(vx)), replace=False)
    ax3.scatter(vx[sel], vy[sel], vz[sel], s=8, c="orange")
    ax3.set_title("F1 (c) Voxel grid\n(occupied cells, CNN-friendly)")

    for ax in (ax1, ax2, ax3):
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.tight_layout()
    fig.savefig(f"{OUT}/f1_representations.png", dpi=120)
    plt.close(fig)


def fig_f2():
    """Depth map, disparity, and two-view geometry intuition."""
    H, W = 120, 160
    Z = np.full((H, W), 8.0)
    Z[40:80, 50:110] = 3.0
    f, B = 525.0, 0.12
    disp = f * B / Z

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(Z, cmap="viridis")
    axes[0].set_title("F2 (a) Depth map Z (m)\ndark=near, bright=far")
    axes[1].imshow(disp, cmap="magma")
    axes[1].set_title("F2 (b) Disparity d=fB/Z\nbright=near (large d)")
    axes[2].axis("off")
    axes[2].set_xlim(-2, 2); axes[2].set_ylim(0, 5)
    axes[2].plot([-1, 1], [0, 0], "ks")
    axes[2].text(-1, 0.2, "Cam L"); axes[2].text(1, 0.2, "Cam R")
    axes[2].plot([0], [3], "ro"); axes[2].text(0.05, 3, "P (3D)")
    axes[2].plot([-1, 0], [0, 3], "b-"); axes[2].plot([1, 0], [0, 3], "b-")
    axes[2].set_title("F2 (c) Two-view: depth from\nbaseline B and shift between views")
    fig.tight_layout()
    fig.savefig(f"{OUT}/f2_depth_disparity.png", dpi=120)
    plt.close(fig)


def fig_f3():
    """Real two-view DLT triangulation of a 3D point."""
    K = np.array([[525.0, 0.0, 320.0],
                  [0.0, 525.0, 240.0],
                  [0.0, 0.0, 1.0]])
    C1 = np.array([-1.0, 0.0, 0.0])
    C2 = np.array([1.0, 0.0, 0.0])
    R = np.eye(3)
    t1 = -R @ C1
    t2 = -R @ C2
    P1 = K @ np.hstack([R, t1.reshape(3, 1)])
    P2 = K @ np.hstack([R, t2.reshape(3, 1)])
    Ptrue = np.array([0.0, 0.3, 4.0])

    x1 = P1 @ np.append(Ptrue, 1); x1 /= x1[2]
    x2 = P2 @ np.append(Ptrue, 1); x2 /= x2[2]

    A = np.zeros((4, 4))
    A[0] = x1[0] * P1[2] - P1[0]
    A[1] = x1[1] * P1[2] - P1[1]
    A[2] = x2[0] * P2[2] - P2[0]
    A[3] = x2[1] * P2[2] - P2[1]
    _, _, Vt = np.linalg.svd(A)
    Xr = Vt[-1]
    Pre = Xr[:3] / Xr[3]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*C1, color="k", s=60, marker="s"); ax.text(*C1, " Cam1")
    ax.scatter(*C2, color="k", s=60, marker="s"); ax.text(*C2, " Cam2")
    ax.scatter(*Ptrue, color="g", s=80, label="true 3D point")
    ax.scatter(*Pre, color="r", s=40, marker="x", label="recovered (triangulation)")
    ax.plot([C1[0], Ptrue[0]], [C1[1], Ptrue[1]], [C1[2], Ptrue[2]], "b--", lw=0.8)
    ax.plot([C2[0], Ptrue[0]], [C2[1], Ptrue[1]], [C2[2], Ptrue[2]], "b--", lw=0.8)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("F3 Two-view triangulation (real DLT computation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/f3_triangulation.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    fig_f0(); fig_f1(); fig_f2(); fig_f3()
    print("saved figures to", OUT)
