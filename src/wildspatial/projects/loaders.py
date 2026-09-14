"""数据集加载器 · 把不同公开数据集归一成 ``Sequence``

支持的场景源（scenario）：
  * ``tum``         —— TUM RGB-D（室内，含真值位姿 + 深度）。真实数据，本地已下 fr1/desk。
  * ``tum_degraded``—— TUM fr3_nostructure（退化室内：无纹理/无结构），真实「搜救式」退化场景。
  * ``4seasons``    —— 4Seasons oldtown_night（驾驶，恶劣天气 GNSS 真值），真实数据，本地已下。
  * ``synthetic``   —— 程序生成（**无需下载**）：绕圈相机 + 若干彩色方块物体 + 已知位姿，
                       用于「无网也能演示多场景框架」和 AR 端侧项目。

所有 loader 都返回同一个 ``Sequence``，上层 Project 完全不感知数据来源。
"""

import os
import math
import zipfile
import numpy as np

from .core import Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
RAW = os.path.join(ROOT, "data", "raw")

TUM_K = {
    "fr1": (517.3, 516.5, 318.6, 255.3),
    "fr2": (520.9, 521.0, 325.1, 249.7),
    "fr3": (535.4, 539.2, 320.1, 247.6),
}


# --------------------------------------------------------------------------- #
# TUM RGB-D（室内，含深度 + 真值位姿）
# --------------------------------------------------------------------------- #
def load_tum(seq="fr1/desk", max_frames=0, stride=1, with_depth=True,
             degrade=None, degrade_sev=0.6):
    """加载 TUM 序列，可选施加 M3 合成退化（模拟退化环境，如搜救/地下）。

    degrade 支持逗号/加号分隔的多种退化，例如 "low_light,motion_blur"。
    纪律：合成退化必须在交付物中明确标注（见 docs/M3、data/degrade.py）。
    """
    from ..data.tum import TUMDataset
    prefix = seq.split("/")[0]
    fx, fy, cx, cy = TUM_K[prefix]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    name_map = {
        "fr1/desk": "rgbd_dataset_freiburg1_desk",
        "fr1/room": "rgbd_dataset_freiburg1_room",
        "fr3/nostructure": "rgbd_dataset_freiburg3_nostructure_notexture_near_withloop",
    }
    root = os.path.join(RAW, name_map.get(seq, seq.replace("/", "_")))
    if not os.path.exists(root):
        raise FileNotFoundError(f"TUM 序列未找到: {root}（请先下载或改 --seq）")
    ds = TUMDataset(root, with_depth=with_depth)

    # 解析退化算子列表
    deg_list = []
    if degrade:
        for d in degrade.replace("+", ",").split(","):
            d = d.strip()
            if d and d != "clean":
                deg_list.append(d)

    rgbs, depths, gts, names = [], [], [], []
    n = len(ds)
    idxs = list(range(0, n, max(1, stride)))
    if max_frames:
        idxs = idxs[:max_frames]
    import cv2
    for k in idxs:
        f = ds[k]
        if "T_wc" not in f:               # 无真值关联的帧跳过，保证对齐
            continue
        if with_depth and f.get("depth") is None:
            continue
        img = f["rgb"]
        if deg_list:
            for d in deg_list:
                img = _degrade(img, d, degrade_sev)
        rgbs.append(img)
        depths.append(f.get("depth"))
        gts.append(f["T_wc"])
        names.append(os.path.basename(f["rgb_path"]))

    gt_poses = np.array(gts) if gts else None
    gt_positions = gt_poses[:, :3, 3] if gt_poses is not None else None
    scen = "indoor" if "fr1" in seq and not deg_list else "degraded"
    name = f"TUM {seq}" + (f" +退化[{degrade}]" if deg_list else "")
    return Sequence(
        name=name, scenario=scen,
        rgbs=rgbs, K=K, depths=depths if with_depth else None,
        gt_poses=gt_poses, gt_positions=gt_positions, frame_names=names,
        meta={"dataset": "TUM RGB-D", "seq": seq, "degrade": degrade or "none",
              "path": root},
    )


def _degrade(img, kind, sev):
    try:
        from ..data.degrade import degrade as _deg_fn
        return _deg_fn(img, kind=kind, severity=sev)
    except Exception:
        return img


def load_tum_degraded(seq="fr1/desk", max_frames=0, stride=3,
                      degrade="low_light,motion_blur", degrade_sev=0.6):
    """退化室内（合成退化模拟搜救/地下环境）—— 注意：退化为合成施加，已明确标注。

    说明：TUM fr3_nostructure 实测缺 rgb.txt/groundtruth.txt，无法定量评估，
    故本项目用「带真值的 fr1/desk + M3 合成退化」来可控地模拟退化环境，
    从而能定量回答「退化到什么程度 VO 开始崩」。
    """
    return load_tum(seq=seq, max_frames=max_frames, stride=stride, with_depth=True,
                    degrade=degrade, degrade_sev=degrade_sev)


# --------------------------------------------------------------------------- #
# 4Seasons（驾驶，恶劣天气 GNSS 真值）
# --------------------------------------------------------------------------- #
def _parse_gnss(raw_bytes):
    ts, pos = [], []
    for line in raw_bytes.decode(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split(",")
        if len(p) < 8:
            continue
        try:
            ts.append(int(p[0])); pos.append([float(p[1]), float(p[2]), float(p[3])])
        except ValueError:
            continue
    return np.array(ts, dtype=np.int64), np.array(pos, dtype=float)


def _parse_times(raw_bytes):
    table = {}
    for line in raw_bytes.decode(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            table[int(parts[0])] = int(parts[1])
        except ValueError:
            continue
    return table


def load_4seasons(split="oldtown_night", start=400, n=200, cam="cam0"):
    """4Seasons 恶劣天气驾驶序列。本地已下 oldtown_night。"""
    base = os.path.join(RAW, "4seasons", split)
    zip_path = os.path.join(base, "stereo.zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"4Seasons 序列未找到: {zip_path}")
    # GNSS 真值
    import glob
    gt_candidates = glob.glob(os.path.join(base, "ref", "recording_*", "GNSSPoses.txt"))
    if not gt_candidates:
        gt_candidates = glob.glob(os.path.join(base, "**", "GNSSPoses.txt"), recursive=True)
    # 内参
    cal = os.path.join(RAW, "4seasons", "calibration", "calibration", "undistorted_calib_0.txt")
    with open(cal) as fh:
        first = fh.readline().split()
    fx, fy, cx, cy = [float(x) for x in first[1:5]]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    with zipfile.ZipFile(zip_path) as z:
        import re
        names = [nm for nm in z.namelist()
                 if re.search(rf"{cam}/[^/]+\.(png|jpg|jpeg)$", nm, re.I)]
        names.sort()
        tnames = [nm for nm in z.namelist() if re.search(rf"{cam}/times\.txt$", nm, re.I)]
        times = _parse_times(z.read(tnames[0])) if tnames else {}
        sel = names[start:start + n] if n else names[start:]
        rgbs, stamps = [], []
        for nm in sel:
            data = z.read(nm)
            import cv2
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            rgbs.append(img)
            fid = int(os.path.basename(nm).split(".")[0])
            stamps.append(times.get(fid, 0))

    gt_ts, gt_pos = _parse_gnss(open(gt_candidates[0], "rb").read())
    order = np.argsort(gt_ts); gts, gp = gt_ts[order], gt_pos[order]
    stamps = np.array(stamps, dtype=np.int64)
    idx = np.clip(np.searchsorted(gts, stamps), 1, len(gts) - 1)
    gt_positions = gp[idx]  # 按时间戳最近邻对齐（Sim3 会吸收残差）
    return Sequence(
        name=f"4Seasons {split}", scenario="driving",
        rgbs=rgbs, K=K, depths=None, gt_poses=None, gt_positions=gt_positions,
        frame_names=[os.path.basename(nm) for nm in sel[:len(rgbs)]],
        meta={"dataset": "4Seasons", "split": split, "path": base},
    )


# --------------------------------------------------------------------------- #
# 合成序列（无需下载）—— 绕圈相机 + 彩色方块 + 已知位姿/深度
# --------------------------------------------------------------------------- #
def load_synthetic(n_frames=100, radius=2.0, n_objects=14, seed=0):
    """程序生成室内巡检场景：相机绕原点转圈，俯视若干「带纹理」彩色方块物体。

    为让单目 VO 能稳定跟踪，刻意给**地面棋盘格纹理**与**物体内部双色纹理**
    （真实场景里随处可见的特征），否则纯色块会让 VO 因特征不足而退化。
    提供**真实**深度（物体像素=物体距离，背景=远平面）与**真实**位姿，
    让 VO / 场景图在「无下载」下也能跑通并自洽。
    """
    import cv2
    rng = np.random.default_rng(seed)
    fx = fy = 525.0; H = W = 320
    cx, cy = W / 2, H / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    # 物体：**悬浮在虚空中的 3D 立方体**（各面不同朝向 → 真·非共面结构）。
    # 关键：场景里不能有任何「大平面」主导（地面/墙会让本质矩阵退化 → 平移≈0）。
    # 立方体沿整条相机路径**密集分布**（x 从 -2 到 4），保证第 0 帧（参考帧）的特征
    # 在 VO 累积视差的等待期内持续可匹配（VO 用固定参考帧，需场景足够一致且特征持久）。
    objs = []
    for i in range(n_objects):
        ox = rng.uniform(-2.0, 4.0)
        oy = rng.uniform(-2.0, 2.0)
        oz = rng.uniform(0.4, 2.5)
        h = rng.uniform(0.3, 0.5)
        color = tuple(rng.integers(80, 255, 3).tolist())
        inner = tuple(rng.integers(0, 120, 3).tolist())
        objs.append({"c": np.array([ox, oy, oz]), "half": h,
                     "color": color, "inner": inner})

    # 投影辅助：4 个世界角点 → 图像多边形 + 该面到相机的距离（写深度用）
    def project_quad(corners_w):
        pts = []
        for cw in corners_w:
            pc = R_wc @ (np.asarray(cw) - C)
            if pc[2] <= 0.05:
                return None
            u = fx * pc[0] / pc[2] + cx
            v = fy * pc[1] / pc[2] + cy
            if not (0 <= u < W and 0 <= v < H):
                return None
            pts.append((int(round(u)), int(round(v))))
        return pts

    rgbs, depths, gt_poses, gt_pos = [], [], [], []
    for f in range(n_frames):
        # 机器人式前进：沿 +X 直行（总位移 4m，保证固定参考帧能累积出足够基线视差），
        # 看向正前方。立方体沿 x∈[-2,4] 密集分布，使第 0 帧特征在等待期内持续可匹配。
        cx0 = -2.0 + 4.0 * f / max(1, n_frames - 1)
        C = np.array([cx0, 0.0, 0.8])
        forward = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up); right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R_wc = np.stack([right, up, forward], axis=0)
        T_wc = np.eye(4); T_wc[:3, :3] = R_wc; T_wc[:3, 3] = C

        # 背景：纯灰 + 轻噪声（无大平面纹理，避免 VO 平面退化）
        canvas = np.clip(np.full((H, W, 3), 150, np.int32)
                         + rng.integers(-6, 6, (H, W, 3)), 0, 255).astype(np.uint8)
        depth = np.full((H, W), 8.0, np.float32)         # 背景远平面

        # 画一个 3D 立方体（非共面：前/顶/右三面），每面带 3×3 内部棋盘格纹理，
        # 给 SIFT/ORB 提供大量可匹配角点（纯色面几乎无可提特征）
        def draw_cube(o):
            c, h = o["c"], o["half"]

            def corner(sx, sy, sz):
                return c + np.array([sx * h, sy * h, sz * h])

            faces = [
                [corner(1, 1, 1), corner(1, -1, 1), corner(-1, -1, 1), corner(-1, 1, 1)],
                [corner(1, 1, 1), corner(1, 1, -1), corner(-1, 1, -1), corner(-1, 1, 1)],
                [corner(1, 1, 1), corner(1, -1, 1), corner(1, -1, -1), corner(1, 1, -1)],
            ]
            for fi, fc in enumerate(faces):
                a, b, cc, d = [np.asarray(x, float) for x in fc]   # 四边形四角
                for iy in range(3):
                    for iz in range(3):
                        u0, u1 = iy / 3.0, (iy + 1) / 3.0
                        v0, v1 = iz / 3.0, (iz + 1) / 3.0
                        def bil(u, v):
                            return (a + u * (b - a) + v * (d - a)
                                    + u * v * ((cc - d) - (b - a)))
                        sub = [bil(u0, v0), bil(u1, v0), bil(u1, v1), bil(u0, v1)]
                        pts = project_quad(sub)
                        if pts is None:
                            continue
                        # 棋盘格明暗 + 各面基础色，形成丰富纹理
                        base = o["color"]
                        shade = (255 - fi * 40) if (iy + iz) % 2 == 0 else 150
                        col = tuple(min(255, int(cc_ * (shade / 255) + 25))
                                    for cc_ in base)
                        cv2.fillPoly(canvas, [np.array(pts, np.int32)], col)
                        dist = float(np.linalg.norm(c - C))
                        cv2.fillPoly(depth, [np.array(pts, np.int32)], int(dist * 1000))

        for o in objs:
            draw_cube(o)
        depth = depth / 1000.0

        rgbs.append(canvas); depths.append(depth)
        gt_poses.append(T_wc); gt_pos.append(C)

    return Sequence(
        name="Synthetic indoor", scenario="synthetic",
        rgbs=rgbs, K=K, depths=depths,
        gt_poses=np.array(gt_poses), gt_positions=np.array(gt_pos),
        frame_names=[f"frame_{i:04d}" for i in range(n_frames)],
        meta={"dataset": "synthetic (procedural)", "n_objects": n_objects},
    )


# --------------------------------------------------------------------------- #
# 统一入口
# --------------------------------------------------------------------------- #
_LOADERS = {
    "tum": load_tum,
    "tum_degraded": load_tum_degraded,
    "4seasons": load_4seasons,
    "synthetic": load_synthetic,
}


def get_sequence(scenario: str, **kw) -> Sequence:
    """按 scenario 名取序列。"""
    if scenario not in _LOADERS:
        raise KeyError(f"未知 scenario: {scenario}（可选 {list(_LOADERS)}）")
    return _LOADERS[scenario](**kw)
