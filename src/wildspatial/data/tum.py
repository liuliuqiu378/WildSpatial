"""TUM RGB-D 数据集加载

目录结构（解压后）：
    rgbd_dataset_freiburg1_desk/
        rgb.txt          每行: "timestamp filename"   （RGB 图像）
        depth.txt        每行: "timestamp filename"   （16-bit 深度图）
        groundtruth.txt  每行: "timestamp tx ty tz qx qy qz qw"
        rgb/  depth/

关键点
------
1. **rgb / depth / groundtruth 的时间戳不同步**（不同传感器不同频率）。
   必须做**时间戳关联**：对每个 rgb 时间戳，找深度和真值中时间差最小的那条。
   TUM 官方提供的 `associate.py` 就是干这个的，默认最大时间差 0.02s。

2. **真值位姿是 T_wc**（相机在世界坐标系下的位姿，即 camera-to-world），
   而我们的管线用 **T_cw**（world-to-camera）。转换：T_cw = inv(T_wc)。
   ⚠️ 这里搞反是 SLAM 最常见的错误之一。

3. 深度图是 16-bit PNG，单位毫米，需除以 `depth_scale`（TUM 为 5000）得到米。
"""

import os
import numpy as np

from ..geometry import lie

__all__ = ["read_tum_file", "read_trajectory", "associate_timestamps",
           "TUMDataset", "quat_to_R", "pose_to_T"]


DEFAULT_DEPTH_SCALE = 5000.0     # TUM: 像素值 5000 → 1.0 米


def read_tum_file(path):
    """读取 "timestamp data..." 格式的文本，跳过 # 注释行。

    返回 (timestamps np.array, data list[list[str]])
    """
    stamps, data = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            stamps.append(float(parts[0]))
            data.append(parts[1:])
    return np.array(stamps), data


def quat_to_R(q):
    """四元数 (x,y,z,w) → 旋转矩阵"""
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_to_T(t, q):
    """TUM 的 (tx,ty,tz, qx,qy,qz,qw) → T_wc (4x4)"""
    T = np.eye(4)
    T[:3, :3] = quat_to_R(q)
    T[:3, 3] = t
    return T


def read_trajectory(path):
    """读取 groundtruth.txt，返回 (timestamps, T_wc 列表)"""
    stamps, data = read_tum_file(path)
    poses = [pose_to_T([float(v) for v in d[:3]], [float(v) for v in d[3:7]])
             for d in data]
    return stamps, poses


def associate_timestamps(stamps_a, stamps_b, max_diff=0.02, offset=0.0):
    """对 A 的每个时间戳，在 B 中找时间差最小的匹配。

    返回匹配索引列表 [(i_a, i_b), ...]，按 A 的顺序。
    （等价于 TUM 官方 associate.py 的默认行为）
    """
    out = []
    b_sorted = np.argsort(stamps_b)
    sb = stamps_b[b_sorted]
    for i_a in range(len(stamps_a)):
        target = stamps_a[i_a] + offset
        j = np.searchsorted(sb, target)
        cand = []
        if j > 0:
            cand.append(j - 1)
        if j < len(sb):
            cand.append(j)
        if not cand:
            continue
        best = min(cand, key=lambda k: abs(sb[k] - target))
        if abs(sb[best] - target) <= max_diff:
            out.append((i_a, int(b_sorted[best])))
    return out


class TUMDataset:
    """TUM RGB-D 序列加载器

    用法：
        ds = TUMDataset("data/raw/rgbd_dataset_freiburg1_desk")
        print(len(ds))
        frame = ds[0]           # dict: rgb, depth, T_wc, T_cw, stamp
    """

    def __init__(self, root, max_time_diff=0.02, depth_scale=DEFAULT_DEPTH_SCALE,
                 with_depth=True):
        self.root = root
        self.depth_scale = depth_scale
        self.with_depth = with_depth

        rgb_txt = os.path.join(root, "rgb.txt")
        depth_txt = os.path.join(root, "depth.txt")
        gt_txt = os.path.join(root, "groundtruth.txt")

        self.rgb_stamps, rgb_files = read_tum_file(rgb_txt)
        self.rgb_paths = [os.path.join(root, f[0]) for f in rgb_files]

        if with_depth and os.path.exists(depth_txt):
            self.depth_stamps, depth_files = read_tum_file(depth_txt)
            self.depth_paths = [os.path.join(root, f[0]) for f in depth_files]
        else:
            self.depth_stamps, self.depth_paths = np.array([]), []

        self.has_gt = os.path.exists(gt_txt)
        if self.has_gt:
            self.gt_stamps, self.gt_poses = read_trajectory(gt_txt)
        else:
            self.gt_stamps, self.gt_poses = np.array([]), []

        # rgb ↔ depth 关联
        if with_depth and len(self.depth_stamps):
            self.rgb_depth_pairs = associate_timestamps(
                self.rgb_stamps, self.depth_stamps, max_time_diff)
        else:
            self.rgb_depth_pairs = [(i, -1) for i in range(len(self.rgb_stamps))]

        # rgb ↔ groundtruth 关联
        if self.has_gt:
            self.rgb_gt_pairs = associate_timestamps(
                self.rgb_stamps, self.gt_stamps, max_time_diff)
        else:
            self.rgb_gt_pairs = []

        self._gt_of_rgb = {i_a: i_b for i_a, i_b in self.rgb_gt_pairs}
        self._depth_of_rgb = {i_a: i_b for i_a, i_b in self.rgb_depth_pairs}

    def __len__(self):
        return len(self.rgb_depth_pairs)

    @property
    def valid_indices(self):
        return [i_a for i_a in self._depth_of_rgb if self._depth_of_rgb[i_a] >= 0]

    def __getitem__(self, k):
        """第 k 个有效帧（k 是对 rgb_depth_pairs 的索引）"""
        import cv2
        i_rgb, i_depth = self.rgb_depth_pairs[k]

        rgb = cv2.imread(self.rgb_paths[i_rgb], cv2.IMREAD_COLOR)
        if rgb is None:
            raise IOError(f"无法读取: {self.rgb_paths[i_rgb]}")

        depth = None
        if i_depth >= 0:
            d = cv2.imread(self.depth_paths[i_depth], cv2.IMREAD_UNCHANGED)
            if d is not None:
                depth = d.astype(np.float32) / self.depth_scale

        out = {"index": i_rgb, "rgb": rgb, "depth": depth,
               "stamp": self.rgb_stamps[i_rgb],
               "rgb_path": self.rgb_paths[i_rgb]}

        if i_rgb in self._gt_of_rgb:
            T_wc = self.gt_poses[self._gt_of_rgb[i_rgb]]
            out["T_wc"] = T_wc
            out["T_cw"] = np.linalg.inv(T_wc)     # ⚠️ 我们的管线统一用 T_cw
        return out

    def trajectory_gt(self):
        """返回与 __getitem__ 顺序一致的真值轨迹（相机位置 Nx3）"""
        if not self.has_gt:
            return None
        pts = []
        for i_rgb, _ in self.rgb_depth_pairs:
            if i_rgb in self._gt_of_rgb:
                T_wc = self.gt_poses[self._gt_of_rgb[i_rgb]]
                pts.append(T_wc[:3, 3])
        return np.array(pts)
