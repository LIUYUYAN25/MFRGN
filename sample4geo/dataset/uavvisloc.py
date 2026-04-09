"""
sample4geo/dataset/uavvisloc.py
================================
UAV-VisLoc 数据集适配层。

核心设计：窗口读取（Windowed Reading）
  - 不把整张卫星图加载到内存
  - 每次 __getitem__ 只用 rasterio 从磁盘读取需要的 patch 区域
  - 内存占用：每个 rasterio 文件句柄 ~几 MB，而非整图的几 GB

CSV 格式（有表头）:
  num | filename | date | lat | lon | height | Omega | Kappa | Phi1 | Phi2

依赖安装:
  pip install rasterio pyproj
"""

import os
import cv2
import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol as geo_rowcol
from pyproj import Transformer
cv2.setNumThreads(0)


# ═══════════════════════════════════════════════════════════════════
#  场景元数据（只存坐标信息，不存图像数据）
# ═══════════════════════════════════════════════════════════════════

class SceneMeta:
    """
    存储单个场景的坐标变换信息，不加载图像数据。

    属性:
        tif_paths  : 有序的 tif 文件路径列表（单文件或多分块）
        tile_grid  : 分块布局 {(row,col): tif_path}，单文件时 {(1,1): path}
        tile_shapes: 每块的 (H, W)，用于拼合时计算偏移
        total_H/W  : 拼合后的总高度和宽度
        transform  : 仿射变换矩阵（左上角块的）
        crs_epsg   : 卫星图坐标系的 EPSG 编号
        _proj      : pyproj.Transformer（WGS84 → 卫星图 CRS）
    """

    def __init__(self, scene_dir: str):
        self.scene_dir = scene_dir
        tif_files = sorted(f for f in os.listdir(scene_dir)
                           if f.lower().endswith('.tif'))
        if not tif_files:
            raise FileNotFoundError(f"{scene_dir} 中没有 .tif 文件")

        self.tile_grid: dict  = {}
        self.tile_shapes: dict = {}   # (row,col) -> (H, W)
        self.tile_offsets: dict = {}  # (row,col) -> (row_offset, col_offset) 在拼合图中的起始像素

        if len(tif_files) == 1:
            path = os.path.join(scene_dir, tif_files[0])
            self.tile_grid  = {(1, 1): path}
            with rasterio.open(path) as ds:
                self.tile_shapes[(1, 1)] = (ds.height, ds.width)
                self.transform = ds.transform
                self.crs_epsg  = ds.crs.to_epsg() if ds.crs else 4326
            self.tile_offsets[(1, 1)] = (0, 0)
            self.total_H, self.total_W = self.tile_shapes[(1, 1)]
        else:
            self._parse_tiles(scene_dir, tif_files)

        self._build_proj()

    def _parse_tiles(self, scene_dir: str, tif_files: list):
        """解析多分块 TIF，命名格式: satelliteNN_RR-CC.tif"""
        parsed = []
        for fname in tif_files:
            stem  = Path(fname).stem
            parts = stem.rsplit('_', 1)
            if len(parts) == 2 and '-' in parts[1]:
                r_s, c_s = parts[1].split('-')
                parsed.append((int(r_s), int(c_s), os.path.join(scene_dir, fname)))
            else:
                # 无法解析，回退为单文件
                path = os.path.join(scene_dir, tif_files[0])
                self.tile_grid = {(1, 1): path}
                with rasterio.open(path) as ds:
                    self.tile_shapes[(1, 1)] = (ds.height, ds.width)
                    self.transform = ds.transform
                    self.crs_epsg  = ds.crs.to_epsg() if ds.crs else 4326
                self.tile_offsets[(1, 1)] = (0, 0)
                self.total_H, self.total_W = self.tile_shapes[(1, 1)]
                return

        max_row = max(p[0] for p in parsed)
        max_col = max(p[1] for p in parsed)

        for row, col, path in parsed:
            self.tile_grid[(row, col)] = path
            with rasterio.open(path) as ds:
                self.tile_shapes[(row, col)] = (ds.height, ds.width)
                if row == 1 and col == 1:
                    self.transform = ds.transform
                    self.crs_epsg  = ds.crs.to_epsg() if ds.crs else 4326

        # 计算每块在拼合图中的像素偏移
        row_heights = {}
        for r in range(1, max_row + 1):
            row_heights[r] = self.tile_shapes[(r, 1)][0]

        col_widths = {}
        for c in range(1, max_col + 1):
            col_widths[c] = self.tile_shapes[(1, c)][1]

        cum_row = 0
        for r in range(1, max_row + 1):
            cum_col = 0
            for c in range(1, max_col + 1):
                self.tile_offsets[(r, c)] = (cum_row, cum_col)
                cum_col += col_widths[c]
            cum_row += row_heights[r]

        self.total_H = sum(row_heights.values())
        self.total_W = sum(col_widths.values())

    def _build_proj(self):
        """构建 WGS84(lat/lon) → 卫星图 CRS 的转换器。"""
        self._proj = None
        try:
            if self.crs_epsg != 4326:
                # always_xy=False: 输入 (lat, lon)，输出 (easting, northing)
                self._proj = Transformer.from_crs(
                    4326, self.crs_epsg, always_xy=False
                )
        except Exception as e:
            print(f"[警告] 坐标转换器构建失败 (epsg={self.crs_epsg}): {e}")

    def gps_to_pixel(self, lat: float, lon: float):
        """
        WGS84(lat, lon) → 拼合卫星图上的 (x_pixel, y_pixel)。
        x_pixel = 列索引（width 方向，东西方向）
        y_pixel = 行索引（height 方向，南北方向）
        """
        if self._proj is not None:
            east, north = self._proj.transform(lat, lon)
            rows, cols = geo_rowcol(self.transform, xs=east, ys=north)
        else:
            # CRS 已是 WGS84: x=lon（东）, y=lat（北）
            rows, cols = geo_rowcol(self.transform, xs=lon, ys=lat)
        return int(cols), int(rows)   # (x_pixel, y_pixel)


# ═══════════════════════════════════════════════════════════════════
#  窗口读取函数（核心：不加载整图）
# ═══════════════════════════════════════════════════════════════════

def read_patch_windowed(scene_meta: SceneMeta,
                        x_pixel: int,
                        y_pixel: int,
                        patch_size: int) -> np.ndarray:
    """
    从磁盘按窗口读取以 (x_pixel, y_pixel) 为中心的 patch_size×patch_size 区域。
    超出卫星图边界部分补零（黑色填充）。

    参数
    ----
    scene_meta : SceneMeta 对象（只含坐标元数据，不含图像数据）
    x_pixel    : 中心列坐标（0 = 左边）
    y_pixel    : 中心行坐标（0 = 上边）
    patch_size : 裁剪边长

    返回
    ----
    patch : (patch_size, patch_size, 3) uint8 RGB ndarray
    """
    half   = patch_size // 2
    # patch 在拼合图中的绝对像素范围
    abs_x1 = x_pixel - half
    abs_y1 = y_pixel - half
    abs_x2 = abs_x1 + patch_size
    abs_y2 = abs_y1 + patch_size

    patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)

    # 遍历每个分块 tile，找出与 patch 有重叠的部分
    for (tile_r, tile_c), tif_path in scene_meta.tile_grid.items():
        tile_row_off, tile_col_off = scene_meta.tile_offsets[(tile_r, tile_c)]
        tile_H, tile_W = scene_meta.tile_shapes[(tile_r, tile_c)]

        # tile 在拼合图中的绝对范围
        tile_x1 = tile_col_off
        tile_y1 = tile_row_off
        tile_x2 = tile_col_off + tile_W
        tile_y2 = tile_row_off + tile_H

        # 计算重叠区域（在拼合图坐标系下）
        overlap_x1 = max(abs_x1, tile_x1)
        overlap_y1 = max(abs_y1, tile_y1)
        overlap_x2 = min(abs_x2, tile_x2)
        overlap_y2 = min(abs_y2, tile_y2)

        if overlap_x1 >= overlap_x2 or overlap_y1 >= overlap_y2:
            continue   # 无重叠，跳过

        # 重叠区在 tile 内的本地坐标
        local_x1 = overlap_x1 - tile_x1
        local_y1 = overlap_y1 - tile_y1
        local_x2 = overlap_x2 - tile_x1
        local_y2 = overlap_y2 - tile_y1

        # 重叠区在 patch 内的坐标
        patch_x1 = overlap_x1 - abs_x1
        patch_y1 = overlap_y1 - abs_y1
        patch_x2 = overlap_x2 - abs_x1
        patch_y2 = overlap_y2 - abs_y1

        # 用 rasterio Window 读取对应区域
        window = Window(
            col_off=local_x1,
            row_off=local_y1,
            width=local_x2 - local_x1,
            height=local_y2 - local_y1,
        )
        try:
            with rasterio.open(tif_path) as ds:
                data = ds.read(window=window)   # (bands, H_window, W_window)
        except Exception as e:
            print(f"[警告] 窗口读取失败 {tif_path}: {e}")
            continue

        # 转为 RGB uint8 (H, W, 3)
        rgb = _bands_to_rgb(data)

        h_read = rgb.shape[0]
        w_read = rgb.shape[1]
        patch[patch_y1: patch_y1 + h_read,
              patch_x1: patch_x1 + w_read] = rgb

    return patch


def _bands_to_rgb(arr: np.ndarray) -> np.ndarray:
    """(bands, H, W) → (H, W, 3) RGB uint8"""
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    elif arr.shape[0] > 3:
        arr = arr[:3]

    img = np.transpose(arr[:3], (1, 2, 0))   # (H, W, 3)

    if img.dtype != np.uint8:
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = ((img.astype(np.float64) - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    return img


# ═══════════════════════════════════════════════════════════════════
#  CSV 解析 + GPS → 像素坐标
# ═══════════════════════════════════════════════════════════════════

def load_scene_samples(data_folder: str,
                       scene_id: str,
                       meta: SceneMeta,
                       label_offset: int = 0):
    """
    解析场景 CSV，将 lat/lon 转换为像素坐标。

    返回: (samples, n_valid)
      samples = [(drone_path, scene_id, x_pixel, y_pixel, label), ...]
    """
    scene_dir = os.path.join(data_folder, scene_id)
    drone_dir = os.path.join(scene_dir, 'drone')
    csv_path  = os.path.join(scene_dir, f'{scene_id}.csv')

    if not os.path.exists(csv_path):
        print(f"[警告] CSV 不存在: {csv_path}")
        return [], 0

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    samples = []
    n_skip  = 0
    margin  = 512   # 允许轻微超界（crop 时填零），完全在外才丢弃

    for _, row in df.iterrows():
        filename = str(row['filename']).strip()
        lat = float(row['lat'])
        lon = float(row['lon'])

        drone_path = os.path.join(drone_dir, filename)
        if not os.path.exists(drone_path):
            n_skip += 1
            continue

        try:
            x_pixel, y_pixel = meta.gps_to_pixel(lat, lon)
        except Exception as e:
            print(f"[警告] 坐标转换失败 {filename}: {e}")
            n_skip += 1
            continue

        # 完全超界才丢弃
        if (x_pixel < -margin or x_pixel >= meta.total_W + margin or
                y_pixel < -margin or y_pixel >= meta.total_H + margin):
            n_skip += 1
            continue

        samples.append(
            (drone_path, scene_id, x_pixel, y_pixel, label_offset + len(samples))
        )

    if n_skip:
        print(f"    跳过 {n_skip} 个无效/越界样本")

    return samples, len(samples)


# ═══════════════════════════════════════════════════════════════════
#  训练集
# ═══════════════════════════════════════════════════════════════════

class UAVVisLocDatasetTrain(Dataset):
    """
    UAV-VisLoc 训练集。

    采用窗口读取策略，__getitem__ 时才从磁盘读取 patch，
    不预加载整张卫星图，内存占用极低。

    参数
    ----
    data_folder          : UAV_VisLoc_dataset 根目录
    scene_ids            : 训练场景 id 列表，e.g. ['01','02',...,'08']
    transforms_query     : 无人机图变换（albumentations，含 Normalize+ToTensor）
    transforms_reference : 卫星 patch 变换
    prob_flip            : 水平翻转概率（query+reference 同步）
    prob_rotate          : 卫星 patch 随机旋转 90/180/270 度的概率
    shuffle_batch_size   : shuffle() 的 batch 大小
    sat_patch_size       : 卫星 patch 边长（原始卫星图像素单位）
                           transforms 里的 Resize 会将其缩放至 img_size
    """

    def __init__(self,
                 data_folder: str,
                 scene_ids: list,
                 transforms_query=None,
                 transforms_reference=None,
                 prob_flip: float = 0.5,
                 prob_rotate: float = 0.5,
                 shuffle_batch_size: int = 128,
                 sat_patch_size: int = 512):
        super().__init__()
        self.data_folder          = data_folder
        self.transforms_query     = transforms_query
        self.transforms_reference = transforms_reference
        self.prob_flip            = prob_flip
        self.prob_rotate          = prob_rotate
        self.shuffle_batch_size   = shuffle_batch_size
        self.sat_patch_size       = sat_patch_size

        # 只存元数据，不存图像
        self.scene_metas: dict = {}
        self.all_samples: list = []
        label_offset = 0

        for scene_id in scene_ids:
            scene_dir = os.path.join(data_folder, scene_id)
            if not os.path.isdir(scene_dir):
                print(f"[警告] 场景目录不存在: {scene_dir}，跳过")
                continue

            print(f"  读取元数据: 场景 {scene_id} ...")
            meta = SceneMeta(scene_dir)
            self.scene_metas[scene_id] = meta
            print(f"    卫星图大小: {meta.total_W}×{meta.total_H}  "
                  f"CRS EPSG: {meta.crs_epsg}")

            samples, n = load_scene_samples(
                data_folder, scene_id, meta, label_offset
            )
            self.all_samples.extend(samples)
            label_offset += n
            print(f"    有效样本: {n}")

        self.train_ids    = list(range(len(self.all_samples)))
        self.current_idxs = copy.deepcopy(self.train_ids)
        print(f"\nUAVVisLocDatasetTrain 总计: {len(self.all_samples)} 个训练样本")
        print("（卫星图采用窗口读取，不占用大块内存）\n")

    def __len__(self):
        return len(self.current_idxs)

    def __getitem__(self, index):
        drone_path, scene_id, x_pixel, y_pixel, label = \
            self.all_samples[self.current_idxs[index]]

        # ── 无人机图（从磁盘读取） ──────────────────────────────────
        query_img = cv2.imread(drone_path)
        if query_img is None:
            query_img = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # ── 卫星 patch（窗口读取，只读这一块） ──────────────────────
        reference_img = read_patch_windowed(
            self.scene_metas[scene_id], x_pixel, y_pixel, self.sat_patch_size
        )

        # ── 同步数据增广 ─────────────────────────────────────────────
        if np.random.random() < self.prob_flip:
            query_img     = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)

        if np.random.random() < self.prob_rotate:
            k = np.random.choice([1, 2, 3])
            reference_img = np.rot90(reference_img, k).copy()

        # ── 变换（Resize + Normalize + ToTensor） ───────────────────
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)['image']

        return query_img, reference_img, torch.tensor(label, dtype=torch.long)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):
        """随机 shuffle，保证每个 batch 内样本不重复。"""
        print("\nShuffle Dataset:")
        idx_pool        = copy.deepcopy(self.train_ids)
        neighbour_split = neighbour_select // 2
        random.shuffle(idx_pool)

        idx_epoch     = set()
        idx_batch     = set()
        batches       = []
        current_batch = []
        break_counter = 0

        pbar = tqdm()
        while True:
            pbar.update()
            if not idx_pool:
                break

            idx = idx_pool.pop(0)
            if (idx not in idx_batch and idx not in idx_epoch
                    and len(current_batch) < self.shuffle_batch_size):
                idx_batch.add(idx)
                current_batch.append(idx)
                idx_epoch.add(idx)
                break_counter = 0

                if sim_dict is not None and \
                        len(current_batch) < self.shuffle_batch_size:
                    near = copy.deepcopy(sim_dict.get(idx, [])[:neighbour_range])
                    near_a = near[:neighbour_split]
                    near_b = near[neighbour_split:]
                    random.shuffle(near_b)
                    for ni in near_a + near_b[:neighbour_split]:
                        if len(current_batch) >= self.shuffle_batch_size:
                            break
                        if ni not in idx_batch and ni not in idx_epoch:
                            idx_batch.add(ni)
                            current_batch.append(ni)
                            idx_epoch.add(ni)
            else:
                if idx not in idx_epoch:
                    idx_pool.append(idx)
                break_counter += 1

            if break_counter >= 1024:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                batches.extend(current_batch)
                idx_batch     = set()
                current_batch = []

        pbar.close()
        time.sleep(0.3)
        self.current_idxs = batches
        print(f"原始: {len(self.train_ids)} → Shuffle 后: {len(self.current_idxs)}")


# ═══════════════════════════════════════════════════════════════════
#  评估集
# ═══════════════════════════════════════════════════════════════════

class UAVVisLocDatasetEval(Dataset):
    """
    UAV-VisLoc 评估集。同样采用窗口读取，不预加载卫星图。

    img_type='query'     → 无人机图
    img_type='reference' → 对应卫星 patch

    相同 index 的 query 和 reference 互为正对（label 相同）。
    """

    def __init__(self,
                 data_folder: str,
                 scene_ids: list,
                 img_type: str,
                 transforms=None,
                 sat_patch_size: int = 512):
        super().__init__()
        assert img_type in ('query', 'reference')

        self.data_folder    = data_folder
        self.img_type       = img_type
        self.transforms     = transforms
        self.sat_patch_size = sat_patch_size

        self.scene_metas: dict = {}
        self._items: list      = []
        self.labels: list      = []
        label_offset = 0

        for scene_id in scene_ids:
            scene_dir = os.path.join(data_folder, scene_id)
            if not os.path.isdir(scene_dir):
                continue

            print(f"  读取元数据: 场景 {scene_id} ...")
            meta = SceneMeta(scene_dir)
            self.scene_metas[scene_id] = meta

            samples, n = load_scene_samples(
                data_folder, scene_id, meta, label_offset
            )
            for drone_path, s_id, x_pixel, y_pixel, lbl in samples:
                if img_type == 'query':
                    self._items.append(drone_path)
                else:
                    self._items.append((s_id, x_pixel, y_pixel))
                self.labels.append(lbl)
            label_offset += n

        print(f"UAVVisLocDatasetEval [{img_type}]: {len(self._items)} 个样本")

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        if self.img_type == 'query':
            img = cv2.imread(self._items[index])
            if img is None:
                img = np.zeros((256, 256, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            scene_id, x_pixel, y_pixel = self._items[index]
            img = read_patch_windowed(
                self.scene_metas[scene_id], x_pixel, y_pixel, self.sat_patch_size
            )

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        return img, torch.tensor(self.labels[index], dtype=torch.long)