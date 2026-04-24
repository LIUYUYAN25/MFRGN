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
import math
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol as geo_rowcol
from pyproj import Transformer

cv2.setNumThreads(0)


# ═══════════════════════════════════════════════════════════════════
#  正北对齐函数
# ═══════════════════════════════════════════════════════════════════
def rotate_uav_image_to_north(image: np.ndarray, phi: float) -> np.ndarray:
    """
    先中心裁剪，再旋转，最后截取内切正方形消除黑边，保持物理比例不失真。
    """
    if pd.isna(phi):
        return image
    
    # angle = 90.0 - float(phi)
    angle = -float(phi)  # 直接使用负的 phi 进行旋转，使得无人机图像正北对齐(进一步检查！！！)
    h, w = image.shape[:2]
    
    # 1. 以最短边为基准，先进行中心裁剪成正方形
    min_side = min(h, w)
    cx, cy = w // 2, h // 2
    cropped = image[cy - min_side//2 : cy + min_side//2, cx - min_side//2 : cx + min_side//2]
    
    # 2. 旋转
    M = cv2.getRotationMatrix2D((min_side//2, min_side//2), angle, 1.0)
    rotated = cv2.warpAffine(cropped, M, (min_side, min_side))
    
    # 3. 裁剪内切正方形以去除旋转产生的黑角，根据旋转角动态计算
    rad = abs(math.radians(angle % 90))
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    # 最大内切正方形边长公式（针对正方形输入）
    safe_side = int(min_side / (cos_a + sin_a)) if (cos_a + sin_a) > 1e-6 else min_side
    safe_side = min(safe_side, min_side)          # 不能超过原边长
    center = min_side // 2
    final_img = rotated[center - safe_side//2 : center + safe_side//2,
                        center - safe_side//2 : center + safe_side//2]
                        
    return final_img

# ═══════════════════════════════════════════════════════════════════
#  场景元数据与窗口读取
# ═══════════════════════════════════════════════════════════════════

class SceneMeta:
    def __init__(self, scene_dir: str):
        self.scene_dir = scene_dir
        tif_files = sorted(f for f in os.listdir(scene_dir) if f.lower().endswith('.tif'))
        if not tif_files:
            raise FileNotFoundError(f"{scene_dir} 中没有 .tif 文件")

        self.tile_grid: dict  = {}
        self.tile_shapes: dict = {}   
        self.tile_offsets: dict = {}  

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
        parsed = []
        for fname in tif_files:
            stem  = Path(fname).stem
            parts = stem.rsplit('_', 1)
            if len(parts) == 2 and '-' in parts[1]:
                r_s, c_s = parts[1].split('-')
                parsed.append((int(r_s), int(c_s), os.path.join(scene_dir, fname)))
            else:
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

        row_heights = {r: self.tile_shapes[(r, 1)][0] for r in range(1, max_row + 1)}
        col_widths = {c: self.tile_shapes[(1, c)][1] for c in range(1, max_col + 1)}

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
        self._proj = None
        try:
            if self.crs_epsg != 4326:
                self._proj = Transformer.from_crs(4326, self.crs_epsg, always_xy=False)
        except Exception as e:
            print(f"[警告] 坐标转换器构建失败 (epsg={self.crs_epsg}): {e}")

    def gps_to_pixel(self, lat: float, lon: float):
        if self._proj is not None:
            east, north = self._proj.transform(lat, lon)
            rows, cols = geo_rowcol(self.transform, xs=east, ys=north)
        else:
            rows, cols = geo_rowcol(self.transform, xs=lon, ys=lat)
        return int(cols), int(rows)


def read_patch_windowed(scene_meta: SceneMeta, x_pixel: int, y_pixel: int, patch_size: int) -> np.ndarray:
    half = patch_size // 2
    abs_x1 = x_pixel - half
    abs_y1 = y_pixel - half
    abs_x2 = abs_x1 + patch_size
    abs_y2 = abs_y1 + patch_size

    patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)

    for (tile_r, tile_c), tif_path in scene_meta.tile_grid.items():
        tile_row_off, tile_col_off = scene_meta.tile_offsets[(tile_r, tile_c)]
        tile_H, tile_W = scene_meta.tile_shapes[(tile_r, tile_c)]

        tile_x1 = tile_col_off
        tile_y1 = tile_row_off
        tile_x2 = tile_col_off + tile_W
        tile_y2 = tile_row_off + tile_H

        overlap_x1 = max(abs_x1, tile_x1)
        overlap_y1 = max(abs_y1, tile_y1)
        overlap_x2 = min(abs_x2, tile_x2)
        overlap_y2 = min(abs_y2, tile_y2)

        if overlap_x1 >= overlap_x2 or overlap_y1 >= overlap_y2:
            continue

        local_x1 = overlap_x1 - tile_x1
        local_y1 = overlap_y1 - tile_y1
        local_x2 = overlap_x2 - tile_x1
        local_y2 = overlap_y2 - tile_y1

        patch_x1 = overlap_x1 - abs_x1
        patch_y1 = overlap_y1 - abs_y1

        window = Window(col_off=local_x1, row_off=local_y1, width=local_x2 - local_x1, height=local_y2 - local_y1)
        try:
            with rasterio.open(tif_path) as ds:
                data = ds.read(window=window)
        except Exception:
            continue

        rgb = _bands_to_rgb(data)
        h_read, w_read = rgb.shape[0], rgb.shape[1]
        patch[patch_y1: patch_y1 + h_read, patch_x1: patch_x1 + w_read] = rgb

    return patch


def _bands_to_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    elif arr.shape[0] > 3:
        arr = arr[:3]

    img = np.transpose(arr[:3], (1, 2, 0))
    if img.dtype != np.uint8:
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = ((img.astype(np.float64) - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    return img


def load_scene_samples(data_folder: str, scene_id: str, meta: SceneMeta, label_offset: int = 0):
    """
    解析场景 CSV，增加了对 Phi (航向角) 的提取
    """
    scene_dir = os.path.join(data_folder, scene_id)
    drone_dir = os.path.join(scene_dir, 'drone_IR')
    csv_path  = os.path.join(scene_dir, f'{scene_id}.csv')

    if not os.path.exists(csv_path):
        return [], 0

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    samples = []
    n_skip  = 0
    margin  = 512

    for _, row in df.iterrows():
        filename = str(row['filename']).strip()
        lat = float(row['lat'])
        lon = float(row['lon'])
        
        # 提取航向角 Phi1
        phi = float(row['phi1']) if 'phi1' in row else 0.0

        # ==================== 提取飞行高度 ====================
        height = float(row['height'])
        if pd.isna(height): 
            print(f"Warning: Query {filename} height is NaN, using default 100.0")
            height = 100.0  # 如果有缺失值，给一个默认基准高度
        # try:
        #     height = float(row['height'])
        #     if pd.isna(height): 
        #         height = 100.0  # 如果有缺失值，给一个默认基准高度
        # except:
        #     height = 100.0
        # =====================================================

        drone_path = os.path.join(drone_dir, filename)
        if not os.path.exists(drone_path):
            n_skip += 1
            continue

        try:
            x_pixel, y_pixel = meta.gps_to_pixel(lat, lon)
        except Exception:
            n_skip += 1
            continue

        if (x_pixel < -margin or x_pixel >= meta.total_W + margin or
                y_pixel < -margin or y_pixel >= meta.total_H + margin):
            n_skip += 1
            continue
            
        samples.append((drone_path, scene_id, x_pixel, y_pixel, lat, lon, label_offset + len(samples), phi, height))

    return samples, len(samples)


# ═══════════════════════════════════════════════════════════════════
#  训练集 (结构与 MFRGN cvact.py 严格对齐)
# ═══════════════════════════════════════════════════════════════════

class UAVVisLocDatasetTrain(Dataset):
    
    def __init__(self,
                 data_folder: str,
                 scene_ids: list,
                 transforms_query=None,
                 transforms_reference=None,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 sat_patch_size=512):
        super().__init__()
        self.data_folder          = data_folder
        self.transforms_query     = transforms_query           # ground/drone
        self.transforms_reference = transforms_reference       # satellite
        self.prob_flip            = prob_flip
        self.prob_rotate          = prob_rotate
        self.shuffle_batch_size   = shuffle_batch_size
        self.sat_patch_size       = sat_patch_size

        self.scene_metas = {}
        self.all_samples_info = [] # 存储完整的 (drone_path, scene_id, x, y, label, phi)
        
        train_ids_list = []
        label_offset = 0

        for scene_id in scene_ids:
            scene_dir = os.path.join(data_folder, scene_id)
            if not os.path.isdir(scene_dir): continue

            meta = SceneMeta(scene_dir)
            self.scene_metas[scene_id] = meta

            samples, n = load_scene_samples(data_folder, scene_id, meta, label_offset)
            
            # 对齐 MFRGN 的 id 管理机制
            for s in samples:
                idx = len(self.all_samples_info)
                self.all_samples_info.append(s)
                train_ids_list.append(idx)
                
            label_offset += n

        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids) # 对应原版的 current_idxs

    def __getitem__(self, index):
        idnum = self.samples[index]
        drone_path, scene_id, x_pixel, y_pixel, lat, lon, label, phi, height = self.all_samples_info[idnum]

        # ── 1. load query -> ground(drone) image ──
        query_img = cv2.imread(drone_path, cv2.IMREAD_GRAYSCALE)
        if query_img is None:
            query_img = np.zeros((256, 256, 1), dtype=np.uint8)
        else:
            query_img = rotate_uav_image_to_north(query_img, phi)  # 【正北对齐】
        
        # ================= 新增：动态尺度计算逻辑 =================
        base_height = 100.0
        
        # 按照高度比例动态缩放裁剪框
        dynamic_patch_size = int(self.sat_patch_size * (height / base_height))
        
        # 设置安全上下限，防止无人机过低(裁剪不到有用信息)或过高(爆显存)
        dynamic_patch_size = max(128, min(dynamic_patch_size, 1024))
        # =========================================================

        # ── 2. load reference -> satellite image ──
        reference_img = read_patch_windowed(self.scene_metas[scene_id], x_pixel, y_pixel, dynamic_patch_size)

        # ── 3. Flip simultaneously query and reference ──
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)

        # ── 4. image transforms ──
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
            
        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)['image']

        # ── 5. Rotate simultaneously query and reference ──
        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1, 2, 3])
            # 注意：无人机和卫星都是俯视图，两边都使用 rot90 (与全景图的 roll 不同)
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))
            query_img = torch.rot90(query_img, k=r, dims=(1, 2))
                   
        label = torch.tensor(label, dtype=torch.long)  
        return query_img, reference_img, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):
        """MFRGN Style Shuffle"""
        print("\nShuffle Dataset:")
        idx_pool = copy.deepcopy(self.train_ids)
        neighbour_split = neighbour_select // 2
        
        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)
            
        random.shuffle(idx_pool)
        
        idx_epoch = set()   
        idx_batch = set()
        batches = []
        current_batch = []
        break_counter = 0
        
        pbar = tqdm()
        while True:
            pbar.update()
            if len(idx_pool) > 0:
                idx = idx_pool.pop(0)
                if idx not in idx_batch and idx not in idx_epoch and len(current_batch) < self.shuffle_batch_size:
                    idx_batch.add(idx)
                    current_batch.append(idx)
                    idx_epoch.add(idx)
                    break_counter = 0
                    
                    if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                        near_similarity = similarity_pool[idx][:neighbour_range]
                        near_neighbours = copy.deepcopy(near_similarity[:neighbour_split])
                        far_neighbours = copy.deepcopy(near_similarity[neighbour_split:])
                        random.shuffle(far_neighbours)
                        far_neighbours = far_neighbours[:neighbour_split]
                        near_similarity_select = near_neighbours + far_neighbours
                        
                        for idx_near in near_similarity_select:
                            if len(current_batch) >= self.shuffle_batch_size:
                                break
                            if idx_near not in idx_batch and idx_near not in idx_epoch:
                                idx_batch.add(idx_near)
                                current_batch.append(idx_near)
                                idx_epoch.add(idx_near)
                                similarity_pool[idx].remove(idx_near)
                                break_counter = 0
                else:
                    if idx not in idx_epoch:
                        idx_pool.append(idx)
                    break_counter += 1
                    
                if break_counter >= 1024:
                    break
            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()
        time.sleep(0.3)
        self.samples = batches
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.train_ids), len(self.samples))) 


# ═══════════════════════════════════════════════════════════════════
#  评估集 (结构与 MFRGN cvact.py 严格对齐)
# ═══════════════════════════════════════════════════════════════════

class UAVVisLocDatasetEval(Dataset):
    
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

        self.scene_metas = {}
        self._items      = []
        self.labels      = []
        self.query_gps   = [] if img_type == 'query' else None
        self.query_scene_id = [] if img_type == 'query' else None
        self.ref_gps     = [] if img_type == 'reference' else None
        label_offset = 0

        for scene_id in scene_ids:
            scene_dir = os.path.join(data_folder, scene_id)
            if not os.path.isdir(scene_dir):
                continue

            meta = SceneMeta(scene_dir)
            self.scene_metas[scene_id] = meta

            samples, n = load_scene_samples(data_folder, scene_id, meta, label_offset)
            
            # 分离 Query 和 Reference 存入 _items
            for drone_path, s_id, x_pixel, y_pixel, lat, lon, lbl, phi, height in samples:
                if img_type == 'query':
                    self._items.append((drone_path, phi, height)) # Eval 也需要存 phi 以便旋转
                    self.query_gps.append((lat, lon))
                    self.query_scene_id.append(scene_id)
                else:
                    self._items.append((s_id, x_pixel, y_pixel, height))
                    self.ref_gps.append((lat, lon))
                self.labels.append(lbl)
                
            label_offset += n

        print(f"UAVVisLocDatasetEval [{img_type}]: {len(self._items)} 个样本")

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        if self.img_type == 'query':
            drone_path, phi, height = self._items[index]
            img = cv2.imread(drone_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((256, 256, 1), dtype=np.uint8)
            else:
                img = rotate_uav_image_to_north(img, phi)  # 【正北对齐】
        else:
            scene_id, x_pixel, y_pixel, height = self._items[index]
            # 动态尺度计算
            base_height = 100.0
            dynamic_patch_size = int(self.sat_patch_size * (height / base_height))
            dynamic_patch_size = max(128, min(dynamic_patch_size, 1024))
            img = read_patch_windowed(self.scene_metas[scene_id], x_pixel, y_pixel, dynamic_patch_size)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = torch.tensor(self.labels[index], dtype=torch.long)
        return img, label