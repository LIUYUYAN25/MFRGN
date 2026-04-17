"""
test_coarse.py
==============
第一阶段粗定位测试脚本（专为修改后的 UAVVisLocDatasetEval 设计）

功能：
1. 加载训练好的 MFRGN 模型
2. 对测试场景（09~11）的所有卫星 patch 提取特征（gallery）
3. 对指定的无人机图像（query）计算 Top-K 最匹配的卫星 patch
4. 输出每个匹配的经纬度、相似度分数
5. 可视化：无人机图 + Top-K 卫星 patch（带经纬度）
6. 保存结果图到本地

使用前请确保：
- 你已经用修改后的 uavvisloc.py 重新运行过训练（或至少生成过 reference dataset）
- 权重路径正确
"""

import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import rasterio
from rasterio.windows import Window

# ====================== 你的项目导入 ======================
from sample4geo.dataset.uavvisloc import UAVVisLocDatasetEval, _bands_to_rgb
from sample4geo.transforms import get_transforms_val
from model.mfrgn import TimmModel_u

# ====================== 配置（请修改这里） ======================
class Config:
    # 模型权重路径（训练完后会生成在 results_uavvisloc/xxx/weights_end.pth）
    weight_path = 'results_uavvisloc/convnext_base.fb_in22k_ft_in1k/mfrgn_uavvisloc_04-16-18-41-38/weights_end.pth'
    
    data_folder = "../Datasets/UAV_VisLoc_dataset"   # ←←← 改成你的数据集路径
    test_scene_ids = ['09', '10', '11']              # 测试场景
    
    img_size = 256                                   # 必须和训练时一致
    sat_patch_size = 512                             # 必须和训练时一致
    batch_size = 128
    top_k = 5                                        # 显示前几个最匹配的粗范围
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = "coarse_test_results"
    
    # 可视化设置
    query_indices = [100, 510, 800, 1000, 1200]                    # 要测试的 query 索引（可多个）
    
    # 大图绘制参数
    bigmap_window_size = 4096                     # 大图窗口边长（像素），可改 1024~4096
    box_size = 512                                # 框的大小（和 sat_patch_size 一致）

config = Config()

os.makedirs(config.save_dir, exist_ok=True)

# ====================== 加载模型 ======================
print("加载模型...")
model = TimmModel_u('convnext_base.fb_in22k_ft_in1k', 
                    psm=True, 
                    img_size=config.img_size)

checkpoint = torch.load(config.weight_path, map_location=config.device)
model.load_state_dict(checkpoint, strict=False)
model = model.to(config.device)
model.eval()

print(f"模型加载完成: {config.weight_path}")

# ====================== 数据变换 ======================
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

sat_transforms_val, ground_transforms_val = get_transforms_val(
    image_size_sat=(config.img_size, config.img_size),
    img_size_ground=(config.img_size, config.img_size),
    mean=mean,
    std=std,
)

# ====================== 准备所有卫星 patch（Reference / Gallery） ======================
print("正在提取所有卫星 patch 的特征（这可能需要几分钟）...")

reference_dataset = UAVVisLocDatasetEval(
    data_folder=config.data_folder,
    scene_ids=config.test_scene_ids,
    img_type='reference',
    transforms=sat_transforms_val,
    sat_patch_size=config.sat_patch_size
)

reference_loader = DataLoader(
    reference_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

ref_features = []
ref_gps_list = []   # 每个 patch 对应的 (lat, lon)

with torch.no_grad():
    for batch_idx, (imgs, labels) in enumerate(tqdm(reference_loader, desc="提取卫星特征")):
        imgs = imgs.to(config.device)
        feats = model(imgs)                     # 模型输出特征
        feats = F.normalize(feats, p=2, dim=1)  # L2 归一化，便于余弦相似度
        
        ref_features.append(feats.cpu())
        
        # 使用我们修改后 dataset 新增的 ref_gps
        start_idx = batch_idx * config.batch_size
        end_idx = start_idx + len(imgs)
        ref_gps_list.extend(reference_dataset.ref_gps[len(ref_gps_list):len(ref_gps_list)+len(imgs)])

ref_features = torch.cat(ref_features, dim=0).to(config.device)
print(f"卫星 patch 总数: {len(ref_features)} 个")
print(f"GPS 信息总数: {len(ref_gps_list)} 个")

# ====================== 测试 Query（无人机图） ======================
query_dataset = UAVVisLocDatasetEval(
    data_folder=config.data_folder,
    scene_ids=config.test_scene_ids,
    img_type='query',
    transforms=ground_transforms_val,
    sat_patch_size=config.sat_patch_size
)

print(f"\n开始测试 {len(config.query_indices)} 张无人机图像...\n")

for q_idx in config.query_indices:
    if q_idx >= len(query_dataset):
        print(f"警告: query index {q_idx} 超出范围，跳过")
        continue
        
    # 取一张 query
    query_tensor, _ = query_dataset[q_idx]           # shape: (3, H, W)
    query_tensor = query_tensor.unsqueeze(0).to(config.device)  # 添加 batch 维度
    
    with torch.no_grad():
        query_feat = model(query_tensor)
        query_feat = F.normalize(query_feat, p=2, dim=1)
    
    # 计算余弦相似度
    similarities = torch.matmul(query_feat, ref_features.T)[0]   # shape: (num_ref,)
    
    # Top-K
    topk_scores, topk_indices = torch.topk(similarities, k=config.top_k)
    
    print(f"\n=== Query {q_idx} ===")
    print(f"Top-{config.top_k} 相似度: {[f'{s:.4f}' for s in topk_scores.tolist()]}")
    
    # ====================== 可视化 ======================
    fig, axs = plt.subplots(1, config.top_k + 1, figsize=(5.5 * (config.top_k + 1), 6))
    
    # 显示 Query（无人机图）
    query_vis = query_tensor[0].cpu().permute(1, 2, 0).numpy()
    query_vis = (query_vis * std + mean) * 255.0
    query_vis = np.clip(query_vis, 0, 255).astype(np.uint8)
    q_lat, q_lon = query_dataset.query_gps[q_idx]

    scene_id = query_dataset.query_scene_id[q_idx]
    meta = query_dataset.scene_metas[scene_id]
    
    axs[0].imshow(query_vis)
    axs[0].set_title(f"Query\nIndex: {q_idx}\n({q_lat:.6f}, {q_lon:.6f})")
    axs[0].axis('off')
    
    # 显示 Top-K 卫星 patch
    for rank, (score, idx) in enumerate(zip(topk_scores, topk_indices)):
        idx = idx.item()
        lat, lon = ref_gps_list[idx]
        
        # 取出对应的卫星图像（从 dataset 重新取）
        ref_img, _ = reference_dataset[idx]   # 注意：这里 index 是全局的
        ref_vis = ref_img.permute(1, 2, 0).numpy()
        ref_vis = (ref_vis * std + mean) * 255.0
        ref_vis = np.clip(ref_vis, 0, 255).astype(np.uint8)
        
        axs[rank + 1].imshow(ref_vis)
        axs[rank + 1].set_title(f"Rank {rank+1}\nScore: {score:.4f}\n({lat:.6f}, {lon:.6f})")
        axs[rank + 1].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(config.save_dir, f"coarse_query_{q_idx:04d}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"可视化结果已保存 → {save_path}")

    # ====================== 大卫星图 + 框（第二阶段核心） ======================
    # 1. 把真实位置转成像素坐标
    true_x, true_y = meta.gps_to_pixel(q_lat, q_lon)
    half = config.bigmap_window_size // 2

    # 限制窗口不超出影像边界，避免空数组
    x_off = max(0, true_x - half)
    y_off = max(0, true_y - half)
    w = min(config.bigmap_window_size, meta.total_W - x_off)
    h = min(config.bigmap_window_size, meta.total_H - y_off)

    # 2. 读取以真实位置为中心的大窗口
    window = Window(col_off=true_x - half, row_off=true_y - half,
                    width=config.bigmap_window_size, height=config.bigmap_window_size)

    try:
        with rasterio.open(list(meta.tile_grid.values())[0]) as ds:
            big_data = ds.read(window=window)
            big_img = _bands_to_rgb(big_data)

        if big_img is None or big_img.size == 0 or big_img.shape[0] == 0:
            print(f"Warning: Query {q_idx} 大图读取为空，跳过 bigmap")
            continue

        big_img = cv2.cvtColor(big_img, cv2.COLOR_RGB2BGR)

        # 真实位置（红框）
        cv2.rectangle(big_img,
                      (half - config.box_size//2, half - config.box_size//2),
                      (half + config.box_size//2, half + config.box_size//2),
                      (0, 0, 255), 8)

        # Top-5 预测位置（绿框）——全部画出
        for rank, idx in enumerate(topk_indices):
            lat, lon = ref_gps_list[idx.item()]
            px, py = meta.gps_to_pixel(lat, lon)
            dx = px - true_x + half
            dy = py - true_y + half

            # 只画在窗口内的框
            if 0 <= dx < big_img.shape[1] and 0 <= dy < big_img.shape[0]:
                color = (0, 255, 0) if rank == 0 else (0, 200, 100)
                thickness = 5 if rank == 0 else 3
                cv2.rectangle(big_img,
                              (dx - config.box_size//2, dy - config.box_size//2),
                              (dx + config.box_size//2, dy + config.box_size//2),
                              color, thickness)
                cv2.putText(big_img, f"R{rank+1}", (dx-35, dy-35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

        big_save = os.path.join(config.save_dir, f"coarse_query_{q_idx:04d}_bigmap.png")
        cv2.imwrite(big_save, big_img)
        print(f"✅ 大卫星图已保存 → {big_save}（红框=真实位置，绿框=Top-5 预测）")

    except Exception as e:
        print(f"Warning: Query {q_idx} 读取大图失败: {e}")

print("\n所有测试完成！结果保存在文件夹：", config.save_dir)