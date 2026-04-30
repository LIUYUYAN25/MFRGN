"""
sample4geo/evaluate/uav_visloc.py
================================
基于物理 GPS 动态距离阈值（按飞行高度自适应）的 UAV-VisLoc 评估脚本。
"""

import os
import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from ..trainer import predict, predict_dual

def haversine_distance_matrix(query_gps, ref_gps):
    """
    计算 Query 和 Reference 之间的真实地理距离矩阵 (单位: 米)
    """
    query_rad = np.radians(query_gps)
    ref_rad = np.radians(ref_gps)
    
    lat1, lon1 = query_rad[:, 0:1], query_rad[:, 1:2]
    lat2, lon2 = ref_rad[:, 0], ref_rad[:, 1]
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000 # 地球平均半径，单位: 米
    
    return c * r

def evaluate(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             is_autocast=True,
             is_dual=False,
             cleanup=True,
             save_csv=None,
             alpha=0.1):  # 【修改】使用比例系数 alpha 取代固定的 threshold_m
    
    print("\nExtract Features:")
    s1 = time.time()
    
    if is_dual:
        query_features, reference_features, query_labels, reference_labels = predict_dual(
            config, model, query_dataloader, reference_dataloader, is_autocast=is_autocast)
    else:
        query_features, query_labels = predict(
            config, model, query_dataloader, is_autocast=is_autocast)
        reference_features, reference_labels = predict(
            config, model, reference_dataloader, is_autocast=is_autocast) 
        
    s2 = time.time()
    print('Extract Features time: {:.2f} s'.format(s2 - s1))
    
    # 获取 GPS 数据
    query_gps = np.array(query_dataloader.dataset.query_gps)
    ref_gps = np.array(reference_dataloader.dataset.ref_gps)
    
    # 【新增】获取无人机的飞行高度
    # 在你的 UAVVisLocDatasetEval 中，_items 存储的是 (drone_path, phi, height)
    query_heights = np.array([item[2] for item in query_dataloader.dataset._items])
    
    print("\nCompute Scores (Dynamic GPS Threshold: alpha = {} * Height):".format(alpha))
    s = time.time()
    r1, r5, r10, r_top1 = calculate_scores_by_dynamic_gps(
        query_features, reference_features, query_gps, ref_gps, query_heights,
        step_size=step_size, ranks=ranks, alpha=alpha)
    e = time.time()
    print('Compute Scores time: {:.2f} s\n'.format(e - s))

    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        torch.cuda.empty_cache()
    
    return r1, r5, r10, r_top1

def calculate_scores_by_dynamic_gps(query_features, reference_features, query_gps, ref_gps, query_heights, step_size=1000, ranks=[1,5,10], alpha=0.1):
    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    # 1. 计算地理距离矩阵 (Q, R)
    distance_matrix = haversine_distance_matrix(query_gps, ref_gps)
    
    # 2. 分块计算余弦相似度矩阵 (Q, R)
    steps = Q // step_size + 1
    similarity = []
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T
        similarity.append(sim_tmp.cpu())
    similarity = torch.cat(similarity, dim=0) # (Q, R)
    
    topk.append(max(1, R // 100))
    max_k = min(max(topk), R)
    
    results = np.zeros([len(topk)])
    
    # 3. 取出前 max_k 个最高相似度的索引
    _, topk_indices = torch.topk(similarity, k=max_k, dim=1)
    topk_indices = topk_indices.numpy() 
    
    valid_queries = 0
    bar = tqdm(range(Q), ncols=100, position=0, leave=True)
    
    # 4. 基于动态距离阈值计算 Recall
    for i in bar:
        dists_i = distance_matrix[i]
        
        # 【核心逻辑修改】：阈值等于当前无人机高度乘以 alpha (最小兜底 20 米)
        dynamic_threshold = max(20.0, query_heights[i] * alpha) 
        
        true_mask = dists_i <= dynamic_threshold
        
        if not np.any(true_mask):
            continue
            
        valid_queries += 1
        
        for j, k in enumerate(topk):
            pred_k_indices = topk_indices[i, :k]
            if np.any(true_mask[pred_k_indices]):
                results[j] += 1.0
                        
    results = results / valid_queries * 100.
    bar.close()
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
    string.append('Recall@top1%: {:.4f}'.format(results[-1]))            
        
    print(' - '.join(string)) 

    return results[0], results[1], results[2], results[-1]