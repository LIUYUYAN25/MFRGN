import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from ..trainer import predict
import torch.nn.functional as F

def evaluate(config,
             model,
             query_loader,
             gallery_loader,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    """
    University-1652 评估函数
    """
    
    print("\nExtract Features (University-1652):")
    s1 = time.time()
    
    # 提取 Query 特征 (通常是 Drone 或 Satellite)
    query_features, query_labels = predict(config, model, query_loader, input_id=1)
    
    # 提取 Gallery 特征 (通常是 Satellite 或 Drone)
    gallery_features, gallery_labels = predict(config, model, gallery_loader, input_id=1)

    s2 = time.time()
    print('Extract Features time: {:.2f}s'.format(s2 - s1))

    print("Compute Scores:")
    # 计算 Recall 和 AP
    r1, mAP = calculate_university_scores(query_features, gallery_features, query_labels, gallery_labels, step_size=step_size, ranks=ranks)

    # 清理显存
    if cleanup:
        del query_features, query_labels, gallery_features, gallery_labels
        gc.collect()
        
    return r1

def calculate_university_scores(query_features, gallery_features, query_labels, gallery_labels, step_size=1000, ranks=[1, 5, 10]):
    """
    针对 University-1652 的评价逻辑，支持多对多匹配
    """
    Q = len(query_features)
    G = len(gallery_features)
    
    # 转换为 numpy 以便处理
    query_labels = query_labels.cpu().numpy()
    gallery_labels = gallery_labels.cpu().numpy()
    
    # 计算相似度矩阵 (Q x G)
    steps = Q // step_size + 1
    similarity = []
    for i in range(steps):
        start = step_size * i
        end = min(start + step_size, Q)
        if start >= end: continue
        
        # 计算余弦相似度
        sim_tmp = query_features[start:end] @ gallery_features.T
        similarity.append(sim_tmp.cpu())
    
    similarity = torch.cat(similarity, dim=0).numpy()

    # 评价指标初始化
    recall = np.zeros(len(ranks))
    ap = 0.0
    
    bar = tqdm(range(Q), desc="Evaluating")
    for i in bar:
        # 获取当前查询的标签和相似度排位
        q_label = query_labels[i]
        sim_vector = similarity[i]
        
        # 排除无效标签 (如果有)
        if q_label == -1:
            continue

        # 降序排列索引
        index = np.argsort(sim_vector)[::-1]
        
        # 寻找匹配的索引（Gallery 中所有与 Query 标签相同的项）
        good_index = np.where(gallery_labels == q_label)[0]
        
        # 计算 AP 和 Recall
        ap_tmp, recall_tmp = compute_ap_recall(index, good_index, ranks)
        ap += ap_tmp
        recall += recall_tmp

    # 平均值计算
    recall = (recall / Q) * 100
    mAP = (ap / Q) * 100
    
    # 打印结果
    result_str = []
    for j in range(len(ranks)):
        result_str.append('Recall@{}: {:.2f}'.format(ranks[j], recall[j]))
    result_str.append('mAP: {:.2f}'.format(mAP))
    print(' - '.join(result_str))

    return recall[0], mAP

def compute_ap_recall(index, good_index, ranks):
    """
    计算单次查询的 AP 和 Recall
    """
    # 检查排序结果中哪些是正确的
    mask = np.in1d(index, good_index)
    
    # 计算 Recall
    recall = np.zeros(len(ranks))
    for j, k in enumerate(ranks):
        if np.any(mask[:k]):
            recall[j] = 1
            
    # 计算 AP
    rows_good = len(good_index)
    if rows_good == 0:
        return 0, recall
    
    # 找到所有匹配项在排序中的位置 (从1开始)
    hit_pos = np.where(mask)[0] + 1
    
    # 计算 Precision 序列并累加
    ap = 0.0
    for j in range(rows_good):
        # Precision = (当前已找到的正确数) / (当前总排序位置)
        precision = (j + 1) / hit_pos[j]
        ap += precision
    ap /= rows_good
    
    return ap, recall