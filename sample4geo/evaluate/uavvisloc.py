"""
sample4geo/evaluate/uavvisloc.py
================================
"""

import os
import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from ..trainer import predict, predict_dual

def evaluate(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             is_autocast=True,
             is_dual=False,
             cleanup=True,
             save_csv=None):
    
    print("\nExtract Features:")
    s1 = time.time()
    
    # 提取特征
    if is_dual:
        query_features, reference_features, query_labels, reference_labels = predict_dual(
            config, model, query_dataloader, reference_dataloader, is_autocast=is_autocast)
    else:
        query_features, query_labels = predict(
            config, model, query_dataloader, is_autocast=is_autocast)
        reference_features, reference_labels = predict(
            config, model, reference_dataloader, is_autocast=is_autocast) 
        
    if save_csv is not None:
        print('Saving features to .csv ...')
        features_q = np.array(query_features.detach().cpu())
        features_r = np.array(reference_features.detach().cpu())
        os.makedirs('results', exist_ok=True)
        np.savetxt(f"results/features_q_{save_csv}.csv", features_q, delimiter=",")
        np.savetxt(f"results/features_r_{save_csv}.csv", features_r, delimiter=",")

    s2 = time.time()
    print('Extract Features time: {:.2f} s'.format(s2 - s1))
    
    print("\nCompute Scores:")
    s = time.time()
    # 获取计算的各项 Recall 结果
    r1, r5, r10, r_top1 = calculate_scores(
        query_features, reference_features, query_labels, reference_labels, 
        step_size=step_size, ranks=ranks)
    e = time.time()
    print('Compute Scores time: {:.2f} s\n'.format(e - s))

    # 释放显存
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        torch.cuda.empty_cache()
    
    # 修改：返回所有 Recall 值，以修复 train_uavvisloc.py 中变量未定义的问题
    return r1, r5, r10, r_top1

def calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):
    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    # 创建字典，将 label 映射到 reference 矩阵的索引
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    similarity = []
    
    # 分块计算余弦相似度矩阵，避免 OOM
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T
        similarity.append(sim_tmp.cpu())
     
    # 得到完整的 Q x R 相似度矩阵
    similarity = torch.cat(similarity, dim=0)
    
    # 追加 top 1% 对应的序号 (R // 100)
    topk.append(max(1, R // 100))
    
    results = np.zeros([len(topk)])
    
    bar = tqdm(range(Q), ncols=100, position=0, leave=True)
    
    for i in bar:
        # 获取真值对 (Ground Truth) 的相似度
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        
        # 统计相似度比真值对还高的数量 (即排名)
        higher_sim = similarity[i,:] > gt_sim
        ranking = higher_sim.sum()
        
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
                        
    results = results / Q * 100.
    bar.close()
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
    string.append('Recall@top1%: {:.4f}'.format(results[-1]))            
        
    print(' - '.join(string)) 

    return results[0], results[1], results[2], results[-1]