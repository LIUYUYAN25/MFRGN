import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn

class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

        return loss  
 


class InfoNCEMargin(nn.Module):
    def __init__(self, loss_function, margin=0.1, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        引入类似 CosFace 的 Margin 机制。
        margin: 角度裕度（建议初值设为 0.1 - 0.2 之间，过大会导致模型难以收敛）
        """
        super().__init__()
        self.loss_function = loss_function
        self.margin = margin
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        # 1. 归一化特征，使点乘结果即为余弦相似度
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        # 2. 计算 Batch 内的相似度矩阵 (B x B)
        sim_matrix = image_features1 @ image_features2.T
        
        # 3. 构造 Margin 矩阵：只在对角线（正样本对）上施加惩罚
        # 为了避免 in-place 操作引发的梯度回传报错，使用构造新矩阵相减的方式
        margin_matrix = torch.zeros_like(sim_matrix)
        margin_matrix.fill_diagonal_(self.margin)
        
        # 对正样本的相似度减去 margin
        sim_matrix_margin = sim_matrix - margin_matrix
        
        # 4. 乘以可学习的温度系数 (logit_scale)
        logits_per_image1 = logit_scale * sim_matrix_margin
        
        # 对于对称损失，转置矩阵
        logits_per_image2 = logit_scale * sim_matrix_margin.T
        
        # 5. 计算对称交叉熵损失
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss = (self.loss_function(logits_per_image1, labels) + 
                self.loss_function(logits_per_image2, labels)) / 2

        return loss