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

class InfoNCEWithEdge(nn.Module):
    def __init__(self, loss_function, margin=0.1, edge_weight=0.1, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        在 InfoNCEMargin 基础上增加 Sobel 边缘约束。
        edge_weight: 边缘损失的权重，建议初值设为 0.1。
        """
        super().__init__()
        self.loss_function = loss_function
        self.margin = margin
        self.edge_weight = edge_weight
        self.device = device
        
        # 初始化 Sobel 算子
        kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        self.kernel_x = torch.FloatTensor(kernel_x).expand(1, 1, 3, 3).to(device)
        self.kernel_y = torch.FloatTensor(kernel_y).expand(1, 1, 3, 3).to(device)

    def get_sobel_edge(self, x):
        # 如果输入是 RGB (B, 3, H, W)，先转为灰度图
        if x.shape[1] == 3:
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        grad_x = F.conv2d(x, self.kernel_x, padding=1)
        grad_y = F.conv2d(x, self.kernel_y, padding=1)
        edge = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        return edge

    def forward(self, image_features1, image_features2, logit_scale, raw_img1=None, raw_img2=None):
        """
        image_features: 模型输出的全局特征向量
        raw_img: 原始输入图像，用于计算 Sobel 边缘 (需在 trainer.py 中传入)
        """
        # 1. 计算原有的 InfoNCEMargin 损失
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        sim_matrix = image_features1 @ image_features2.T
        margin_matrix = torch.zeros_like(sim_matrix)
        margin_matrix.fill_diagonal_(self.margin)
        
        sim_matrix_margin = sim_matrix - margin_matrix
        logits_per_image1 = logit_scale * sim_matrix_margin
        logits_per_image2 = logit_scale * sim_matrix_margin.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        contrastive_loss = (self.loss_function(logits_per_image1, labels) + 
                            self.loss_function(logits_per_image2, labels)) / 2

        # 2. 计算边缘对齐辅助损失 (如果传入了原始图像)
        edge_loss = 0.0
        if raw_img1 is not None and raw_img2 is not None:
            edge1 = self.get_sobel_edge(raw_img1)
            edge2 = self.get_sobel_edge(raw_img2)
            # 约束两个模态的边缘图在空间布局上的一致性
            edge_loss = F.mse_loss(edge1, edge2)

        # 3. 总损失加权
        total_loss = contrastive_loss + self.edge_weight * edge_loss
        return total_loss

class MultiSimilarityLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=50.0, base=0.5, margin=0.1):
        """
        Multi-Similarity Loss (对称/双向版本)
        alpha: 控制正样本惩罚的尺度 (建议 2.0)
        beta: 控制负样本惩罚的尺度 (建议 40.0 - 50.0)
        base: 相似度基准值 (建议 0.5)
        margin: 困难负样本挖掘的边界，只有相似度大于 (正样本相似度 - margin) 的负样本才会产生 Loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.margin = margin

    def forward(self, features1, features2, logit_scale=None, **kwargs):
        """
        利用 logit_scale=None 和 **kwargs 吸收掉 trainer.py 传过来的额外参数(如 raw_img)，
        这样你就不需要去改动 trainer.py 里的前向传播逻辑了。
        """
        # L2 归一化特征
        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)

        # 计算 Batch 内的相似度矩阵 (B, B)
        sim_mat = features1 @ features2.T
        
        # 内部函数：计算单向的 MS Loss
        def compute_ms_loss(sim_matrix):
            loss = 0.0
            batch_size = sim_matrix.size(0)
            
            for i in range(batch_size):
                # 1. 取出正样本相似度 (对角线元素)
                pos_pair_sim = sim_matrix[i, i]
                
                # 2. 取出负样本相似度 (非对角线元素)
                neg_pair_sim = torch.cat([sim_matrix[i, :i], sim_matrix[i, i+1:]])

                # 3. MS Loss 核心机制：只保留“有威胁的”困难负样本
                # 条件：负样本的相似度 + margin > 正样本的相似度
                neg_stats = neg_pair_sim[neg_pair_sim + self.margin > pos_pair_sim]
                
                # 4. 计算正样本 Loss 分量
                pos_loss = 1.0 / self.alpha * torch.log(1 + torch.exp(-self.alpha * (pos_pair_sim - self.base)))
                
                # 5. 计算负样本 Loss 分量 (指数加权，越相似的负样本惩罚越重)
                if len(neg_stats) > 0:
                    neg_loss = 1.0 / self.beta * torch.log(1 + torch.sum(torch.exp(self.beta * (neg_stats - self.base))))
                else:
                    neg_loss = 0.0
                    
                loss += (pos_loss + neg_loss)
                
            return loss / batch_size

        # 计算双向 Loss 并求平均 (UAV检索Sat，以及Sat检索UAV)
        loss_i2t = compute_ms_loss(sim_mat)        # features1 -> features2
        loss_t2i = compute_ms_loss(sim_mat.T)      # features2 -> features1
        
        return (loss_i2t + loss_t2i) / 2.0