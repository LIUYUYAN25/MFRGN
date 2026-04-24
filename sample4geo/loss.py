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