import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedLoss(nn.Module):
    """
    高级损失模块，集成了多种损失函数，用于多模态学习任务。
    """
    def __init__(self, temperature=0.1, weights=None):
        """
        初始化 AdvancedLoss 模块。

        Args:
            temperature (float): InfoNCE 损失和对比损失中的温度参数。
            weights (dict, optional): 一个字典，包含各种损失组件的权重。
                                      如果为 None，则使用默认权重。
        """
        super().__init__()
        self.temperature = temperature  # 温度参数，用于缩放 logits
        self.ce_loss = nn.CrossEntropyLoss()  # 交叉熵损失函数
        # 线性投影层，用于将 ID 嵌入投影到较低维度空间
        self.id_embed_projector = nn.Linear(768, 256)
        # 线性投影层，用于将衣物嵌入投影到较低维度空间
        self.cloth_embed_projector = nn.Linear(768, 256)
        # 初始化损失权重，如果未提供则使用默认值
        self.weights = weights if weights is not None else {
            'info_nce': 1.0,  # InfoNCE 损失权重
            'cls': 1.0,       # ID 分类损失权重
            'bio': 0.5,       # 生物特征（ID）对比损失权重
            'cloth': 0.5,     # 衣物对比损失权重
            'cloth_adv': 0.1, # 衣物对抗损失权重
            'cloth_match': 1.0, # 衣物匹配损失权重
            'decouple': 0.2,  # 解耦损失权重，提高权重以支持交叉注意力
            'gate_regularization': 0.01  # 门控正则化损失权重
        }

    def gate_regularization_loss(self, gate):
        """
        计算门控正则化损失，鼓励门控值分布平衡，使其接近0.5。
        这有助于确保解耦模块的两个分支（例如，身份和衣物）得到同等程度的关注。

        Args:
            gate (Tensor): 门控向量，形状通常为 [batch_size, dim] 或 [batch_size, 1]。

        Returns:
            Tensor: 计算得到的均方误差损失值。
        """
        # 创建一个与 gate 形状相同，值全为 0.5 的目标张量
        target = torch.full_like(gate, 0.5)
        # 计算门控向量与目标张量之间的均方误差损失
        return F.mse_loss(gate, target)

    def info_nce_loss(self, image_embeds, text_embeds):
        """
        计算 InfoNCE (Noise Contrastive Estimation) 损失。
        用于对齐图像嵌入和文本嵌入。

        Args:
            image_embeds (Tensor): 图像嵌入，形状 [batch_size, embed_dim]。
            text_embeds (Tensor): 文本嵌入，形状 [batch_size, embed_dim]。

        Returns:
            Tensor: 计算得到的 InfoNCE 损失。
        """
        batch_size = image_embeds.size(0)
        # 对嵌入进行 L2 归一化
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        # 计算图像嵌入和文本嵌入之间的相似度矩阵，并用温度参数缩放
        logits = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
        # 创建标签，对于匹配的样本，标签为对角线索引
        labels = torch.arange(batch_size, device=image_embeds.device)
        # 计算图像到文本的损失
        loss_i2t = self.ce_loss(logits, labels)
        # 计算文本到图像的损失
        loss_t2i = self.ce_loss(logits.t(), labels)
        # 返回平均损失
        return (loss_i2t + loss_t2i) / 2

    def id_classification_loss(self, id_logits, pids):
        """
        计算 ID 分类损失。

        Args:
            id_logits (Tensor): ID 分类的 logits 输出，形状 [batch_size, num_classes]。
            pids (Tensor): 真实的行人 ID 标签，形状 [batch_size]。

        Returns:
            Tensor: 计算得到的交叉熵损失。
        """
        return self.ce_loss(id_logits, pids)

    def bio_contrastive_loss(self, id_embeds, text_embeds):
        """
        计算生物特征（ID）相关的对比损失。
        用于对齐 ID 嵌入和相关的文本嵌入。

        Args:
            id_embeds (Tensor): ID 嵌入，形状 [batch_size, embed_dim]。
            text_embeds (Tensor): 相关的文本嵌入，形状 [batch_size, embed_dim]。

        Returns:
            Tensor: 计算得到的对比损失。
        """
        batch_size = id_embeds.size(0)
        # 投影并归一化 ID 嵌入
        id_embeds = F.normalize(self.id_embed_projector(id_embeds), dim=-1)
        # 归一化文本嵌入
        text_embeds = F.normalize(text_embeds, dim=-1)
        # 计算 ID 嵌入和文本嵌入之间的相似度矩阵，并用温度参数缩放
        logits = torch.matmul(id_embeds, text_embeds.t()) / self.temperature
        # 创建标签
        labels = torch.arange(batch_size, device=id_embeds.device)
        return self.ce_loss(logits, labels)

    def cloth_contrastive_loss(self, cloth_embeds, cloth_text_embeds):
        """
        计算衣物相关的对比损失。
        用于对齐衣物嵌入和相关的衣物文本嵌入。

        Args:
            cloth_embeds (Tensor): 衣物嵌入，形状 [batch_size, embed_dim]。
            cloth_text_embeds (Tensor): 衣物相关的文本嵌入，形状 [batch_size, embed_dim]。

        Returns:
            Tensor: 计算得到的对比损失。如果输入为 None，则返回 0。
        """
        # 如果衣物嵌入或衣物文本嵌入为空，则返回 0 损失
        if cloth_embeds is None or cloth_text_embeds is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        batch_size = cloth_embeds.size(0)
        # 投影并归一化衣物嵌入
        cloth_embeds = F.normalize(self.cloth_embed_projector(cloth_embeds), dim=-1)
        # 归一化衣物文本嵌入
        cloth_text_embeds = F.normalize(cloth_text_embeds, dim=-1)
        # 计算衣物嵌入和衣物文本嵌入之间的相似度矩阵，并用温度参数缩放
        logits = torch.matmul(cloth_embeds, cloth_text_embeds.t()) / self.temperature
        # 创建标签
        labels = torch.arange(batch_size, device=cloth_embeds.device)
        return self.ce_loss(logits, labels)

    def cloth_adversarial_loss(self, cloth_embeds, cloth_text_embeds, epoch=None):
        """
        计算衣物相关的对抗性损失。
        目标是使得衣物嵌入与其不匹配的文本嵌入之间的相似度降低。

        Args:
            cloth_embeds (Tensor): 衣物嵌入。
            cloth_text_embeds (Tensor): 衣物文本嵌入。
            epoch (int, optional): 当前训练周期，用于调整对抗性权重。

        Returns:
            Tensor: 计算得到的对抗性损失。如果输入为 None，则返回 0。
        """
        # 如果衣物嵌入或衣物文本嵌入为空，则返回 0 损失
        if cloth_embeds is None or cloth_text_embeds is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        batch_size = cloth_embeds.size(0)
        # 投影并归一化衣物嵌入
        cloth_embeds = F.normalize(self.cloth_embed_projector(cloth_embeds), dim=-1)
        # 归一化衣物文本嵌入
        cloth_text_embeds = F.normalize(cloth_text_embeds, dim=-1)
        # 计算相似度矩阵
        logits = torch.matmul(cloth_embeds, cloth_text_embeds.t()) / self.temperature
        # 创建标签
        labels = torch.arange(batch_size, device=cloth_embeds.device)
        # 计算负样本的 logits，将对角线（正样本）元素置零
        neg_logits = logits - torch.diag(torch.diagonal(logits))
        # 计算负样本的 log_softmax 损失的均值，并取负，以最大化负样本的预测概率（即最小化其相似度）
        neg_loss = -F.log_softmax(neg_logits, dim=1).mean()
        # 如果提供了 epoch，则动态调整对抗性权重
        if epoch is not None:
            adv_weight = min(1.0, 0.2 + epoch * 0.05) # 权重随 epoch 增加而增加，但最大不超过 1.0
            neg_loss = adv_weight * neg_loss
        return neg_loss

    def compute_cloth_matching_loss(self, cloth_image_embeds, cloth_text_embeds, is_matched):
        """
        计算衣物匹配损失，类似于 InfoNCE，但专门用于衣物图像和文本嵌入。
        `is_matched` 参数在这里没有直接使用，因为损失的构造方式是基于批内正负样本对。

        Args:
            cloth_image_embeds (Tensor): 衣物图像嵌入。
            cloth_text_embeds (Tensor): 衣物文本嵌入。
            is_matched (Tensor, optional): 指示图像和文本是否匹配的布尔张量。当前实现未使用。

        Returns:
            Tensor: 计算得到的衣物匹配损失。如果输入为 None，则返回 0。
        """
        # 如果任何一个输入为空，则返回 0 损失
        if cloth_image_embeds is None or cloth_text_embeds is None or is_matched is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        batch_size = cloth_image_embeds.size(0)
        # 归一化嵌入
        cloth_image_embeds = F.normalize(cloth_image_embeds, dim=-1)
        cloth_text_embeds = F.normalize(cloth_text_embeds, dim=-1)
        # 计算相似度矩阵
        sim = torch.matmul(cloth_image_embeds, cloth_text_embeds.t()) / self.temperature
        # 创建标签
        labels = torch.arange(batch_size, device=cloth_image_embeds.device)
        # 计算图像到文本的损失
        loss_i2t = self.ce_loss(sim, labels)
        # 计算文本到图像的损失
        loss_t2i = self.ce_loss(sim.t(), labels)
        # 返回平均损失
        return (loss_i2t + loss_t2i) / 2

    def compute_decoupling_loss(self, id_embeds, cloth_embeds):
        """
        计算解耦损失，使用 Hilbert-Schmidt Independence Criterion (HSIC) 的一个变体。
        目标是使 ID 嵌入和衣物嵌入之间的相关性最小化，以促进特征解耦。

        Args:
            id_embeds (Tensor): ID 嵌入。
            cloth_embeds (Tensor): 衣物嵌入。

        Returns:
            Tensor: 计算得到的 HSIC 解耦损失。如果输入为 None，则返回 0。
        """
        # 如果任何一个输入为空，则返回 0 损失
        if id_embeds is None or cloth_embeds is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        # 投影并归一化 ID 嵌入
        id_embeds = F.normalize(self.id_embed_projector(id_embeds), dim=-1)
        # 投影并归一化衣物嵌入
        cloth_embeds = F.normalize(self.cloth_embed_projector(cloth_embeds), dim=-1)
        batch_size = id_embeds.size(0)
        # 计算 ID 嵌入的核矩阵 (Gram matrix)
        id_kernel = torch.matmul(id_embeds, id_embeds.t())
        # 计算衣物嵌入的核矩阵
        cloth_kernel = torch.matmul(cloth_embeds, cloth_embeds.t())
        # 将核矩阵的对角线元素置零，以排除自身与自身的比较
        id_kernel = id_kernel - torch.diag(torch.diagonal(id_kernel))
        cloth_kernel = cloth_kernel - torch.diag(torch.diagonal(cloth_kernel))
        # 计算 HSIC：两个中心化核矩阵的点积的均值
        # (batch_size - 1) 用于无偏估计
        hsic = torch.mean(id_kernel * cloth_kernel) / (batch_size - 1) if batch_size > 1 else torch.tensor(0.0, device=id_embeds.device)
        return hsic

    def forward(self, image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                cloth_embeds, cloth_text_embeds, cloth_image_embeds, pids, is_matched, epoch=None, gate=None):
        """
        前向传播函数，计算所有定义的损失。

        Args:
            image_embeds (Tensor): 整体图像嵌入。
            id_text_embeds (Tensor): 与 ID 相关的文本嵌入。
            fused_embeds (Tensor): 融合后的嵌入 (当前未使用)。
            id_logits (Tensor): ID 分类的 logits。
            id_embeds (Tensor): 从图像中提取的 ID 特征嵌入。
            cloth_embeds (Tensor): 从图像中提取的衣物特征嵌入。
            cloth_text_embeds (Tensor): 与衣物相关的文本嵌入。
            cloth_image_embeds (Tensor): 衣物的图像嵌入 (可能与整体图像嵌入不同，例如裁剪后的衣物区域)。
            pids (Tensor): 行人 ID 标签。
            is_matched (Tensor): 指示图像-文本对是否匹配的布尔张量。
            epoch (int, optional): 当前训练周期，用于某些损失的动态调整。
            gate (Tensor, optional): 从解耦模块输出的门控向量，用于门控正则化损失。

        Returns:
            dict: 包含所有计算得到的损失及其加权总和的字典。
        """
        losses = {} # 初始化损失字典
        # 计算各项损失，如果相关输入为 None，则损失值为 0.0
        losses['info_nce'] = self.info_nce_loss(image_embeds, id_text_embeds) if image_embeds is not None and id_text_embeds is not None else torch.tensor(0.0, device=next(self.parameters()).device)
        losses['cls'] = self.id_classification_loss(id_logits, pids) if id_logits is not None and pids is not None else torch.tensor(0.0, device=next(self.parameters()).device)
        losses['bio'] = self.bio_contrastive_loss(id_embeds, id_text_embeds) if id_embeds is not None and id_text_embeds is not None else torch.tensor(0.0, device=next(self.parameters()).device)
        losses['cloth'] = self.cloth_contrastive_loss(cloth_embeds, cloth_text_embeds)
        losses['cloth_adv'] = self.cloth_adversarial_loss(cloth_embeds, cloth_text_embeds, epoch)
        losses['cloth_match'] = self.compute_cloth_matching_loss(cloth_image_embeds, cloth_text_embeds, is_matched)
        losses['decouple'] = self.compute_decoupling_loss(id_embeds, cloth_embeds)
        losses['gate_regularization'] = self.gate_regularization_loss(gate) if gate is not None else torch.tensor(0.0, device=next(self.parameters()).device)

        # 计算加权总损失
        total_loss = sum(self.weights[k] * v for k, v in losses.items() if isinstance(v, torch.Tensor) and k in self.weights)
        losses['total'] = total_loss
        return losses