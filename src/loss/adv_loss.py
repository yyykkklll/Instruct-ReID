import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedLoss(nn.Module):
    """
    高级损失函数模块，包含 InfoNCE、身份分类、生物对比、衣物对比和衣物对抗损失
    """

    def __init__(self, temperature=0.07, weights=None):
        """
        初始化损失函数模块

        Args:
            temperature (float): InfoNCE 损失的温度参数，默认为 0.07
            weights (dict): 损失权重，包含 'info_nce', 'cls', 'bio', 'cloth', 'cloth_adv'，默认从配置加载
        """
        super().__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()  # 交叉熵损失
        self.id_embed_projector = nn.Linear(768, 256)  # 投影 id_embeds 到 256 维
        self.cloth_embed_projector = nn.Linear(768, 256)  # 投影 cloth_embeds 到 256 维
        self.weights = weights if weights is not None else {
            'info_nce': 1.0,
            'cls': 0.5,
            'bio': 0.1,
            'cloth': 0.5,
            'cloth_adv': 0.5
        }

    def info_nce_loss(self, image_embeds, text_embeds):
        """
        计算 InfoNCE 损失，用于图像-文本匹配
        """
        batch_size = image_embeds.size(0)
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        logits = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
        labels = torch.arange(batch_size, device=image_embeds.device)
        loss_i2t = self.ce_loss(logits, labels)
        loss_t2i = self.ce_loss(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2

    def id_classification_loss(self, id_logits, pids):
        """
        计算身份分类损失（交叉熵）
        """
        return self.ce_loss(id_logits, pids)

    def bio_contrastive_loss(self, id_embeds, text_embeds):
        """
        计算生物对比损失，对齐身份特征和文本特征
        """
        batch_size = id_embeds.size(0)
        id_embeds = F.normalize(self.id_embed_projector(id_embeds), dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        logits = torch.matmul(id_embeds, text_embeds.t()) / self.temperature
        labels = torch.arange(batch_size, device=id_embeds.device)
        return self.ce_loss(logits, labels)

    def cloth_contrastive_loss(self, cloth_embeds, cloth_text_embeds):
        """
        计算衣物对比损失，对齐图像衣物特征和衣物文本特征
        """
        if cloth_embeds is None or cloth_text_embeds is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        batch_size = cloth_embeds.size(0)
        cloth_embeds = F.normalize(self.cloth_embed_projector(cloth_embeds), dim=-1)
        cloth_text_embeds = F.normalize(cloth_text_embeds, dim=-1)
        logits = torch.matmul(cloth_embeds, cloth_text_embeds.t()) / self.temperature
        labels = torch.arange(batch_size, device=cloth_embeds.device)
        return self.ce_loss(logits, labels)

    def cloth_adversarial_loss(self, cloth_embeds, cloth_text_embeds):
        """
        计算衣物对抗损失，最大化衣物特征与文本特征的差异（通过 GRL 实现）
        """
        if cloth_embeds is None or cloth_text_embeds is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        batch_size = cloth_embeds.size(0)
        cloth_embeds = F.normalize(self.cloth_embed_projector(cloth_embeds), dim=-1)
        cloth_text_embeds = F.normalize(cloth_text_embeds, dim=-1)
        # 计算负对数似然，鼓励特征分散（对抗目标）
        logits = torch.matmul(cloth_embeds, cloth_text_embeds.t()) / self.temperature
        labels = torch.arange(batch_size, device=cloth_embeds.device)
        # 负损失：最大化差异，等效于最小化负的 InfoNCE 损失
        return -self.ce_loss(logits, labels)

    def forward(self, image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                cloth_embeds, cloth_text_embeds, pids):
        """
        前向传播，计算总损失

        Args:
            image_embeds: 图像特征，形状 [batch_size, feat_dim]
            id_text_embeds: 身份文本特征，形状 [batch_size, feat_dim]
            fused_embeds: 融合特征，形状 [batch_size, feat_dim]
            id_logits: 身份分类 logits，形状 [batch_size, num_classes]
            id_embeds: 身份特征，形状 [batch_size, text_width]
            cloth_embeds: 衣物特征，形状 [batch_size, text_width]
            cloth_text_embeds: 衣物文本特征，形状 [batch_size, feat_dim]
            pids: 身份标签，形状 [batch_size]

        Returns:
            dict: 包含各损失分量和总损失
        """
        losses = {}
        losses['info_nce'] = self.info_nce_loss(image_embeds, id_text_embeds) if image_embeds is not None and id_text_embeds is not None else 0.0
        losses['cls'] = self.id_classification_loss(id_logits, pids) if id_logits is not None and pids is not None else 0.0
        losses['bio'] = self.bio_contrastive_loss(id_embeds, id_text_embeds) if id_embeds is not None and id_text_embeds is not None else 0.0
        losses['cloth'] = self.cloth_contrastive_loss(cloth_embeds, cloth_text_embeds)
        losses['cloth_adv'] = self.cloth_adversarial_loss(cloth_embeds, cloth_text_embeds)

        total_loss = sum(self.weights[k] * losses[k] for k in self.weights)
        losses['total'] = total_loss
        return losses
