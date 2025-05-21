import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedLoss(nn.Module):
    def __init__(self, temperature=0.1, weights=None):
        super().__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.id_embed_projector = nn.Linear(768, 256)
        self.cloth_embed_projector = nn.Linear(768, 256)
        self.weights = weights if weights is not None else {
            'info_nce': 1.0,
            'cls': 1.0,
            'bio': 0.5,
            'cloth': 0.5,
            'cloth_adv': 0.1,
            'cloth_match': 1.0,
            'decouple': 0.2,
            'gate_regularization': 0.01,
            'fusion_gate_regularization': 0.01  # 新增融合门控正则化
        }

    def gate_regularization_loss(self, gate):
        target = torch.full_like(gate, 0.5)
        return F.mse_loss(gate, target)

    def fusion_gate_regularization_loss(self, fusion_gate):
        target = torch.full_like(fusion_gate, 0.5)
        return F.mse_loss(fusion_gate, target)

    def info_nce_loss(self, image_embeds, text_embeds):
        batch_size = image_embeds.size(0)
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        logits = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
        labels = torch.arange(batch_size, device=image_embeds.device)
        loss_i2t = self.ce_loss(logits, labels)
        loss_t2i = self.ce_loss(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2

    def id_classification_loss(self, id_logits, pids):
        return self.ce_loss(id_logits, pids)

    def bio_contrastive_loss(self, id_embeds, text_embeds):
        batch_size = id_embeds.size(0)
        id_embeds = F.normalize(self.id_embed_projector(id_embeds), dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        logits = torch.matmul(id_embeds, text_embeds.t()) / self.temperature
        labels = torch.arange(batch_size, device=id_embeds.device)
        return self.ce_loss(logits, labels)

    def cloth_contrastive_loss(self, cloth_embeds, cloth_text_embeds):
        if cloth_embeds is None or cloth_text_embeds is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        batch_size = cloth_embeds.size(0)
        cloth_embeds = F.normalize(self.cloth_embed_projector(cloth_embeds), dim=-1)
        cloth_text_embeds = F.normalize(cloth_text_embeds, dim=-1)
        logits = torch.matmul(cloth_embeds, cloth_text_embeds.t()) / self.temperature
        labels = torch.arange(batch_size, device=cloth_embeds.device)
        return self.ce_loss(logits, labels)

    def cloth_adversarial_loss(self, cloth_embeds, cloth_text_embeds, epoch=None):
        if cloth_embeds is None or cloth_text_embeds is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        batch_size = cloth_embeds.size(0)
        cloth_embeds = F.normalize(self.cloth_embed_projector(cloth_embeds), dim=-1)
        cloth_text_embeds = F.normalize(cloth_text_embeds, dim=-1)
        logits = torch.matmul(cloth_embeds, cloth_text_embeds.t()) / self.temperature
        labels = torch.arange(batch_size, device=cloth_embeds.device)
        neg_logits = logits - torch.diag(torch.diagonal(logits))
        neg_loss = -F.log_softmax(neg_logits, dim=1).mean()
        if epoch is not None:
            adv_weight = min(1.0, 0.2 + epoch * 0.05)
            neg_loss = adv_weight * neg_loss
        return neg_loss

    def compute_cloth_matching_loss(self, cloth_image_embeds, cloth_text_embeds, is_matched):
        if cloth_image_embeds is None or cloth_text_embeds is None or is_matched is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        batch_size = cloth_image_embeds.size(0)
        cloth_image_embeds = F.normalize(cloth_image_embeds, dim=-1)
        cloth_text_embeds = F.normalize(cloth_text_embeds, dim=-1)
        sim = torch.matmul(cloth_image_embeds, cloth_text_embeds.t()) / self.temperature
        labels = torch.arange(batch_size, device=cloth_image_embeds.device)
        loss_i2t = self.ce_loss(sim, labels)
        loss_t2i = self.ce_loss(sim.t(), labels)
        return (loss_i2t + loss_t2i) / 2

    def compute_decoupling_loss(self, id_embeds, cloth_embeds):
        if id_embeds is None or cloth_embeds is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        id_embeds = F.normalize(self.id_embed_projector(id_embeds), dim=-1)
        cloth_embeds = F.normalize(self.cloth_embed_projector(cloth_embeds), dim=-1)
        batch_size = id_embeds.size(0)
        id_kernel = torch.matmul(id_embeds, id_embeds.t())
        cloth_kernel = torch.matmul(cloth_embeds, cloth_embeds.t())
        id_kernel = id_kernel - torch.diag(torch.diagonal(id_kernel))
        cloth_kernel = cloth_kernel - torch.diag(torch.diagonal(cloth_kernel))
        hsic = torch.mean(id_kernel * cloth_kernel) / (batch_size - 1)
        return hsic

    def forward(self, image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                cloth_embeds, cloth_text_embeds, cloth_image_embeds, pids, is_matched, epoch=None, 
                gate=None, fusion_gate=None):
        losses = {}
        losses['info_nce'] = self.info_nce_loss(image_embeds, id_text_embeds) if image_embeds is not None and id_text_embeds is not None else 0.0
        losses['cls'] = self.id_classification_loss(id_logits, pids) if id_logits is not None and pids is not None else 0.0
        losses['bio'] = self.bio_contrastive_loss(id_embeds, id_text_embeds) if id_embeds is not None and id_text_embeds is not None else 0.0
        losses['cloth'] = self.cloth_contrastive_loss(cloth_embeds, cloth_text_embeds)
        losses['cloth_adv'] = self.cloth_adversarial_loss(cloth_embeds, cloth_text_embeds, epoch)
        losses['cloth_match'] = self.compute_cloth_matching_loss(cloth_image_embeds, cloth_text_embeds, is_matched)
        losses['decouple'] = self.compute_decoupling_loss(id_embeds, cloth_embeds)
        losses['gate_regularization'] = self.gate_regularization_loss(gate) if gate is not None else 0.0
        losses['fusion_gate_regularization'] = self.fusion_gate_regularization_loss(fusion_gate) if fusion_gate is not None else 0.0

        total_loss = sum(self.weights[k] * losses[k] for k in self.weights)
        losses['total'] = total_loss
        return losses