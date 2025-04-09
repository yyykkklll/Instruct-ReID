import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all processes, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        if dist.is_initialized():
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logits = torch.matmul(image_features, text_features.t()) / self.temperature
        labels = torch.arange(image_features.shape[0], device=image_features.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2


class CLIPLoss(nn.Module):
    """CLIP-style contrastive loss with symmetric image-text alignment."""

    def __init__(self, temperature=0.1):
        super(CLIPLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 计算相似度矩阵
        logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
        logits_per_text = logits_per_image.t()

        # 标签：对角线上的正样本
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)

        # 双向损失
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        return (loss_img + loss_txt) / 2


class TripletLoss(nn.Module):
    """Triplet loss with hard negative mining."""

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, image_features, text_features, labels):
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 计算距离矩阵
        dist_mat = torch.cdist(image_features, text_features, p=2)

        # 构建三元组
        batch_size = labels.shape[0]
        positive_dist = dist_mat.diag()  # 正样本距离
        mask = labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t())

        # 负样本距离（最硬的负样本）
        negative_dist = dist_mat.masked_fill(mask, float('inf')).min(dim=1)[0]

        # 三元组损失
        loss = F.relu(positive_dist - negative_dist + self.margin).mean()
        return loss


class CombinedLoss(nn.Module):
    """Combined loss with InfoNCE, CLIP, and Triplet components."""

    def __init__(self, temperature_info=0.07, temperature_clip=0.1, margin_triplet=0.3, weights=(1.0, 1.0, 1.0)):
        super(CombinedLoss, self).__init__()
        self.info_nce = InfoNCELoss(temperature=temperature_info)
        self.clip_loss = CLIPLoss(temperature=temperature_clip)
        self.triplet_loss = TripletLoss(margin=margin_triplet)
        self.weights = weights  # (info_nce_weight, clip_weight, triplet_weight)

    def forward(self, image_features, text_features, labels):
        info_nce_loss = self.info_nce(image_features, text_features)
        clip_loss = self.clip_loss(image_features, text_features)
        triplet_loss = self.triplet_loss(image_features, text_features, labels)

        total_loss = (self.weights[0] * info_nce_loss +
                      self.weights[1] * clip_loss +
                      self.weights[2] * triplet_loss)
        return {
            'info_nce_loss': info_nce_loss,
            'clip_loss': clip_loss,
            'triplet_loss': triplet_loss,
            'total_loss': total_loss
        }