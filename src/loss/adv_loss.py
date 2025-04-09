import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """跨进程收集张量，支持反向传播"""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        if image_features is None or text_features is None:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu', requires_grad=True)

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
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        if image_features is None or text_features is None:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu', requires_grad=True)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
        logits_per_text = logits_per_image.t()
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        return (loss_img + loss_txt) / 2


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, image_features, text_features, labels):
        if image_features is None or text_features is None:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu', requires_grad=True)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        dist_mat = torch.cdist(image_features, text_features, p=2)
        positive_dist = dist_mat.diag()
        mask = labels.expand(len(labels), len(labels)).eq(labels.expand(len(labels), len(labels)).t())
        negative_dist = dist_mat.masked_fill(mask, float('inf')).min(dim=1)[0]
        loss = F.relu(positive_dist - negative_dist + self.margin).mean()
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, temperature_info=0.07, temperature_clip=0.1, margin_triplet=0.3, weights=(1.0, 1.0, 1.0)):
        super().__init__()
        self.info_nce = InfoNCELoss(temperature=temperature_info)
        self.clip_loss = CLIPLoss(temperature=temperature_clip)
        self.triplet_loss = TripletLoss(margin=margin_triplet)
        self.weights = weights

    def forward(self, image_features, text_features, labels):
        if image_features is None or text_features is None or labels is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                'info_nce_loss': zero_loss,
                'clip_loss': zero_loss,
                'triplet_loss': zero_loss,
                'total_loss': zero_loss
            }

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