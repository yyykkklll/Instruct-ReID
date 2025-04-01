import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

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
        # 确保输入形状为 [batch_size, feat_dim]
        image_features = F.normalize(image_features, dim=-1)  # [batch_size, feat_dim], e.g., [128, 768]
        text_features = F.normalize(text_features, dim=-1)  # [batch_size, feat_dim], e.g., [128, 768]
        logits = torch.matmul(image_features,
                              text_features.mT) / self.temperature  # [batch_size, batch_size], e.g., [128, 128]
        labels = torch.arange(image_features.shape[0], device=image_features.device)
        # 双向 InfoNCE 损失
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.mT, labels)
        return (loss_i2t + loss_t2i) / 2


class CosFaceLoss(nn.Module):
    """CosFace Loss based on the predictions of classifier.
    Reference:
        Wang et al. CosFace: Large Margin Cosine Loss for Deep Face Recognition. In CVPR, 2018.

    Args:
        scale (float): scaling factor.
        margin (float): pre-defined margin.
    """

    def __init__(self, scale=16, margin=0.1, **kwargs):
        super().__init__()
        self.s = scale
        self.m = margin

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        one_hot = torch.zeros_like(inputs)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)
        output = self.s * (inputs - one_hot * self.m)
        return F.cross_entropy(output, targets)
