from __future__ import absolute_import

from .adv_loss import InfoNCELoss, CosFaceLoss  # 移除 ClothesBasedAdversarialLoss

__all__ = [
    'InfoNCELoss',
    'CosFaceLoss'
]
