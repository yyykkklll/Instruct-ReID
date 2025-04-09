from __future__ import absolute_import

from .adv_loss import InfoNCELoss, CLIPLoss, TripletLoss, CombinedLoss

__all__ = [
    'InfoNCELoss',
    'CLIPLoss',
    'TripletLoss',
    'CombinedLoss'
]
