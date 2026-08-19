"""Loss functions for imbalanced segmentation."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-5, squared: bool = True) -> None:
        super().__init__()
        self.smooth = smooth
        self.squared = squared

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        if self.squared:
            intersection = (probs * targets).sum(dim=dims)
            denominator = (probs**2).sum(dim=dims) + (targets**2).sum(dim=dims)
        else:
            intersection = (probs * targets).sum(dim=dims)
            denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """Combined Dice + weighted BCE loss for imbalanced binary segmentation."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0, bce_pos_weight: float | None = None) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss()
        pos_weight = torch.tensor([bce_pos_weight]) if bce_pos_weight is not None else None
        self.bce: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.dice_weight * self.dice(logits, targets) + self.bce_weight * self.bce(logits, targets)


class FocalLoss(nn.Module):
    """Focal loss for binary segmentation to address class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = torch.where(targets > 0.5, probs, 1 - probs)
        alpha_t = torch.where(targets > 0.5, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()
