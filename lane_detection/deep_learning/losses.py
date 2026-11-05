"""Loss (how wrong) and IoU (how much overlap with the answer sheet)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits, targets, eps: float = 1.0):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    total = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + eps) / (total + eps)).mean()


class BCEDiceLoss(nn.Module):
    """Pixel-by-pixel error (BCE) + overlap error (Dice)."""

    def __init__(self, pos_weight: float = 5.0):
        super().__init__()
        # Lane pixels are RARE (thin lines), so we make mistakes on them count more.
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight)
        return bce + dice_loss(logits, targets)


@torch.no_grad()
def batch_iou(logits, targets, thr: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * targets).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - inter
    return ((inter + 1e-6) / (union + 1e-6)).mean().item()
