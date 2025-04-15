import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # (N, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (N, C, H, W)

        probs = probs.contiguous().view(probs.size(0), probs.size(1), -1)
        targets_one_hot = targets_one_hot.contiguous().view(
            targets_one_hot.size(0), targets_one_hot.size(1), -1
        )

        intersection = (probs * targets_one_hot).sum(dim=2)
        union = probs.sum(dim=2) + targets_one_hot.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice.mean()

        return loss


class SmoothDiceLoss(DiceLoss):
    def forward(self, logits, targets):
        return torch.log(torch.cosh(super().forward(logits, targets)))
