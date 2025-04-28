import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (
    unary_from_softmax,
    create_pairwise_bilateral,
    create_pairwise_gaussian,
)


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


class SECLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=1.0, gamma=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).cuda()

    def seed_loss(self, pred_mask, seed_mask):
        """
        pred_mask: (N, C, H, W) - model output (after softmax)
        seed_mask: (N, C, H, W) - CAM-like seeds (only confident regions non-zero)
        """
        loss = -(seed_mask * torch.log(pred_mask + 1e-7)).sum(dim=1).mean()
        return loss

    def expand_loss(self, pred_mask, labels):
        """
        pred_mask: (N, C, H, W)
        labels: (N, C) - binary labels (1 if class is in image)
        """
        pred_max = pred_mask.view(pred_mask.size(0), pred_mask.size(1), -1).max(dim=2)[
            0
        ]  # (N, C)
        loss = F.cross_entropy(pred_max, labels)
        return loss

    def constrain_loss(self, pred_mask, image):
        """
        pred_mask: (N, C, H, W)
        image: (N, H, W, 3) numpy RGB
        """
        loss = 0
        for i in range(pred_mask.size(0)):
            probs = pred_mask[i].detach().cpu().numpy()
            crf_output = apply_dense_crf(denorm_image(image[i]), probs)
            crf_output = torch.tensor(crf_output).to(pred_mask.device)
            pred = pred_mask[i]
            loss += F.kl_div(pred.log(), crf_output, reduction="batchmean")
        return loss / pred_mask.size(0)

    def forward(
        self,
        pred_mask,
        seed_mask,
        image,
        labels,
    ):
        L_seed = self.seed_loss(pred_mask, seed_mask)
        L_expand = self.expand_loss(pred_mask, labels)
        L_constrain = self.constrain_loss(pred_mask, image)

        total_loss = (
            self.alpha * L_seed + self.beta * L_expand + self.gamma * L_constrain
        )
        return total_loss


def denorm_image(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).cuda()

    denormalized_image = image * std[:, None, None] + mean[:, None, None]
    denormalized_image = (
        denormalized_image * 255
    )  # Convert back to pixel range [0, 255]
    return (
        denormalized_image.permute(1, 2, 0).byte().cpu().numpy()
    )  # Convert to numpy for CRF


def apply_dense_crf(image, probs):
    """
    image: (H, W, 3)
    probs: (C, H, W)
    """
    H, W = image.shape[:2]
    d = dcrf.DenseCRF2D(W, H, probs.shape[0])
    unary = unary_from_softmax(probs)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image.copy(order="C"), compat=10)
    Q = d.inference(5)
    return np.array(Q).reshape((probs.shape[0], H, W))


def batched_cam_to_crf(cam, imgs, labels):
    out = torch.zeros(
        (
            imgs.size(0),
            imgs.size(2),
            imgs.size(3),
        ),
        dtype=torch.long,
    ).cuda()

    for batch_id in range(imgs.size(0)):
        crf_input = np.zeros((3, imgs.size(2), imgs.size(3)))
        crf_input[labels[batch_id]] = cam[batch_id].detach().cpu().numpy()

        background = 1.0 - np.max(crf_input, axis=0, keepdims=True)
        background = np.clip(background, 0, 1)

        crf_input[0] = background

        eps = 1e-8
        crf_input = crf_input / (np.sum(crf_input, axis=0, keepdims=True) + eps)

        crf_output = apply_dense_crf(denorm_image(imgs[batch_id]), crf_input)
        crf_output = torch.tensor(crf_output).cuda()

        out[batch_id] = torch.argmax(crf_output, 0)

    return out
