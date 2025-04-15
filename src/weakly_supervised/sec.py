import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import numpy as np
from tqdm import tqdm

from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian



def seed_loss(pred_mask, seed_mask):
    """
    pred_mask: (N, C, H, W) - model output (after softmax)
    seed_mask: (N, C, H, W) - CAM-like seeds (only confident regions non-zero)
    """
    loss = -(seed_mask * torch.log(pred_mask + 1e-7)).sum(dim=1).mean()
    return loss

def expand_loss(pred_mask, labels):
    """
    pred_mask: (N, C, H, W)
    labels: (N, C) - binary labels (1 if class is in image)
    """
    pred_max = pred_mask.view(pred_mask.size(0), pred_mask.size(1), -1).max(dim=2)[0]  # (N, C)
    loss = F.cross_entropy(pred_max, labels)
    return loss

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
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image.copy(order='C'), compat=10)
    Q = d.inference(5)
    return np.array(Q).reshape((probs.shape[0], H, W))

def constrain_loss(pred_mask, image):
    """
    pred_mask: (N, C, H, W)
    image: (N, H, W, 3) numpy RGB
    """
    loss = 0
    for i in range(pred_mask.size(0)):
        probs = pred_mask[i].detach().cpu().numpy()
        crf_output = apply_dense_crf(image[i], probs)
        crf_output = torch.tensor(crf_output).to(pred_mask.device)
        pred = pred_mask[i]
        loss += F.kl_div(pred.log(), crf_output, reduction='batchmean')
    return loss / pred_mask.size(0)

def sec_loss(pred_mask, seed_mask, image, labels, alpha=1.0, beta=0.5, gamma=0.5):
    L_seed = seed_loss(pred_mask, seed_mask)
    L_expand = expand_loss(pred_mask, labels)
    L_constrain = constrain_loss(pred_mask, image)

    total_loss = alpha * L_seed + beta * L_expand + gamma * L_constrain
    return total_loss