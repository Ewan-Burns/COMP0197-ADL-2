import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian


class MultiHeadResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base.children())[:-2])  # up to last conv
        self.features.requires_grad = False
        self.pool = base.avgpool
        self.class_head = nn.Linear(512, num_classes)
        self.segm_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        feats = self.features(x)  # (B, 512, H, W)
        out = self.pool(feats).squeeze(-1).squeeze(-1)

        segm = self.segm_head(feats)
        cl = self.class_head(out)

        return feats, segm, cl

def generate_cams(feats, fc_weights, class_idx):
    """
    feats: (B, C, H, W)
    fc_weights: (num_classes, C)
    class_idx: (B,) or scalar
    Returns CAMs: (B, 1, H, W)
    """
    B, C, H, W = feats.shape
    cams = []

    with torch.no_grad():
        for i in range(B):
            w = fc_weights[class_idx[i]]  # shape (C,)
            cam = torch.sum(w.view(C, 1, 1) * feats[i], dim=0)  # (H, W)
            cam = torch.relu(cam)  # ReLU for better localization
            cam2 = cam - torch.min(cam)
            cam3 = cam2 / (torch.max(cam) + 1e-6)
            cams.append(cam3.unsqueeze(0))  # (1, H, W)

    return torch.stack(cams)  # (B, 1, H, W)


