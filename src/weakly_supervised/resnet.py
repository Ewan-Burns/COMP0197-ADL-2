import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (
    unary_from_softmax,
    create_pairwise_bilateral,
    create_pairwise_gaussian,
)


class MultiHeadResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base.children())[:-2])  # up to last conv
        self.features.requires_grad = False
        self.pool = base.avgpool
        self.segm_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        feats = self.features(x)  # (B, 512, H, W)
        segm = self.segm_head(feats)
        return segm
