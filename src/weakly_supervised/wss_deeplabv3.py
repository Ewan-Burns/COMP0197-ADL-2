import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torchvision.models.segmentation as models
from torchcam.methods import GradCAMpp
import numpy as np


class WSSDeepLabV3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.base = models.deeplabv3_resnet50(
            pretrained_backbone=True, num_classes=num_classes
        )
        self.num_classes = num_classes

    def forward(self, x):
        pred_mask = self.base(x)["out"]
        return pred_mask
