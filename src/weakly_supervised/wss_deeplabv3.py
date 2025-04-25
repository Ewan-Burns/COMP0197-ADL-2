import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.segmentation as models
from torchcam.methods import GradCAMpp
from src.utils.sec import apply_crf_to_heatmap
import numpy as np


class MultiHeadDeepLabV3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = models.deeplabv3_resnet50(
            pretrained_backbone=True, num_classes=num_classes
        )
        self.cam_extractor = GradCAMpp(self.base, target_layer="backbone.layer4.2")
        self.num_classes = num_classes

    def _get_cam(self, input, output):
        cam = torch.zeros(
            (output.size(0), output.size(1), input.size(2), input.size(3))
        ).cuda()

        for class_idx in range(output.size(1)):
            activation_map = self.cam_extractor(class_idx, output, retain_graph=True)
            c = F.interpolate(
                activation_map[0].unsqueeze(0),
                (input.size(2), input.size(3)),
                mode="bilinear",
            ).squeeze()
            cam[:, class_idx, :, :] = c

        return cam

    def forward(self, x):
        pred_mask = self.base(x)["out"]
        return pred_mask, self._get_cam(x, pred_mask)
