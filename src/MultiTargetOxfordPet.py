import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.optim as optim
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from unet import UNet
from tqdm import tqdm

def show_prediction(img, mask):

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title("Predicted Segmentation")
    plt.axis("off")
    plt.show()


class MultiTargetOxfordPet(Dataset):
    def __init__(self):
        super().__init__()
        self.base = OxfordIIITPet(root='./data', split='trainval', target_types=['segmentation', 'binary-category'], download=True)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.segm_transform = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224), interpolation=Image.NEAREST),
        ])

    def __getitem__(self, item):
        image, (mask, category) = self.base[item]
        mask = (np.array(mask) != 2).astype(np.long)
        mask *= category + 1

        if np.random.rand() < 0.5:
            return F.hflip(self.transform(image)), F.hflip(self.segm_transform(mask))

        return self.transform(image), self.segm_transform(mask)

    def __len__(self):
        return len(self.base)