import numpy as np
import torchvision.transforms.functional as F
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from src.utils.dataset import ResNetTransform


class MultiTargetOxfordPet(Dataset):
    def __init__(self, use_breed=False, random_hflip=True, random_vflip=True):
        super().__init__()
        self.base = OxfordIIITPet(
            root="./data",
            split="trainval",
            target_types=[
                "segmentation",
                "category" if use_breed else "binary-category",
            ],
            download=True,
        )
        self.transform = ResNetTransform
        self.segm_transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((224, 224), interpolation=Image.NEAREST),
            ]
        )

        self.random_hflip = random_hflip
        self.random_vflip = random_vflip

    def __getitem__(self, item):
        image, (mask, category) = self.base[item]
        mask = (np.array(mask) != 2).astype(np.long)

        shifted_category = category + 1
        mask *= shifted_category

        image = self.transform(image)
        mask = self.segm_transform(mask)

        if np.random.rand() < 0.5 and self.random_hflip:
            image = F.hflip(image)
            mask = F.hflip(mask)

        if np.random.rand() < 0.5 and self.random_vflip:
            image = F.vflip(image)
            mask = F.vflip(mask)

        return image, mask, shifted_category

    def __len__(self):
        return len(self.base)
