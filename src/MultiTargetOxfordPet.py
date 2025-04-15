import numpy as np
import torchvision.transforms.functional as F
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from src.utils.dataset import ResNetTransform


class MultiTargetOxfordPet(Dataset):
    def __init__(self, use_breed=False):
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

    def __getitem__(self, item):
        image, (mask, category) = self.base[item]
        mask = (np.array(mask) != 2).astype(np.long)
        mask *= category + 1

        if np.random.rand() < 0.5:
            return F.hflip(self.transform(image)), F.hflip(self.segm_transform(mask))

        return self.transform(image), self.segm_transform(mask)

    def __len__(self):
        return len(self.base)
