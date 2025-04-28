import numpy as np
import torchvision.transforms.functional as F
import torch
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from src.utils.dataset import ResNetTransform


import os
import os.path
import pathlib
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset

import xml.etree.ElementTree as ET


class OxfordIIITPetWithBoxes(VisionDataset):
    """`Oxford-IIIT Pet Dataset   <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"trainval"`` (default) or ``"test"``.
        target_types (string, sequence of strings, optional): Types of target to use. Can be ``category`` (default) or
            ``segmentation``. Can also be a list to output a tuple with all specified target types. The types represent:

                - ``category`` (int): Label for one of the 37 pet categories.
                - ``binary-category`` (int): Binary label for cat or dog.
                - ``segmentation`` (PIL image): Segmentation trimap of the image.

            If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into
            ``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
    """

    _RESOURCES = (
        (
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            "5c4f3ee8e5d25df40f4fd59a7f44e54c",
        ),
        (
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            "95a8c909bbe2e81eed6a22bccdf3f68f",
        ),
    )
    _VALID_TARGET_TYPES = ("category", "binary-category", "segmentation", "bbox")

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        split: str = "trainval",
        target_types: Union[Sequence[str], str] = "category",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self._split = verify_str_arg(split, "split", ("trainval", "test"))
        if isinstance(target_types, str):
            target_types = [target_types]
        self._target_types = [
            verify_str_arg(target_type, "target_types", self._VALID_TARGET_TYPES)
            for target_type in target_types
        ]

        super().__init__(
            root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self._base_folder = pathlib.Path(self.root) / "oxford-iiit-pet"
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._segs_folder = self._anns_folder / "trimaps"
        self._xmls_folder = self._anns_folder / "xmls"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        image_ids = []
        self._labels = []
        self._bin_labels = []
        self._bboxes = []
        with open(self._anns_folder / f"{self._split}.txt") as file:
            for line in file:
                image_id, label, bin_label, _ = line.strip().split()
                image_ids.append(image_id)
                self._labels.append(int(label) - 1)
                self._bin_labels.append(int(bin_label) - 1)

        self.bin_classes = ["Cat", "Dog"]
        self.classes = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls, _ in sorted(
                {
                    (image_id.rsplit("_", 1)[0], label)
                    for image_id, label in zip(image_ids, self._labels)
                },
                key=lambda image_id_and_label: image_id_and_label[1],
            )
        ]
        self.bin_class_to_idx = dict(
            zip(self.bin_classes, range(len(self.bin_classes)))
        )
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self._images = [
            self._images_folder / f"{image_id}.jpg" for image_id in image_ids
        ]
        self._segs = [self._segs_folder / f"{image_id}.png" for image_id in image_ids]

        self._bboxes = [
            self._parse_bounding_box(self._xmls_folder / f"{image_id}.xml")
            for image_id in image_ids
        ]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = Image.open(self._images[idx]).convert("RGB")

        target: Any = []
        for target_type in self._target_types:
            if target_type == "category":
                target.append(self._labels[idx])
            elif target_type == "binary-category":
                target.append(self._bin_labels[idx])
            elif target_type == "bbox":
                target.append(self._bboxes[idx])
            else:  # target_type == "segmentation"
                target.append(Image.open(self._segs[idx]))

        if not target:
            target = None
        elif len(target) == 1:
            target = target[0]
        else:
            target = tuple(target)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def _check_exists(self) -> bool:
        for folder in (self._images_folder, self._anns_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True

    def _download(self) -> None:
        if self._check_exists():
            return

        for url, md5 in self._RESOURCES:
            download_and_extract_archive(
                url, download_root=str(self._base_folder), md5=md5
            )

    def _parse_bounding_box(self, xml_path):

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            obj = root.find("object")
            bbox = obj.find("bndbox")

            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            return torch.tensor(
                [xmin, ymin, xmax, ymax], dtype=torch.float32
            ).unsqueeze(0)

        except FileNotFoundError:
            return torch.tensor([0, 0, 1, 1], dtype=torch.float32).unsqueeze(0)


class MultiTargetOxfordPet(Dataset):
    def __init__(self, use_breed=False, random_hflip=True, random_vflip=True):
        super().__init__()

        target_types = [
            "segmentation",
            "category" if use_breed else "binary-category",
            "bbox",
        ]

        self.base = OxfordIIITPetWithBoxes(
            root="./data",
            split="trainval",
            target_types=target_types,
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
        image, (mask, category, bbox) = self.base[item]
        mask = (np.array(mask) != 2).astype(np.long)

        shifted_category = category + 1
        mask *= shifted_category

        imWidth = image.width
        imHeight = image.height

        image = self.transform(image)
        mask = self.segm_transform(mask)

        if np.random.rand() < 0.5 and self.random_hflip:
            image = F.hflip(image)
            mask = F.hflip(mask)
            bbox[0][0] = imWidth - bbox[0][0]
            bbox[0][2] = imWidth - bbox[0][2]

        if np.random.rand() < 0.5 and self.random_vflip:
            image = F.vflip(image)
            mask = F.vflip(mask)

            bbox[0][1] = imHeight - bbox[0][1]
            bbox[0][3] = imHeight - bbox[0][3]

        bbox[0][0] *= 224 / imWidth
        bbox[0][2] *= 224 / imWidth
        bbox[0][1] *= 224 / imHeight
        bbox[0][3] *= 224 / imHeight

        return image, mask, bbox, shifted_category

    def __len__(self):
        return len(self.base)
