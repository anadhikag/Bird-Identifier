"""Dataset utilities for the CUB-200-2011 bird species dataset.

This module is responsible for ONE thing: loading the CUB-200-2011 dataset
using its official train/test split and exposing PyTorch-ready Dataset and
DataLoader objects.

It has no knowledge of models, training loops, or inference logic, so it
can be reused unchanged in Stage 2 (FastAPI backend) and Stage 3 (Azure
deployment) without modification.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_SIZE: int = 384

# ImageNet normalization statistics, required because EfficientNetV2-S is
# pretrained on ImageNet and expects inputs normalized the same way.
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class CUBPaths:
    """Resolves the filesystem layout of an extracted CUB-200-2011 archive."""

    root_dir: str

    @property
    def images_dir(self) -> str:
        return os.path.join(self.root_dir, "images")

    @property
    def images_txt(self) -> str:
        return os.path.join(self.root_dir, "images.txt")

    @property
    def labels_txt(self) -> str:
        return os.path.join(self.root_dir, "image_class_labels.txt")

    @property
    def split_txt(self) -> str:
        return os.path.join(self.root_dir, "train_test_split.txt")

    @property
    def classes_txt(self) -> str:
        return os.path.join(self.root_dir, "classes.txt")


def load_class_names(paths: CUBPaths) -> list[str]:
    """Load the ordered list of the 200 CUB species names.

    The returned list is 0-indexed (index 0 corresponds to class_id 1 in
    the raw dataset files) so it can be used directly as ``nn.Linear``
    output indices and as a lookup table at inference time.

    Args:
        paths: Resolved CUB-200-2011 filesystem paths.

    Returns:
        Ordered list of 200 human-readable class names, e.g. the raw
        "001.Black_footed_Albatross" becomes "Black footed Albatross".
    """
    classes_df = pd.read_csv(
        paths.classes_txt, sep=" ", header=None, names=["class_id", "class_name"]
    )
    classes_df = classes_df.sort_values("class_id")
    cleaned = classes_df["class_name"].str.split(".", n=1).str[1].str.replace("_", " ")
    return cleaned.tolist()


class CUB200Dataset(Dataset):
    """PyTorch Dataset for CUB-200-2011 using the official train/test split.

    This class only reads metadata and images from disk and applies image
    transforms. It contains no training or model logic so it remains valid
    for both the training pipeline (Stage 1) and any future data-serving
    endpoint (Stage 2/3).
    """

    def __init__(
        self,
        root_dir: str,
        train: bool,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            root_dir: Path to the extracted CUB_200_2011 directory (must
                contain images/, images.txt, image_class_labels.txt,
                train_test_split.txt, and classes.txt).
            train: If True, load the official training split; otherwise
                load the official test split.
            transform: Optional torchvision transform pipeline applied to
                each PIL image. If None, no transform is applied.
        """
        self._paths = CUBPaths(root_dir)
        self.transform = transform
        self.train = train
        self.class_names = load_class_names(self._paths)
        self._samples = self._build_sample_list()

    def _build_sample_list(self) -> list[tuple[str, int]]:
        """Merge CUB metadata files into a list of (image_path, label)."""
        images = pd.read_csv(
            self._paths.images_txt,
            sep=" ",
            header=None,
            names=["image_id", "file_path"],
        )
        labels = pd.read_csv(
            self._paths.labels_txt,
            sep=" ",
            header=None,
            names=["image_id", "class_id"],
        )
        split = pd.read_csv(
            self._paths.split_txt,
            sep=" ",
            header=None,
            names=["image_id", "is_training_image"],
        )

        merged = images.merge(labels, on="image_id").merge(split, on="image_id")
        wanted_flag = 1 if self.train else 0
        merged = merged[merged["is_training_image"] == wanted_flag]

        samples: list[tuple[str, int]] = [
            (
                os.path.join(self._paths.images_dir, row.file_path),
                int(row.class_id) - 1,  # convert to 0-indexed label
            )
            for row in merged.itertuples(index=False)
        ]
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        image_path, label = self._samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Build the training-time augmentation/preprocessing pipeline.

    Args:
        image_size: Target square image size expected by the model.

    Returns:
        Composed torchvision transform with augmentation suitable for
        fine-grained bird species classification.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Build the deterministic evaluation/inference preprocessing pipeline.

    This exact pipeline (minus augmentation) is reused by infer.py so that
    train-time and inference-time preprocessing never drift apart.

    Args:
        image_size: Target square image size expected by the model.

    Returns:
        Composed torchvision transform with no random augmentation.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = IMAGE_SIZE,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Build train and test DataLoaders using the official CUB split.

    Args:
        root_dir: Path to the extracted CUB_200_2011 directory.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses used for data loading.
        image_size: Target square image size expected by the model.

    Returns:
        A tuple of (train_loader, test_loader, class_names).
    """
    train_dataset = CUB200Dataset(
        root_dir=root_dir, train=True, transform=get_train_transforms(image_size)
    )
    test_dataset = CUB200Dataset(
        root_dir=root_dir, train=False, transform=get_eval_transforms(image_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, train_dataset.class_names
