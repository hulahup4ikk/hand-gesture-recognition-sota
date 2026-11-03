"""
Dataset and data loading utilities for Hand Gesture Recognition.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASSES = ["like", "palm", "peace"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_default_transforms(img_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """Returns standard torchvision transforms with optional augmentation for training."""
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class GestureDataset(Dataset):
    """Gesture image dataset loading from class subdirectories."""

    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        for cls_name in CLASSES:
            cls_folder = self.root_dir / cls_name
            if not cls_folder.exists():
                continue
            cls_idx = CLASS_TO_IDX[cls_name]
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                for p in cls_folder.glob(ext):
                    self.samples.append((p, cls_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid images found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    data_path = Path(data_dir)
    train_ds = GestureDataset(str(data_path / "train"), transform=get_default_transforms(img_size, is_train=True))
    val_ds = GestureDataset(str(data_path / "val"), transform=get_default_transforms(img_size, is_train=False))
    test_ds = GestureDataset(str(data_path / "test"), transform=get_default_transforms(img_size, is_train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

