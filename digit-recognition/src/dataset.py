"""
Data loading and augmentation utilities for MNIST digit recognition.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(train: bool = True) -> transforms.Compose:
    base = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST global mean/std
    ]

    if train:
        augment = [
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
        return transforms.Compose(augment + base)

    return transforms.Compose(base)


def get_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    val_split: float = 0.1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader)."""
    root = Path(data_dir)

    full_train = datasets.MNIST(root, train=True, download=True, transform=get_transforms(train=True))
    test_set = datasets.MNIST(root, train=False, download=True, transform=get_transforms(train=False))

    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    train_set, val_set = torch.utils.data.random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Override val transform (no augmentation)
    val_set.dataset = datasets.MNIST(root, train=True, download=False, transform=get_transforms(train=False))

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
