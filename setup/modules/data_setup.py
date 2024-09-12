"""
Utility functions for creating PyTorch DataLoaders for various types of datasets
(e.g., image classification, generic datasets). This module provides flexibility to work
with multiple dataset formats.
"""

import os
from typing import Tuple, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: Optional[str] = None,
    test_dir: Optional[str] = None,
    dataset: Optional[Dataset] = None,
    transform: Optional[transforms.Compose] = None,
    batch_size: int = 32,
    num_workers: int = NUM_WORKERS,
    shuffle_train: bool = True,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[List[str]]]:
    """
    Creates PyTorch DataLoaders for training and testing data.

    Args:
        train_dir (Optional[str]): Path to training directory (for folder-based datasets like images).
        test_dir (Optional[str]): Path to testing directory (for folder-based datasets like images).
        dataset (Optional[Dataset]): A custom PyTorch Dataset (if using non-folder based datasets).
        transform (Optional[transforms.Compose]): Transforms to be applied on the data.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of workers for DataLoader.
        shuffle_train (bool): Whether to shuffle the training data.
        pin_memory (bool): Whether to use pinned memory in DataLoader (recommended for CUDA).

    Returns:
        Tuple[DataLoader, DataLoader, Optional[List[str]]]: Train DataLoader, Test DataLoader, and list of class names (if applicable).
    """

    if dataset is not None:
        # Custom Dataset case
        train_data, test_data = dataset['train'], dataset['test']
        class_names = dataset.get('classes', None)
    elif train_dir and test_dir:
        # Image folder case (directory based)
        train_data = datasets.ImageFolder(train_dir, transform=transform)
        test_data = datasets.ImageFolder(test_dir, transform=transform)
        class_names = train_data.classes
    else:
        raise ValueError("You must provide either `train_dir` and `test_dir` for folder-based datasets or a `dataset`.")

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test data
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_dataloader, test_dataloader, class_names


def get_default_transforms(data_type: str = 'image') -> transforms.Compose:
    """
    Provides default transforms for datasets based on the type of data.

    Args:
        data_type (str): Type of the dataset (e.g., 'image', 'text', etc.)

    Returns:
        transforms.Compose: Default transformations applied to the data.
    """
    if data_type == 'image':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise NotImplementedError(f"Transforms for {data_type} are not implemented yet.")
