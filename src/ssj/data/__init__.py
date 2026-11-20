"""Dataset building and augmentation.

``ssj.data.loader`` (PyTorch dataset and batching) is imported explicitly by the
training code so that this package stays importable without torch.
"""

from .augment import AugmentConfig, augment
from .build import DatasetConfig, FeatureDataset, Item, assign_splits, build_dataset

__all__ = [
    "AugmentConfig",
    "DatasetConfig",
    "FeatureDataset",
    "Item",
    "assign_splits",
    "augment",
    "build_dataset",
]
