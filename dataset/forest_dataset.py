"""
PyTorch Dataset for the Forest Semantic Segmentation dataset

This module provides the ForestDataset class, responsible for:
    - loading RGB images
    - loading RGB annotation masks
    - converting RGB masks into class-index masks
    - returning image/mask pairs for training

The dataset is compatible with semantic segmentation models such as: SegFormer, DeepLabV3, Mask2Former...
"""

from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from dataset.dataset_config_loader import (
    TRAIN_IMAGE_DIR,
    TEST_IMAGE_DIR,
    TRAIN_MASK_DIR,
    TEST_MASK_DIR,
)

from dataset.dataset_info import load_class_mapping

class ForestDataset(Dataset):
    """
    PyTorch Dataset for the Forest Semantic Segmentation dataset
    """

    def __init__(self, split="train", indicies=None, transform=None):

        self.split = split.lower()
        if self.split not in ["train", "test"]:
            raise ValueError("split must be 'train' or 'test'")

        self.indicies = indicies
        self.transform = transform

        if self.split == "train":
            self.image_paths = TRAIN_IMAGE_DIR
            self.mask_paths = TRAIN_MASK_DIR
        else:
            self.image_paths = TEST_IMAGE_DIR
            self.mask_paths = TEST_MASK_DIR

        # Load class mapping
        self.class_mapping = load_class_mapping()
        self.rgb_to_class_index = self.rgb_to_class_index()

        # Load image and mask file paths
        self.image_files = sorted(self.image_paths.glob("*.png"))
        self.mask_files = []

        for image_file in self.image_files:
            mask_file = self.mask_paths / image_file.name
            if mask_file.exists():
                self.mask_files.append(mask_file)
            else:
                raise FileNotFoundError(f"Mask file {mask_file} not found for image {image_file}")

        # Indices filtering
        if self.indicies is not None:
            self.image_files = self.image_files[self.indicies] # self.image_files = self.image_files[self.image_files[i] for i in self.indicies]
            self.mask_files = self.mask_files[self.indicies]

    def rgb_to_class_index(self):
        """
        Build a mapping from RGB images to class indexes
        """

        rgb_to_index = {}
        for _, row in self.class_mapping.iterrows():
            rgb_to_index[row["rgb"]] = int(row["id"])

        return rgb_to_index




