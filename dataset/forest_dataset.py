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

    def __init__(self, split="train", indices=None, transform=None):

        self.split = split.lower()
        if self.split not in ["train", "test"]:
            raise ValueError("split must be 'train' or 'test'")

        self.indices = indices
        self.transform = transform

        if self.split == "train":
            self.image_paths = TRAIN_IMAGE_DIR
            self.mask_paths = TRAIN_MASK_DIR
        else:
            self.image_paths = TEST_IMAGE_DIR
            self.mask_paths = TEST_MASK_DIR

        # Load class mapping
        self.class_mapping = load_class_mapping()
        self.num_classes = len(self.class_mapping)
        self.class_names = self.class_mapping["class"].tolist()
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
        if self.indices is not None:
            self.image_files = [self.image_files[i] for i in self.indices]
            self.mask_files = [self.mask_files[i] for i in self.indices]

    def rgb_to_class_index(self):
        """
        Build a mapping from RGB images to class indexes
        """

        rgb_to_index = {}
        for _, row in self.class_mapping.iterrows():
            rgb_to_index[row["rgb"]] = int(row["id"])

        return rgb_to_index

    def rgb_mask_to_class_mask(self, rgb_mask):
        """
        Convert an RGB mask to a class index mask

        rgb_mask: numpy array of shape (H, W, 3) representing the RGB mask
        return: numpy array of shape (H, W) representing the class index mask
        """

        h, w, _ = rgb_mask.shape
        class_mask = np.zeros((h, w), dtype=np.uint8)

        for rgb, class_id in self.rgb_to_class_index.items():
            mask = np.all(rgb_mask == rgb, axis=2) # axis=2 checks if all channels match the rgb value
            class_mask[mask] = class_id

        return class_mask

    def __len__(self):
        """
        Return the number of samples
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Get an image from the dataset

        idx: index of the image
        return: image and mask
        """

        img = np.array(Image.open(self.image_files[idx]).convert("RGB"))
        rgb_mask = np.array(Image.open(self.mask_files[idx]).convert("RGB"))
        mask = self.rgb_mask_to_class_mask(rgb_mask)

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        return img, mask