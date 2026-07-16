"""
Albumentations transformations for the Forest Semantic Segmentation dataset.

This module provides image preprocessing and data augmentation
pipelines for semantic segmentation.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

class ForestTransforms:

    def __init__(self, image_size=(512, 512), augmentation = False, train=True):
        self.image_size = image_size
        self.augmentation = augmentation
        self.train = train

    def get_transforms(self):
        """
        Get the Albumentations transformation pipeline

        return: Albumentations Compose object
        """
        if self.augmentation and self.train:
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                A.ColorJitter(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])