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
                A.RandomScale(scale_limit=(-0.3, 0.3), p=0.5),
                A.PadIfNeeded(min_height=self.image_size[0], min_width=self.image_size[1], border_mode=0),
                A.RandomCrop(height=self.image_size[0], width=self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=1.0),
                ], p=0.7),
                A.RandomShadow(shadow_roi=(0, 0.3, 1, 1), num_shadows_limit=(1, 3), p=0.4),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(std_range=(0.04, 0.14), p=1.0),
                ], p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])