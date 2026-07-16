"""
Loss functions for semantic segmentation.

This module provides a wrapper around different loss functions
used for training semantic segmentation models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.training_config_loader import (
    LOSS_FUNCTION,
    IGNORE_INDEX,
    CLASS_WEIGHTS,
)

class Losses:

    def __init__(self):

        self.loss_function = LOSS_FUNCTION
        self.ignore_index = IGNORE_INDEX
        self.class_weights = CLASS_WEIGHTS

        self.loss = self.build_loss()

    def build_loss(self):
        """
        Build loss function for semantic segmentation

        return: loss function
        """

        if self.loss_function == "cross_entropy":
            return self.cross_entropy()

        elif self.loss_function == "dice":
            return self.dice_loss()

        elif self.loss_function == "dice_cross_entropy":
            return self.dice_cross_entropy()

        elif self.loss_function == "focal":
            return self.focal_loss()

        elif self.loss_function == "tversky":
            return self.tversky_loss()

        else:
            raise ValueError("Loss function not recognized")

    def cross_entropy(self):
        """
        Build loss function for semantic segmentation
        """

        if self.class_weights is not None:
            class_weights = torch.tensor(self.class_weights, dtype=torch.float32)
            return nn.CrossEntropyLoss(weight=class_weights, ignore_index=self.ignore_index)
        else:
            return nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def dice_loss(self):
        """
        Build loss function for semantic segmentation
        """

        def loss_fn(logits, targets):

            num_classes = logits.shape[1]
            probs = F.softmax(logits, dim=1)

            targets = targets.clone()
            valid_mask = (targets != self.ignore_index)
            targets[~valid_mask] = 0
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
            valid_mask = valid_mask.unsqueeze(1)

            probs = probs * valid_mask
            targets_one_hot = targets_one_hot * valid_mask

            intersection = torch.sum(probs * targets_one_hot, dim=(0, 2, 3))
            union = torch.sum(probs + targets_one_hot, dim=(0, 2, 3))

            dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
            dice_loss = 1.0 - dice.mean()

            return dice_loss

        return loss_fn