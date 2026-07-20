"""
Evaluation metrics for semantic segmentation

This module provides a wrapper around different evaluation metrics
used for semantic segmentation models
"""

import torch

class Metrics:

    def __init__(self, num_classes, ignore_index=255):

        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def flatten(self, y_pred, y_true):
        """
        Flattens the predictions and ground truth labels for evaluation

        y_pred: (torch.Tensor) The predicted labels
        y_true: (torch.Tensor) The ground truth labels
        """

        # Flatten the predictions and ground truth labels
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # Remove the ignored index from the ground truth labels
        mask = y_true != self.ignore_index
        y_pred = y_pred[mask]
        y_true = y_true[mask]

        return y_pred, y_true

    def pixel_accuracy(self, y_pred, y_true):
        """
        Computes the pixel accuracy for semantic segmentation
        """

        # Flatten the predictions and ground truth labels
        y_pred, y_true = self.flatten(y_pred, y_true)

        # Compute the pixel accuracy
        correct = (y_pred == y_true).sum().float()
        total = y_true.numel() # numel() returns the total number of elements in the tensor

        return correct / total if total > 0 else 0.0

    def mean_pixel_accuracy(self, y_pred, y_true):
        """
        Computes the mean pixel accuracy for semantic segmentation
        """

        # Flatten the predictions and ground truth labels
        y_pred, y_true = self.flatten(y_pred, y_true)

        # Compute the mean pixel accuracy
        mean_acc = 0.0
        for c in range(self.num_classes):
            mask = y_true == c
            if mask.sum() > 0:
                mean_acc += (y_pred[mask] == c).sum().float() / mask.sum().float()

        return mean_acc / self.num_classes if self.num_classes > 0 else 0.0

    def iou(self, y_pred, y_true):
        """
        Computes the intersection over union for semantic segmentation
        """

        y_pred, y_true = self.flatten(y_pred, y_true)

        ious = []
        for c in range(self.num_classes):
            mask = y_true == c
            pred = y_pred == c

            intersection = (pred & mask).sum().float()
            union = (pred | mask).sum().float()

            if union > 0:
                ious.append(intersection / union)
            else:
                ious.append(torch.tensor(0.0))

        return torch.stack(ious) if len(ious) > 0 else torch.tensor(0.0)

    def mean_iou(self, y_pred, y_true):
        """
        Computes the mean intersection over union for semantic segmentation
        """

        iou = self.iou(y_pred, y_true)
        return iou.mean() if len(iou) > 0 else torch.tensor(0.0)

    def dice(self, y_pred, y_true):
        """
        Computes the dice coefficient for semantic segmentation
        """

        y_pred, y_true = self.flatten(y_pred, y_true)

        dices = []
        eps = 1e-6 # Epsilon to avoid division by zero
        for c in range(self.num_classes):
            mask = y_true == c
            pred = y_pred == c

            intersection = (pred & mask).sum().float()
            dice_score = (2 * intersection) / (pred.sum().float() + mask.sum().float() + eps)
            dices.append(dice_score)

        return torch.stack(dices) if len(dices) > 0 else torch.tensor(0.0)

    def mean_dice(self, y_pred, y_true):
        """
        Computes the mean dice coefficient for semantic segmentation
        """

        dice = self.dice(y_pred, y_true)
        return dice.mean() if len(dice) > 0 else torch.tensor(0.0)

    def frequency_weighted_iou(self, y_pred, y_true):
        """
        Computes the frequency weighted intersection over union for semantic segmentation
        """

        y_pred, y_true = self.flatten(y_pred, y_true)

        freq_weighted_iou = 0.0
        total_pixels = len(y_true)

        for c in range(self.num_classes):
            mask = y_true == c
            pred = y_pred == c

            intersection = (pred & mask).sum().float()
            union = (pred | mask).sum().float()

            if union > 0:
                freq_weighted_iou += (mask.sum().float() / total_pixels) * (intersection / union)

        return freq_weighted_iou if total_pixels > 0 else torch.tensor(0.0)

    def compute_metrics(self, y_pred, y_true):
        """
        Computes the metrics for semantic segmentation

        return: (dict) A dictionary containing the computed metrics
        """

        metrics = {
            "pixel_accuracy": self.pixel_accuracy(y_pred, y_true),
            "mean_pixel_accuracy": self.mean_pixel_accuracy(y_pred, y_true),
            "iou": self.iou(y_pred, y_true),
            "mean_iou": self.mean_iou(y_pred, y_true),
            "dice": self.dice(y_pred, y_true),
            "mean_dice": self.mean_dice(y_pred, y_true),
            "frequency_weighted_iou": self.frequency_weighted_iou(y_pred, y_true)
        }

        return metrics