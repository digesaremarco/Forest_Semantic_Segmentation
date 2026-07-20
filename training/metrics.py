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