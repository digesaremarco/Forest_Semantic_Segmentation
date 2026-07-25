"""
This module evaluates a trained semantic segmentation model on the test dataset

The evaluation consists of:
    - loading the best checkpoint
    - computing all segmentation metrics
    - computing the confusion matrix
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from training.checkpoint import Checkpoint
from training.metrics import Metrics


class QuantitativeEvaluation:

    def __init__(self, model, test_loader, metrics, device):

        self.model = model
        self.test_loader = test_loader
        self.metrics = metrics
        self.device = device

        self.checkpoint = Checkpoint()
        self.results = None

    def load_model(self):
        """
        Load the trained model from the checkpoint
        """

        self.checkpoint.load_checkpoint(self.model)
        self.model.eval()

    def evaluate(self):
        """
        Evaluate the model on the test set
        """

        self.load_model()

        running_metrics = {
            "pixel_accuracy": 0.0,
            "mean_pixel_accuracy": 0.0,
            "mean_iou": 0.0,
            "mean_dice": 0.0,
            "frequency_weighted_iou": 0.0,
        }
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, masks in self.test_loader:
                images, masks = images.to(self.device), masks.to(self.device)

                outputs = self.model(images)
                logits = outputs.logits
                logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                predictions = torch.argmax(logits, dim=1)
                metrics = self.metrics.compute_metrics(predictions, masks)

                for key in running_metrics:
                    running_metrics[key] += float(metrics[key])

                all_predictions.append(predictions.cpu().numpy().reshape(-1))
                all_targets.append(masks.cpu().numpy().reshape(-1))

            n_batches = len(self.test_loader)

            for key in running_metrics:
                running_metrics[key] /= n_batches # Average over all batches

        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        valid_mask = all_targets != self.metrics.ignore_index

        confusion = confusion_matrix(all_targets[valid_mask], all_predictions[valid_mask],
                                            labels=np.arange(self.metrics.num_classes))
        running_metrics["confusion_matrix"] = confusion
        self.results = running_metrics

        return running_metrics

    def show_results(self):
        """
        Print evaluation results
        """

        if self.results is None:
            raise RuntimeError("Run evaluate() before show_results()")

        print("\nTEST RESULTS:\n")

        print(f"Pixel Accuracy        : {self.results['pixel_accuracy']:.4f}")
        print(f"Mean Pixel Accuracy   : {self.results['mean_pixel_accuracy']:.4f}")
        print(f"Mean IoU              : {self.results['mean_iou']:.4f}")
        print(f"Mean Dice             : {self.results['mean_dice']:.4f}")
        print(f"Frequency Weighted IoU: {self.results['frequency_weighted_iou']:.4f}")