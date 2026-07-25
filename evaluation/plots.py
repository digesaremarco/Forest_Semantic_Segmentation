"""
This module provides plotting utilities used during
training and evaluation

Supported plots:
    - single prediction
    - prediction grids
    - overlay visualization
    - confusion matrix
    - training curves
"""

from pathlib import Path
import numpy as np
import math
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from dataset.dataset_info import (
    load_class_mapping,
)

class Plots:

    def __init__(self, save_directory=None):

        self.mapping = load_class_mapping()
        self.class_names = self.mapping["class"].tolist()

        self.colors_rgb = np.array(self.mapping["rgb"].tolist(), dtype=np.uint8)
        self.colors = self.colors_rgb.astype(float) / 255.0
        self.save_directory = save_directory

        if self.save_directory:
            self.save_directory = Path(self.save_directory)
            self.save_directory.mkdir(parents=True, exist_ok=True)

    def prediction_to_rgb(self, prediction):
        """
        Convert a prediction to an RGB image using the class mapping

        prediction: prediction made by the model, ndarray (H, W)
        return: RGB image
        """

        rgb_image = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)

        for class_id, color in enumerate(self.colors_rgb):
            rgb_image[prediction == class_id] = color

        return rgb_image

    def overlay_prediction(self, image, prediction, alpha=0.5):
        """
        Overlay a prediction on top of an image
        """

        rgb_prediction = self.prediction_to_rgb(prediction)

        overlayed_image = (1 - alpha) * image + alpha * rgb_prediction
        overlayed_image = np.clip(overlayed_image, 0, 255).astype(np.uint8)

        return overlayed_image

    def add_legend(self, figure):
        """
        Add a legend to the figure
        """

        legend_elements = []

        for class_name, color in zip(self.class_names, self.colors):
            legend_elements.append(
                Patch(facecolor=color, edgecolor="black", label=class_name)
            )

        figure.legend(handles=legend_elements, loc="lower center", ncol=6, fontsize=9,
                      frameon=True, bbox_to_anchor=(0.5, 0.01))

    def save_figure(self, fig, save_name):
        """
        Save the figure
        """

        if self.save_directory:
            save_path = self.save_directory / save_name
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"Figure saved to {save_path}")
        else:
            print("Save directory not specified. Figure not saved.")

    def show_prediction(self, image, prediction, figsize=(16,6), save_name=None):
        """
         Display image, prediction and overlay
        """

        prediction = self.prediction_to_rgb(prediction)
        overlay = self.overlay_prediction(image, prediction)

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[1].imshow(prediction)
        axes[1].set_title("Prediction")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")

        for ax in axes:
            ax.axis("off")

        self.add_legend(fig)
        plt.tight_layout(rect=[0, 0.12, 1, 1])

        if save_name:
            self.save_figure(fig, save_name)

        plt.show()

    def show_prediction_grid(self, images, predictions, figsize=(16,6), save_name=None):
        """
        Display multiple predictions
        Layout: Image | Prediction | Overlay
        """

        num_samples = len(images)
        fig, axes = plt.subplots(num_samples, 3, figsize=(16, 5 * num_samples), squeeze=False)

        for i in range(num_samples):
            prediction_rgb = self.prediction_to_rgb(predictions[i])
            overlay = self.overlay_prediction(images[i], predictions[i])

            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title("Image")
            axes[i, 1].imshow(prediction_rgb)
            axes[i, 1].set_title("Prediction")
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title("Overlay")

            for ax in axes[i]:
                ax.axis("off")

        self.add_legend(fig)
        plt.tight_layout(rect=[0, 0.08, 1, 1], h_pad=3)

        if save_name:
            self.save_figure(fig, save_name)

        plt.show()


    def confusion_matrix(self, confusion_matrix, normalize=False, figsize=(12, 12), save_name=None):
        """
        Plot the confusion matrix
        """

        fig, axes = plt.subplots(figsize=figsize)
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=self.class_names)
        disp.plot(ax=axes, cmap=plt.cm.Blues, colorbar=False, xticks_rotation="vertical", values_format=".2f" if normalize else "d")
        plt.tight_layout()
        plt.title("Confusion Matrix")

        if save_name:
            self.save_figure(fig, save_name)

        plt.show()

    def load_logs(self, logger):
        """
        Load training logs from the logger

        return: dictionary containing all logged values
        """

        logs = logger.read_logs()

        if len(logs) <= 1:
            raise RuntimeError("No training logs found.")

        header = logs[0]
        rows = logs[1:]
        data = {column: [] for column in header}

        for row in rows:
            for key, value in zip(header, row):
                data[key].append(float(value))

        return data

    def plot_metric(self, logger, metric_name, figsize=(7, 5), save_name=None):
        """
        Plot a single metric stored inside the logger
        """

        data = self.load_logs(logger)

        if metric_name not in data:
            raise ValueError(f"Metric '{metric_name}' not found")

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(data["epoch"], data[metric_name], linewidth=2)
        ax.set_title(metric_name.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.grid(True)
        plt.tight_layout()

        if save_name:
            self.save_figure(fig, save_name)

        plt.show()

    def plot_all_metrics(self, logger, figsize=(15, 10), save_name=None):
        """
        Plot all metrics stored inside the logger
        """

        data = self.load_logs(logger)

        metrics = [key for key in data.keys() if key != "epoch"]
        n_metrics = len(metrics)

        n_cols = 2
        n_rows = math.ceil(n_metrics / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).reshape(-1)

        for ax, metric in zip(axes, metrics):
            ax.plot(data["epoch"], data[metric], linewidth=2)
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("Epoch")
            ax.grid(True)

        # Hide unused axes
        for ax in axes[len(metrics):]:
            ax.axis("off")

        plt.tight_layout()

        if save_name:
            self.save_figure(fig, save_name)

        plt.show()