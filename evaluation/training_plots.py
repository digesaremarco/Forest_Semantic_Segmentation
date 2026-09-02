"""
Visualize training metrics

This module loads the training logs and displays
all the curves generated during training
"""

from training.logger import Logger
from evaluation.plots import Plots

from training.training_config_loader import (
    LOG_DIRECTORY,
)


class TrainingPlots:

    def __init__(self, save_directory=None):

        self.logger = Logger()
        self.plots = Plots(save_directory=save_directory)

    def plot_all(self):
        """
        Plot every metric stored in the logger
        """

        self.plots.plot_all_metrics(self.logger)

    def plot_metric(self, metric_name):
        """
        Plot a single metric

        metric_name: name of the metric to display
        """

        self.plots.plot_metric(logger=self.logger, metric_name=metric_name)

    def compare_metric(self, metric_name):
        """
        Compare one metric across all training runs
        """

        log_names = [
            "segformerb1_focal_log.csv",
            "segformerb1_tversky_log.csv",
            "segformerb1_dice_ce_log.csv",
            "segformerb1_dice_log.csv",
        ]

        labels = [
            "B1 - Focal",
            "B1 - Tversky",
            "B1 - Dice + CE",
            "B1 - Dice",
        ]

        self.plots.compare_metric(
            metric_name=metric_name,
            log_directory=LOG_DIRECTORY,
            log_names=log_names,
            labels=labels,
        )