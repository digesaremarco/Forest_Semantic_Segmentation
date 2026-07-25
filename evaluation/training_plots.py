"""
Visualize training metrics.

This module loads the training logs and displays
all the curves generated during training.
"""

from training.logger import Logger
from evaluation.plots import Plots


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