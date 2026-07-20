"""
Early stopping utility.

This module provides an EarlyStopping class used to stop the
training process when the monitored metric does not improve.
"""

import math

from training.training_config_loader import (
    EARLY_STOPPING,
    PATIENCE,
    EARLY_STOPPING_MONITOR,
)

class EarlyStopping:

    def __init__(self):

        self.enabled = EARLY_STOPPING
        self.patience = PATIENCE
        self.monitor = EARLY_STOPPING_MONITOR

        if "loss" in self.monitor.lower():
            self.mode = "min"
            self.best_score = math.inf # Initialize best_score to infinity for loss
        else:
            self.mode = "max"
            self.best_score = -math.inf

        self.counter = 0 # Initialize counter to track the number of epochs without improvement
        self.early_stop = False

    def reset(self):
        """
        resets the early stopping status
        """

        if "loss" in self.monitor.lower():
            self.best_score = math.inf
        else:
            self.best_score = -math.inf

        self.counter = 0
        self.early_stop = False

    def step(self, current_score):
        """
        Checks if the current score is better than the best score and updates the early stopping status accordingly

        current_score: (float) The current score of the monitored metric
        return: (bool) True if the current score is better than the best score, False otherwise
        """

        if not self.enabled:
            return False

        improvement = False # Flag to indicate if there is an improvement in the monitored metric

        if self.mode == "min":
            if current_score < self.best_score:
                self.best_score = current_score
                self.counter = 0
                improvement = True
            else:
                self.counter += 1
        else: # mode == "max"
            if current_score > self.best_score:
                self.best_score = current_score
                self.counter = 0
                improvement = True
            else:
                self.counter += 1

        # Check if the counter has reached the patience threshold and set early_stop to True if it has
        if self.counter >= self.patience:
            self.early_stop = True

        return improvement

    def should_stop(self):
        """
        Returns True if the training process should be stopped early, False otherwise
        """

        return self.early_stop

    def get_best_score(self):
        """
        Returns the best score of the monitored metric
        """

        return self.best_score

    def show_info(self):
        """
        Prints  the early stopping configuration and status
        """

        print(f"Enabled   : {self.enabled}")
        print(f"Monitor   : {self.monitor}")
        print(f"Mode      : {self.mode}")
        print(f"Patience  : {self.patience}")
        print(f"Best      : {self.best_score}")
        print(f"Counter   : {self.counter}")
        print(f"Stop      : {self.early_stop}")