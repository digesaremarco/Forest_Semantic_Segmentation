"""
Learning rate schedulers for semantic segmentation

This module provides a wrapper around different learning rate
schedulers used for training semantic segmentation models
"""

import torch.optim.lr_scheduler as lr_scheduler

from training.training_config_loader import (
    SCHEDULER,
    SCHEDULER_PARAMETERS,
    EPOCHS,
)


class Scheduler:

    def __init__(self, optimizer):

        self.optimizer = optimizer

        self.scheduler_name = SCHEDULER.lower()
        self.scheduler_parameters = SCHEDULER_PARAMETERS

        self.scheduler = self.build_scheduler()

    def build_scheduler(self):
        """
        Build scheduler and update learning rate

        return: torch.optim.lr_scheduler._LRScheduler
        """

        if self.scheduler_name == "cosine":
            params = self.scheduler_parameters["cosine"]

            scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=EPOCHS,
                eta_min=params["eta_min"],
            )

        elif self.scheduler_name == "polynomial":
            params = self.scheduler_parameters["polynomial"]

            scheduler = lr_scheduler.PolynomialLR(
                self.optimizer,
                total_iters=EPOCHS,
                power=params["power"],
            )

        elif self.scheduler_name == "step":
            params = self.scheduler_parameters["step"]

            scheduler = lr_scheduler.StepLR(
                self.optimizer,
                step_size=params["step_size"],
                gamma=params["gamma"],
            )

        elif self.scheduler_name == "none":
            scheduler = None

        else:
            raise ValueError(f"Scheduler '{self.scheduler_name}' not supported")

        return scheduler