"""
Optimizers for semantic segmentation

This module provides a wrapper around different optimizers
used for training semantic segmentation models
"""

import torch.optim as optim

from training.training_config_loader import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    OPTIMIZER,
    OPTIMIZER_PARAMETERS,
)

class Optimizer:

    def __init__(self, model):

        self.model = model

        self.optimizer_name = OPTIMIZER.lower()
        self. weight_decay = WEIGHT_DECAY
        self.learning_rate = LEARNING_RATE
        self.optimizer_parameters = OPTIMIZER_PARAMETERS

        self.optimizer = self.build_optimizer()

    def build_optimizer(self):
        """
        Build optimizer and update learning rate

        return: torch.optim.Optimizer
        """

        model_parameters = self.model.parameters()

        if self.optimizer_name == "adam":
            optimizer = optim.Adam(
                model_parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        elif self.optimizer_name == "adamw":
            params = self.optimizer_parameters["adamw"]

            optimizer = optim.AdamW(
                model_parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=tuple(params["betas"]),
                eps=params["eps"],
            )

        elif self.optimizer_name == "sgd":
            params = self.optimizer_parameters["sgd"]

            optimizer = optim.SGD(
                model_parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=params["momentum"],
                nesterov=params["nesterov"],
            )

        else:
            raise ValueError("Unknown optimizer name")

        return optimizer

    def get_optimizer(self):
        """
        Return the optimizer
        """

        return self.optimizer

    def show_info(self):
        """
        Show info about the optimizer
        """

        print(f"Optimizer      : {self.optimizer_name}")
        print(f"Learning rate  : {self.learning_rate}")
        print(f"Weight decay   : {self.weight_decay}")

        if self.optimizer_name == "adamw":
            params = self.optimizer_parameters["adamw"]

            print(f"Betas          : {params['betas']}")
            print(f"Eps            : {params['eps']}")

        elif self.optimizer_name == "sgd":
            params = self.optimizer_parameters["sgd"]

            print(f"Momentum       : {params['momentum']}")
            print(f"Nesterov       : {params['nesterov']}")