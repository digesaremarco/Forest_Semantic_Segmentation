"""
Training logger, this module provides utilities for logging the training
progress during semantic segmentation experiments.
"""

from pathlib import Path
import csv
from datetime import datetime

from training.training_config_loader import (
    LOG_DIRECTORY,
    LOG_FILENAME,
)


class Logger:

    def __init__(self):

        self.log_directory = LOG_DIRECTORY
        self.log_file_name = LOG_FILENAME

        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_directory / self.log_file_name

        self.header = [
            "epoch",
            "train_loss",
            "validation_loss",
            "learning_rate",
            "pixel_accuracy",
            "mean_pixel_accuracy",
            "mean_iou",
            "mean_dice",
            "frequency_weighted_iou",
        ]

    def initialize_logger(self):
        """
        Creates a new log file and writes the header row to it
        """

        with open(self.log_file, "w") as log_file:
            writer = csv.writer(log_file, delimiter=",")
            writer.writerow(self.header)

    def log(self, epoch, train_loss, validation_loss, learning_rate, metrics):
        """
        Logs the training progress for a given epoch
        """

        row = [
            epoch,
            float(train_loss),
            float(validation_loss),
            float(learning_rate),
            float(metrics["pixel_accuracy"]),
            float(metrics["mean_pixel_accuracy"]),
            float(metrics["mean_iou"]),
            float(metrics["mean_dice"]),
            float(metrics["frequency_weighted_iou"]),
        ]

        with open(self.log_file, "a") as log_file:
            writer = csv.writer(log_file, delimiter=",")
            writer.writerow(row)

    def read_logs(self):
        """
        Reads the log file and returns the logged data as a list of dictionaries

        return: list of dictionaries, each dictionary represents a row in the log file
        """

        if not self.log_file.exists():
            return []

        with open(self.log_file, "r") as log_file:
            reader = csv.reader(log_file)
            return list(reader)

    def clear_logs(self):
        """
        Clears the log file by deleting it and creating a new one with the header row
        """

        if self.log_file.exists():
            self.log_file.unlink()

    def show_info(self):

        print("Logger")
        print(f"Directory : {self.log_directory}")
        print(f"File      : {self.log_file}")


if __name__ == "__main__":

    logger = Logger()

    # Cancella eventuali log precedenti
    logger.clear_logs()

    # Inizializza il file CSV
    logger.initialize_logger()

    # Simulazione di alcune epoche di training
    for epoch in range(1, 6):

        train_loss = 1.0 / epoch
        validation_loss = 1.2 / epoch
        learning_rate = 0.001

        metrics = {
            "pixel_accuracy": 0.80 + epoch * 0.02,
            "mean_pixel_accuracy": 0.75 + epoch * 0.025,
            "mean_iou": 0.60 + epoch * 0.03,
            "mean_dice": 0.70 + epoch * 0.025,
            "frequency_weighted_iou": 0.58 + epoch * 0.03,
        }

        logger.log(
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
            learning_rate=learning_rate,
            metrics=metrics,
        )

    # Legge e stampa il contenuto del log
    print("\n=== Contenuto del file di log ===")
    logs = logger.read_logs()

    for row in logs:
        print(row)

    # Informazioni sul logger
    print("\n=== Info Logger ===")
    logger.show_info()