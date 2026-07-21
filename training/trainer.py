"""
Trainer for semantic segmentation.

This module provides the training loop used for semantic
segmentation models.
"""

import time

import torch
from tqdm import tqdm

from training.metrics import Metrics
from training.training_config_loader import (
    EPOCHS,
    MIXED_PRECISION,
    GRADIENT_CLIPPING,
)

class Trainer:

    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, metrics,
                 logger, checkpoint, early_stopping, device):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.metrics = metrics
        self.logger = logger
        self.checkpoint = checkpoint
        self.early_stopping = early_stopping

        self.device = device


        self.epochs = EPOCHS
        self.mixed_precision = MIXED_PRECISION
        self.gradient_clipping = GRADIENT_CLIPPING

        self.best_metric = float("-inf")

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision) # Initialize GradScaler for mixed precision training

    def train_epoch(self, epoch):
        """
        Train the model for one epoch

        epoch: int, current epoch number
        return: float, average loss for the epoch
        """

        self.model.train()

        running_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Train]", leave=False)

        for images, masks in progress_bar:
            images = images.to(self.device, non_blocking=True) # non_blocking=True allows for asynchronous data transfer to GPU
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True) # set_to_none=True can improve performance by avoiding unnecessary memory allocations

            with torch.amp.autocast(device_type=self.device.type, enabled=self.mixed_precision): # Enable mixed precision training if specified
                outputs = self.model(images)
                logits = outputs.logits
                # Interpolate the logits to match the size of the masks (for example, SegFormer outputs may be smaller than the input image size)
                logits = torch.nn.functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = self.criterion(logits, masks)

            self.scaler.scale(loss).backward() # Scale the loss for mixed precision training

            if self.gradient_clipping:
                self.scaler.unscale_(self.optimizer) # Unscale the gradients before clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping) # Clip gradients to prevent exploding gradients

            self.scaler.step(self.optimizer) # Update the model parameters
            self.scaler.update() # Update the scale for mixed precision training

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")

        epoch_loss = running_loss / len(self.train_loader)

        return epoch_loss

    def validation_epoch(self, epoch):
        """
        Validate the model for one epoch

        epoch: int, current epoch number
        return: tuple, validation loss and metrics
        """

        self.model.eval()

        running_loss = 0.0
        running_metrics = {
            "pixel_accuracy": 0.0,
            "mean_pixel_accuracy": 0.0,
            "mean_iou": 0.0,
            "mean_dice": 0.0,
            "frequency_weighted_iou": 0.0,
        }
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Validation]", leave=False)

        with torch.no_grad():
            for images, masks in progress_bar:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                outputs = self.model(images)
                logits = outputs.logits
                logits = torch.nn.functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

                loss = self.criterion(logits, masks)
                running_loss += loss.item()

                predictions = torch.argmax(logits, dim=1) # Get the predicted class for each pixel
                metrics = self.metrics.compute_metrics(predictions, masks)
                for key in running_metrics:
                    running_metrics[key] += float(metrics[key])

                progress_bar.set_postfix(loss=f"{loss.item():.4f}", mIoU=f"{metrics['mean_iou']:.4f}")

        validation_loss = running_loss / len(self.val_loader)
        for key in running_metrics:
            running_metrics[key] /= len(self.val_loader)

        return validation_loss, running_metrics

    def train(self):
        """
        Execute the training loop

        return: float, best metric value achieved during training
        """

        self.logger.clear_logs()
        self.logger.initialize_logger()

        print("Starting training...")
        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start_time = time.time()

            train_loss = self.train_epoch(epoch) # Training
            validation_loss, running_metrics = self.validation_epoch(epoch) # Validation

            if self.scheduler:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            self.logger.log(epoch + 1, train_loss, validation_loss, current_lr, running_metrics) # Log the results

            current_metric = running_metrics["mean_iou"]
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.checkpoint.save_checkpoint(self.model, self.optimizer, self.scheduler, epoch + 1, self.best_metric)

            self.early_stopping.step(current_metric)
            if self.early_stopping.should_stop():
                print("Early stopping triggered")
                break

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            print(
                f"Epoch [{epoch + 1}/{self.epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Validation Loss: {validation_loss:.4f} | "
                f"mIoU: {running_metrics['mean_iou']:.4f} | "
                f"Dice: {running_metrics['mean_dice']:.4f} | "
                f"Pixel Acc: {running_metrics['pixel_accuracy']:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_duration:.1f}s"
            )

        total_duration = time.time() - start_time
        print("Training completed")
        print(f"Best mIoU : {self.best_metric:.4f}")
        print(f"Total time: {total_duration:.1f} s")

        return self.best_metric