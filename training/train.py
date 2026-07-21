"""
Training entry point

This script initializes all training components and
starts the semantic segmentation training.
"""

import torch
from pip._vendor.resolvelib.resolvers import criterion
from torch.utils.data import DataLoader

from dataset.dataloader import ForestDataLoader
from dataset.transforms import ForestTransforms

from dataset.dataset_info import load_class_mapping

from models.segformer import SegFormer
from training import checkpoint

from training.losses import Losses
from training.optimizer import Optimizer
from training.scheduler import Scheduler
from training.metrics import Metrics
from training.logger import Logger
from training.checkpoint import Checkpoint
from training.early_stopping import EarlyStopping
from training.trainer import Trainer

from training.training_config_loader import (
    BATCH_SIZE,
    VALIDATION_SPLIT,
    SHUFFLE,
    RANDOM_SEED,
    NUM_WORKERS,
    PIN_MEMORY,
)

from models.model_config_loader import (
    DEVICE,
    ARCHITECTURE,
)

def main():

    print("Forest Semantic Segmentation Training")

    # Device settings
    if DEVICE.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(DEVICE.lower())
    print("Device:", device)

    # Load dataset
    train_transforms = ForestTransforms().get_transforms()
    test_transforms = ForestTransforms(train=False).get_transforms()

    dataloader = ForestDataLoader(
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        shuffle=SHUFFLE,
        random_seed=RANDOM_SEED,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        train_transform=train_transforms,
        test_transform=test_transforms,
    )
    dataloader.create_dataloaders()
    train_loader = dataloader.train_loader
    validation_loader = dataloader.validation_loader

    # Class mapping
    class_mapping = load_class_mapping()
    num_classes = len(class_mapping)

    # Model
    if ARCHITECTURE.lower() == "segformer":
        model = SegFormer().get_model()
    else:
        raise ValueError(f"Unsupported architecture: {ARCHITECTURE.lower()}")

    # Training components
    criterion = Losses().get_loss()
    optimizer = Optimizer(model).get_optimizer()
    scheduler = Scheduler(optimizer).get_scheduler()
    metrics = Metrics(num_classes)
    logger = Logger()
    checkpoint = Checkpoint()
    early_stopping = EarlyStopping()

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        logger=logger,
        checkpoint=checkpoint,
        early_stopping=early_stopping,
        device=device,
    )
    trainer.train()

if __name__ == "__main__":
    main()