"""
Utilities for creating PyTorch datasets and dataloaders

This module is responsible for:
    - creating the train/validation split
    - instantiating ForestDataset objects
    - creating PyTorch DataLoaders
"""

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset.forest_dataset import ForestDataset


class ForestDataLoader:
    """
    Create train, validation and test DataLoaders for the
    Forest Semantic Segmentation dataset.
    """

    def __init__(self, batch_size=8, validation_split=0.10, shuffle=True, random_seed=42,
        num_workers=4, pin_memory=True, train_transform=None, test_transform=None):

        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory # Enables faster data transfer to GPU

        self.train_transform = train_transform
        self.test_transform = test_transform

        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None

    def create_train_validation_split(self):
        """
        Create a reproducible train / validation split
        """

        dataset = ForestDataset(split="train")
        indices = list(range(len(dataset)))

        train_indices, val_indices = train_test_split(
            indices,
            test_size=self.validation_split,
            random_state=self.random_seed,
            shuffle=self.shuffle,
        )

        return train_indices, val_indices


    def create_datasets(self):
        """
        Create train, validation and test datasets
        """

        train_indices, val_indices = self.create_train_validation_split()

        self.train_dataset = ForestDataset(
            split="train", indices=train_indices, transform=self.train_transform
        )
        self.validation_dataset = ForestDataset(
            split="train", indices=val_indices, transform=self.test_transform
        )
        self.test_dataset = ForestDataset(split="test", transform=self.test_transform)


    def create_dataloaders(self):
        """
        Create PyTorch DataLoaders
        """

        self.create_datasets()

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        self.validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle validation data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle test data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


    def show_info(self):
        """
        Print information about the datasets and dataloaders
        """

        print("Train Dataset:")
        print(f"  Number of samples: {len(self.train_dataset)}")
        print(f"  Number of classes: {self.train_dataset.num_classes}")
        print(f"  Class names: {self.train_dataset.class_names}")

        print("\nValidation Dataset:")
        print(f"  Number of samples: {len(self.validation_dataset)}")
        print(f"  Number of classes: {self.validation_dataset.num_classes}")
        print(f"  Class names: {self.validation_dataset.class_names}")

        print("\nTest Dataset:")
        print(f"  Number of samples: {len(self.test_dataset)}")
        print(f"  Number of classes: {self.test_dataset.num_classes}")
        print(f"  Class names: {self.test_dataset.class_names}")