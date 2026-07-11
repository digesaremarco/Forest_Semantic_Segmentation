"""
Dataset exploration utilities for Forest Semantic Segmentation

This module provides functions to inspect:
    - dataset structure
    - number of files
    - image properties
    - dataset statistics

"""
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from PIL import Image

from dataset.dataset_config_loader import (
    TRAIN_IMAGE_DIR,
    TEST_IMAGE_DIR,
    TRAIN_MASK_DIR,
    TEST_MASK_DIR,
    DATA_DIR,
    MAPPING_FILE
)


# ----------------------------------------------------------------------------
# Dataset structure
# ----------------------------------------------------------------------------

def print_dataset_structure(root_dir=DATA_DIR, max_depth=3, max_files=3):
    """
    Print the directory structure of the dataset up to a specified depth

    :param root_dir: root directory of dataset
    :param max_depth: maximum depth to display
    :param max_files: maximum number of files to display per directory
    """

    root_dir = Path(root_dir)

    print("\nDataset structure:")

    files_per_dir = {}

    for path in sorted(root_dir.rglob("*")):
        depth = len(path.relative_to(root_dir).parts)

        if depth <= max_depth:
            indent = "    " * (depth - 1)

            if path.is_dir():
                print(f"{indent}[DIR] {path.name}")
                files_per_dir[path] = 0

            else:
                parent = path.parent
                count = files_per_dir.get(parent, 0)

                if count < max_files:
                    print(f"{indent}[FILE] {path.name}")
                elif count == max_files:
                    print(f"{indent}...")

                files_per_dir[parent] = count + 1


def count_files(root_dir=DATA_DIR):
    """
    Count files in the train and test folders

    :param root_dir: root directory of dataset
    """

    root_dir = Path(root_dir)

    train_dir = root_dir / "images" / "train"
    test_dir = root_dir / "images" / "test"

    train_count = sum(1 for p in train_dir.iterdir() if p.is_file())
    test_count = sum(1 for p in test_dir.iterdir() if p.is_file())

    print(f"Train files: {train_count}")
    print(f"Test files:  {test_count}")
    print(f"Total files: {train_count + test_count}")

    return train_count, test_count



# ----------------------------------------------------------------------------
# Image analysis
# ----------------------------------------------------------------------------

def find_images(root_dir=DATA_DIR):
    """
    Find images in the train and test folders

    :param root_dir: root directory of dataset
    :return: list of image paths
    """

    root_dir = Path(root_dir)

    train_dir = root_dir / "images" / "train"
    test_dir = root_dir / "images" / "test"

    train_images = list(train_dir.glob("*.png"))
    test_images = list(test_dir.glob("*.png"))

    return train_images, test_images

def image_summary(root_dir=DATA_DIR):
    """
    Print basic image statistics

    :param root_dir: root directory of dataset
    """

    train_images, test_images = find_images(root_dir)
    all_images = train_images + test_images
    print(f"Number of images: {len(all_images)}")

    # Get image sizes
    sizes = [Image.open(img).size for img in all_images]
    widths, heights = zip(*sizes)

    print(f"Image width range: {min(widths)} - {max(widths)}")
    print(f"Max image width: {max(widths)}")
    print(f"Min image width: {min(widths)}")
    print(f"Image height range: {min(heights)} - {max(heights)}")
    print(f"Max image height: {max(heights)}")
    print(f"Min image height: {min(heights)}")

    # Count unique sizes
    size_counts = Counter(sizes)
    print(f"Unique image sizes: {len(size_counts)}")
    for size, count in size_counts.items():
        print(f"Size {size}: {count} images")


def compute_rgb_statistics(root_dir=DATA_DIR, max_images=None):
    """
    Compute RGB mean and standard deviation (Useful for normalization before training)

    :param root_dir: root directory of dataset
    :param max_images: maximum number of images to use for statistics (None for all)
    :return: mean, std (each as a 3-element array for R, G, B channels)
    """

    train_images, test_images = find_images(root_dir)
    all_images = train_images + test_images

    if max_images:
        all_images = all_images[:max_images]

    pixels = []

    for img_path in all_images:
        img = Image.open(img_path).convert("RGB")
        img = np.array(img) / 255.0
        pixels.append(img.reshape(-1, 3))

    pixels = np.concatenate(pixels, axis=0)
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)

    return mean, std


# ----------------------------------------------------------------------------
# Mapping utilities
# ----------------------------------------------------------------------------

def load_class_mapping():
    """
    Load the segmentation mapping file
    """

    mapping = pd.read_excel(MAPPING_FILE, header=1)

    mapping = mapping.iloc[:, :3] # First 3 columns
    mapping.columns = ["id", "rgb", "class"]

    mapping = mapping.dropna(subset=["id", "rgb", "class"])
    mapping["id"] = mapping["id"].astype(int)

    # RGB string -> tuple
    mapping["rgb"] = mapping["rgb"].apply(
        lambda x: tuple(
            int(v.strip())
            for v in str(x).split(",")
        )
    )

    return mapping

def list_classes():
    """
    Print all dataset classes.
    """

    mapping = load_class_mapping()

    print("\nDataset classes\n")

    for _, row in mapping.iterrows():

        print(f"{row['id']:>2} : {row['class']}")


# ----------------------------------------------------------------------------
# Dataset summary
# ----------------------------------------------------------------------------

def dataset_summary():
    """
    Print a complete dataset summary
    """

    print("=" * 70)
    print("Forest Semantic Segmentation Dataset")
    print("=" * 70)

    count_files()

    print()

    image_summary()

    print()

    list_classes()

    print("\nRGB statistics")

    mean, std = compute_rgb_statistics()
    print(f"Mean: {mean}")
    print(f"Std : {std}")