"""
Dataset distribution analysis utilities for Forest Semantic Segmentation

This module provides functions to analyze:
    - pixel distribution per class
    - class imbalance
    - mask RGB labels
    - class statistics plots
"""

from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from dataset.dataset_config_loader import TRAIN_MASK_DIR, TEST_MASK_DIR
from dataset.dataset_info import load_class_mapping


# ----------------------------------------------------------------------------
# Class distribution
# ----------------------------------------------------------------------------


def class_distribution(split="train"):
    """
    Compute the number of pixels belonging to each class.

    :param split: either 'train' or 'test'
    """

    if split == "train":
        mask_dir = TRAIN_MASK_DIR
    elif split == "test":
        mask_dir = TEST_MASK_DIR
    else:
        raise ValueError("split must be 'train' or 'test'")

    mapping = load_class_mapping()
    rgb_to_class = {row["rgb"]: row["class"] for _, row in mapping.iterrows()}

    distribution = Counter()
    mask_paths = sorted(mask_dir.glob("*.png"))

    for mask_path in mask_paths:
        mask = np.asarray(Image.open(mask_path).convert("RGB"))
        colors, counts = np.unique(mask.reshape(-1, 3), axis=0, return_counts=True)
        for color, count in zip(colors, counts):
            rgb = tuple(color.tolist())
            class_name = rgb_to_class.get(rgb, "Unknown")
            distribution[class_name] += int(count)

    return distribution


# ----------------------------------------------------------------------------
# Printing statistics
# ----------------------------------------------------------------------------


def print_class_distribution(split="train"):
    """
    Print percentage of pixels for each class

    :param split: either 'train' or 'test'
    """

    distribution = class_distribution(split)
    total_pixels = sum(distribution.values())

    print("\nClass distribution")
    print("-" * 60)

    for cls, pixels in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        percentage = pixels / total_pixels * 100
        print(f"{cls:<30}{percentage:>8.2f}%")


# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------


def plot_class_distribution(split="train", normalize=True):
    """
    Plot class distribution

    :param split: either 'train' or 'test'
    :param normalize: whether to normalize the class distribution
    """

    distribution = class_distribution(split)
    labels = list(distribution.keys())
    values = np.array(list(distribution.values()), dtype=np.float64)

    if normalize:
        values = (values / values.sum()) * 100
        ylabel = "Percentage of pixels (%)"
    else:
        ylabel = "Pixel count"

    order = np.argsort(values)[::-1]
    labels = [labels[i] for i in order]
    values = values[order]

    plt.figure(figsize=(14, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"Class distribution ({split})")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# Mask inspection
# ----------------------------------------------------------------------------


def mask_statistics(split="train"):
    """
    Analyze RGB labels present in masks
    """

    if split == "train":
        mask_dir = TRAIN_MASK_DIR
    elif split == "test":
        mask_dir = TEST_MASK_DIR
    else:
        raise ValueError("split must be 'train' or 'test'")

    unique_colors = set()
    mask_paths = sorted(mask_dir.glob("*.png"))

    for mask_path in mask_paths:
        mask = np.asarray(Image.open(mask_path).convert("RGB"))
        colors = np.unique(mask.reshape(-1, 3), axis=0)
        for color in colors:
            unique_colors.add(tuple(color.tolist()))

    print(f"\nMasks analyzed: {len(mask_paths)}")
    print(f"Unique RGB labels found: {len(unique_colors)}")

    return unique_colors



def print_unknown_colors(split="train"):
    """
    Print RGB colors not present in mapping file
    """

    colors = mask_statistics(split)
    mapping = load_class_mapping()
    valid_colors = {row["rgb"] for _, row in mapping.iterrows()}
    unknown = colors - valid_colors

    print("\nUnknown colors:")

    if len(unknown) == 0:
        print("None")
    else:
        for c in sorted(unknown):
            print(c)