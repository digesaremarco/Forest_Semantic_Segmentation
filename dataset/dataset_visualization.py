"""
Dataset visualization utilities for Forest Semantic Segmentation

This module provides functions to visualize:
    - RGB images
    - segmentation masks
    - image/mask pairs
    - image-mask overlays
    - random dataset samples
    - sample comparisons
"""

from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
import textwrap

from PIL import Image

from dataset.dataset_config_loader import TRAIN_IMAGE_DIR, TEST_IMAGE_DIR, TRAIN_MASK_DIR, TEST_MASK_DIR
from dataset.dataset_info import load_class_mapping


# ----------------------------------------------------------------------------
# Visualization utilities
# ----------------------------------------------------------------------------


def show_image(image_path):
    """
    Display an RGB image

    image_path: path to image
    """

    image = Image.open(image_path).convert("RGB")

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(Path(image_path).name)
    plt.axis("off")
    plt.show()

def show_mask(mask_path):
    """
    Display a segmentation mask

    mask_path: path to segmentation mask
    """

    mask = Image.open(mask_path).convert("RGB")

    plt.figure(figsize=(8, 8))
    plt.imshow(mask)
    plt.title(Path(mask_path).name)
    plt.axis("off")
    plt.show()

def show_sample(image_path, mask_path):
    """
    Display an image and its annotation
    """

    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("RGB"))

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].imshow(image)
    ax[0].set_title("RGB Image")
    ax[0].axis("off")
    ax[1].imshow(mask)
    ax[1].set_title("Annotation")
    ax[1].axis("off")
    plt.tight_layout()
    plt.show()

def show_overlay(image_path, mask_path, alpha=0.45):
    """
    Display an RGB image with segmentation overlay

    alpha: transparency of the overlay (0.0 to 1.0)
    """

    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("RGB"))

    overlay = (image.astype(np.float32) * (1 - alpha) + mask.astype(np.float32) * alpha).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(overlay)
    ax.set_title("Overlay")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

def show_random_sample(split="train", overlay=False):
    """
    Display a random sample

    overlay: whether to show the overlay of image and mask
    """

    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'.")

    if split == "train":
        image_dir, mask_dir = TRAIN_IMAGE_DIR, TRAIN_MASK_DIR
    else:
        image_dir, mask_dir = TEST_IMAGE_DIR, TEST_MASK_DIR

    image_paths = sorted(image_dir.glob("*.png"))
    image_path = random.choice(image_paths)
    mask_path = mask_dir / image_path.name

    print(f"Sample: {image_path.name}")

    if overlay:
        show_overlay(image_path, mask_path)
    else:
        show_sample(image_path, mask_path)

def show_samples_grid(split="train", n=9):
    """
    Display a grid of random RGB images

    n: number of random images
    """

    image_dir = TRAIN_IMAGE_DIR if split == "train" else TEST_IMAGE_DIR
    image_paths = sorted(image_dir.glob("*.png"))

    n = min(n, len(image_paths))
    samples = random.sample(image_paths, n)

    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, img_path in zip(axes, samples):
        image = Image.open(img_path)
        ax.imshow(image)
        ax.set_title(img_path.stem, fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def show_random_overlay(split="train"):
    """
    Display a random overlay
    """

    show_random_sample(split=split, overlay=True)

def compare_samples(indices=None, split="train"):
    """
    Compare multiple samples

    indices: list of indices to compare (None for random)
    """

    if split == "train":
        image_dir, mask_dir = TRAIN_IMAGE_DIR, TRAIN_MASK_DIR
    else:
        image_dir, mask_dir = TEST_IMAGE_DIR, TEST_MASK_DIR

    image_paths = sorted(image_dir.glob("*.png"))

    if indices is None:
        indices = random.sample(range(len(image_paths)), 3)

    fig, axes = plt.subplots(len(indices), 3, figsize=(15, 5 * len(indices)))

    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, idx in enumerate(indices):
        image_path = image_paths[idx]
        mask_path = mask_dir / image_path.name
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"))
        overlay = (image.astype(np.float32) * 0.6 + mask.astype(np.float32) * 0.4).astype(np.uint8)
        axes[row, 0].imshow(image)
        axes[row, 0].set_title("Image")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(mask)
        axes[row, 1].set_title("Mask")
        axes[row, 1].axis("off")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay")
        axes[row, 2].axis("off")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# Class visualization
# ----------------------------------------------------------------------------


def show_class_legend():
    """
    Display dataset classes with RGB colors in a compact grid layout
    """

    mapping = load_class_mapping()
    n_classes = len(mapping)

    n_cols = 8
    n_rows = -(-n_classes // n_cols) # ceil division to get number of rows
    cell_w, cell_h = 1.4, 1.3  # cell width and height in inches

    fig, ax = plt.subplots(figsize=(n_cols * cell_w * 0.9, n_rows * cell_h * 0.9))
    ax.axis("off")

    for i, (_, row) in enumerate(mapping.iterrows()):
        col = i % n_cols
        r = i // n_cols

        x = col * cell_w
        y = (n_rows - r - 1) * cell_h

        rgb = row["rgb"]
        color = np.array(rgb) / 255.0

        ax.add_patch(plt.Rectangle((x, y + 0.55), 1.0, 0.45, color=color, ec="black", linewidth=0.3))
        label = f"{row['id']} - {row['class']}"
        wrapped = "\n".join(textwrap.wrap(label, width=16))
        ax.text(x + 0.5, y + 0.45, wrapped, ha="center", va="top", fontsize=7.5)
        ax.text(x + 0.5, y + 0.05, f"{rgb}", ha="center", va="top", fontsize=6, color="gray") # RBG under the label

    ax.set_xlim(-0.2, n_cols * cell_w)
    ax.set_ylim(-0.2, n_rows * cell_h)
    ax.set_aspect("equal")
    plt.title("Forest Dataset Classes", fontsize=14)
    plt.tight_layout()
    plt.show()