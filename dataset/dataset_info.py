"""
Dataset exploration utilities for Forest Semantic Segmentation.

This module provides functions to inspect:
    - dataset structure
    - number of files
    - image properties
    - dataset statistics

"""
import os
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "dataset" / "data"


# =============================================================================
# Dataset structure
# =============================================================================

def print_dataset_structure(root_dir=DATA_DIR, max_depth=3, max_files=3):
    """
    Print the directory structure of the dataset up to a specified depth.

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


