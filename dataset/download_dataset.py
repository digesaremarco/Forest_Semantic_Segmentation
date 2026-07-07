"""
Download and extract the Forest Semantic Segmentation dataset.

The dataset is downloaded from Google Drive, extracted into:

    datasets/data/

The ZIP archive is automatically removed after extraction.
"""

from pathlib import Path
import zipfile
import shutil

import gdown


# =============================================================================
# Configuration
# =============================================================================

FILE_ID = "13kJXmbjWLZlh45KpUkN2rqEfRbf64Iqs"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "datasets" / "data"
ZIP_PATH = DATA_DIR / "dataset.zip"


# =============================================================================
# Functions
# =============================================================================

def dataset_exists() -> bool:
    """
    Check if the dataset exists

    Returns: True if the dataset exists, False otherwise
    """

    return DATA_DIR.exists() and any(DATA_DIR.iterdir())

def download_and_extract():
    """
    Download and extract the dataset from Google Drive.

    The dataset is downloaded as a ZIP archive, extracted into the data directory,
    and the ZIP archive is removed after extraction.
    """

    # Check if the dataset already exists
    if dataset_exists():
        print(f"Dataset already exists in {DATA_DIR}. Skipping download.")
        return

    # Create the data directory if it doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download the dataset ZIP archive
    print(f"Downloading dataset from {DOWNLOAD_URL}...")
    gdown.download(DOWNLOAD_URL, str(ZIP_PATH), quiet=False)

    # Extract the ZIP archive
    print(f"Extracting dataset to {DATA_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    # Remove the ZIP archive
    print(f"Removing ZIP archive {ZIP_PATH}...")
    ZIP_PATH.unlink()

    print("Dataset download and extraction complete.")