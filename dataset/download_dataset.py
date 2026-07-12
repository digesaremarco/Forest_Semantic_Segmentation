"""
Download and extract the Forest Semantic Segmentation dataset
This script downloads:
    - Raw RGB images
    - RGB annotation images
    - Segmentation mapping

The downloaded files are stored inside dataset/data/
"""

from pathlib import Path
import zipfile

import gdown

from dataset.dataset_config_loader import (
    DATA_DIR,
    RAW_IMAGES_ID,
    ANNOTATIONS_ID,
    MAPPING_ID,
)

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------

def _download_zip(file_id: str, output_dir: Path, archive_name: str):
    """
    Download and extract a ZIP archive from Google Drive

    file_id: id of the file to download
    output_dir: directory to download and extract the archive
    archive_name: name of the ZIP archive
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / archive_name

    if any(output_dir.iterdir()):
        print(f"{output_dir.name} already exists. Skipping.")
        return

    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"Downloading {output_dir.name}...")
    gdown.download(url, str(zip_path), quiet=False)

    print(f"Extracting {output_dir.name}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    zip_path.unlink()

    print(f"{output_dir.name} downloaded successfully.\n")


def _download_file(file_id: str, output_path: Path):
    """
    Download a single file from Google Drive

    file_id: id of the file to download
    output_path: path to save the downloaded file
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"{output_path.name} already exists. Skipping.")
        return

    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"Downloading {output_path.name}...")
    gdown.download(url, str(output_path), quiet=False)

    print(f"{output_path.name} downloaded successfully.\n")


def download_dataset():
    """
    Download the complete Forest Semantic Segmentation dataset
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    _download_zip(
        RAW_IMAGES_ID,
        DATA_DIR / "images",
        "images.zip"
    )

    _download_zip(
        ANNOTATIONS_ID,
        DATA_DIR / "annotations",
        "annotations.zip"
    )

    _download_file(
        MAPPING_ID,
        DATA_DIR / "segmentation_mapping.xlsx"
    )

    print("Dataset download completed.")
