from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


CONFIG_FILE = (
    PROJECT_ROOT /
    "configurations" /
    "dataset_configuration.yaml"
)


def load_dataset_config():
    with open(CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)

    return config["dataset"]


CONFIG = load_dataset_config()


DATA_DIR = PROJECT_ROOT / CONFIG["root"]


# Images
IMAGE_DIR = DATA_DIR / CONFIG["images"]["directory"]

TRAIN_IMAGE_DIR = IMAGE_DIR / CONFIG["images"]["train"]
TEST_IMAGE_DIR = IMAGE_DIR / CONFIG["images"]["test"]


# Annotations
ANNOTATION_DIR = DATA_DIR / CONFIG["annotations"]["directory"]

TRAIN_MASK_DIR = ANNOTATION_DIR / CONFIG["annotations"]["train"]
TEST_MASK_DIR = ANNOTATION_DIR / CONFIG["annotations"]["test"]


# Mapping
MAPPING_FILE = DATA_DIR / CONFIG["mapping_file"]


# Download IDs
RAW_IMAGES_ID = CONFIG["download"]["raw_images_id"]
ANNOTATIONS_ID = CONFIG["download"]["annotations_id"]
MAPPING_ID = CONFIG["download"]["mapping_id"]