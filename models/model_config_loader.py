from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


CONFIG_FILE = (
    PROJECT_ROOT /
    "configurations" /
    "model_configuration.yaml"
)


def load_model_config():
    """
    Load the model configuration file.
    """

    with open(CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)

    return config["model"]


CONFIG = load_model_config()


# General configuration
ARCHITECTURE = CONFIG["architecture"]
BACKBONE = CONFIG["backbone"]
NUM_CLASSES = CONFIG["num_classes"]
PRETRAINED = CONFIG["pretrained"]
IGNORE_INDEX = CONFIG["ignore_index"]
DEVICE = CONFIG["device"]


# Checkpoints
CHECKPOINT_DIR = (
    PROJECT_ROOT /
    CONFIG["checkpoint"]["save_directory"]
)
SAVE_BEST_ONLY = CONFIG["checkpoint"]["save_best_only"]


# Hugging Face models
SEGFORMER_MODELS = CONFIG["huggingface"]["segformer"]