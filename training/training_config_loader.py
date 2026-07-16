from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


CONFIG_FILE = (
    PROJECT_ROOT /
    "configurations" /
    "training_configuration.yaml"
)


def load_training_config():
    """
    Load the training configuration file.
    """

    with open(CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)

    return config["training"]


CONFIG = load_training_config()


# General training parameters
EPOCHS = CONFIG["epochs"]
BATCH_SIZE = CONFIG["batch_size"]
LEARNING_RATE = CONFIG["learning_rate"]
WEIGHT_DECAY = CONFIG["weight_decay"]
OPTIMIZER = CONFIG["optimizer"]
SCHEDULER = CONFIG["scheduler"]
WARMUP_EPOCHS = CONFIG["warmup_epochs"]
VALIDATION_SPLIT = CONFIG["validation_split"]
SHUFFLE = CONFIG["shuffle"]
RANDOM_SEED = CONFIG["random_seed"]
NUM_WORKERS = CONFIG["num_workers"]
PIN_MEMORY = CONFIG["pin_memory"]
MIXED_PRECISION = CONFIG["mixed_precision"]
GRADIENT_CLIPPING = CONFIG["gradient_clipping"]


# Early stopping
EARLY_STOPPING = CONFIG["early_stopping"]["enabled"]
PATIENCE = CONFIG["early_stopping"]["patience"]
EARLY_STOPPING_MONITOR = CONFIG["early_stopping"]["monitor"]


# Loss
LOSS_FUNCTION = CONFIG["loss"]["function"]
IGNORE_INDEX = CONFIG["loss"]["ignore_index"]
CLASS_WEIGHTS = CONFIG["loss"]["class_weights"]


# Optimizer parameters
OPTIMIZER_PARAMETERS = CONFIG["optimizer_parameters"]


# Scheduler parameters
SCHEDULER_PARAMETERS = CONFIG["scheduler_parameters"]