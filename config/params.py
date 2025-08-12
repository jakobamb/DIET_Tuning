"""Default parameters for DIET finetuning framework command-line interface."""

import torch
from config import DEFAULT_PARAMS
from config.training_config import DEVICE

# Re-export the default params from the config package for command-line usage
BACKBONE_TYPE = DEFAULT_PARAMS["backbone_type"]
MODEL_SIZE = DEFAULT_PARAMS["model_size"]
DATASET_NAME = DEFAULT_PARAMS["dataset_name"]
LIMIT_DATA = DEFAULT_PARAMS["limit_data"]
NUM_EPOCHS = DEFAULT_PARAMS["num_epochs"]
BATCH_SIZE = DEFAULT_PARAMS["batch_size"]
LEARNING_RATE = DEFAULT_PARAMS["learning_rate"]
WEIGHT_DECAY = DEFAULT_PARAMS["weight_decay"]
DA_STRENGTH = DEFAULT_PARAMS["da_strength"]
RESUME_FROM = DEFAULT_PARAMS["resume_from"]
LABEL_SMOOTHING = DEFAULT_PARAMS["label_smoothing"]
NUM_DIET_CLASSES = DEFAULT_PARAMS["num_diet_classes"]
PROJECTION_DIM = DEFAULT_PARAMS["projection_dim"]
RUN_SANITY_CHECK = DEFAULT_PARAMS["run_sanity_check"]
EXPECTED_THRESHOLD = DEFAULT_PARAMS["expected_threshold"]