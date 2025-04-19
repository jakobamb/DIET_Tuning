"""Configuration settings for DIET finetuning framework."""
import torch
import numpy as np

# Hardware settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Default configuration parameters for DIET finetuning experiments.
These values are used as defaults for command-line arguments in main.py.
"""

# Model parameters
BACKBONE_TYPE = "ijepa"  # Options: "resnet50", "dinov2", "mae", "mambavision", "ijepa", "aim"
MODEL_SIZE = "b16_1k"     # Size depends on backbone type:
                         # - resnet50: N/A (uses ImageNet weights) x
                         # - dinov2: "small", "base", "large", "giant" x
                        # - ijepa: "b16_1k", "b16_22k", "l14_22k", "h14_1k" 

                         # - mae: "base", "large", "huge"
                         # - mambavision: "T", "T2", "S", "B", "L", "L2", "B-21K", "L-21K", "L2-512-21K", "L3-256-21K", "L3-512-21K"
                         # - aim: "600M", "1B", "3B", "7B"

# Dataset parameters
DATASET_NAME = "cifar10"  # Options: "cifar10", "pathmnist", "chestmnist", "dermamnist", "plantnet300k", "galaxy10_decals"
LIMIT_DATA = 1000  # Set to very large number (e.g., 1000000) for full dataset

# Training parameters
NUM_EPOCHS = 30  # Number of epochs to train
BATCH_SIZE = 20
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.05
DA_STRENGTH = 1  # Data augmentation strength (0-3)
RESUME_FROM = None  # Path to checkpoint to resume training from (e.g., "checkpoints/checkpoint_epoch_25.pt,None")

# DIET parameters
LABEL_SMOOTHING = 0.3  # Set to 0.0 to disable DIET (baseline)
NUM_DIET_CLASSES = 10
PROJECTION_DIM = 256  # Projection head output dimension

# Sanity check configuration
RUN_SANITY_CHECK = True
EXPECTED_THRESHOLD = 0.85