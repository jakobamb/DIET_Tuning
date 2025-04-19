"""Configuration module for DIET finetuning."""
import torch
import numpy as np

# Hardware settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default model settings
DEFAULT_MODEL_SETTINGS = {
    "resnet50": {
        "sizes": ["default"],
        "embedding_dims": {"default": 2048},
        "checkpoint_paths": {}
    },
    "dinov2": {
        "sizes": ["small", "base", "large", "giant"],
        "embedding_dims": {"small": 384, "base": 768, "large": 1024, "giant": 1536},
        "checkpoint_paths": {}
    },
    "mae": {
        "sizes": ["base", "large", "huge"],
        "embedding_dims": {"base": 768, "large": 1024, "huge": 1280},
        "checkpoint_paths": {}
    },
    "mambavision": {
        "sizes": ["T", "S", "B", "L", "L2", "L3-512", "L3-256"],
        "embedding_dims": {
            "T": 512, "S": 768, "B": 1024, "L": 1280, "L2": 1408, 
            "L3-512": 1568, "L3-256": 1568
        },
        "checkpoint_paths": {}
    },
    "ijepa": {
        "sizes": ["b16_1k", "b16_22k", "l14_22k", "h14_1k"],
        "embedding_dims": {
            "b16_1k": 1280, "b16_22k": 1280, "l14_22k": 1280, "h14_1k": 1280
        },
        "checkpoint_paths": {}
    },
    "aim": {
        "sizes": ["600M", "1B", "3B", "7B"],
        "embedding_dims": {"600M": 768, "1B": 1024, "3B": 1536, "7B": 2048},
        "checkpoint_paths": {}
    }
}

# Default dataset settings
DATASET_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "input_size": 32,
        "is_rgb": True,
        "num_classes": 10
    },
    "pathmnist": {
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "input_size": 28,
        "is_rgb": True,
        "num_classes": 9
    },
    "chestmnist": {
        "mean": (0.4984,),
        "std": (0.2483,),
        "input_size": 28,
        "is_rgb": False,
        "num_classes": 2
    },
    "dermamnist": {
        "mean": (0.7634, 0.5423, 0.5698),
        "std": (0.0841, 0.1246, 0.1043),
        "input_size": 28,
        "is_rgb": True,
        "num_classes": 7
    },
    "octmnist": {
        "mean": (0.1778,),
        "std": (0.1316,),
        "input_size": 28,
        "is_rgb": False,
        "num_classes": 4
    },
    "pneumoniamnist": {
        "mean": (0.5060,),
        "std": (0.2537,),
        "input_size": 28,
        "is_rgb": False,
        "num_classes": 2
    },
    "plantnet300k": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 85
    },
    "galaxy10_decals": {
        "mean": (0.097, 0.097, 0.097),
        "std": (0.174, 0.164, 0.156),
        "input_size": 256,  
        "is_rgb": True,
        "num_classes": 10
    },
    "crop14_balance": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "input_size": 512,
        "is_rgb": True,
        "num_classes": 14
    }
}

# Default training settings
DEFAULT_TRAINING_SETTINGS = {
    "num_epochs": 30,
    "batch_size": 20,
    "learning_rate": 5e-4,
    "weight_decay": 0.05,
    "da_strength": 1,
    "label_smoothing": 0.3,
    "num_diet_classes": 10,
    "projection_dim": 256,
    "temperature": 3.0,
    "limit_data": 1000,  # Set to np.inf for full dataset
    "eval_frequency": 5
}

# Wandb settings
WANDB_SETTINGS = {
    "project": "DIET-Finetuning",
    "enable_logging": True
}

# Expected performance thresholds for sanity checks
SANITY_CHECK_THRESHOLDS = {
    "dinov2": 0.91,
    "mae": 0.85,
    "mambavision": 0.85,
    "ijepa": 0.85,
    "aim": 0.75,
    "resnet50": 0.80
}

# Default paths
PATHS = {
    "checkpoints": "checkpoints",
    "results": "results",
    "data": "data"
}

def get_model_embedding_dim(backbone_type, model_size):
    """Get embedding dimension for specified model configuration.
    
    Args:
        backbone_type: Model type
        model_size: Model size
        
    Returns:
        int: Embedding dimension
    """
    if backbone_type in DEFAULT_MODEL_SETTINGS:
        if model_size in DEFAULT_MODEL_SETTINGS[backbone_type]["embedding_dims"]:
            return DEFAULT_MODEL_SETTINGS[backbone_type]["embedding_dims"][model_size]
    
    # Default fallback values
    fallback_dims = {
        "resnet50": 2048,
        "dinov2": 384,
        "mae": 768,
        "mambavision": 512,
        "ijepa": 1280,
        "aim": 768
    }
    return fallback_dims.get(backbone_type, 768)

def get_dataset_stats(dataset_name):
    """Get statistics for the specified dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        dict: Dataset statistics
    """
    return DATASET_STATS.get(dataset_name.lower(), {
        "mean": (0.5,),
        "std": (0.5,),
        "input_size": 28,
        "is_rgb": False,
        "num_classes": 10
    })

def get_training_settings(**overrides):
    """Get training settings with optional overrides.
    
    Args:
        **overrides: Override default settings
        
    Returns:
        dict: Training settings
    """
    settings = DEFAULT_TRAINING_SETTINGS.copy()
    settings.update(overrides)
    return settings