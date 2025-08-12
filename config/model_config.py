"""Model configuration for DIET finetuning."""

from typing import Dict, List


# Model architecture specifications
MODEL_SPECS = {
    "resnet50": {
        "sizes": [
            "default", "a1_in1k", "clip.cc12m", "clip.openai", 
            "tv2_in1k", "gluon_in1k", "fb_swsl_ig1b_ft_in1k", "ram_in1k"
        ],
        "embedding_dims": {
            "default": 2048,
            "a1_in1k": 2048,
            "clip.cc12m": 2048,
            "clip.openai": 2048,
            "tv2_in1k": 2048,
            "gluon_in1k": 2048,
            "fb_swsl_ig1b_ft_in1k": 2048,
            "ram_in1k": 2048,
        },
    },
    "dinov2": {
        "sizes": ["small", "base", "large", "giant"],
        "embedding_dims": {
            "small": 384, 
            "base": 768, 
            "large": 1024, 
            "giant": 1536
        },
    },
    "mae": {
        "sizes": ["base", "large", "huge"],
        "embedding_dims": {
            "base": 768, 
            "large": 1024, 
            "huge": 1280
        },
    },
    "mambavision": {
        "sizes": ["T", "S", "B", "L", "L2", "L3-512", "L3-256"],
        "embedding_dims": {
            "T": 512,
            "S": 768,
            "B": 1024,
            "L": 1280,
            "L2": 1408,
            "L3-512": 1568,
            "L3-256": 1568,
        },
    },
    "ijepa": {
        "sizes": ["b16_1k", "b16_22k", "l14_22k", "h14_1k"],
        "embedding_dims": {
            "b16_1k": 1280,
            "b16_22k": 1280,
            "l14_22k": 1280,
            "h14_1k": 1280,
        },
    },
    "aim": {
        "sizes": ["600M", "1B", "3B", "7B"],
        "embedding_dims": {
            "600M": 768, 
            "1B": 1024, 
            "3B": 1536, 
            "7B": 2048
        },
    },
    "simclr": {
        "sizes": ["resnet50-1x", "resnet50-2x", "resnet50-4x"],
        "embedding_dims": {
            "resnet50-1x": 2048,
            "resnet50-2x": 2048,
            "resnet50-4x": 2048,
        }
    },
}

# Default model configuration
DEFAULT_MODEL_CONFIG = {
    "backbone_type": "simclr",
    "model_size": "resnet50-1x",
    "projection_dim": 256,
    "temperature": 3.0,
}

# Expected performance thresholds for sanity checks
SANITY_CHECK_THRESHOLDS = {
    "dinov2": 0.91,
    "mae": 0.85,
    "mambavision": 0.85,
    "ijepa": 0.85,
    "aim": 0.75,
    "resnet50": 0.80,
    "simclr": 0.80,
}


def get_model_embedding_dim(backbone_type: str, model_size: str) -> int:
    """Get embedding dimension for specified model configuration.

    Args:
        backbone_type: Model type
        model_size: Model size

    Returns:
        Embedding dimension for the specified model
    """
    if backbone_type in MODEL_SPECS:
        if model_size in MODEL_SPECS[backbone_type]["embedding_dims"]:
            return MODEL_SPECS[backbone_type]["embedding_dims"][model_size]

    # Default fallback values
    fallback_dims = {
        "resnet50": 2048,
        "dinov2": 384,
        "mae": 768,
        "mambavision": 512,
        "ijepa": 1280,
        "aim": 768,
        "simclr": 2048,
    }
    return fallback_dims.get(backbone_type, 768)


def get_available_model_sizes(backbone_type: str) -> List[str]:
    """Get available sizes for a specific model backbone.
    
    Args:
        backbone_type: Model type
        
    Returns:
        List of available model sizes
    """
    if backbone_type in MODEL_SPECS:
        return MODEL_SPECS[backbone_type]["sizes"]
    return []
