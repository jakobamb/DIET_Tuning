"""Data configuration for DIET finetuning."""

from typing import Dict


# Dataset statistics for normalization and metadata
DATASET_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "input_size": 32,
        "is_rgb": True,
        "num_classes": 10,
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "input_size": 32,
        "is_rgb": True,
        "num_classes": 100,
    },
    "food101": {
        "mean": (
            0.485,
            0.456,
            0.406,
        ),  # ImageNet stats - good baseline for natural food images
        "std": (
            0.229,
            0.224,
            0.225,
        ),  # Food images have similar distribution to ImageNet
        "input_size": 224,  # Standard size for food classification (rescaled from 512)
        "is_rgb": True,
        "num_classes": 101,
    },
    "fgvc_aircraft": {
        "mean": (
            0.485,
            0.456,
            0.406,
        ),  # ImageNet stats - good baseline for aircraft images
        "std": (
            0.229,
            0.224,
            0.225,
        ),  # Aircraft images are natural outdoor scenes similar to ImageNet
        "input_size": 224,  # Standard size for fine-grained classification
        "is_rgb": True,
        "num_classes": 100,  # FGVC-Aircraft has 100 aircraft model classes
    },
    "pathmnist": {
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 9,
    },
    "chestmnist": {
        "mean": (0.4984,),
        "std": (0.2483,),
        "input_size": 224,
        "is_rgb": False,
        "num_classes": 2,
    },
    "dermamnist": {
        "mean": (0.7634, 0.5423, 0.5698),
        "std": (0.0841, 0.1246, 0.1043),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 7,
    },
    "octmnist": {
        "mean": (0.1778,),
        "std": (0.1316,),
        "input_size": 224,
        "is_rgb": False,
        "num_classes": 4,
    },
    "pneumoniamnist": {
        "mean": (0.5060,),
        "std": (0.2537,),
        "input_size": 224,
        "is_rgb": False,
        "num_classes": 2,
    },
    "plantnet300k": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 85,
    },
    "galaxy10_decals": {
        "mean": (0.097, 0.097, 0.097),
        "std": (0.174, 0.164, 0.156),
        "input_size": 256,
        "is_rgb": True,
        "num_classes": 10,
    },
    "crop14_balance": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "input_size": 512,
        "is_rgb": True,
        "num_classes": 14,
    },
}




def get_dataset_stats(dataset_name: str) -> Dict:
    """Get statistics for the specified dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dictionary with dataset statistics
    """
    return DATASET_STATS.get(
        dataset_name.lower(),
        {
            "mean": (0.5,),
            "std": (0.5,),
            "input_size": 224,
            "is_rgb": False,
            "num_classes": 10,
        },
    )
