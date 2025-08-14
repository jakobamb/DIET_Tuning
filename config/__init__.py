"""Main configuration for DIET finetuning."""

from typing import Dict, Any

# Re-export key configuration components for easy imports
from config.data_config import get_dataset_stats
from config.model_config import get_model_embedding_dim
from config.training_config import TrainerConfig, DEVICE


def create_trainer_config(args) -> TrainerConfig:
    """Create a TrainerConfig from command line arguments.

    Args:
        args: Command line arguments

    Returns:
        Configuration for the trainer
    """
    dataset_info = get_dataset_stats(args.dataset)

    config_dict = {
        # Model configuration
        "backbone_type": args.backbone,
        "model_size": args.model_size,
        "num_classes": dataset_info["num_classes"],
        "projection_dim": args.projection_dim,
        # Data configuration
        "dataset_name": args.dataset,
        "batch_size": args.batch_size,
        "limit_data": args.limit_data,
        "num_diet_classes": args.num_diet_classes,
        "input_size": dataset_info["input_size"],
        "is_rgb": dataset_info["is_rgb"],
        "dataset_mean": dataset_info["mean"],
        "dataset_std": dataset_info["std"],
        # Training configuration
        "num_epochs": args.num_epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "training_mode": args.training_mode,
        "eval_frequency": args.eval_frequency,
        "checkpoint_freq": args.checkpoint_freq,
        # Paths
        "checkpoint_dir": args.checkpoint_dir,
        # Logging settings
        "enable_wandb": args.use_wandb,
    }

    return TrainerConfig.from_dict(config_dict)


def create_experiment_config(args, embedding_dim: int, dataset_info: Dict) -> Dict:
    """Create experiment configuration dictionary for wandb.

    Args:
        args: Command line arguments
        embedding_dim: Model embedding dimension
        dataset_info: Dataset information

    Returns:
        Experiment configuration
    """
    return {
        # Model parameters
        "backbone_type": args.backbone,
        "model_size": args.model_size,
        "embedding_dim": embedding_dim,
        "projection_dim": args.projection_dim,
        # Dataset parameters
        "dataset_name": args.dataset,
        "num_classes": dataset_info["num_classes"],
        "num_diet_classes": args.num_diet_classes,
        "input_size": dataset_info["input_size"],
        "is_rgb": dataset_info["is_rgb"],
        "limit_data": args.limit_data,
        # Training parameters
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "da_strength": args.da_strength,
        "label_smoothing": args.label_smoothing,
        "training_mode": args.training_mode,
        "checkpoint_freq": args.checkpoint_freq,
        "resume_from": args.resume_from,
        # Logging
        "wandb_dir": args.wandb_dir,
        "wandb_prefix": args.wandb_prefix,
        # DIET-specific settings
        "is_diet_active": args.label_smoothing > 0,
    }
