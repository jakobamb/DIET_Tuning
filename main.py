#!/usr/bin/env python
"""
Main entry point for running DIET finetuning experiments.
"""
# Standard library imports
import os
import sys
import argparse
from datetime import datetime

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Third-party imports
import torch
import wandb

# Configuration imports
from config import create_experiment_config, create_trainer_config
from config.training_config import DEVICE
from config.params import (
    BACKBONE_TYPE,
    MODEL_SIZE,
    DATASET_NAME,
    LIMIT_DATA,
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DA_STRENGTH,
    RESUME_FROM,
    LABEL_SMOOTHING,
    NUM_DIET_CLASSES,
    PROJECTION_DIM,
)

# Data loading
from loaders.data_loader import prepare_data_loaders

# Training components
from training.trainer import DIETTrainer, create_projection_head

# Utility modules
from utils.wandb_logger import (
    init_wandb,
    create_experiment_dashboard,
    log_model_architecture,
)
from utils.sanity_check import unified_sanity_check

# Model implementations
from models.aim import get_aim_model
from models.dinov2 import get_dinov2_model
from models.ijepa import get_ijepa_model
from models.mae import get_mae_model
from models.mambavision import get_mambavision_model
from models.resnet50 import get_resnet50_model
from models.simclr import get_simclr_model


def get_model(backbone_type, model_size):
    """Load the appropriate model based on backbone type"""

    # ... (model selection variables) ...
    sanity_results = unified_sanity_check(backbone_type, model_size)
    print(sanity_results)

    print(f"Creating {backbone_type}-{model_size} model...")

    if backbone_type == "resnet50":
        return get_resnet50_model(DEVICE, model_size=model_size)

    elif backbone_type == "dinov2":
        return get_dinov2_model(DEVICE, model_size=model_size)

    elif backbone_type == "mae":
        return get_mae_model(DEVICE, model_size=model_size)

    elif backbone_type == "ijepa":
        return get_ijepa_model(DEVICE, model_size=model_size)

    elif backbone_type == "mambavision":
        return get_mambavision_model(DEVICE, model_variant=model_size)

    elif backbone_type == "aim":
        return get_aim_model(DEVICE, model_size=model_size)

    elif backbone_type == "simclr":
        return get_simclr_model(DEVICE, model_size=model_size)

    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


def train(args):
    """Main training function"""
    print("\n" + "=" * 70)
    print(
        f"DIET FINETUNING EXPERIMENT: {args.backbone.upper()}-{args.model_size} on {args.dataset}"
    )
    print("=" * 70)

    # Basic settings
    device = DEVICE
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    train_loader, test_loader, dataset_info = prepare_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_diet_classes=args.num_diet_classes,
        da_strength=args.da_strength,
        limit_data=args.limit_data,
    )

    num_classes = dataset_info["num_classes"]
    print(f"Loaded dataset with {num_classes} classes")

    # Create the backbone model
    net, embedding_dim = get_model(args.backbone, args.model_size)
    print(f"Created backbone with embedding dimension: {embedding_dim}")

    # Create projection head for DIET
    projection_head = create_projection_head(
        embedding_dim, embedding_dim, args.projection_dim, device
    )

    # Create classification heads
    W_probe = torch.nn.Linear(embedding_dim, num_classes).to(device)
    W_diet = torch.nn.Linear(args.projection_dim, args.num_diet_classes, bias=False).to(
        device
    )

    # Create optimizer
    print(f"Creating optimizer with lr={args.lr}, weight_decay={args.weight_decay}")
    optimizer = torch.optim.AdamW(
        list(net.parameters())
        + list(W_probe.parameters())
        + list(W_diet.parameters())
        + list(projection_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Add learning rate scheduler
    print("Creating learning rate scheduler (cosine annealing)")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-5
    )

    # Load from checkpoint if specified
    start_epoch = 0
    if args.resume_from:
        checkpoint_path = (
            os.path.join(args.checkpoint_dir, args.resume_from)
            if not os.path.isabs(args.resume_from)
            else args.resume_from
        )
        if os.path.exists(checkpoint_path):
            print(f"\nLoading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )

            # Load model and optimizer states
            net.load_state_dict(checkpoint["model_state_dict"])
            projection_head.load_state_dict(checkpoint["projection_head_state_dict"])
            W_probe.load_state_dict(checkpoint["W_probe_state_dict"])
            W_diet.load_state_dict(checkpoint["W_diet_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Get the starting epoch
            start_epoch = checkpoint["epoch"]
            print(f"Resuming from epoch {start_epoch}")

            # Adjust scheduler to the correct epoch
            for _ in range(start_epoch):
                scheduler.step()

            print(
                f"Successfully loaded checkpoint. Current learning rate: {scheduler.get_last_lr()[0]:.6f}"
            )
        else:
            print(
                f"Warning: Checkpoint file {checkpoint_path} not found. Starting from scratch."
            )

    # Create experiment configuration dictionary for wandb using our new config structure
    experiment_config = create_experiment_config(args, embedding_dim, dataset_info)

    # Add any additional fields that aren't in the config structure
    experiment_config["start_epoch"] = start_epoch

    # Initialize wandb if enabled
    run = None
    if args.use_wandb:
        run = init_wandb(experiment_config)
        log_model_architecture(run, net, projection_head, W_probe, W_diet)

    # Create a trainer config object
    trainer_config = create_trainer_config(args)

    # Create the trainer with our config
    trainer = DIETTrainer(
        model=net,
        projection_head=projection_head,
        W_probe=W_probe,
        W_diet=W_diet,
        device=device,
        num_classes=num_classes,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smoothing=args.label_smoothing,
        checkpoint_dir=args.checkpoint_dir,
        config=trainer_config,
    )

    # Run the training
    metrics_history, initial_results, final_results = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=args.num_epochs,
        run=run,
        eval_frequency=args.eval_frequency,
        start_epoch=start_epoch,
    )

    # Create experiment dashboard in wandb
    if args.use_wandb and run is not None:
        create_experiment_dashboard(
            run, metrics_history, initial_results, final_results, experiment_config
        )

        # Finish the wandb run
        run.finish()

    return metrics_history, initial_results, final_results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DIET Finetuning Framework")

    # Model arguments
    parser.add_argument(
        "--backbone",
        type=str,
        default=BACKBONE_TYPE,
        choices=["resnet50", "dinov2", "mae", "mambavision", "ijepa", "aim"],
        help="Backbone model type",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=MODEL_SIZE,
        help="Model size (depends on backbone type)",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset", type=str, default=DATASET_NAME, help="Dataset name"
    )
    parser.add_argument(
        "--limit-data",
        type=int,
        default=LIMIT_DATA,
        help="Maximum number of training samples",
    )

    # Training arguments
    parser.add_argument(
        "--num-epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=WEIGHT_DECAY, help="Weight decay"
    )
    parser.add_argument(
        "--da-strength",
        type=int,
        default=DA_STRENGTH,
        help="Data augmentation strength (0-3)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=RESUME_FROM,
        help="Resume training from a checkpoint file (e.g., checkpoint_epoch_25.pt)",
    )

    # DIET arguments
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=LABEL_SMOOTHING,
        help="Label smoothing strength (0 to disable DIET)",
    )
    parser.add_argument(
        "--num-diet-classes",
        type=int,
        default=NUM_DIET_CLASSES,
        help="Number of random classes for DIET method",
    )
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=PROJECTION_DIM,
        help="Projection head output dimension",
    )

    # Evaluation arguments
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=5,
        help="Run zero-shot evaluation every N epochs",
    )

    # Logging and saving arguments
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=True,
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--no-wandb",
        dest="use_wandb",
        action="store_false",
        help="Disable Weights & Biases logging",
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Run the training
    train(args)


if __name__ == "__main__":
    main()
