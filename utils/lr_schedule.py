"""
Learning rate scheduling utilities for DIET tuning.
"""

import math
import torch


def create_warmup_cosine_scheduler(
    optimizer, num_epochs, base_lr, warmup_ratio=0.1, eta_min=1e-5
):
    """
    Create a learning rate scheduler with linear warmup followed by cosine annealing.

    Args:
        optimizer: PyTorch optimizer
        num_epochs: Total number of training epochs
        base_lr: Base learning rate (target learning rate after warmup)
        warmup_ratio: Fraction of epochs to use for warmup (default: 0.1 for 10%)
        eta_min: Minimum learning rate at the end of cosine annealing

    Returns:
        torch.optim.lr_scheduler.LambdaLR: Configured scheduler
    """
    warmup_epochs = int(warmup_ratio * num_epochs)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup: from 0 to target lr
            return epoch / warmup_epochs
        else:
            # Cosine annealing: from target lr to eta_min
            cos_epoch = epoch - warmup_epochs
            cos_total = num_epochs - warmup_epochs
            eta_min_ratio = eta_min / base_lr
            cos_val = math.cos(math.pi * cos_epoch / cos_total)
            cosine_factor = 0.5 * (1 + cos_val)
            return cosine_factor * (1.0 - eta_min_ratio) + eta_min_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_diet_warmup_cosine_scheduler(
    optimizer,
    num_epochs,
    base_lr,
    diet_only_epochs,
    diet_lr=1e-3,
    warmup_ratio=0.1,
    eta_min=1e-5,
):
    """
    Create a learning rate scheduler with DIET-only constant LR, then warmup + cosine annealing.

    Args:
        optimizer: PyTorch optimizer
        num_epochs: Total number of training epochs
        base_lr: Base learning rate (target learning rate after warmup)
        diet_only_epochs: Number of epochs to use constant DIET-only learning rate
        diet_lr: Learning rate during DIET-only phase (default: 1e-3)
        warmup_ratio: Fraction of remaining epochs to use for warmup (default: 0.1)
        eta_min: Minimum learning rate at the end of cosine annealing

    Returns:
        torch.optim.lr_scheduler.LambdaLR: Configured scheduler
    """
    remaining_epochs = num_epochs - diet_only_epochs
    warmup_epochs = int(warmup_ratio * remaining_epochs)

    def lr_lambda(epoch):
        if epoch < diet_only_epochs:
            # Constant learning rate for DIET-only phase
            return diet_lr / base_lr
        elif epoch < diet_only_epochs + warmup_epochs:
            # Linear warmup: from 0 to base_lr
            warmup_progress = (epoch - diet_only_epochs) / warmup_epochs
            current_lr = warmup_progress * base_lr  # Start from 0, go to base_lr
            return current_lr / base_lr
        else:
            # Cosine annealing: from base_lr to eta_min
            cos_epoch = epoch - diet_only_epochs - warmup_epochs
            cos_total = num_epochs - diet_only_epochs - warmup_epochs
            eta_min_ratio = eta_min / base_lr
            cos_val = math.cos(math.pi * cos_epoch / cos_total)
            cosine_factor = 0.5 * (1 + cos_val)
            return cosine_factor * (1.0 - eta_min_ratio) + eta_min_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_scheduler_info(num_epochs, warmup_ratio=0.1):
    """
    Get information about the scheduler configuration.

    Args:
        num_epochs: Total number of training epochs
        warmup_ratio: Fraction of epochs to use for warmup

    Returns:
        dict: Information about the scheduler configuration
    """
    warmup_epochs = int(warmup_ratio * num_epochs)
    return {
        "total_epochs": num_epochs,
        "warmup_epochs": warmup_epochs,
        "warmup_ratio": warmup_ratio,
        "cosine_epochs": num_epochs - warmup_epochs,
    }


def get_diet_scheduler_info(num_epochs, diet_only_epochs, warmup_ratio=0.1):
    """
    Get information about the DIET scheduler configuration.

    Args:
        num_epochs: Total number of training epochs
        diet_only_epochs: Number of epochs for DIET-only phase
        warmup_ratio: Fraction of remaining epochs to use for warmup

    Returns:
        dict: Information about the scheduler configuration
    """
    remaining_epochs = num_epochs - diet_only_epochs
    warmup_epochs = int(warmup_ratio * remaining_epochs)
    cosine_epochs = num_epochs - diet_only_epochs - warmup_epochs

    return {
        "total_epochs": num_epochs,
        "diet_only_epochs": diet_only_epochs,
        "warmup_epochs": warmup_epochs,
        "cosine_epochs": cosine_epochs,
        "warmup_ratio": warmup_ratio,
    }
