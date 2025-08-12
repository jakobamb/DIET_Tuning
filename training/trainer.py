"""Training components for DIET finetuning framework."""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.wandb_logger import (
    log_training_metrics,
    log_evaluation_metrics,
    log_zero_shot_metrics,
    save_model_checkpoint,
)
from evaluation.metrics import zero_shot_eval
from training.metrics_utils import (
    MetricsTracker,
    aggregate_metrics,
)


class DIETTrainer:
    """Trainer class for DIET finetuning."""

    def __init__(
        self,
        model,
        projection_head,
        W_probe,
        W_diet,
        device,
        num_classes,
        optimizer,
        scheduler=None,
        label_smoothing=0.0,
        checkpoint_dir="checkpoints",
        checkpoint_freq=10,
        training_mode="combined",
        config=None,
    ):
        """Initialize the trainer.

        Args:
            model: The backbone model
            projection_head: The projection head
            W_probe: The probe linear layer
            W_diet: The DIET linear layer
            device: The device to use
            num_classes: Number of classes in the dataset
            optimizer: The optimizer
            scheduler: Optional learning rate scheduler
            label_smoothing: Label smoothing value for DIET loss
            checkpoint_dir: Directory to save checkpoints
            checkpoint_freq: Frequency of saving checkpoints
            training_mode: Training mode to use:
                - "combined": Uses weighted combination of DIET and probe losses
                - "diet_only": Uses only the DIET loss (requires n)
                - "probe_only": Uses only the probe loss
            config: Optional TrainerConfig object with additional settings
        """
        self.model = model
        self.projection_head = projection_head
        self.W_probe = W_probe
        self.W_diet = W_diet
        self.device = device
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.is_diet_active = label_smoothing > 0
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.training_mode = training_mode
        self.config = config

        # Set temperature from config if available
        self.temperature = getattr(config, "temperature", 3.0) if config else 3.0

        # Create loss functions
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
        self.criterion_diet = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()

        # For backward compatibility
        self.metrics_history = {
            "train_loss_diet": [],
            "train_loss_probe": [],
            "train_acc": [],
            "test_acc": [],
            "zero_shot_metrics": {},
        }

    def train(
        self,
        train_loader,
        test_loader,
        num_epochs,
        run=None,
        eval_frequency=5,
        start_epoch=0,
    ):
        """Train the model.

        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            num_epochs: Number of epochs to train
            run: Optional wandb run object
            eval_frequency: How often to run zero-shot evaluation
            start_epoch: Epoch to start training from (for resuming training)

        Returns:
            metrics_history: Dictionary of training metrics
        """
        # Run initial zero-shot evaluation
        print("\n" + "=" * 50)
        print("INITIAL ZERO-SHOT EVALUATION (BEFORE TRAINING)")
        print("=" * 50)
        initial_time = time.time()
        initial_results = zero_shot_eval(
            self.model, test_loader, self.num_classes, self.device, eval_id=0
        )
        print(f"Initial evaluation completed in {time.time() - initial_time:.2f}s")

        # Store initial results
        self.metrics_history["zero_shot_metrics"][0] = initial_results.copy()

        # Log initial results to wandb
        if run is not None:
            log_zero_shot_metrics(run, initial_results, 0)

        # Validate that the training data matches the training mode
        if self.training_mode == "diet_only":
            # Check first batch for diet classes
            first_batch = next(iter(train_loader))
            if len(first_batch) < 3 or first_batch[2] is None:
                raise ValueError(
                    "DIET-only training requires diet classes (n) in the dataloader"
                )

        # Begin training
        train_start_time = time.time()
        epoch_times = []

        # Count trainable parameters
        num_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        num_trainable_params += sum(
            p.numel() for p in self.projection_head.parameters() if p.requires_grad
        )
        num_trainable_params += sum(
            p.numel() for p in self.W_probe.parameters() if p.requires_grad
        )
        num_trainable_params += sum(
            p.numel() for p in self.W_diet.parameters() if p.requires_grad
        )

        # Training loop
        for epoch in range(start_epoch, start_epoch + num_epochs):
            # Start training epoch
            epoch_start = time.time()
            batch_metrics_list = []

            # Set models to training mode
            self.model.train()
            self.projection_head.train()
            self.W_probe.train()
            self.W_diet.train()

            print("\n==========================")
            print(
                f"Starting epoch {epoch+1}/{start_epoch + num_epochs} "
                f"at {time.strftime('%H:%M:%S')}"
            )
            print("==========================\n")
            print("Initializing training loop...")
            print("\n")

            # Iterate through training data (without tqdm)
            for i, batch in enumerate(train_loader):
                # Flexible batch unpacking: support (x, y, n) and (x, y)
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 3:
                        x, y, n = batch
                    elif len(batch) == 2:
                        x, y = batch
                        n = None  # No diet class provided
                    else:
                        print(
                            f"Warning: Unexpected batch length ({len(batch)}) at batch {i}. Skipping."
                        )
                        continue
                else:
                    print(f"Warning: Unexpected batch type at batch {i}. Skipping.")
                    continue

                # Send tensors to device
                x = x.to(self.device)
                y = y.to(self.device)

                # Ensure y is 1D (flatten if needed)
                if y.dim() > 1:
                    y = y.view(-1)

                if n is not None:
                    n = n.to(self.device).long()
                    if n.dim() > 1:
                        n = n.view(-1)

                # Forward pass
                z = self.model(x)  # Original features
                z_norm = F.normalize(z, p=2, dim=1)  # L2 normalize
                z_proj = self.projection_head(z_norm)  # Projection through MLP

                # Calculate diet loss with temperature scaling
                if n is not None:
                    logits_diet = self.W_diet(z_proj) / self.temperature
                    loss_diet = self.criterion_diet(logits_diet, n)
                else:
                    loss_diet = torch.tensor(0.0, device=self.device)

                # Calculate probe loss (only if needed)
                loss_probe = None
                logits_probe = None
                batch_acc = None
                if self.training_mode != "diet_only":
                    logits_probe = self.W_probe(z)
                    loss_probe = self.criterion(logits_probe, y)
                    # Calculate accuracy for tracking
                    preds = logits_probe.argmax(dim=1)
                    batch_acc = torch.mean((y == preds).float()).item()

                # Determine which loss to use based on training mode
                if self.training_mode == "diet_only":
                    if n is not None:
                        loss = loss_diet
                    else:
                        raise ValueError(
                            "DIET-only training requires diet classes (n) to be provided"
                        )
                elif self.training_mode == "probe_only":
                    loss = loss_probe
                else:  # Combined mode (default)
                    # Combine losses dynamically based on training phase
                    total_epochs = start_epoch + num_epochs
                    if n is not None:
                        if epoch < total_epochs * 0.5:  # First 50% of training
                            loss = 0.6 * loss_diet + 0.4 * loss_probe
                        elif epoch < total_epochs * 0.8:  # Next 30% of training
                            loss = 0.4 * loss_diet + 0.6 * loss_probe
                        else:  # Final 20% of training
                            loss = 0.2 * loss_diet + 0.8 * loss_probe
                    else:
                        loss = loss_probe

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()

                # Calculate gradient norm
                total_grad_norm = 0.0
                for p in (
                    list(self.model.parameters())
                    + list(self.projection_head.parameters())
                    + list(self.W_probe.parameters())
                    + list(self.W_diet.parameters())
                ):
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm**0.5

                # Update parameters
                self.optimizer.step()

                # Track batch metrics
                batch_metrics = {
                    "batch_loss_diet": loss_diet.item(),
                    "batch_grad_norm": total_grad_norm,
                }

                if self.training_mode != "diet_only" and loss_probe is not None:
                    batch_metrics["batch_loss_probe"] = loss_probe.item()
                    batch_metrics["batch_acc"] = batch_acc

                batch_metrics_list.append(batch_metrics)

                # Print batch update every 10 batches
                if i % 10 == 0:
                    print(
                        f"Batch {i}: grad_norm={total_grad_norm:.4f} across {num_trainable_params} trainable params"
                    )

            # End of epoch processing
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            print(f"\nEpoch {epoch+1} completed in {epoch_time:.2f}s\n")

            # Aggregate metrics for the epoch
            epoch_metrics = aggregate_metrics(
                batch_metrics_list,
                prefix="train_",
            )

            # Update metrics tracker
            self.metrics_tracker.update(epoch_metrics)

            # For backward compatibility
            self.metrics_history["train_loss_diet"].append(
                epoch_metrics.get("train_batch_loss_diet", float("nan"))
            )

            if "train_batch_loss_probe" in epoch_metrics:
                self.metrics_history["train_loss_probe"].append(
                    epoch_metrics["train_batch_loss_probe"]
                )
                self.metrics_history["train_acc"].append(
                    epoch_metrics["train_batch_acc"]
                )
            else:
                # Use NaN for metrics that weren't tracked
                self.metrics_history["train_loss_probe"].append(float("nan"))
                self.metrics_history["train_acc"].append(float("nan"))

            # Step the learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Learning rate updated to: {current_lr:.6f}")
            else:
                current_lr = None

            # Log training metrics to wandb
            if run is not None:
                # Create a clean dict for wandb
                wandb_metrics = {
                    "diet_loss": epoch_metrics.get("train_batch_loss_diet", 0)
                }

                # Add probe metrics if available
                if "train_batch_loss_probe" in epoch_metrics:
                    wandb_metrics.update(
                        {
                            "probe_loss": epoch_metrics["train_batch_loss_probe"],
                            "accuracy": epoch_metrics["train_batch_acc"],
                        }
                    )

                log_training_metrics(run, wandb_metrics, epoch + 1, current_lr)

            # Print epoch summary
            if self.training_mode == "diet_only":
                print(
                    f"Epoch {epoch+1} Metrics - DIET Loss: "
                    f"{epoch_metrics.get('train_batch_loss_diet', float('nan')):.4e}"
                )
            else:
                print(
                    f"Epoch {epoch+1} Metrics - DIET Loss: "
                    f"{epoch_metrics.get('train_batch_loss_diet', float('nan')):.4e}, "
                    f"Probe Loss: {epoch_metrics.get('train_batch_loss_probe', float('nan')):.4e}, "
                    f"Accuracy: {epoch_metrics.get('train_batch_acc', float('nan')):.4f}"
                )

            # Evaluate on test set
            test_acc = self.evaluate_test_set(test_loader)
            self.metrics_history["test_acc"].append(test_acc)

            # Log evaluation metrics to wandb
            if run is not None:
                log_evaluation_metrics(run, {"accuracy": test_acc}, epoch + 1)

            # Print epoch summary
            print(f"Epoch {epoch+1}/{start_epoch + num_epochs} summary:")
            if self.training_mode == "diet_only":
                print(
                    f"  Train - DIET loss: "
                    f"{epoch_metrics.get('train_batch_loss_diet', float('nan')):.4e}"
                )
            else:
                print(
                    f"  Train - DIET loss: "
                    f"{epoch_metrics.get('train_batch_loss_diet', float('nan')):.4e}, "
                    f"Probe loss: {epoch_metrics.get('train_batch_loss_probe', float('nan')):.4e}, "
                    f"Acc: {epoch_metrics.get('train_batch_acc', float('nan')):.4f}"
                )
            print(f"  Test  - Acc: {test_acc:.4f}")

            # Zero-shot evaluation every few epochs
            if (
                epoch + 1
            ) % eval_frequency == 0 or epoch == start_epoch + num_epochs - 1:
                print(f"\nRunning zero-shot evaluation at epoch {epoch+1}...")

                try:
                    epoch_zero_shot = zero_shot_eval(
                        self.model,
                        test_loader,
                        self.num_classes,
                        self.device,
                        eval_id=epoch + 1,
                    )
                    self.metrics_history["zero_shot_metrics"][
                        epoch + 1
                    ] = epoch_zero_shot.copy()

                    # Log zero-shot metrics to wandb
                    if run is not None:
                        log_zero_shot_metrics(
                            run, epoch_zero_shot, epoch + 1, initial_results
                        )

                except Exception as e:
                    print(f"Error in zero-shot evaluation: {e}")

            # Save checkpoint
            if run is not None and (epoch + 1) % self.checkpoint_freq == 0:
                if self.training_mode == "diet_only":
                    checkpoint_metrics = {
                        "test_acc": test_acc,
                        "train_loss_diet": epoch_metrics.get(
                            "train_batch_loss_diet", 0
                        ),
                    }
                else:
                    checkpoint_metrics = {
                        "train_acc": epoch_metrics.get("train_batch_acc", 0),
                        "test_acc": test_acc,
                        "train_loss_diet": epoch_metrics.get(
                            "train_batch_loss_diet", 0
                        ),
                        "train_loss_probe": epoch_metrics.get(
                            "train_batch_loss_probe", 0
                        ),
                    }
                save_model_checkpoint(
                    run,
                    self.model,
                    self.optimizer,
                    self.projection_head,
                    self.W_probe,
                    self.W_diet,
                    epoch + 1,
                    checkpoint_metrics,
                    save_dir=self.checkpoint_dir,
                )

        # End of training
        training_time = time.time() - train_start_time
        print(f"\nTraining completed in {training_time:.2f}s")

        # Final zero-shot evaluation
        print("\n" + "=" * 50)
        print("FINAL ZERO-SHOT EVALUATION (AFTER TRAINING)")
        print("=" * 50)
        final_time = time.time()
        final_results = zero_shot_eval(
            self.model,
            test_loader,
            self.num_classes,
            self.device,
            eval_id=start_epoch + num_epochs + 1,
        )
        print(f"Final evaluation completed in {time.time() - final_time:.2f}s")

        # Log final zero-shot metrics to wandb
        if run is not None:
            log_zero_shot_metrics(
                run, final_results, start_epoch + num_epochs + 1, initial_results
            )

        # Calculate improvements
        improvements = {
            f"improvement_{k}": final_results[k] - initial_results[k]
            for k in initial_results.keys()
        }

        # Print results summary
        print("\n" + "=" * 50)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 50)
        print(f"Number of classes: {self.num_classes}")
        print(f"Training mode: {self.training_mode}")
        print(
            f"DIET label smoothing: {self.criterion_diet.label_smoothing}"
            + (" (DIET active)" if self.is_diet_active else " (DIET inactive)")
        )
        print("\nZero-shot performance:")
        print("-" * 60)

        print(
            f"{'Metric':<15} {'Initial':<10} {'Final':<10} {'Improvement':<10} {'Relative %':<10}"
        )
        print("-" * 60)
        metrics = list(initial_results.keys())
        for metric in metrics:
            initial = initial_results[metric]
            final = final_results[metric]
            imp = improvements[f"improvement_{metric}"]
            rel_imp = (imp / initial) * 100 if initial > 0 else float("inf")
            print(
                f"{metric:<15} {initial:.4f}     {final:.4f}     {imp:+.4f}     {rel_imp:+.2f}%"
            )

        print("\nCONCLUSION:")
        avg_improvement = np.mean(
            [improvements[f"improvement_{k}"] for k in initial_results.keys()]
        )
        if avg_improvement > 0:
            print(
                f"DIET finetuning {'improved' if self.is_diet_active else 'would likely improve'} zero-shot performance "
                + f"by an average of {avg_improvement:.4f} ({(avg_improvement / np.mean(list(initial_results.values()))) * 100:.2f}%)"
            )
        else:
            print(
                f"DIET finetuning {'did not improve' if self.is_diet_active else 'would likely not improve'} zero-shot performance"
            )

        return self.metrics_history, initial_results, final_results

    def evaluate_test_set(self, test_loader):
        """Evaluate model on test set with probe classifier.

        Args:
            test_loader: DataLoader for test data

        Returns:
            float: Test accuracy
        """
        print("\nStarting evaluation on test set:")
        self.model.eval()
        with torch.no_grad():
            run_acc_test = []
            for i, batch in enumerate(test_loader):
                # Flexible unpacking: works with (x, y) or (x, y, n)
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 3:
                        x, y, _ = batch
                    elif len(batch) == 2:
                        x, y = batch
                    else:
                        print(
                            f"Warning: Unexpected test batch length ({len(batch)}) at batch {i}. Skipping."
                        )
                        continue
                else:
                    print(
                        f"Warning: Unexpected test batch type at batch {i}. Skipping."
                    )
                    continue

                x = x.to(self.device)
                y = y.to(self.device)
                z = self.model(x)
                logits_probe = self.W_probe(z)

                # Adjust dimensions if necessary
                if y.dim() != logits_probe.argmax(1).dim():
                    y = (
                        y.squeeze()
                        if y.dim() > logits_probe.argmax(1).dim()
                        else y.unsqueeze(0)
                    )
                batch_acc = torch.mean((y == logits_probe.argmax(1)).float()).item()
                run_acc_test.append(batch_acc)

                if i % 10 == 0:  # Print only every 10 batches to reduce verbosity
                    print(f"Test Batch {i}: Accuracy={batch_acc:.4f}")

            test_acc = np.mean(run_acc_test) if run_acc_test else 0
            print(f"\nOverall Test Accuracy: {test_acc:.4f}")
        return test_acc

    def create_visualizations(self, tracked_epochs=None):
        """Create visualization plots for training metrics.

        Args:
            tracked_epochs: List of epochs to include in zero-shot plots

        Returns:
            Tuple of (training_fig, zero_shot_fig)
        """
        # Training progress plot
        training_fig = plt.figure(figsize=(15, 5))

        # Plot loss
        plt.subplot(1, 3, 1)
        plt.plot(self.metrics_history["train_loss_diet"], label="DIET Loss")

        # Only plot probe loss if not in diet_only mode
        if self.training_mode != "diet_only":
            plt.plot(self.metrics_history["train_loss_probe"], label="Probe Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.grid(True)

        # Plot accuracy
        plt.subplot(1, 3, 2)

        # Only plot train accuracy if not in diet_only mode
        if self.training_mode != "diet_only":
            plt.plot(self.metrics_history["train_acc"], label="Train Accuracy")

        plt.plot(self.metrics_history["test_acc"], label="Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy")
        plt.legend()
        plt.grid(True)

        # Plot zero-shot metrics
        plt.subplot(1, 3, 3)

        # Get initial and final zero-shot metrics
        if tracked_epochs is None:
            tracked_epochs = sorted(self.metrics_history["zero_shot_metrics"].keys())

        if len(tracked_epochs) >= 2:
            initial_epoch = tracked_epochs[0]
            final_epoch = tracked_epochs[-1]

            metrics = list(
                self.metrics_history["zero_shot_metrics"][initial_epoch].keys()
            )
            x = range(len(metrics))
            width = 0.35

            # Plot bar chart comparing initial and final metrics
            plt.bar(
                x,
                [
                    self.metrics_history["zero_shot_metrics"][initial_epoch][m]
                    for m in metrics
                ],
                width,
                label="Initial",
            )
            plt.bar(
                [i + width for i in x],
                [
                    self.metrics_history["zero_shot_metrics"][final_epoch][m]
                    for m in metrics
                ],
                width,
                label="Final",
            )
            plt.xlabel("Metrics")
            plt.ylabel("Score")
            plt.title("Zero-Shot Performance")
            plt.xticks([i + width / 2 for i in x], metrics)
            plt.legend()
            plt.grid(True)

        training_fig.tight_layout()

        # Zero-shot progression plot
        if len(tracked_epochs) >= 2:
            zero_shot_fig = plt.figure(figsize=(15, 10))

            metrics_list = list(
                self.metrics_history["zero_shot_metrics"][tracked_epochs[0]].keys()
            )

            # Plot each metric's progression
            for i, metric in enumerate(metrics_list):
                plt.subplot(2, 2, i + 1)
                values = [
                    self.metrics_history["zero_shot_metrics"][e][metric]
                    for e in tracked_epochs
                ]
                plt.plot(tracked_epochs, values, marker="o", linewidth=2)
                plt.xlabel("Epoch")
                plt.ylabel(f"{metric} Score")
                plt.title(f"Zero-shot {metric} Progression")
                plt.grid(True)

                # Add initial and final values as text annotations
                plt.annotate(
                    f"{values[0]:.4f}",
                    (tracked_epochs[0], values[0]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                )
                plt.annotate(
                    f"{values[-1]:.4f}",
                    (tracked_epochs[-1], values[-1]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                )

            zero_shot_fig.tight_layout()
            zero_shot_fig.suptitle(
                "Zero-shot Metrics Progression During Training", fontsize=16
            )
            zero_shot_fig.subplots_adjust(top=0.9)

            return training_fig, zero_shot_fig

        return training_fig, None


def create_projection_head(in_dim, hidden_dim, out_dim, device):
    """Create projection head for DIET method.

    Args:
        in_dim: Input dimension
        hidden_dim: Hidden dimension
        out_dim: Output dimension
        device: Device to put the projection head on

    Returns:
        Projection head module
    """
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden_dim),
        torch.nn.BatchNorm1d(hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(hidden_dim, out_dim),
    ).to(device)


# Add plot_metrics method to DIETTrainer class
def plot_training_metrics(self, tracked_epochs=None, figsize=(18, 6)):
    """Plot training metrics.

    Args:
        tracked_epochs: List of epochs for zero-shot metrics
        figsize: Figure size tuple

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Use the new plot_metrics utility if metrics_tracker has data
    if self.metrics_tracker.history:
        # Define which metrics to plot
        metrics_to_plot = []

        # Always include DIET loss
        metrics_to_plot.append("train_batch_loss_diet")

        # Include probe metrics if not in diet_only mode
        if self.training_mode != "diet_only":
            metrics_to_plot.extend(["train_batch_loss_probe", "train_batch_acc"])

        # Include test accuracy and gradient norm
        metrics_to_plot.append("train_batch_grad_norm")

        # Get all metrics data
        metrics_history = self.metrics_tracker.get_all()

        # Add test accuracy from the old metrics_history
        metrics_history["test_acc"] = self.metrics_history["test_acc"]
        metrics_to_plot.append("test_acc")

        # Create plot using the utility function
        from training.metrics_utils import plot_metrics as plot_metrics_util

        return plot_metrics_util(
            metrics_history, metrics_to_plot=metrics_to_plot, figsize=figsize
        )

    # Fallback to legacy plotting
    fig = plt.figure(figsize=figsize)

    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(self.metrics_history["train_loss_diet"], label="DIET Loss")

    if self.training_mode != "diet_only":
        plt.plot(self.metrics_history["train_loss_probe"], label="Probe Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 3, 2)

    if self.training_mode != "diet_only":
        plt.plot(self.metrics_history["train_acc"], label="Train Accuracy")

    plt.plot(self.metrics_history["test_acc"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.legend()
    plt.grid(True)

    # Plot zero-shot metrics if available
    plt.subplot(1, 3, 3)

    if tracked_epochs is None:
        tracked_epochs = sorted(self.metrics_history["zero_shot_metrics"].keys())

    if len(tracked_epochs) >= 2:
        initial_epoch = tracked_epochs[0]
        final_epoch = tracked_epochs[-1]

        metrics = list(self.metrics_history["zero_shot_metrics"][initial_epoch].keys())
        x = range(len(metrics))
        width = 0.35

        # Plot bar chart comparing initial and final metrics
        plt.bar(
            x,
            [
                self.metrics_history["zero_shot_metrics"][initial_epoch][m]
                for m in metrics
            ],
            width,
            label=f"Epoch {initial_epoch}",
        )
        plt.bar(
            [i + width for i in x],
            [
                self.metrics_history["zero_shot_metrics"][final_epoch][m]
                for m in metrics
            ],
            width,
            label=f"Epoch {final_epoch}",
        )

        plt.xlabel("Metrics")
        plt.ylabel("Score")
        plt.title("Zero-Shot Evaluation")
        plt.xticks([i + width / 2 for i in x], metrics, rotation=45)
        plt.legend()
    else:
        plt.title("Zero-Shot Evaluation\n(Not enough data)")

    plt.tight_layout()
    return fig


# Add the plot_metrics method to the DIETTrainer class
DIETTrainer.plot_metrics = plot_training_metrics
