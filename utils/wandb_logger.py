"""Logging utilities for DIET finetuning using Weights & Biases."""

import os
import wandb
import torch
import numpy as np
from datetime import datetime
from io import BytesIO

# Import from our metrics module
from evaluation.metrics import (
    create_zero_shot_progression_plot,
    create_training_progress_plot,
)


def init_wandb(args):
    """Initialize wandb for experiment tracking

    Args:
        args: Dictionary containing experiment configuration parameters

    Returns:
        run: wandb run object
    """
    # Create experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"DIET_{args['backbone_type']}_{args['model_size']}_{args['dataset_name']}_{timestamp}"

    # Initialize wandb run
    run = wandb.init(
        project="DIET-Finetuning",
        name=experiment_name,
        config=args,
        settings=wandb.Settings(start_method="thread"),
        tags=[
            args["backbone_type"],
            args["model_size"],
            args["dataset_name"],
            "DIET" if args["label_smoothing"] > 0 else "Baseline",
        ],
    )

    print(f"WandB initialized: {run.name}")
    return run


def log_training_metrics(run, metrics, epoch, lr=None):
    """Log training metrics to wandb

    Args:
        run: wandb run object
        metrics: Dictionary of training metrics
        epoch: Current epoch number
        lr: Current learning rate (optional)
    """
    log_dict = {
        "train/diet_loss": metrics["diet_loss"],
        "train/probe_loss": metrics["probe_loss"],
        "train/accuracy": metrics["accuracy"],
        "epoch": epoch,
    }

    # Add learning rate if provided
    if lr is not None:
        log_dict["train/learning_rate"] = lr

    # Log metrics to wandb
    run.log(log_dict)


def log_evaluation_metrics(run, metrics, epoch):
    """Log evaluation metrics to wandb

    Args:
        run: wandb run object
        metrics: Dictionary of evaluation metrics
        epoch: Current epoch number
    """
    log_dict = {"eval/accuracy": metrics["accuracy"], "epoch": epoch}

    # Log metrics to wandb
    run.log(log_dict)


def log_zero_shot_metrics(run, metrics, epoch, initial_metrics=None):
    """Log zero-shot evaluation metrics to wandb

    Args:
        run: wandb run object
        metrics: Dictionary of zero-shot metrics
        epoch: Current epoch number
        initial_metrics: Initial zero-shot metrics for comparison (optional)
    """
    log_dict = {"epoch": epoch}

    # Log each zero-shot metric
    for metric_name, value in metrics.items():
        log_dict[f"zero_shot/{metric_name}"] = value

        # Log improvements if initial metrics are provided
        if initial_metrics is not None:
            improvement = value - initial_metrics[metric_name]
            relative_improvement = (
                (improvement / initial_metrics[metric_name]) * 100
                if initial_metrics[metric_name] > 0
                else float("inf")
            )
            log_dict[f"zero_shot/{metric_name}_improvement"] = improvement
            log_dict[f"zero_shot/{metric_name}_relative_improvement"] = (
                relative_improvement
            )

    # Calculate average improvement if initial metrics are provided
    if initial_metrics is not None:
        improvements = [metrics[m] - initial_metrics[m] for m in metrics.keys()]
        avg_improvement = np.mean(improvements)
        log_dict["zero_shot/average_improvement"] = avg_improvement

    # Log metrics to wandb
    run.log(log_dict)


def save_model_checkpoint(
    run,
    model,
    optimizer,
    projection_head,
    W_probe,
    W_diet,
    epoch,
    metrics,
    save_dir="checkpoints",
):
    """Save model checkpoint and log it to wandb

    Args:
        run: wandb run object
        model: The backbone model
        optimizer: Optimizer
        projection_head: DIET projection head
        W_probe: Probe linear layer
        W_diet: DIET linear layer
        epoch: Current epoch number
        metrics: Metrics to determine if this is a best checkpoint
        save_dir: Directory to save checkpoints locally
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Determine if this is the best checkpoint based on test accuracy
    test_acc = metrics.get("test_acc", 0)
    is_best = test_acc > getattr(save_model_checkpoint, "best_acc", 0)
    if is_best:
        save_model_checkpoint.best_acc = test_acc

    # Create checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "projection_head_state_dict": projection_head.state_dict(),
        "W_probe_state_dict": W_probe.state_dict(),
        "W_diet_state_dict": W_diet.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_acc": test_acc,
        "metrics": metrics,
    }

    # Save checkpoint locally
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)

    # Save best checkpoint separately
    if is_best:
        best_path = os.path.join(save_dir, "best_checkpoint.pt")
        torch.save(checkpoint, best_path)

        # Log best checkpoint to wandb
        best_artifact = wandb.Artifact(
            name=f"best_model_{run.id}",
            type="model",
            description=f"Best model checkpoint (epoch {epoch}, acc={test_acc:.4f})",
        )
        best_artifact.add_file(best_path)
        run.log_artifact(best_artifact)

    # Log regular checkpoint to wandb every 5 epochs or final epoch
    if epoch % 5 == 0 or is_best:
        artifact = wandb.Artifact(
            name=f"model_e{epoch}_{run.id}",
            type="model",
            description=f"Model checkpoint from epoch {epoch}",
        )
        artifact.add_file(checkpoint_path)
        run.log_artifact(artifact)


# Initialize static variable for best accuracy
save_model_checkpoint.best_acc = 0


def log_model_architecture(run, model, projection_head, W_probe, W_diet):
    """Log model architecture details to wandb

    Args:
        run: wandb run object
        model: The backbone model
        projection_head: DIET projection head
        W_probe: Probe linear layer
        W_diet: DIET linear layer
    """
    # Log architecture as a text table
    architecture_text = "# Model Architecture\n\n"

    # Count parameters
    model_params = sum(p.numel() for p in model.parameters())
    trainable_model_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    projection_params = sum(p.numel() for p in projection_head.parameters())
    W_probe_params = sum(p.numel() for p in W_probe.parameters())
    W_diet_params = sum(p.numel() for p in W_diet.parameters())
    total_params = model_params + projection_params + W_probe_params + W_diet_params
    total_trainable = (
        trainable_model_params + projection_params + W_probe_params + W_diet_params
    )

    # Add parameter counts
    architecture_text += f"## Parameter Counts\n\n"
    architecture_text += (
        f"| Component | Total Parameters | Trainable Parameters | % of Total |\n"
    )
    architecture_text += (
        f"|-----------|-----------------|----------------------|------------|\n"
    )
    architecture_text += f"| Backbone Model | {model_params:,} | {trainable_model_params:,} | {100 * model_params / total_params:.2f}% |\n"
    architecture_text += f"| Projection Head | {projection_params:,} | {projection_params:,} | {100 * projection_params / total_params:.2f}% |\n"
    architecture_text += f"| Classification Head | {W_probe_params:,} | {W_probe_params:,} | {100 * W_probe_params / total_params:.2f}% |\n"
    architecture_text += f"| DIET Head | {W_diet_params:,} | {W_diet_params:,} | {100 * W_diet_params / total_params:.2f}% |\n"
    architecture_text += (
        f"| **Total** | **{total_params:,}** | **{total_trainable:,}** | **100%** |\n\n"
    )

    # Log the architecture text
    run.log({"model_architecture": wandb.Html(architecture_text)})


def log_figure_to_wandb(run, figure, name):
    """Convert matplotlib figure to wandb Image and log it

    Args:
        run: wandb run object
        figure: Matplotlib figure
        name: Name for the logged figure
    """
    # Save figure to a BytesIO object
    buf = BytesIO()
    figure.savefig(buf, format="png")
    buf.seek(0)

    # Convert BytesIO to PIL Image before logging to wandb
    try:
        from PIL import Image

        img = Image.open(buf)
        run.log({name: wandb.Image(img)})
    except ImportError:
        # If PIL is not available, just log a message and continue
        print(f"Warning: Could not log {name} to wandb - PIL is not installed")


def log_metrics_table(
    run, metrics_history, initial_results=None, final_results=None, enhanced=False
):
    """
    Log metrics tables to W&B with varying levels of detail.

    Args:
        run: The wandb run object
        metrics_history: Dictionary containing metrics history with zero_shot_metrics
        initial_results: Dictionary with initial metrics (optional)
        final_results: Dictionary with final metrics (optional)
        enhanced: Whether to create enhanced tables with trend indicators (optional)
    """
    # Get all tracked epochs and metrics
    tracked_epochs = sorted(metrics_history["zero_shot_metrics"].keys())
    if not tracked_epochs:
        print("No epochs to log in metrics history")
        return

    metrics_list = list(metrics_history["zero_shot_metrics"][tracked_epochs[0]].keys())

    # Create standard metrics table
    columns = ["epoch"]
    for metric in metrics_list:
        columns.extend([f"{metric}", f"{metric}_change", f"{metric}_relative_change"])

    zero_shot_table = wandb.Table(columns=columns)

    # Initial values for calculating changes
    initial_values = metrics_history["zero_shot_metrics"][tracked_epochs[0]]

    # Add data for each epoch
    for epoch in tracked_epochs:
        current_metrics = metrics_history["zero_shot_metrics"][epoch]

        # Create a row for this epoch
        row = [epoch]

        # Add metrics, absolute change and relative change for each metric
        for metric in metrics_list:
            current_value = current_metrics[metric]
            change = current_value - initial_values[metric]
            rel_change = (
                (change / initial_values[metric]) * 100
                if initial_values[metric] > 0
                else float("inf")
            )

            # Add to row: current value, absolute change, relative change
            row.extend([current_value, change, rel_change])

        # Add row to table
        zero_shot_table.add_data(*row)

    # Log the detailed table
    run.log({"zero_shot_metrics_table": zero_shot_table})

    # Create a more user-friendly table for viewing trends
    metric_names = {
        "knn_acc": "K-NN Accuracy",
        "kmeans_ari": "K-Means ARI",
        "kmeans_nmi": "K-Means NMI",
        "linear_acc": "Linear Probe Accuracy",
    }

    columns = ["Epoch"]
    for metric in metrics_list:
        # Use friendly names if available
        nice_name = metric_names.get(metric, metric)
        columns.append(nice_name)

    friendly_table = wandb.Table(columns=columns)

    # Add a row for each epoch
    for epoch in tracked_epochs:
        row = [epoch]
        for metric in metrics_list:
            value = metrics_history["zero_shot_metrics"][epoch][metric]
            row.append(value)
        friendly_table.add_data(*row)

    # Log the user-friendly table
    run.log({"zero_shot_progression": friendly_table})

    # If initial and final results are provided, create summary tables
    if initial_results and final_results:
        # Basic summary table
        summary_columns = ["metric", "initial", "final", "change", "relative_change"]
        summary_table = wandb.Table(columns=summary_columns)

        for metric in metrics_list:
            initial = initial_results[metric]
            final = final_results[metric]
            change = final - initial
            rel_change = (change / initial) * 100 if initial > 0 else float("inf")

            summary_table.add_data(metric, initial, final, change, rel_change)

        # Log the summary table
        run.log({"zero_shot_summary_table": summary_table})

        # If enhanced mode is requested, create a more detailed summary
        if enhanced:
            summary_columns = [
                "metric",
                "initial",
                "final",
                "absolute_change",
                "percent_change",
                "trend",
                "significance",
            ]
            enhanced_table = wandb.Table(columns=summary_columns)

            # Add each metric to the summary
            for metric in metrics_list:
                initial = initial_results[metric]
                final = final_results[metric]
                change = final - initial
                percent = (change / initial) * 100 if initial > 0 else 0

                # Create a trend indicator
                trend = "↑" if change > 0 else "↓" if change < 0 else "→"

                # Significance level (arbitrary thresholds)
                if abs(percent) > 20:
                    significance = "High"
                elif abs(percent) > 5:
                    significance = "Medium"
                else:
                    significance = "Low"

                # Add to summary table
                enhanced_table.add_data(
                    metric, initial, final, change, percent, trend, significance
                )

            # Log the enhanced summary table
            run.log({"metrics_final_summary": enhanced_table})


def log_sanity_check_results(run, results_dict, model_type):
    """
    Log sanity check results to W&B tables

    Args:
        run: W&B run object
        results_dict: Dictionary of sanity check results
        model_type: Type of model that was evaluated
    """
    if results_dict is None:
        print(f"No results to log for {model_type}")
        return

    # Extract data
    k_values = results_dict["k_values"]
    accuracies = results_dict["accuracies"]
    best_acc = results_dict["best_acc"]
    best_k = results_dict["best_k"]

    # Check if we have linear probe results (for IJEPA)
    has_linear_probe = "linear_probe_acc" in results_dict

    # Create accuracy table by k value
    k_table = wandb.Table(columns=["model", "k_value", "accuracy"])

    # Add data for each k value
    for k, acc in zip(k_values, accuracies):
        k_table.add_data(model_type, k, acc * 100)  # Convert to percentage

    # Log the table
    run.log({f"sanity_check_{model_type}_knn": k_table})

    # Create summary table
    if has_linear_probe:
        summary_columns = ["model", "method", "accuracy", "best_k"]
        summary_table = wandb.Table(columns=summary_columns)

        # Add k-NN result
        summary_table.add_data(model_type, "k-NN", best_acc * 100, best_k)

        # Add linear probe result
        summary_table.add_data(
            model_type, "Linear Probe", results_dict["linear_probe_acc"] * 100, "N/A"
        )
    else:
        summary_columns = ["model", "best_k_value", "best_accuracy"]
        summary_table = wandb.Table(columns=summary_columns)
        summary_table.add_data(model_type, best_k, best_acc * 100)

    # Log the summary table
    run.log({f"sanity_check_{model_type}_summary": summary_table})


def log_combined_sanity_check_results(run, results_dict):
    """
    Log a combined table of all sanity check results

    Args:
        run: W&B run object
        results_dict: Dictionary mapping model types to their sanity check results
    """
    # Create table for combined results
    combined_table = wandb.Table(
        columns=["model", "method", "best_accuracy", "best_k", "passed_check"]
    )

    # Expected thresholds for each model type
    thresholds = {
        "dinov2": 0.91,
        "mae": 0.85,
        "mambavision": 0.85,
        "ijepa": 0.85,
        "aim": 0.75,
    }

    # Add data for each model
    for model_type, results in results_dict.items():
        if results is None:
            continue

        # Get model's threshold
        threshold = thresholds.get(model_type, 0.85)

        # Add k-NN result
        best_acc = results["best_acc"]
        best_k = results["best_k"]
        passed = best_acc >= threshold

        combined_table.add_data(
            model_type,
            "k-NN",
            best_acc * 100,  # Convert to percentage
            best_k,
            "✓" if passed else "✗",
        )

        # Add linear probe result if available
        if "linear_probe_acc" in results:
            linear_acc = results["linear_probe_acc"]
            linear_passed = linear_acc >= threshold

            combined_table.add_data(
                model_type,
                "Linear Probe",
                linear_acc * 100,  # Convert to percentage
                "N/A",
                "✓" if linear_passed else "✗",
            )

    # Log the combined table
    run.log({"sanity_check_combined_results": combined_table})


def create_experiment_dashboard(
    run, metrics_history, initial_results, final_results, experiment_config
):
    """
    Create a comprehensive dashboard visualization for the experiment

    Args:
        run: The wandb run object
        metrics_history: Dictionary with metrics history
        initial_results: Dictionary with initial zero-shot metrics
        final_results: Dictionary with final zero-shot metrics
        experiment_config: Dictionary with experiment configuration
    """
    # Create main summary as HTML
    summary_text = f"""
    <h1>DIET Finetuning Experiment Dashboard</h1>
    <h2>Configuration</h2>
    <table>
        <tr><td><b>Model:</b></td><td>{experiment_config['backbone_type']} ({experiment_config['model_size']})</td></tr>
        <tr><td><b>Dataset:</b></td><td>{experiment_config['dataset_name']}</td></tr>
        <tr><td><b>DIET Status:</b></td><td>{'Active' if experiment_config['is_diet_active'] else 'Inactive'}</td></tr>
        <tr><td><b>Label Smoothing:</b></td><td>{experiment_config['label_smoothing']}</td></tr>
        <tr><td><b>Training Samples:</b></td><td>{experiment_config.get('limit_data', 'Full Dataset')}</td></tr>
        <tr><td><b>Training Epochs:</b></td><td>{experiment_config['num_epochs']}</td></tr>
        <tr><td><b>Learning Rate:</b></td><td>{experiment_config['lr']}</td></tr>
    </table>
    
    <h2>Performance Summary</h2>
    <table>
        <tr><th>Metric</th><th>Initial</th><th>Final</th><th>Improvement</th><th>Relative %</th></tr>
    """

    # Add metrics to summary
    metrics_list = list(initial_results.keys())
    for metric in metrics_list:
        initial = initial_results[metric]
        final = final_results[metric]
        change = final - initial
        rel_change = (change / initial) * 100 if initial > 0 else float("inf")

        # Add trend indicator
        trend = "⬆️" if change > 0 else "⬇️" if change < 0 else "↔️"

        summary_text += f"""
        <tr>
            <td>{metric}</td>
            <td>{initial:.4f}</td>
            <td>{final:.4f}</td>
            <td>{change:+.4f} {trend}</td>
            <td>{rel_change:+.2f}%</td>
        </tr>
        """

    # Calculate average improvement
    avg_improvement = np.mean(
        [final_results[m] - initial_results[m] for m in metrics_list]
    )
    avg_rel_improvement = (
        avg_improvement / np.mean(list(initial_results.values()))
    ) * 100

    # Close the table and add conclusion
    summary_text += f"""
    </table>
    
    <h2>Conclusion</h2>
    <p>
        DIET finetuning {'improved' if avg_improvement > 0 else 'did not improve'} zero-shot performance by an average of 
        {avg_improvement:.4f} ({avg_rel_improvement:.2f}%).
    </p>
    """

    # Log this summary as HTML
    run.log({"experiment_dashboard": wandb.Html(summary_text)})

    # Create and log the combined figures
    # First create the training progress plot
    training_fig = create_training_progress_plot(metrics_history)
    log_figure_to_wandb(run, training_fig, "training_progress")

    # Then create the zero-shot progression plot
    tracked_epochs = sorted(metrics_history["zero_shot_metrics"].keys())
    zero_shot_fig = create_zero_shot_progression_plot(
        metrics_history, tracked_epochs, metrics_list
    )
    log_figure_to_wandb(run, zero_shot_fig, "zero_shot_progression")

    # Log tables
    log_metrics_table(
        run, metrics_history, initial_results, final_results, enhanced=True
    )

    # Create a report (only available in paid tiers)
    try:
        report = wandb.Table(columns=["component", "content"])
        report.add_data("summary", wandb.Html(summary_text))
        report.add_data("training_progress", wandb.Image(training_fig))
        report.add_data("zero_shot_progression", wandb.Image(zero_shot_fig))

        run.log({"experiment_report": report})
    except:
        print("Failed to create report table - continuing without it")

    return
