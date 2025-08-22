"""Utilities for metrics management in DIET training."""

from typing import Dict, List, Any, Optional, Union, Callable
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections import defaultdict


class MetricsTracker:
    """Base class for tracking metrics."""

    def __init__(self):
        self.history = defaultdict(list)

    def update(self, metrics_dict: Dict[str, float]):
        """Update metrics history with new values.

        Args:
            metrics_dict: Dictionary of metric name to value
        """
        for name, value in metrics_dict.items():
            self.history[name].append(value)

    def get(self, name: str) -> List[float]:
        """Get history for a specific metric.

        Args:
            name: Metric name

        Returns:
            List of metric values
        """
        return self.history.get(name, [])

    def get_latest(self, name: str) -> Optional[float]:
        """Get latest value for a specific metric.

        Args:
            name: Metric name

        Returns:
            Latest metric value or None if no values
        """
        history = self.get(name)
        if history:
            return history[-1]
        return None

    def get_all(self) -> Dict[str, List[float]]:
        """Get all metrics history.

        Returns:
            Dictionary of metric name to list of values
        """
        return dict(self.history)


# Utility functions for metrics computation
def calculate_batch_metrics(
    diet_loss: torch.Tensor,
) -> Dict[str, float]:
    """Calculate batch-level metrics for DIET training.

    Args:
        diet_loss: DIET loss value

    Returns:
        Dictionary of metric name to value
    """
    metrics = {
        "batch_loss_diet": diet_loss.item(),
    }
    return metrics


def aggregate_metrics(
    batch_metrics_list: List[Dict[str, float]],
    prefix: str = "",
    metrics_to_aggregate: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Aggregate batch metrics into epoch metrics.

    Args:
        batch_metrics_list: List of batch metrics dictionaries
        prefix: Optional prefix to add to metric names
        metrics_to_aggregate: Optional list of metrics to aggregate

    Returns:
        Dictionary of aggregated metrics
    """
    if not batch_metrics_list:
        return {}

    # Determine which metrics to aggregate
    if metrics_to_aggregate is None:
        # Use keys from first batch
        metrics_to_aggregate = list(batch_metrics_list[0].keys())

    # Aggregate metrics
    aggregated = {}
    for metric in metrics_to_aggregate:
        values = [
            batch[metric]
            for batch in batch_metrics_list
            if metric in batch and not np.isnan(batch[metric])
        ]
        if values:
            metric_name = f"{prefix}{metric}" if prefix else metric
            aggregated[metric_name] = np.mean(values)

    return aggregated


def plot_metrics(
    metrics_history: Dict[str, List[float]],
    metrics_to_plot: Optional[List[str]] = None,
    figsize: tuple = (12, 8),
    style: str = "seaborn-v0_8-whitegrid",
) -> Figure:
    """Plot training metrics history.

    Args:
        metrics_history: Dictionary of metric name to list of values
        metrics_to_plot: Optional list of metrics to plot
        figsize: Figure size
        style: Matplotlib style

    Returns:
        Matplotlib figure
    """
    if metrics_to_plot is None:
        # Filter out metrics with all NaN values
        metrics_to_plot = [
            name
            for name, values in metrics_history.items()
            if not all(np.isnan(v) for v in values)
        ]

    # Use specified style
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize)

        for name in metrics_to_plot:
            if name in metrics_history and metrics_history[name]:
                values = metrics_history[name]
                # Replace NaN values with None for plotting
                values = [None if np.isnan(v) else v for v in values]
                epochs = range(1, len(values) + 1)
                ax.plot(epochs, values, marker="o", linestyle="-", label=name)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title("Training Metrics")
        ax.legend()
        ax.grid(True)

    return fig
