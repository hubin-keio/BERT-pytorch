"""Visualization utilities for BERT-pytorch"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from .logger import get_logger

logger = get_logger(__name__)


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    output_path: str = "training_curves.png",
    title: str = "Training Progress"
) -> None:
    """
    Plot training and validation curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch (optional)
        train_accs: List of training accuracies per epoch (optional)
        val_accs: List of validation accuracies per epoch (optional)
        output_path: Path to save the plot
        title: Title for the plot
    """
    try:
        num_plots = sum([True, val_losses is not None, train_accs is not None, val_accs is not None])
        num_plots = max(2, num_plots)  # At least 2 subplots

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot losses
        epochs = range(1, len(train_losses) + 1)
        axes[0].plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=4)
        if val_losses:
            axes[0].plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Loss Curves', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot accuracies
        if train_accs:
            axes[1].plot(epochs, train_accs, 'g-o', label='Training Accuracy', linewidth=2, markersize=4)
        if val_accs:
            axes[1].plot(epochs, val_accs, 'm-s', label='Validation Accuracy', linewidth=2, markersize=4)
        if train_accs or val_accs:
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Accuracy (%)', fontsize=12)
            axes[1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No accuracy data available',
                        ha='center', va='center', transform=axes[1].transAxes)

        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {output_path}")

        plt.close()

    except Exception as e:
        logger.error(f"Failed to plot training curves: {e}")


def plot_loss_components(
    mlm_losses: List[float],
    nsp_losses: List[float],
    total_losses: List[float],
    output_path: str = "loss_components.png",
    title: str = "Loss Components"
) -> None:
    """
    Plot MLM and NSP loss components.

    Args:
        mlm_losses: List of MLM losses per epoch
        nsp_losses: List of NSP losses per epoch
        total_losses: List of total losses per epoch
        output_path: Path to save the plot
        title: Title for the plot
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = range(1, len(mlm_losses) + 1)
        ax.plot(epochs, mlm_losses, 'b-o', label='MLM Loss', linewidth=2, markersize=4)
        ax.plot(epochs, nsp_losses, 'r-s', label='NSP Loss', linewidth=2, markersize=4)
        ax.plot(epochs, total_losses, 'g-^', label='Total Loss', linewidth=2, markersize=4)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Loss components plot saved to {output_path}")

        plt.close()

    except Exception as e:
        logger.error(f"Failed to plot loss components: {e}")


def plot_metrics_summary(
    metrics: Dict[str, List[float]],
    output_path: str = "metrics_summary.png",
    title: str = "Training Metrics Summary"
) -> None:
    """
    Plot multiple metrics in a grid layout.

    Args:
        metrics: Dictionary of metric names to lists of values
        output_path: Path to save the plot
        title: Title for the plot
    """
    try:
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]

        colors = plt.cm.tab10(np.linspace(0, 1, n_metrics))

        for idx, (metric_name, values) in enumerate(metrics.items()):
            epochs = range(1, len(values) + 1)
            axes[idx].plot(epochs, values, color=colors[idx], marker='o', linewidth=2, markersize=4)
            axes[idx].set_xlabel('Epoch', fontsize=10)
            axes[idx].set_ylabel(metric_name, fontsize=10)
            axes[idx].set_title(metric_name, fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics summary saved to {output_path}")

        plt.close()

    except Exception as e:
        logger.error(f"Failed to plot metrics summary: {e}")
