"""Unit tests for utilities."""

import pytest
import torch
import numpy as np
import random
from pathlib import Path
import tempfile

from bert_pytorch.utils import set_seed, get_logger
from bert_pytorch.utils.visualization import (
    plot_training_curves,
    plot_loss_components,
    plot_metrics_summary,
)


class TestSeedManagement:
    """Test seed management."""

    def test_set_seed_reproducibility(self):
        """Test that set_seed ensures reproducibility."""
        set_seed(42)
        tensor1 = torch.randn(10)
        numpy1 = np.random.randn(10)
        random1 = [random.random() for _ in range(10)]

        set_seed(42)
        tensor2 = torch.randn(10)
        numpy2 = np.random.randn(10)
        random2 = [random.random() for _ in range(10)]

        assert torch.allclose(tensor1, tensor2)
        assert np.allclose(numpy1, numpy2)
        assert random1 == random2

    def test_set_seed_different_values(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        tensor1 = torch.randn(10)

        set_seed(123)
        tensor2 = torch.randn(10)

        assert not torch.allclose(tensor1, tensor2)

    def test_set_seed_invalid_input(self):
        """Test that invalid inputs raise errors."""
        with pytest.raises(TypeError):
            set_seed("42")

        with pytest.raises(ValueError):
            set_seed(-1)

    def test_set_seed_cuda_determinism(self):
        """Test CUDA determinism (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        set_seed(42)
        tensor1 = torch.randn(10, device="cuda")

        set_seed(42)
        tensor2 = torch.randn(10, device="cuda")

        assert torch.allclose(tensor1, tensor2)


class TestLogging:
    """Test logging utilities."""

    def test_get_logger_creation(self):
        """Test logger creation."""
        logger = get_logger("test_module")
        assert logger is not None
        assert logger.name == "test_module"

    def test_get_logger_same_name(self):
        """Test that loggers with same name are the same."""
        logger1 = get_logger("test")
        logger2 = get_logger("test")
        assert logger1 is logger2

    def test_logger_methods(self):
        """Test logger methods work without errors."""
        logger = get_logger("test")
        # These should not raise errors
        logger.info("Test info")
        logger.warning("Test warning")
        logger.error("Test error")
        logger.debug("Test debug")


class TestVisualization:
    """Test visualization utilities."""

    def test_plot_training_curves(self):
        """Test plotting training curves."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "curves.png"

            train_losses = [0.5, 0.4, 0.3, 0.2]
            val_losses = [0.6, 0.5, 0.4, 0.3]
            train_accs = [80, 85, 88, 90]
            val_accs = [75, 80, 83, 85]

            plot_training_curves(
                train_losses=train_losses,
                val_losses=val_losses,
                train_accs=train_accs,
                val_accs=val_accs,
                output_path=str(output_path),
            )

            assert output_path.exists()

    def test_plot_training_curves_minimal(self):
        """Test plotting with minimal arguments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "curves.png"

            train_losses = [0.5, 0.4, 0.3]

            plot_training_curves(
                train_losses=train_losses,
                output_path=str(output_path),
            )

            assert output_path.exists()

    def test_plot_loss_components(self):
        """Test plotting loss components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "losses.png"

            mlm_losses = [0.3, 0.25, 0.2]
            nsp_losses = [0.2, 0.15, 0.1]
            total_losses = [0.5, 0.4, 0.3]

            plot_loss_components(
                mlm_losses=mlm_losses,
                nsp_losses=nsp_losses,
                total_losses=total_losses,
                output_path=str(output_path),
            )

            assert output_path.exists()

    def test_plot_metrics_summary(self):
        """Test plotting metrics summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.png"

            metrics = {
                "Loss": [0.5, 0.4, 0.3],
                "Accuracy": [0.8, 0.85, 0.9],
                "F1 Score": [0.75, 0.8, 0.85],
            }

            plot_metrics_summary(
                metrics=metrics,
                output_path=str(output_path),
            )

            assert output_path.exists()

    def test_plot_single_metric(self):
        """Test plotting with single metric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metric.png"

            metrics = {"Loss": [0.5, 0.4, 0.3]}

            plot_metrics_summary(
                metrics=metrics,
                output_path=str(output_path),
            )

            assert output_path.exists()

    def test_plot_creates_directory(self):
        """Test that plotting creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "curves.png"

            train_losses = [0.5, 0.4]

            plot_training_curves(
                train_losses=train_losses,
                output_path=str(output_path),
            )

            assert output_path.exists()


if __name__ == "__main__":
    pytest.main([__file__])
