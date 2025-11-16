"""PyTorch Lightning module for BERT training"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import Adam
from typing import Tuple, Optional

from ..model import BERTLM, BERT
from .optim_schedule import ScheduledOptim
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BERTLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for BERT pretraining with MLM and NSP tasks.

    Handles:
    - Forward pass through BERT with task heads
    - Loss computation for both MLM and NSP
    - Optimization with learning rate scheduling
    - TensorBoard logging per epoch
    - Automatic checkpoint management
    """

    def __init__(
        self,
        bert: BERT,
        vocab_size: int,
        lr: float = 1e-4,
        warmup_steps: int = 10000,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        log_freq: int = 10
    ) -> None:
        """
        Initialize BERT Lightning module.

        Args:
            bert: BERT encoder model
            vocab_size: vocabulary size
            lr: learning rate
            warmup_steps: number of warmup steps for LR scheduler
            weight_decay: weight decay for Adam optimizer
            betas: beta parameters for Adam optimizer
            log_freq: logging frequency (batches)
        """
        super().__init__()

        self.bert_model = BERTLM(bert, vocab_size)
        self.criterion = nn.NLLLoss(ignore_index=0)
        self.log_freq = log_freq

        # Store hyperparameters
        self.save_hyperparameters({
            'lr': lr,
            'warmup_steps': warmup_steps,
            'weight_decay': weight_decay,
            'betas': betas,
            'vocab_size': vocab_size
        })

        # Training metrics
        self.train_loss = 0.0
        self.train_mlm_loss = 0.0
        self.train_nsp_loss = 0.0
        self.train_correct = 0
        self.train_total = 0

        # Validation metrics
        self.val_loss = 0.0
        self.val_mlm_loss = 0.0
        self.val_nsp_loss = 0.0
        self.val_correct = 0
        self.val_total = 0

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = ScheduledOptim(
            optimizer,
            d_model=self.bert_model.bert.hidden,
            n_warmup_steps=self.hparams.warmup_steps
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

    def forward(self, x: torch.Tensor, segment_label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: input token indices
            segment_label: segment labels

        Returns:
            tuple of (nsp_logits, mlm_logits)
        """
        return self.bert_model(x, segment_label)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: batch of data
            batch_idx: batch index

        Returns:
            loss value
        """
        # Forward pass
        next_sent_output, mask_lm_output = self.forward(batch["bert_input"], batch["segment_label"])

        # Compute losses
        next_loss = self.criterion(next_sent_output, batch["is_next"])
        mask_loss = self.criterion(mask_lm_output.transpose(1, 2), batch["bert_label"])
        loss = next_loss + mask_loss

        # Compute accuracy for NSP task
        correct = next_sent_output.argmax(dim=-1).eq(batch["is_next"]).sum().item()

        # Update running metrics
        self.train_loss += loss.item()
        self.train_mlm_loss += mask_loss.item()
        self.train_nsp_loss += next_loss.item()
        self.train_correct += correct
        self.train_total += batch["is_next"].nelement()

        # Log per batch
        if (batch_idx + 1) % self.log_freq == 0:
            avg_loss = self.train_loss / (batch_idx + 1)
            avg_acc = (self.train_correct / self.train_total * 100) if self.train_total > 0 else 0
            self.log('train/loss', avg_loss, on_step=True, on_epoch=False)
            self.log('train/nsp_loss', self.train_nsp_loss / (batch_idx + 1), on_step=True, on_epoch=False)
            self.log('train/mlm_loss', self.train_mlm_loss / (batch_idx + 1), on_step=True, on_epoch=False)
            self.log('train/acc', avg_acc, on_step=True, on_epoch=False)

        return loss

    def on_train_epoch_end(self) -> None:
        """Called at end of training epoch."""
        if self.train_total > 0:
            avg_loss = self.train_loss / max(1, len(self.trainer.train_dataloader))
            avg_acc = self.train_correct / self.train_total * 100
            avg_mlm_loss = self.train_mlm_loss / max(1, len(self.trainer.train_dataloader))
            avg_nsp_loss = self.train_nsp_loss / max(1, len(self.trainer.train_dataloader))

            self.log('train/epoch_loss', avg_loss, on_epoch=True)
            self.log('train/epoch_nsp_loss', avg_nsp_loss, on_epoch=True)
            self.log('train/epoch_mlm_loss', avg_mlm_loss, on_epoch=True)
            self.log('train/epoch_acc', avg_acc, on_epoch=True)

            logger.info(
                f"Epoch {self.current_epoch} - "
                f"Train Loss: {avg_loss:.4f}, "
                f"MLM Loss: {avg_mlm_loss:.4f}, "
                f"NSP Loss: {avg_nsp_loss:.4f}, "
                f"Accuracy: {avg_acc:.2f}%"
            )

            # Reset metrics
            self.train_loss = 0.0
            self.train_mlm_loss = 0.0
            self.train_nsp_loss = 0.0
            self.train_correct = 0
            self.train_total = 0

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Validation step.

        Args:
            batch: batch of data
            batch_idx: batch index

        Returns:
            loss value
        """
        # Forward pass
        next_sent_output, mask_lm_output = self.forward(batch["bert_input"], batch["segment_label"])

        # Compute losses
        next_loss = self.criterion(next_sent_output, batch["is_next"])
        mask_loss = self.criterion(mask_lm_output.transpose(1, 2), batch["bert_label"])
        loss = next_loss + mask_loss

        # Compute accuracy
        correct = next_sent_output.argmax(dim=-1).eq(batch["is_next"]).sum().item()

        # Update running metrics
        self.val_loss += loss.item()
        self.val_mlm_loss += mask_loss.item()
        self.val_nsp_loss += next_loss.item()
        self.val_correct += correct
        self.val_total += batch["is_next"].nelement()

        return loss

    def on_validation_epoch_end(self) -> None:
        """Called at end of validation epoch."""
        if self.val_total > 0:
            avg_loss = self.val_loss / max(1, len(self.trainer.val_dataloaders))
            avg_acc = self.val_correct / self.val_total * 100
            avg_mlm_loss = self.val_mlm_loss / max(1, len(self.trainer.val_dataloaders))
            avg_nsp_loss = self.val_nsp_loss / max(1, len(self.trainer.val_dataloaders))

            self.log('val/epoch_loss', avg_loss, on_epoch=True)
            self.log('val/epoch_nsp_loss', avg_nsp_loss, on_epoch=True)
            self.log('val/epoch_mlm_loss', avg_mlm_loss, on_epoch=True)
            self.log('val/epoch_acc', avg_acc, on_epoch=True)

            logger.info(
                f"Epoch {self.current_epoch} - "
                f"Val Loss: {avg_loss:.4f}, "
                f"MLM Loss: {avg_mlm_loss:.4f}, "
                f"NSP Loss: {avg_nsp_loss:.4f}, "
                f"Accuracy: {avg_acc:.2f}%"
            )

            # Reset metrics
            self.val_loss = 0.0
            self.val_mlm_loss = 0.0
            self.val_nsp_loss = 0.0
            self.val_correct = 0
            self.val_total = 0
