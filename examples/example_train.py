#!/usr/bin/env python3
"""
Example training script for BERT protein language model.

This script demonstrates how to:
1. Prepare training data for protein sequences
2. Build vocabulary from protein corpus
3. Train BERT using PyTorch Lightning
4. Monitor training with TensorBoard
5. Save and visualize results

Example protein corpus format:
```
MKTIIALSYIFCLVFADYKDDDKWEE    TAVG
NNKEIANDKPLPSTTEKFPYDSAE     SPTQ
...
```

Tab-separated pairs of protein sequences.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from bert_pytorch.model import BERT
from bert_pytorch.trainer import BERTLightningModule
from bert_pytorch.dataset import BERTDataset, WordVocab
from bert_pytorch.utils import set_seed, get_logger
from bert_pytorch.utils.visualization import plot_training_curves, plot_loss_components

logger = get_logger(__name__)


def create_sample_protein_corpus(output_file: Path, num_samples: int = 100) -> None:
    """
    Create a sample protein corpus for testing.

    Args:
        output_file: Path to save corpus
        num_samples: Number of sentence pairs to generate
    """
    logger.info(f"Creating sample protein corpus with {num_samples} samples")

    # Sample amino acids and proteins
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    sample_proteins = [
        "MKTIIALSYIFCLVFADYKDDDKWEE",
        "NNKEIANDKPLPSTTEKFPYDSAE",
        "MAVHQVSTPTLVEVSR",
        "MKLAVLGLSLVLTVACSASCSGGGGSG",
        "METAYLGDSNPAQPPGLAK",
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for i in range(num_samples):
            # Create two random protein sequences
            prot1 = "".join(
                [amino_acids[j % len(amino_acids)] for j in range(20 + i % 10)]
            )
            prot2 = "".join(
                [amino_acids[(i + j) % len(amino_acids)] for j in range(20 + (i + 1) % 10)]
            )

            # Occasionally use real samples
            if i % 10 < len(sample_proteins):
                prot1 = sample_proteins[i % len(sample_proteins)]
                prot2 = sample_proteins[(i + 1) % len(sample_proteins)]

            f.write(f"{prot1}\t{prot2}\n")

    logger.info(f"Sample corpus saved to {output_file}")


def build_vocabulary(
    corpus_path: Path, vocab_path: Path, vocab_size: Optional[int] = None
) -> WordVocab:
    """
    Build vocabulary from corpus.

    Args:
        corpus_path: Path to corpus file
        vocab_path: Path to save vocabulary
        vocab_size: Maximum vocabulary size (optional)

    Returns:
        WordVocab object
    """
    logger.info(f"Building vocabulary from {corpus_path}")

    with open(corpus_path, "r") as f:
        vocab = WordVocab(f, max_size=vocab_size, min_freq=1)

    vocab.save_vocab(str(vocab_path))
    logger.info(f"Vocabulary saved to {vocab_path}")
    logger.info(f"Vocabulary size: {len(vocab)}")

    return vocab


def setup_training(
    train_corpus: Path,
    test_corpus: Optional[Path],
    vocab_path: Path,
    output_dir: Path,
    hidden_size: int = 256,
    num_layers: int = 8,
    attn_heads: int = 8,
    seq_len: int = 512,
    batch_size: int = 32,
    num_workers: int = 4,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    seed: int = 42,
) -> Tuple[pl.Trainer, BERTLightningModule, DataLoader, Optional[DataLoader]]:
    """
    Setup training components.

    Args:
        train_corpus: Path to training corpus
        test_corpus: Path to test corpus (optional)
        vocab_path: Path to vocabulary
        output_dir: Output directory for checkpoints
        hidden_size: Hidden size of model
        num_layers: Number of transformer layers
        attn_heads: Number of attention heads
        seq_len: Maximum sequence length
        batch_size: Training batch size
        num_workers: Number of DataLoader workers
        epochs: Number of epochs
        learning_rate: Learning rate
        seed: Random seed

    Returns:
        Tuple of (trainer, module, train_loader, val_loader)
    """
    # Set seed for reproducibility
    set_seed(seed)

    # Load vocabulary
    logger.info(f"Loading vocabulary from {vocab_path}")
    vocab = WordVocab.load_vocab(str(vocab_path))
    logger.info(f"Vocabulary size: {len(vocab)}")

    # Load datasets
    logger.info(f"Loading training dataset from {train_corpus}")
    train_dataset = BERTDataset(
        str(train_corpus), vocab, seq_len=seq_len, on_memory=True
    )
    logger.info(f"Training dataset size: {len(train_dataset)}")

    test_dataset = None
    if test_corpus and test_corpus.exists():
        logger.info(f"Loading test dataset from {test_corpus}")
        test_dataset = BERTDataset(
            str(test_corpus), vocab, seq_len=seq_len, on_memory=True
        )
        logger.info(f"Test dataset size: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if test_dataset:
        val_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

    # Build model
    logger.info("Building BERT model")
    bert = BERT(
        vocab_size=len(vocab),
        hidden=hidden_size,
        n_layers=num_layers,
        attn_heads=attn_heads,
    )

    # Create Lightning module
    module = BERTLightningModule(
        bert=bert,
        vocab_size=len(vocab),
        lr=learning_rate,
        warmup_steps=min(10000, len(train_loader) * epochs // 10),
    )

    # Setup TensorBoard logger
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir), name="logs", version=0, log_graph=True
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="bert-{epoch:02d}-{val/epoch_loss:.2f}" if val_loader else "bert-{epoch:02d}",
        monitor="val/epoch_loss" if val_loader else "train/epoch_loss",
        mode="min",
        save_top_k=3,
        verbose=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val/epoch_loss" if val_loader else "train/epoch_loss",
        patience=5,
        verbose=True,
        mode="min",
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        enable_checkpointing=True,
        enable_progress_bar=True,
        deterministic=True,
    )

    return trainer, module, train_loader, val_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Example BERT training script for protein sequences"
    )
    parser.add_argument(
        "--train_corpus",
        type=Path,
        default=Path("data/protein_train.txt"),
        help="Path to training corpus",
    )
    parser.add_argument(
        "--test_corpus",
        type=Path,
        default=Path("data/protein_test.txt"),
        help="Path to test corpus (optional)",
    )
    parser.add_argument(
        "--vocab_path", type=Path, default=Path("data/vocab.pkl"), help="Path to vocabulary"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/protein_bert"),
        help="Output directory",
    )
    parser.add_argument(
        "--create_sample_data",
        action="store_true",
        help="Create sample protein corpus for testing",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="Hidden size"
    )
    parser.add_argument(
        "--num_layers", type=int, default=8, help="Number of layers"
    )
    parser.add_argument(
        "--attn_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--seq_len", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of DataLoader workers"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("BERT Protein Language Model Training Example")
    logger.info("=" * 80)

    # Create sample data if requested
    if args.create_sample_data:
        args.train_corpus.parent.mkdir(parents=True, exist_ok=True)
        create_sample_protein_corpus(args.train_corpus, num_samples=500)
        if not args.test_corpus.exists():
            create_sample_protein_corpus(args.test_corpus, num_samples=100)

    # Build vocabulary if needed
    if not args.vocab_path.exists():
        build_vocabulary(args.train_corpus, args.vocab_path, vocab_size=500)

    # Setup training
    trainer, module, train_loader, val_loader = setup_training(
        train_corpus=args.train_corpus,
        test_corpus=args.test_corpus,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        attn_heads=args.attn_heads,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save final model
    final_model_path = args.output_dir / "bert_protein_final.pth"
    torch.save(module.bert_model.bert.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Logs: {args.output_dir / 'logs'}")
    logger.info(f"Checkpoints: {args.output_dir / 'checkpoints'}")
    logger.info("=" * 80)

    # Print TensorBoard command
    logger.info(f"View results with: tensorboard --logdir {args.output_dir / 'logs'}")


if __name__ == "__main__":
    main()
