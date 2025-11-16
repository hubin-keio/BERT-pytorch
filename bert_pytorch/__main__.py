import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from .model import BERT
from .trainer import BERTLightningModule
from .dataset import BERTDataset, WordVocab
from .utils import set_seed, get_logger, plot_training_curves
from .utils.visualization import plot_loss_components

logger = get_logger(__name__)


def str_to_bool(v: str) -> bool:
    """Convert string to boolean value"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train():
    """Main training function with PyTorch Lightning"""
    parser = argparse.ArgumentParser(
        description="Train BERT model with Masked Language Modeling and Next Sentence Prediction"
    )

    # Dataset arguments
    parser.add_argument("-c", "--train_dataset", required=True, type=str,
                       help="path to training dataset (tab-separated sentence pairs)")
    parser.add_argument("-t", "--test_dataset", type=str, default=None,
                       help="path to test/validation dataset")
    parser.add_argument("-v", "--vocab_path", required=True, type=str,
                       help="path to vocabulary file (built with bert-vocab)")
    parser.add_argument("-o", "--output_path", required=True, type=str,
                       help="directory to save model checkpoints and logs")

    # Model architecture arguments
    parser.add_argument("-hs", "--hidden", type=int, default=256,
                       help="hidden size of transformer model (default: 256)")
    parser.add_argument("-l", "--layers", type=int, default=8,
                       help="number of transformer layers (default: 8)")
    parser.add_argument("-a", "--attn_heads", type=int, default=8,
                       help="number of attention heads (default: 8)")
    parser.add_argument("-s", "--seq_len", type=int, default=20,
                       help="maximum sequence length (default: 20)")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="dropout rate (default: 0.1)")

    # Training arguments
    parser.add_argument("-b", "--batch_size", type=int, default=64,
                       help="batch size (default: 64)")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                       help="number of epochs (default: 10)")
    parser.add_argument("-w", "--num_workers", type=int, default=5,
                       help="number of DataLoader workers (default: 5)")

    # Optimizer arguments
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="learning rate (default: 1e-3)")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01,
                       help="weight decay for Adam optimizer (default: 0.01)")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                       help="beta1 for Adam optimizer (default: 0.9)")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                       help="beta2 for Adam optimizer (default: 0.999)")
    parser.add_argument("--warmup_steps", type=int, default=10000,
                       help="number of warmup steps (default: 10000)")

    # Device and utility arguments
    parser.add_argument("--with_cuda", type=str_to_bool, default=True,
                       help="enable CUDA training (default: true)")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None,
                       help="CUDA device IDs to use")
    parser.add_argument("--log_freq", type=int, default=10,
                       help="logging frequency in batches (default: 10)")
    parser.add_argument("--corpus_lines", type=int, default=None,
                       help="total number of lines in corpus")
    parser.add_argument("--on_memory", type=str_to_bool, default=True,
                       help="load dataset into memory (default: true)")
    parser.add_argument("--seed", type=int, default=42,
                       help="random seed for reproducibility (default: 42)")
    parser.add_argument("--patience", type=int, default=5,
                       help="early stopping patience in epochs (default: 5)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("BERT Pretraining with PyTorch Lightning")
    logger.info("=" * 80)

    # Load vocabulary
    logger.info(f"Loading vocabulary from {args.vocab_path}")
    vocab = WordVocab.load_vocab(args.vocab_path)
    logger.info(f"Vocabulary size: {len(vocab)}")

    # Load datasets
    logger.info(f"Loading training dataset from {args.train_dataset}")
    train_dataset = BERTDataset(
        args.train_dataset, vocab, seq_len=args.seq_len,
        corpus_lines=args.corpus_lines, on_memory=args.on_memory
    )
    logger.info(f"Training dataset size: {len(train_dataset)}")

    test_dataset = None
    if args.test_dataset is not None:
        logger.info(f"Loading test dataset from {args.test_dataset}")
        test_dataset = BERTDataset(
            args.test_dataset, vocab, seq_len=args.seq_len, on_memory=args.on_memory
        )
        logger.info(f"Test dataset size: {len(test_dataset)}")

    # Create data loaders
    logger.info("Creating data loaders")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, pin_memory=args.with_cuda
    )

    val_loader = None
    if test_dataset is not None:
        val_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=False, pin_memory=args.with_cuda
        )

    # Build BERT model
    logger.info("Building BERT model")
    logger.info(f"  - Hidden size: {args.hidden}")
    logger.info(f"  - Layers: {args.layers}")
    logger.info(f"  - Attention heads: {args.attn_heads}")
    logger.info(f"  - Vocabulary size: {len(vocab)}")

    bert = BERT(
        vocab_size=len(vocab),
        hidden=args.hidden,
        n_layers=args.layers,
        attn_heads=args.attn_heads,
        dropout=args.dropout
    )

    # Create Lightning module
    logger.info("Creating BERT Lightning module")
    lightning_module = BERTLightningModule(
        bert=bert,
        vocab_size=len(vocab),
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.adam_weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        log_freq=args.log_freq
    )

    # Setup TensorBoard logger
    logger.info(f"Setting up TensorBoard logging to {output_dir / 'logs'}")
    tb_logger = TensorBoardLogger(
        save_dir=output_dir,
        name="logs",
        version=0,
        log_graph=True
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="bert-{epoch:02d}-{val/epoch_loss:.2f}",
        monitor="val/epoch_loss" if val_loader else "train/epoch_loss",
        mode="min",
        save_top_k=3,
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val/epoch_loss" if val_loader else "train/epoch_loss",
        patience=args.patience,
        verbose=True,
        mode="min"
    )

    callbacks = [checkpoint_callback, early_stop_callback]

    # Determine device
    gpus = None
    if args.with_cuda and torch.cuda.is_available():
        gpus = args.cuda_devices if args.cuda_devices else [0]
        logger.info(f"Using GPUs: {gpus}")
    else:
        logger.info("Using CPU")

    # Create PyTorch Lightning trainer
    logger.info("Creating PyTorch Lightning trainer")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=gpus,
        strategy="ddp" if (args.with_cuda and torch.cuda.device_count() > 1) else None,
        logger=tb_logger,
        callbacks=callbacks,
        log_every_n_steps=args.log_freq,
        enable_checkpointing=True,
        enable_model_summary=True,
        enable_progress_bar=True,
        deterministic=True
    )

    # Train the model
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    # Save final model
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)

    final_model_path = output_dir / "bert_final.pth"
    torch.save(bert.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    logger.info(f"Logs and checkpoints saved to {output_dir}")
    logger.info(f"View results with: tensorboard --logdir {output_dir / 'logs'}")


if __name__ == "__main__":
    train()
