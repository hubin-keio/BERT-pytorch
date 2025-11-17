# BERT-pytorch

[![LICENSE](https://img.shields.io/github/license/codertimo/BERT-pytorch.svg)](https://github.com/codertimo/BERT-pytorch/blob/master/LICENSE)
![GitHub issues](https://img.shields.io/github/issues/codertimo/BERT-pytorch.svg)
[![GitHub stars](https://img.shields.io/github/stars/codertimo/BERT-pytorch.svg)](https://github.com/codertimo/BERT-pytorch/stargazers)
[![CircleCI](https://circleci.com/gh/codertimo/BERT-pytorch.svg?style=shield)](https://circleci.com/gh/codertimo/BERT-pytorch)
[![PyPI](https://img.shields.io/pypi/v/bert-pytorch.svg)](https://pypi.org/project/bert_pytorch/)
[![PyPI - Status](https://img.shields.io/pypi/status/bert-pytorch.svg)](https://pypi.org/project/bert_pytorch/)
[![Documentation Status](https://readthedocs.org/projects/bert-pytorch/badge/?version=latest)](https://bert-pytorch.readthedocs.io/en/latest/?badge=latest)

PyTorch implementation of Google AI's 2018 BERT, optimized for protein language modeling with PyTorch Lightning and TensorBoard support.

> **BERT**: Pre-training of Deep Bidirectional Transformers for Language Understanding
> Paper URL: https://arxiv.org/abs/1810.04805

**Latest Release**: November 2025 - Enhanced with PyTorch Lightning, TensorBoard, type hints, and comprehensive utilities


## Introduction

Google AI's BERT paper shows the amazing result on various NLP task (new 17 NLP tasks SOTA), 
including outperform the human F1 score on SQuAD v1.1 QA task. 
This paper proved that Transformer(self-attention) based encoder can be powerfully used as 
alternative of previous language model with proper language model training method. 
And more importantly, they showed us that this pre-trained language model can be transfer 
into any NLP task without making task specific model architecture.

This amazing result would be record in NLP history, 
and I expect many further papers about BERT will be published very soon.

This repo is implementation of BERT. Code is very simple and easy to understand fastly.
Some of these codes are based on [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

Currently this project is working on progress. And the code is not verified yet.

## Installation
```bash
pip install -e .
# or
pip install -r requirements.txt
```

### Requirements
- PyTorch >= 1.9.0
- PyTorch Lightning >= 1.5.0
- TensorBoard >= 2.7.0
- NumPy >= 1.19.0
- tqdm >= 4.60.0
- matplotlib >= 3.3.0

## Key Features

### ✨ New in November 2025 Release
- **PyTorch Lightning Integration**: Simplified training with automatic distributed support
- **TensorBoard Logging**: Real-time monitoring of training metrics per epoch
- **Type Hints**: Full type annotations for better IDE support and error detection
- **Input Validation**: Automatic parameter checking to prevent silent failures
- **Reproducibility**: Seed management for deterministic training
- **Visualization Tools**: Matplotlib-based plotting functions for analysis
- **Comprehensive Logging**: Structured logging throughout training pipeline
- **Early Stopping & Checkpointing**: Automatic model management and overfitting prevention

## Quickstart

### 0. Prepare your corpus
```
Welcome to the \t the jungle\n
I can stay \t here all night\n
```

or tokenized corpus (tokenization is not in package)
```
Wel_ _come _to _the \t _the _jungle\n
_I _can _stay \t _here _all _night\n
```


### 1. Building vocab based on your corpus
```shell
bert-vocab -c data/corpus.small -o data/vocab.small
```

### 2. Train your own BERT model
```bash
bert -c data/corpus.small -v data/vocab.small -o output/bert_training \
  --hidden 256 \
  --layers 8 \
  --attn_heads 8 \
  --epochs 20 \
  --batch_size 64 \
  --seed 42 \
  --patience 5
```

### 3. Monitor training with TensorBoard
```bash
tensorboard --logdir output/bert_training/logs
```

Then open http://localhost:6006 in your browser.

## Advanced Training Usage

### Full Training Example with All Options
```bash
bert \
  --train_dataset data/corpus.train \
  --test_dataset data/corpus.test \
  --vocab_path data/vocab.pkl \
  --output_path output/bert_model \
  --hidden 384 \
  --layers 12 \
  --attn_heads 12 \
  --seq_len 512 \
  --dropout 0.1 \
  --batch_size 32 \
  --epochs 100 \
  --num_workers 8 \
  --lr 0.0001 \
  --warmup_steps 10000 \
  --with_cuda true \
  --seed 42 \
  --patience 5
```

### Command-Line Arguments

#### Dataset Arguments
- `-c, --train_dataset` (required): Path to training dataset
- `-t, --test_dataset` (optional): Path to test/validation dataset
- `-v, --vocab_path` (required): Path to vocabulary file
- `-o, --output_path` (required): Output directory for checkpoints and logs

#### Model Architecture
- `--hidden` (default: 256): Hidden size of transformer
- `--layers` (default: 8): Number of transformer layers
- `--attn_heads` (default: 8): Number of attention heads
- `--seq_len` (default: 20): Maximum sequence length (protein sequences can be up to 2048)
- `--dropout` (default: 0.1): Dropout rate

#### Training Configuration
- `-b, --batch_size` (default: 64): Batch size
- `-e, --epochs` (default: 10): Number of epochs
- `-w, --num_workers` (default: 5): DataLoader workers
- `--lr` (default: 0.001): Learning rate
- `--warmup_steps` (default: 10000): LR warmup steps
- `--adam_weight_decay` (default: 0.01): Weight decay
- `--patience` (default: 5): Early stopping patience

#### Utility Options
- `--with_cuda` (default: true): Use CUDA if available
- `--cuda_devices`: Specific GPU IDs to use
- `--seed` (default: 42): Random seed for reproducibility
- `--log_freq` (default: 10): Logging frequency in batches
- `--on_memory` (default: true): Load dataset into memory

## Using the Utilities

### Reproducibility with Seed Management
```python
from bert_pytorch.utils import set_seed

# Set seed for reproducible training
set_seed(42)
# This sets seeds for Python, NumPy, PyTorch, and CUDA
```

### Logging
```python
from bert_pytorch.utils import get_logger

logger = get_logger(__name__)
logger.info("Training started")
logger.warning("This is a warning")
logger.error("An error occurred")
```

### Visualization
```python
from bert_pytorch.utils.visualization import (
    plot_training_curves,
    plot_loss_components,
    plot_metrics_summary
)

# Plot training curves
plot_training_curves(
    train_losses=[0.5, 0.4, 0.3],
    val_losses=[0.6, 0.5, 0.4],
    train_accs=[80, 85, 88],
    val_accs=[75, 80, 83],
    output_path="results/training_curves.png",
    title="Training Progress"
)

# Plot loss components (MLM vs NSP)
plot_loss_components(
    mlm_losses=[0.3, 0.25, 0.2],
    nsp_losses=[0.2, 0.15, 0.1],
    total_losses=[0.5, 0.4, 0.3],
    output_path="results/loss_components.png"
)

# Plot multiple metrics
metrics = {
    "MLM Loss": [0.3, 0.25, 0.2],
    "NSP Loss": [0.2, 0.15, 0.1],
    "Accuracy": [80, 85, 88]
}
plot_metrics_summary(metrics, output_path="results/metrics.png")
```

## Python API Usage

### Loading a Pre-trained Model
```python
import torch
from bert_pytorch import BERT
from bert_pytorch.model import BERTLM

# Load vocabulary
vocab = WordVocab.load_vocab('vocab.pkl')

# Initialize model
bert = BERT(vocab_size=len(vocab), hidden=256, n_layers=8, attn_heads=8)
lm = BERTLM(bert, vocab_size=len(vocab))

# Load weights
state_dict = torch.load('bert_final.pth')
bert.load_state_dict(state_dict)

# Inference
bert.eval()
with torch.no_grad():
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    segment_labels = torch.tensor([[1, 1, 1, 2, 2]])
    output = bert(input_ids, segment_labels)
    # output shape: [batch_size, seq_len, hidden_size]
```

### Using the Lightning Module
```python
from bert_pytorch.trainer import BERTLightningModule
import pytorch_lightning as pl

# Create module
module = BERTLightningModule(
    bert=bert,
    vocab_size=len(vocab),
    lr=1e-4,
    warmup_steps=10000
)

# Create trainer
trainer = pl.Trainer(max_epochs=20, gpus=1)

# Train
trainer.fit(module, train_dataloaders, val_dataloaders)
```

## Language Model Pre-training

In the paper, authors shows the new language model training methods, 
which are "masked language model" and "predict next sentence".


### Masked Language Model 

> Original Paper : 3.3.1 Task #1: Masked LM 

```
Input Sequence  : The man went to [MASK] store with [MASK] dog
Target Sequence :                  the                his
```

#### Rules:
Randomly 15% of input token will be changed into something, based on under sub-rules

1. Randomly 80% of tokens, gonna be a `[MASK]` token
2. Randomly 10% of tokens, gonna be a `[RANDOM]` token(another word)
3. Randomly 10% of tokens, will be remain as same. But need to be predicted.

### Predict Next Sentence

> Original Paper : 3.3.2 Task #2: Next Sentence Prediction

```
Input : [CLS] the man went to the store [SEP] he bought a gallon of milk [SEP]
Label : Is Next

Input = [CLS] the man heading to the store [SEP] penguin [MASK] are flight ##less birds [SEP]
Label = NotNext
```

"Is this sentence can be continuously connected?"

 understanding the relationship, between two text sentences, which is
not directly captured by language modeling

#### Rules:

1. Randomly 50% of next sentence, gonna be continuous sentence.
2. Randomly 50% of next sentence, gonna be unrelated sentence.


## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch_size 16`
- Reduce hidden size: `--hidden 128`
- Reduce sequence length: `--seq_len 256`
- Enable gradient checkpointing (for advanced users)

### Training is Slow
- Increase number of workers: `--num_workers 8`
- Use multiple GPUs: `--cuda_devices 0 1 2 3`
- Reduce logging frequency: `--log_freq 50`

### Model Not Converging
- Reduce learning rate: `--lr 1e-5`
- Increase warmup steps: `--warmup_steps 50000`
- Check corpus quality and format (tab-separated sentences)
- Try different random seed: `--seed 123`

### TensorBoard Not Showing Data
- Ensure training completed at least one epoch
- Check logs directory: `output/bert_training/logs`
- Restart TensorBoard: `tensorboard --logdir output/bert_training/logs`
- Clear browser cache if using http://localhost:6006

## FAQ

**Q: How do I handle protein sequences longer than default sequence length?**
A: Set `--seq_len 512` or higher (up to 2048). Consider your GPU memory accordingly.

**Q: Can I use this for other languages besides protein sequences?**
A: Yes! This is a general BERT implementation suitable for any tokenized text. Prepare corpus in the same format.

**Q: How do I resume training from a checkpoint?**
A: Load the checkpoint via `torch.load()` and pass to the model before calling `trainer.fit()`.

**Q: What's the recommended batch size?**
A: Start with 64. If OOM errors occur, reduce to 32 or 16. Larger batches (128+) may improve convergence but require more GPU memory.

**Q: How often should I validate?**
A: If you have a test set, validation happens every epoch. Otherwise, early stopping monitors training loss.

**Q: Can I use multiple GPUs?**
A: Yes, if multiple GPUs are available, specify: `--cuda_devices 0 1 2 3`

**Q: How do I extract features from a trained model?**
A: Use the BERT encoder output (hidden states) after loading the pre-trained weights:
```python
bert.eval()
with torch.no_grad():
    output = bert(input_ids, segment_labels)  # [batch, seq_len, hidden]
```

## Publications & Projects Using BERT-pytorch

If you use this implementation in your research, please cite the original BERT paper and this repository.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Original Author
Junseong Kim, Scatter Lab (codertimo@gmail.com / junseong.kim@scatterlab.co.kr)

## Contributors (November 2025)
- PyTorch Lightning integration
- TensorBoard support
- Type hints and validation
- Comprehensive utilities and documentation

## License

This project follows Apache 2.0 License as written in LICENSE file

Copyright 2018 Junseong Kim, Scatter Lab, respective BERT contributors

Copyright (c) 2018 Alexander Rush: [The Annotated Transformer](https://github.com/harvardnlp/annotated-transformer)
