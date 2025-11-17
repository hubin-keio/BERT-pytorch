#!/usr/bin/env python3
"""
Example inference script for BERT protein language model.

This script demonstrates how to:
1. Load a pre-trained BERT model
2. Perform inference on protein sequences
3. Extract contextual embeddings
4. Use the model for downstream tasks
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from bert_pytorch.model import BERT, BERTLM
from bert_pytorch.dataset import WordVocab
from bert_pytorch.utils import get_logger

logger = get_logger(__name__)


class BERTInference:
    """Wrapper for BERT inference."""

    def __init__(self, model_path: Path, vocab_path: Path, device: str = "cpu"):
        """
        Initialize BERT inference model.

        Args:
            model_path: Path to saved model weights
            vocab_path: Path to vocabulary file
            device: Device to load model on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)

        # Load vocabulary
        logger.info(f"Loading vocabulary from {vocab_path}")
        self.vocab = WordVocab.load_vocab(str(vocab_path))
        logger.info(f"Vocabulary size: {len(self.vocab)}")

        # Build model
        logger.info("Building BERT model")
        self.bert = BERT(
            vocab_size=len(self.vocab), hidden=256, n_layers=8, attn_heads=8
        )

        # Load weights
        logger.info(f"Loading model weights from {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)
        self.bert.load_state_dict(state_dict)
        self.bert.to(self.device)
        self.bert.eval()

        logger.info("Model ready for inference")

    def tokenize(self, sequence: str) -> List[int]:
        """
        Tokenize protein sequence.

        Args:
            sequence: Protein sequence (single letter amino acid codes)

        Returns:
            List of token IDs
        """
        # Add special tokens
        tokens = [self.vocab.sos_index]  # [CLS]

        # Tokenize individual amino acids
        for aa in sequence:
            token_id = self.vocab.stoi.get(aa, self.vocab.unk_index)
            tokens.append(token_id)

        tokens.append(self.vocab.eos_index)  # [SEP]

        return tokens

    def pad_sequence(self, tokens: List[int], max_len: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad sequence to fixed length.

        Args:
            tokens: Token IDs
            max_len: Maximum sequence length

        Returns:
            Tuple of (padded tokens, segment labels)
        """
        # Pad or truncate
        if len(tokens) < max_len:
            tokens = tokens + [self.vocab.pad_index] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]

        # Create segment labels (all 1s for single sequence)
        segment_labels = [1 if token != self.vocab.pad_index else 0 for token in tokens]

        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(segment_labels, dtype=torch.long),
        )

    @torch.no_grad()
    def get_embeddings(self, sequences: List[str], max_len: int = 512) -> torch.Tensor:
        """
        Get contextual embeddings for sequences.

        Args:
            sequences: List of protein sequences
            max_len: Maximum sequence length

        Returns:
            Embeddings of shape [batch_size, seq_len, hidden_size]
        """
        # Tokenize and pad
        batch_input_ids = []
        batch_segment_ids = []

        for seq in sequences:
            tokens = self.tokenize(seq)
            input_ids, segment_ids = self.pad_sequence(tokens, max_len)
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(segment_ids)

        # Stack into batch
        input_ids = torch.stack(batch_input_ids).to(self.device)
        segment_ids = torch.stack(batch_segment_ids).to(self.device)

        # Forward pass
        embeddings = self.bert(input_ids, segment_ids)

        return embeddings

    @torch.no_grad()
    def get_sequence_embedding(self, sequences: List[str], max_len: int = 512) -> torch.Tensor:
        """
        Get aggregated sequence embeddings (mean of non-pad tokens).

        Args:
            sequences: List of protein sequences
            max_len: Maximum sequence length

        Returns:
            Sequence embeddings of shape [batch_size, hidden_size]
        """
        embeddings = self.get_embeddings(sequences, max_len)

        # Create mask for non-pad tokens
        batch_input_ids = []
        for seq in sequences:
            tokens = self.tokenize(seq)
            input_ids, _ = self.pad_sequence(tokens, max_len)
            batch_input_ids.append(input_ids)

        input_ids = torch.stack(batch_input_ids).to(self.device)
        mask = (input_ids != self.vocab.pad_index).float()

        # Mean pooling over valid tokens
        masked_embeddings = embeddings * mask.unsqueeze(-1)
        sum_embeddings = masked_embeddings.sum(dim=1)
        sum_mask = mask.sum(dim=1, keepdim=True)

        sequence_embeddings = sum_embeddings / (sum_mask + 1e-9)

        return sequence_embeddings

    @torch.no_grad()
    def get_cls_embedding(self, sequences: List[str], max_len: int = 512) -> torch.Tensor:
        """
        Get [CLS] token embeddings.

        Args:
            sequences: List of protein sequences
            max_len: Maximum sequence length

        Returns:
            [CLS] embeddings of shape [batch_size, hidden_size]
        """
        embeddings = self.get_embeddings(sequences, max_len)
        return embeddings[:, 0, :]  # First token is [CLS]

    def similarity(self, seq1: str, seq2: str, max_len: int = 512) -> float:
        """
        Compute cosine similarity between two sequences.

        Args:
            seq1: First protein sequence
            seq2: Second protein sequence
            max_len: Maximum sequence length

        Returns:
            Cosine similarity score
        """
        embeddings = self.get_sequence_embedding([seq1, seq2], max_len)
        emb1 = embeddings[0]
        emb2 = embeddings[1]

        similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        return similarity


def main():
    """Main inference example."""
    parser = argparse.ArgumentParser(
        description="Example inference with BERT protein language model"
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("output/protein_bert/bert_protein_final.pth"),
        help="Path to saved model",
    )
    parser.add_argument(
        "--vocab_path",
        type=Path,
        default=Path("data/vocab.pkl"),
        help="Path to vocabulary",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="embeddings",
        choices=["embeddings", "similarity", "interactive"],
        help="Inference task",
    )

    args = parser.parse_args()

    # Initialize inference
    inference = BERTInference(args.model_path, args.vocab_path, device=args.device)

    if args.task == "embeddings":
        # Example 1: Get embeddings for protein sequences
        logger.info("Example 1: Getting embeddings for protein sequences")

        sequences = [
            "MKTIIALSYIFCLVFADYKDDDKWEE",
            "NNKEIANDKPLPSTTEKFPYDSAE",
            "MAVHQVSTPTLVEVSR",
        ]

        # Get full contextual embeddings
        embeddings = inference.get_embeddings(sequences)
        logger.info(f"Full embeddings shape: {embeddings.shape}")
        logger.info(f"  [batch_size, seq_len, hidden_size] = {embeddings.shape}")

        # Get sequence-level embeddings
        seq_embeddings = inference.get_sequence_embedding(sequences)
        logger.info(f"Sequence embeddings shape: {seq_embeddings.shape}")

        # Get [CLS] embeddings
        cls_embeddings = inference.get_cls_embedding(sequences)
        logger.info(f"[CLS] embeddings shape: {cls_embeddings.shape}")

    elif args.task == "similarity":
        # Example 2: Compute sequence similarity
        logger.info("Example 2: Computing sequence similarity")

        sequences = [
            "MKTIIALSYIFCLVFADYKDDDKWEE",
            "NNKEIANDKPLPSTTEKFPYDSAE",
            "MKTIIALSYIFCLVFADYKDDDKWEE",  # Same as first
        ]

        logger.info("Sequences:")
        for i, seq in enumerate(sequences):
            logger.info(f"  {i}: {seq}")

        logger.info("Pairwise similarities:")
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                sim = inference.similarity(sequences[i], sequences[j])
                logger.info(f"  Seq{i} <-> Seq{j}: {sim:.4f}")

    elif args.task == "interactive":
        # Example 3: Interactive inference
        logger.info("Interactive mode - Enter protein sequences (q to quit)")
        logger.info("Examples:")
        logger.info("  1. Get embedding: MKTIIALSYIFCLVFADYKDDDKWEE")
        logger.info("  2. Compare: MKTIIALSYIFCLVFADYKDDDKWEE | NNKEIANDKPLPSTTEKFPYDSAE")

        while True:
            user_input = input("\n> ").strip()

            if user_input.lower() == "q":
                logger.info("Exiting...")
                break

            if "|" in user_input:
                # Similarity task
                parts = user_input.split("|")
                if len(parts) == 2:
                    seq1 = parts[0].strip()
                    seq2 = parts[1].strip()
                    sim = inference.similarity(seq1, seq2)
                    logger.info(f"Similarity: {sim:.4f}")
            else:
                # Embedding task
                embeddings = inference.get_sequence_embedding([user_input])
                logger.info(f"Embedding shape: {embeddings.shape}")
                logger.info(f"Mean value: {embeddings.mean().item():.4f}")
                logger.info(f"Std value: {embeddings.std().item():.4f}")


if __name__ == "__main__":
    main()
