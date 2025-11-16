import torch
import torch.nn as nn
from typing import Tuple

from .bert import BERT


class BERTLM(nn.Module):
    """
    BERT Language Model with MLM and NSP tasks.

    Combines BERT encoder with task-specific heads for:
    - Masked Language Model (MLM): Token prediction
    - Next Sentence Prediction (NSP): Binary classification
    """

    def __init__(self, bert: BERT, vocab_size: int) -> None:
        """
        Initialize BERT Language Model.

        :param bert: BERT encoder model
        :param vocab_size: total vocabulary size
        :raises ValueError: if vocab_size is not positive
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x: torch.Tensor, segment_label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of BERT Language Model.

        :param x: input token indices, shape [batch_size, seq_len]
        :param segment_label: segment labels, shape [batch_size, seq_len]
        :return: tuple of (nsp_logits, mlm_logits)
                 - nsp_logits: [batch_size, 2]
                 - mlm_logits: [batch_size, seq_len, vocab_size]
        """
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)


class NextSentencePrediction(nn.Module):
    """
    Next Sentence Prediction (NSP) task head.

    Binary classification to predict if two sentences are consecutive in the corpus.
    """

    def __init__(self, hidden: int) -> None:
        """
        Initialize NSP head.

        :param hidden: hidden size from BERT model
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of NSP head.

        :param x: BERT output, shape [batch_size, seq_len, hidden]
        :return: logits for binary classification, shape [batch_size, 2]
        """
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """
    Masked Language Model (MLM) task head.

    Predicts masked tokens from the input sequence.
    Multi-class classification problem where n_classes = vocab_size.
    """

    def __init__(self, hidden: int, vocab_size: int) -> None:
        """
        Initialize MLM head.

        :param hidden: hidden size from BERT model
        :param vocab_size: total vocabulary size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MLM head.

        :param x: BERT output, shape [batch_size, seq_len, hidden]
        :return: logits for token prediction, shape [batch_size, seq_len, vocab_size]
        """
        return self.softmax(self.linear(x))
