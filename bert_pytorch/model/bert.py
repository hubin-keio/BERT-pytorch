import torch
import torch.nn as nn

from .transformer import TransformerBlock
from .embedding import BERTEmbedding


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size: int, hidden: int = 768, n_layers: int = 12,
                 attn_heads: int = 12, dropout: float = 0.1) -> None:
        """
        Initialize BERT model.

        :param vocab_size: total vocabulary size
        :param hidden: hidden size of transformer model
        :param n_layers: number of transformer blocks (layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        :raises ValueError: if hidden is not divisible by attn_heads
        """

        super().__init__()

        # Validate parameters
        if hidden % attn_heads != 0:
            raise ValueError(f"hidden size ({hidden}) must be divisible by attn_heads ({attn_heads})")
        if dropout < 0 or dropout > 1:
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        if attn_heads <= 0:
            raise ValueError(f"attn_heads must be positive, got {attn_heads}")

        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, segment_info: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of BERT model.

        :param x: input token indices, shape [batch_size, seq_len]
        :param segment_info: segment labels, shape [batch_size, seq_len]
        :return: contextual representations, shape [batch_size, seq_len, hidden]
        """
        mask = (x > 0).unsqueeze(1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
