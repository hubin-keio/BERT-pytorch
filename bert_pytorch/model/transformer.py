import torch
import torch.nn as nn

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward


class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block.

    Single layer of bidirectional transformer consisting of:
    - Multi-headed self-attention
    - Position-wise feed-forward network
    - Residual connections and layer normalization
    """

    def __init__(self, hidden: int, attn_heads: int, feed_forward_hidden: int,
                 dropout: float) -> None:
        """
        Initialize Transformer block.

        :param hidden: hidden size of transformer
        :param attn_heads: number of attention heads
        :param feed_forward_hidden: feed-forward hidden size (usually 4*hidden_size)
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Transformer block.

        :param x: input embeddings, shape [batch_size, seq_len, hidden]
        :param mask: attention mask, shape [batch_size, 1, 1, seq_len]
        :return: output embeddings, shape [batch_size, seq_len, hidden]
        """
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
