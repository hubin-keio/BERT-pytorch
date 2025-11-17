"""Unit tests for BERT model components."""

import pytest
import torch
from bert_pytorch.model import BERT, BERTLM, NextSentencePrediction, MaskedLanguageModel


class TestBERT:
    """Test BERT model."""

    def test_bert_initialization(self):
        """Test BERT initialization."""
        bert = BERT(vocab_size=1000, hidden=256, n_layers=8, attn_heads=8)
        assert bert.hidden == 256
        assert bert.n_layers == 8
        assert bert.attn_heads == 8

    def test_bert_forward_pass(self):
        """Test BERT forward pass."""
        bert = BERT(vocab_size=1000, hidden=256, n_layers=4, attn_heads=8)
        bert.eval()

        batch_size = 2
        seq_len = 50
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        segment_labels = torch.ones_like(input_ids)

        output = bert(input_ids, segment_labels)

        assert output.shape == (batch_size, seq_len, 256)

    def test_bert_hidden_divisible_by_attn_heads(self):
        """Test that hidden size must be divisible by attention heads."""
        with pytest.raises(ValueError):
            BERT(vocab_size=1000, hidden=250, n_layers=8, attn_heads=8)

    def test_bert_dropout_range(self):
        """Test that dropout must be between 0 and 1."""
        with pytest.raises(ValueError):
            BERT(vocab_size=1000, hidden=256, dropout=1.5)

        with pytest.raises(ValueError):
            BERT(vocab_size=1000, hidden=256, dropout=-0.1)

    def test_bert_positive_parameters(self):
        """Test that model parameters must be positive."""
        with pytest.raises(ValueError):
            BERT(vocab_size=-1, hidden=256)

        with pytest.raises(ValueError):
            BERT(vocab_size=1000, hidden=256, n_layers=0)

        with pytest.raises(ValueError):
            BERT(vocab_size=1000, hidden=256, attn_heads=0)

    def test_bert_attention_mask(self):
        """Test attention mask computation."""
        bert = BERT(vocab_size=1000, hidden=128, n_layers=2, attn_heads=4)
        bert.eval()

        batch_size = 2
        seq_len = 30

        # Create input with padding
        input_ids = torch.tensor([
            [1, 2, 3, 4, 5, 0, 0, 0],  # Padded
            [1, 2, 3, 0, 0, 0, 0, 0],  # Padded
        ])
        segment_labels = torch.ones_like(input_ids)

        with torch.no_grad():
            output = bert(input_ids, segment_labels)
            assert output.shape[0] == 2
            assert output.shape[2] == 128


class TestBERTLM:
    """Test BERT Language Model."""

    def test_bertlm_initialization(self):
        """Test BERTLM initialization."""
        bert = BERT(vocab_size=1000, hidden=256, n_layers=4)
        lm = BERTLM(bert, vocab_size=1000)

        assert isinstance(lm.next_sentence, NextSentencePrediction)
        assert isinstance(lm.mask_lm, MaskedLanguageModel)

    def test_bertlm_forward_pass(self):
        """Test BERTLM forward pass."""
        bert = BERT(vocab_size=1000, hidden=256, n_layers=4, attn_heads=8)
        lm = BERTLM(bert, vocab_size=1000)
        lm.eval()

        batch_size = 2
        seq_len = 50
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        segment_labels = torch.ones_like(input_ids)

        nsp_logits, mlm_logits = lm(input_ids, segment_labels)

        assert nsp_logits.shape == (batch_size, 2)
        assert mlm_logits.shape == (batch_size, seq_len, 1000)

    def test_bertlm_invalid_vocab_size(self):
        """Test BERTLM with invalid vocab size."""
        bert = BERT(vocab_size=1000, hidden=256, n_layers=4)

        with pytest.raises(ValueError):
            BERTLM(bert, vocab_size=0)

        with pytest.raises(ValueError):
            BERTLM(bert, vocab_size=-1)


class TestNextSentencePrediction:
    """Test NSP head."""

    def test_nsp_initialization(self):
        """Test NSP head initialization."""
        nsp = NextSentencePrediction(hidden=256)
        assert nsp.linear.in_features == 256
        assert nsp.linear.out_features == 2

    def test_nsp_forward_pass(self):
        """Test NSP forward pass."""
        nsp = NextSentencePrediction(hidden=256)
        nsp.eval()

        batch_size = 4
        seq_len = 50
        bert_output = torch.randn(batch_size, seq_len, 256)

        with torch.no_grad():
            nsp_logits = nsp(bert_output)

        assert nsp_logits.shape == (batch_size, 2)
        # Check that outputs are log probabilities
        assert torch.allclose(nsp_logits.exp().sum(dim=1), torch.ones(batch_size), atol=1e-5)


class TestMaskedLanguageModel:
    """Test MLM head."""

    def test_mlm_initialization(self):
        """Test MLM head initialization."""
        mlm = MaskedLanguageModel(hidden=256, vocab_size=1000)
        assert mlm.linear.in_features == 256
        assert mlm.linear.out_features == 1000

    def test_mlm_forward_pass(self):
        """Test MLM forward pass."""
        mlm = MaskedLanguageModel(hidden=256, vocab_size=1000)
        mlm.eval()

        batch_size = 4
        seq_len = 50
        bert_output = torch.randn(batch_size, seq_len, 256)

        with torch.no_grad():
            mlm_logits = mlm(bert_output)

        assert mlm_logits.shape == (batch_size, seq_len, 1000)
        # Check that outputs are log probabilities
        assert torch.allclose(mlm_logits.exp().sum(dim=2), torch.ones(batch_size, seq_len), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
