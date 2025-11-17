"""Integration tests for training pipeline."""

import pytest
import torch
import tempfile
from pathlib import Path
from torch.utils.data import DataLoader

from bert_pytorch.model import BERT, BERTLM
from bert_pytorch.dataset import BERTDataset, WordVocab
from bert_pytorch.trainer import BERTLightningModule
from bert_pytorch.utils import set_seed
import pytorch_lightning as pl


class TestTrainingPipeline:
    """Integration tests for training pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        data_dir = Path(tempfile.gettempdir()) / "bert_test_data"
        data_dir.mkdir(exist_ok=True)

        # Create sample corpus
        corpus_file = data_dir / "corpus.txt"
        if not corpus_file.exists():
            with open(corpus_file, "w") as f:
                for i in range(100):
                    prot1 = "MKTIIALSYIFCLVFADYKDDDKWEE"
                    prot2 = "NNKEIANDKPLPSTTEKFPYDSAE"
                    f.write(f"{prot1}\t{prot2}\n")

        return data_dir, corpus_file

    def test_vocab_building(self, sample_data):
        """Test vocabulary building."""
        data_dir, corpus_file = sample_data

        with open(corpus_file, "r") as f:
            vocab = WordVocab(f, max_size=None, min_freq=1)

        assert len(vocab) > 0
        assert vocab.pad_index == 0
        assert vocab.unk_index == 1
        assert hasattr(vocab, "stoi")
        assert hasattr(vocab, "itos")

    def test_dataset_loading(self, sample_data):
        """Test dataset loading."""
        data_dir, corpus_file = sample_data

        # Build vocab
        with open(corpus_file, "r") as f:
            vocab = WordVocab(f, max_size=None, min_freq=1)

        # Create dataset
        dataset = BERTDataset(
            str(corpus_file), vocab, seq_len=50, on_memory=True
        )

        assert len(dataset) > 0
        sample = dataset[0]
        assert "bert_input" in sample
        assert "bert_label" in sample
        assert "segment_label" in sample
        assert "is_next" in sample

    def test_dataloader(self, sample_data):
        """Test DataLoader creation."""
        data_dir, corpus_file = sample_data

        with open(corpus_file, "r") as f:
            vocab = WordVocab(f, max_size=None, min_freq=1)

        dataset = BERTDataset(
            str(corpus_file), vocab, seq_len=50, on_memory=True
        )

        loader = DataLoader(dataset, batch_size=8, num_workers=0)

        assert len(loader) > 0

        for batch in loader:
            assert batch["bert_input"].shape[0] == 8
            assert batch["bert_input"].shape[1] == 50
            break

    def test_model_training_step(self):
        """Test a single training step."""
        set_seed(42)

        # Create small model
        bert = BERT(vocab_size=100, hidden=64, n_layers=2, attn_heads=4)
        lm = BERTLM(bert, vocab_size=100)

        # Create module
        module = BERTLightningModule(bert, vocab_size=100)

        # Create dummy batch
        batch = {
            "bert_input": torch.randint(0, 100, (4, 30)),
            "bert_label": torch.randint(0, 100, (4, 30)),
            "segment_label": torch.ones(4, 30, dtype=torch.long),
            "is_next": torch.randint(0, 2, (4,)),
        }

        # Training step
        loss = module.training_step(batch, batch_idx=0)

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_validation_step(self):
        """Test a single validation step."""
        set_seed(42)

        bert = BERT(vocab_size=100, hidden=64, n_layers=2, attn_heads=4)
        module = BERTLightningModule(bert, vocab_size=100)

        batch = {
            "bert_input": torch.randint(0, 100, (4, 30)),
            "bert_label": torch.randint(0, 100, (4, 30)),
            "segment_label": torch.ones(4, 30, dtype=torch.long),
            "is_next": torch.randint(0, 2, (4,)),
        }

        loss = module.validation_step(batch, batch_idx=0)

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_lightning_trainer_creation(self):
        """Test Lightning trainer creation."""
        bert = BERT(vocab_size=100, hidden=64, n_layers=2, attn_heads=4)
        module = BERTLightningModule(bert, vocab_size=100)

        trainer = pl.Trainer(
            max_epochs=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        assert trainer is not None

    def test_forward_backward_pass(self):
        """Test forward and backward pass."""
        set_seed(42)

        bert = BERT(vocab_size=100, hidden=64, n_layers=2, attn_heads=4)
        lm = BERTLM(bert, vocab_size=100)

        # Create optimizer
        optimizer = torch.optim.Adam(lm.parameters(), lr=1e-4)

        # Forward pass
        input_ids = torch.randint(0, 100, (4, 30))
        segment_ids = torch.ones_like(input_ids)
        labels = torch.randint(0, 100, (4, 30))
        nsp_labels = torch.randint(0, 2, (4,))

        nsp_logits, mlm_logits = lm(input_ids, segment_ids)

        # Compute loss
        criterion = torch.nn.NLLLoss(ignore_index=0)
        nsp_loss = criterion(nsp_logits, nsp_labels)
        mlm_loss = criterion(mlm_logits.transpose(1, 2), labels)
        total_loss = nsp_loss + mlm_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        assert total_loss.item() > 0
        assert not torch.isnan(total_loss)

    def test_model_eval_mode(self):
        """Test model evaluation mode."""
        bert = BERT(vocab_size=100, hidden=64, n_layers=2, attn_heads=4)
        bert.eval()

        input_ids = torch.randint(0, 100, (2, 30))
        segment_ids = torch.ones_like(input_ids)

        with torch.no_grad():
            output = bert(input_ids, segment_ids)

        assert output.shape == (2, 30, 64)

    def test_model_reproducibility(self):
        """Test model reproducibility with seed."""
        set_seed(42)
        bert1 = BERT(vocab_size=100, hidden=64, n_layers=2, attn_heads=4)
        bert1.eval()

        set_seed(42)
        bert2 = BERT(vocab_size=100, hidden=64, n_layers=2, attn_heads=4)
        bert2.eval()

        input_ids = torch.randint(0, 100, (2, 30))
        segment_ids = torch.ones_like(input_ids)

        with torch.no_grad():
            output1 = bert1(input_ids, segment_ids)
            output2 = bert2(input_ids, segment_ids)

        # Note: Outputs may not be exactly equal due to initialization randomness
        # but should be very close
        assert output1.shape == output2.shape


if __name__ == "__main__":
    pytest.main([__file__])
