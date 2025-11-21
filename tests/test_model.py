"""Tests for model architecture."""

import pytest
import torch

from traincw.models.cnn_lstm_ctc import CWDecoder, compute_ctc_loss, create_model_from_config
from traincw.models.decoder import (
    decode_predictions,
    get_vocabulary_size,
    indices_to_text,
    text_to_indices,
)
from traincw.models.encoder import CNNEncoder
from traincw.utils.config import Config


class TestCNNEncoder:
    """Test CNN encoder."""

    def test_encoder_forward(self):
        """Test encoder forward pass."""
        encoder = CNNEncoder(
            input_channels=1,
            cnn_channels=[32, 64],
            kernel_sizes=[3, 3],
            strides=[1, 1],
            pooling=[2, 2],
        )

        # Test with 3D input (batch, freq, time)
        x = torch.randn(2, 64, 100)
        output = encoder(x)

        assert output.ndim == 3  # (batch, time, features)
        assert output.shape[0] == 2  # batch size
        assert output.shape[1] < 100  # time reduced by pooling

    def test_encoder_with_4d_input(self):
        """Test encoder with 4D input."""
        encoder = CNNEncoder()

        x = torch.randn(2, 1, 64, 100)
        output = encoder(x)

        assert output.ndim == 3

    def test_encoder_output_dim(self):
        """Test get_output_dim method."""
        encoder = CNNEncoder()
        output_dim = encoder.get_output_dim(64)

        assert isinstance(output_dim, int)
        assert output_dim > 0


class TestCWDecoder:
    """Test CW decoder model."""

    def test_model_creation(self):
        """Test model creation."""
        model = CWDecoder(n_mels=64)

        assert isinstance(model, torch.nn.Module)
        assert model.num_classes == 45

    def test_model_forward(self):
        """Test model forward pass."""
        model = CWDecoder(n_mels=64)

        # Input: (batch, n_mels, time)
        x = torch.randn(2, 64, 100)
        output = model(x)

        # Output: (batch, time', num_classes)
        assert output.ndim == 3
        assert output.shape[0] == 2  # batch size
        assert output.shape[2] == 45  # num_classes

    def test_model_decode(self):
        """Test model decode method."""
        model = CWDecoder(n_mels=64)
        model.eval()

        x = torch.randn(2, 64, 100)
        texts = model.decode(x, method="greedy")

        assert len(texts) == 2
        assert all(isinstance(text, str) for text in texts)

    def test_model_parameters(self):
        """Test getting model parameters."""
        model = CWDecoder()

        num_params = model.get_num_parameters()
        assert num_params > 0
        assert isinstance(num_params, int)

        size_mb = model.get_model_size_mb()
        assert size_mb > 0
        assert isinstance(size_mb, float)

    def test_create_model_from_config(self):
        """Test creating model from config."""
        config = Config()
        model = create_model_from_config(config)

        assert isinstance(model, CWDecoder)
        assert model.n_mels == config.audio.n_mels


class TestDecoder:
    """Test CTC decoder utilities."""

    def test_text_to_indices(self):
        """Test text to indices conversion."""
        text = "CQ"
        indices = text_to_indices(text)

        assert isinstance(indices, list)
        assert len(indices) == 2
        assert all(isinstance(idx, int) for idx in indices)

    def test_indices_to_text(self):
        """Test indices to text conversion."""
        indices = [3, 17]  # Example indices
        text = indices_to_text(indices)

        assert isinstance(text, str)

    def test_text_roundtrip(self):
        """Test text -> indices -> text roundtrip."""
        original_text = "TEST"
        indices = text_to_indices(original_text)
        reconstructed_text = indices_to_text(indices)

        assert original_text == reconstructed_text

    def test_vocabulary_size(self):
        """Test vocabulary size."""
        vocab_size = get_vocabulary_size()

        assert vocab_size == 45  # 44 chars + blank

    def test_decode_predictions(self):
        """Test decode predictions."""
        # Create random logits
        logits = torch.randn(2, 100, 45)

        # Greedy decoding
        texts = decode_predictions(logits, method="greedy")
        assert len(texts) == 2
        assert all(isinstance(text, str) for text in texts)

        # Beam search decoding
        texts_beam = decode_predictions(logits, method="beam_search", beam_width=5)
        assert len(texts_beam) == 2

    def test_prosigns(self):
        """Test prosign encoding."""
        text = "CQ <AR> <SK>"
        indices = text_to_indices(text)

        # Should have 4 indices: C, Q, AR, SK
        assert len(indices) == 4

        reconstructed = indices_to_text(indices)
        assert "<AR>" in reconstructed
        assert "<SK>" in reconstructed


class TestCTCLoss:
    """Test CTC loss computation."""

    def test_ctc_loss_computation(self):
        """Test CTC loss computation."""
        # Create dummy data
        batch_size = 2
        time_steps = 100
        num_classes = 45

        logits = torch.randn(batch_size, time_steps, num_classes)
        targets = torch.randint(1, num_classes, (batch_size, 10))
        input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long)
        target_lengths = torch.tensor([8, 6], dtype=torch.long)

        # Compute loss
        loss = compute_ctc_loss(logits, targets, input_lengths, target_lengths)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_ctc_loss_gradient(self):
        """Test that CTC loss is differentiable."""
        logits = torch.randn(2, 100, 45, requires_grad=True)
        targets = torch.randint(1, 45, (2, 10))
        input_lengths = torch.full((2,), 100, dtype=torch.long)
        target_lengths = torch.tensor([8, 6], dtype=torch.long)

        loss = compute_ctc_loss(logits, targets, input_lengths, target_lengths)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
