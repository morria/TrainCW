"""Tests for audio preprocessing utilities."""

import numpy as np
import pytest
import torch

from traincw.utils.audio import (
    compute_mel_spectrogram,
    normalize_audio,
    pad_or_trim_spectrogram,
    spectrogram_to_time_length,
    time_length_to_audio_length,
)


class TestMelSpectrogram:
    """Test mel-spectrogram computation."""

    def test_compute_from_numpy(self):
        """Test computing spectrogram from numpy array."""
        audio = np.random.randn(16000)  # noqa: NPY002
        spec = compute_mel_spectrogram(audio)

        assert isinstance(spec, torch.Tensor)
        assert spec.shape[0] == 64  # n_mels
        assert spec.shape[1] > 0  # time steps

    def test_compute_from_torch(self):
        """Test computing spectrogram from torch tensor."""
        audio = torch.randn(16000)
        spec = compute_mel_spectrogram(audio)

        assert isinstance(spec, torch.Tensor)
        assert spec.shape[0] == 64

    def test_normalization(self):
        """Test spectrogram normalization."""
        audio = np.random.randn(16000)  # noqa: NPY002

        spec_normalized = compute_mel_spectrogram(audio, normalize=True)
        spec_unnormalized = compute_mel_spectrogram(audio, normalize=False)

        # Normalized should have approximately zero mean
        assert abs(spec_normalized.mean().item()) < 0.1

        # Unnormalized should not
        assert spec_normalized.std() != spec_unnormalized.std()

    def test_different_parameters(self):
        """Test with different audio parameters."""
        audio = np.random.randn(8000)  # noqa: NPY002

        spec = compute_mel_spectrogram(
            audio,
            sample_rate=16000,
            n_mels=32,  # Fewer mel bins
        )

        assert spec.shape[0] == 32


class TestAudioNormalization:
    """Test audio normalization."""

    def test_normalize_numpy(self):
        """Test normalizing numpy array."""
        audio = np.random.randn(16000) * 0.1  # noqa: NPY002
        normalized = normalize_audio(audio, target_db=-20.0)

        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == audio.shape

        # Check RMS level is approximately target
        rms = np.sqrt(np.mean(normalized**2))
        target_rms = 10 ** (-20.0 / 20.0)
        assert abs(rms - target_rms) < 0.01

    def test_normalize_torch(self):
        """Test normalizing torch tensor."""
        audio = torch.randn(16000) * 0.1
        normalized = normalize_audio(audio, target_db=-20.0)

        assert isinstance(normalized, torch.Tensor)
        assert normalized.shape == audio.shape

    def test_silent_audio(self):
        """Test normalizing silent audio."""
        audio = np.zeros(16000)
        normalized = normalize_audio(audio, target_db=-20.0)

        # Should handle zeros gracefully
        assert not np.isnan(normalized).any()
        assert not np.isinf(normalized).any()


class TestSpectrogramPadding:
    """Test spectrogram padding and trimming."""

    def test_pad_spectrogram(self):
        """Test padding spectrogram to target length."""
        spec = torch.randn(64, 50)  # (n_mels, time)
        padded = pad_or_trim_spectrogram(spec, target_length=100)

        assert padded.shape == (64, 100)

    def test_trim_spectrogram(self):
        """Test trimming spectrogram to target length."""
        spec = torch.randn(64, 150)
        trimmed = pad_or_trim_spectrogram(spec, target_length=100)

        assert trimmed.shape == (64, 100)

    def test_exact_length(self):
        """Test when spectrogram is already target length."""
        spec = torch.randn(64, 100)
        result = pad_or_trim_spectrogram(spec, target_length=100)

        assert result.shape == spec.shape
        assert torch.equal(result, spec)

    def test_batch_padding(self):
        """Test padding with batch dimension."""
        spec = torch.randn(2, 64, 50)  # (batch, n_mels, time)
        padded = pad_or_trim_spectrogram(spec, target_length=100)

        assert padded.shape == (2, 64, 100)


class TestLengthConversions:
    """Test audio/spectrogram length conversions."""

    def test_audio_to_spectrogram_length(self):
        """Test converting audio length to spectrogram time steps."""
        audio_length = 16000  # 1 second at 16kHz
        time_steps = spectrogram_to_time_length(audio_length, hop_length=160)

        # With hop_length=160, 1 second should give ~100 time steps
        assert 90 < time_steps < 110

    def test_spectrogram_to_audio_length(self):
        """Test converting spectrogram time steps to audio length."""
        time_steps = 100
        audio_length = time_length_to_audio_length(time_steps, hop_length=160)

        # Should be close to 16000 samples (1 second)
        assert 15000 < audio_length < 17000

    def test_roundtrip_conversion(self):
        """Test roundtrip audio -> spectrogram -> audio length."""
        audio_length = 16000
        time_steps = spectrogram_to_time_length(audio_length, hop_length=160)
        reconstructed_length = time_length_to_audio_length(time_steps, hop_length=160)

        # Should be close (not exact due to windowing)
        assert abs(audio_length - reconstructed_length) < 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
