"""Audio preprocessing utilities for CW decoding.

This module provides functions for:
- Computing mel-spectrograms from audio waveforms
- Audio normalization
- Feature extraction for the neural network
"""

import numpy as np
import torch
import torchaudio.transforms as transforms


def compute_mel_spectrogram(
    waveform: np.ndarray | torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_mels: int = 64,
    f_min: float = 0.0,
    f_max: float | None = 8000.0,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute mel-spectrogram from audio waveform.

    This function converts audio to a mel-spectrogram representation suitable
    for the CNN-LSTM-CTC model. The spectrogram is computed using STFT followed
    by mel filterbank transformation.

    Args:
        waveform: Input audio waveform (1D array or tensor)
        sample_rate: Audio sample rate in Hz (default: 16000)
        n_fft: FFT size (default: 512)
        hop_length: Hop size in samples, 10ms at 16kHz (default: 160)
        win_length: Window size in samples, 25ms at 16kHz (default: 400)
        n_mels: Number of mel filterbanks (default: 64)
        f_min: Minimum frequency (default: 0.0)
        f_max: Maximum frequency (default: 8000.0, Nyquist at 16kHz)
        normalize: Whether to normalize the spectrogram (default: True)

    Returns:
        Mel-spectrogram as torch.Tensor of shape (n_mels, time_steps)

    Example:
        >>> audio = np.random.randn(16000)  # 1 second of audio at 16kHz
        >>> spectrogram = compute_mel_spectrogram(audio)
        >>> spectrogram.shape
        torch.Size([64, 101])  # 64 mel bins, ~101 time steps
    """
    # Convert to torch tensor if numpy array
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()
    else:
        waveform = waveform.float()

    # Ensure 1D waveform
    if waveform.ndim > 1:
        waveform = waveform.squeeze()

    # Create mel spectrogram transform
    mel_transform = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=2.0,  # Power spectrogram
        normalized=False,
    )

    # Compute mel spectrogram
    mel_spec = mel_transform(waveform)

    # Convert to log scale (dB)
    mel_spec_db = transforms.AmplitudeToDB(stype="power", top_db=80)(mel_spec)

    # Normalize if requested
    if normalize:
        # Per-sample normalization: mean 0, std 1
        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        mel_spec_db = (mel_spec_db - mean) / std if std > 1e-6 else mel_spec_db - mean

    return mel_spec_db


def normalize_audio(
    audio: np.ndarray | torch.Tensor,
    target_db: float = -20.0,
    eps: float = 1e-8,
) -> np.ndarray | torch.Tensor:
    """
    Normalize audio to target RMS level in dB.

    Args:
        audio: Input audio waveform
        target_db: Target RMS level in dB (default: -20.0)
        eps: Small constant to avoid log(0) (default: 1e-8)

    Returns:
        Normalized audio waveform (same type as input)

    Example:
        >>> audio = np.random.randn(16000) * 0.1  # Quiet audio
        >>> normalized = normalize_audio(audio, target_db=-20.0)
        >>> # normalized will have RMS level of -20dB
    """
    is_numpy = isinstance(audio, np.ndarray)

    # Convert to numpy for processing
    audio_np = (audio.cpu().numpy() if audio.is_cuda else audio.numpy()) if not is_numpy else audio

    # Calculate current RMS
    rms = np.sqrt(np.mean(audio_np**2) + eps)

    # Convert target dB to linear scale
    target_rms = 10 ** (target_db / 20.0)

    # Scale audio
    if rms > eps:
        scale_factor = target_rms / rms
        audio_normalized = audio_np * scale_factor
    else:
        audio_normalized = audio_np

    # Convert back to original type
    if not is_numpy:
        return torch.from_numpy(audio_normalized).to(audio.device).type(audio.dtype)

    return audio_normalized


def pad_or_trim_spectrogram(
    spectrogram: torch.Tensor,
    target_length: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Pad or trim spectrogram to target length in time dimension.

    Args:
        spectrogram: Input spectrogram of shape (n_mels, time) or (batch, n_mels, time)
        target_length: Target length in time steps
        pad_value: Value to use for padding (default: 0.0)

    Returns:
        Spectrogram of shape (..., n_mels, target_length)
    """
    current_length = spectrogram.shape[-1]

    if current_length == target_length:
        return spectrogram
    elif current_length < target_length:
        # Pad
        pad_amount = target_length - current_length
        if spectrogram.ndim == 2:
            # (n_mels, time)
            padding = torch.full((spectrogram.shape[0], pad_amount), pad_value)
            return torch.cat([spectrogram, padding], dim=1)
        else:
            # (batch, n_mels, time)
            padding = torch.full(
                (spectrogram.shape[0], spectrogram.shape[1], pad_amount), pad_value
            )
            return torch.cat([spectrogram, padding], dim=2)
    else:
        # Trim
        if spectrogram.ndim == 2:
            return spectrogram[:, :target_length]
        else:
            return spectrogram[:, :, :target_length]


def spectrogram_to_time_length(
    audio_length: int,
    sample_rate: int = 16000,
    hop_length: int = 160,
) -> int:
    """
    Calculate the number of time steps in the spectrogram given audio length.

    Args:
        audio_length: Length of audio in samples
        sample_rate: Sample rate in Hz
        hop_length: Hop length in samples

    Returns:
        Number of time steps in spectrogram
    """
    return (audio_length - 1) // hop_length + 1


def time_length_to_audio_length(
    time_length: int,
    hop_length: int = 160,
) -> int:
    """
    Calculate the minimum audio length needed for a given spectrogram time length.

    Args:
        time_length: Number of time steps in spectrogram
        hop_length: Hop length in samples

    Returns:
        Minimum audio length in samples
    """
    return (time_length - 1) * hop_length + 1
