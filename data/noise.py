"""
Noise generation for CW training data.
"""

import numpy as np
import scipy.signal
from typing import Optional


def add_white_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add white Gaussian noise to achieve target SNR.

    Args:
        audio: Input audio signal
        snr_db: Target signal-to-noise ratio in dB

    Returns:
        Audio with added noise
    """
    # Calculate signal power
    signal_power = np.mean(audio ** 2)

    # Calculate noise power needed for target SNR
    # SNR_dB = 10 * log10(P_signal / P_noise)
    # P_noise = P_signal / 10^(SNR_dB/10)
    noise_power = signal_power / (10 ** (snr_db / 10))

    # Generate white noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))

    return audio + noise


def generate_pink_noise(n_samples: int) -> np.ndarray:
    """
    Generate pink noise (1/f spectrum).

    Uses the Voss-McCartney algorithm.

    Args:
        n_samples: Number of samples to generate

    Returns:
        Pink noise array
    """
    # Simple pink noise using multiple white noise sources
    # at different update rates
    n_rows = 16
    array = np.zeros((n_rows, n_samples))

    # Initialize
    for i in range(n_rows):
        array[i, :] = np.random.randn(n_samples)

    # Accumulate with different update rates
    pink = np.zeros(n_samples)
    for i in range(n_samples):
        # Update rows based on bit pattern
        for j in range(n_rows):
            if i & (1 << j):
                array[j, i] = np.random.randn()
        pink[i] = array[:, i].sum()

    # Normalize
    pink = pink / np.std(pink)

    return pink


def add_pink_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add pink noise (1/f spectrum) to achieve target SNR.

    Args:
        audio: Input audio signal
        snr_db: Target signal-to-noise ratio in dB

    Returns:
        Audio with added pink noise
    """
    # Calculate signal power
    signal_power = np.mean(audio ** 2)

    # Calculate noise power needed for target SNR
    noise_power = signal_power / (10 ** (snr_db / 10))

    # Generate pink noise
    noise = generate_pink_noise(len(audio))

    # Scale to desired power
    noise = noise * np.sqrt(noise_power / np.mean(noise ** 2))

    return audio + noise


def add_band_limited_noise(audio: np.ndarray, snr_db: float,
                          center_freq: float, bandwidth: float,
                          sample_rate: int = 16000) -> np.ndarray:
    """
    Add band-limited noise around signal frequency.

    Args:
        audio: Input audio signal
        snr_db: Target signal-to-noise ratio in dB
        center_freq: Center frequency in Hz
        bandwidth: Bandwidth in Hz (±bandwidth around center)
        sample_rate: Sample rate in Hz

    Returns:
        Audio with added band-limited noise
    """
    # Generate white noise
    noise = np.random.randn(len(audio))

    # Design bandpass filter
    low_freq = max(1, center_freq - bandwidth)
    high_freq = min(sample_rate / 2 - 1, center_freq + bandwidth)

    # Butterworth bandpass filter
    sos = scipy.signal.butter(4, [low_freq, high_freq], btype='band',
                            fs=sample_rate, output='sos')
    filtered_noise = scipy.signal.sosfilt(sos, noise)

    # Calculate signal power
    signal_power = np.mean(audio ** 2)

    # Scale noise to achieve target SNR
    noise_power = signal_power / (10 ** (snr_db / 10))
    filtered_noise = filtered_noise * np.sqrt(noise_power / np.mean(filtered_noise ** 2))

    return audio + filtered_noise


def sample_snr(phase: int = 3) -> float:
    """
    Sample SNR from realistic distribution.

    Args:
        phase: Curriculum phase (1=foundation, 2=expansion, 3=mastery)

    Returns:
        SNR in dB
    """
    if phase == 1:
        # Phase 1: Clean signals (15-25 dB)
        return np.random.uniform(15, 25)
    elif phase == 2:
        # Phase 2: Moderate noise (10-25 dB)
        return np.random.uniform(10, 25)
    else:
        # Phase 3: Full range (-5 to 30 dB)
        # Distribution from training plan
        choice = np.random.random()
        if choice < 0.10:
            return np.random.uniform(25, 30)  # Excellent
        elif choice < 0.25:
            return np.random.uniform(20, 25)  # Very Good
        elif choice < 0.45:
            return np.random.uniform(15, 20)  # Good
        elif choice < 0.70:
            return np.random.uniform(10, 15)  # Fair
        elif choice < 0.85:
            return np.random.uniform(5, 10)   # Poor
        elif choice < 0.95:
            return np.random.uniform(0, 5)    # Very Poor
        else:
            return np.random.uniform(-5, 0)   # Barely Readable


def add_noise(audio: np.ndarray, snr_db: float, center_freq: float,
             sample_rate: int = 16000) -> np.ndarray:
    """
    Add noise to audio based on random selection of noise types.

    Args:
        audio: Input audio signal
        snr_db: Target SNR in dB
        center_freq: Signal center frequency
        sample_rate: Sample rate in Hz

    Returns:
        Audio with added noise
    """
    # Always add white noise as base
    audio_noisy = add_white_noise(audio, snr_db)

    # 40% chance of adding pink noise on top
    if np.random.random() < 0.40:
        # Add pink noise at slightly higher SNR (so it doesn't dominate)
        audio_noisy = add_pink_noise(audio_noisy, snr_db + 3)

    # 30% chance of band-limited noise
    if np.random.random() < 0.30:
        bandwidth = np.random.uniform(200, 500)
        audio_noisy = add_band_limited_noise(audio_noisy, snr_db + 3,
                                            center_freq, bandwidth, sample_rate)

    return audio_noisy


def generate_impulse_noise(duration: float, sample_rate: int,
                          impulse_rate: float, impulse_duration: float = 0.03,
                          impulse_amplitude: float = 10.0) -> np.ndarray:
    """
    Generate QRN-style impulse noise (static crashes).

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        impulse_rate: Impulses per second (0.5-3 typical)
        impulse_duration: Duration of each impulse in seconds (10-50ms)
        impulse_amplitude: Amplitude relative to 1.0 (10-30 typical)

    Returns:
        Impulse noise array
    """
    n_samples = int(duration * sample_rate)
    noise = np.zeros(n_samples)

    # Number of impulses
    n_impulses = int(impulse_rate * duration)

    for _ in range(n_impulses):
        # Random position
        pos = np.random.randint(0, n_samples)

        # Random duration
        impulse_samples = int(np.random.uniform(0.010, 0.050) * sample_rate)
        impulse_samples = min(impulse_samples, n_samples - pos)

        # Generate impulse with exponential decay
        t = np.arange(impulse_samples) / sample_rate
        envelope = np.exp(-t / 0.010)  # 10ms decay

        # Random amplitude
        amplitude = np.random.uniform(10, 30)

        # Add impulse
        impulse = np.random.randn(impulse_samples) * envelope * amplitude
        noise[pos:pos+impulse_samples] += impulse

    return noise


def add_impulse_noise(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Add QRN-style impulse noise to audio.

    Args:
        audio: Input audio signal
        sample_rate: Sample rate in Hz

    Returns:
        Audio with added impulse noise
    """
    if np.random.random() >= 0.20:  # 20% of samples get impulse noise
        return audio

    duration = len(audio) / sample_rate
    impulse_rate = np.random.uniform(0.5, 3.0)  # Impulses per second

    impulses = generate_impulse_noise(duration, sample_rate, impulse_rate)

    return audio + impulses
