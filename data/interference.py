"""
Interference generation: QRM (other CW signals) and propagation effects.
"""

import numpy as np
from typing import List, Tuple
from morse.morse_code import MorseCode
from morse.timing import TimingCalculator, sample_wpm, select_operator_style
from .audio_synthesis import generate_cw_audio, sample_frequency, sample_envelope_type


def generate_qrm_signal(main_frequency: float, duration: float,
                       sample_rate: int = 16000) -> Tuple[np.ndarray, dict]:
    """
    Generate a QRM signal (interfering CW station).

    Args:
        main_frequency: Frequency of main signal (to avoid)
        duration: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Tuple of (audio, metadata)
    """
    morse = MorseCode()

    # Random frequency offset from main signal
    offset = np.random.choice([
        np.random.uniform(-800, -100),  # Below main signal
        np.random.uniform(100, 800)      # Above main signal
    ])
    qrm_frequency = main_frequency + offset

    # Ensure frequency is in valid range
    qrm_frequency = np.clip(qrm_frequency, 400, 900)

    # Random speed (independent of main signal)
    qrm_wpm = sample_wpm(phase=3)

    # Random timing variance
    timing_variance = select_operator_style()

    # Generate random text (callsigns and random characters)
    from .text_generator import generate_random_text, generate_callsign
    if np.random.random() < 0.5:
        text = generate_callsign()
    else:
        text = generate_random_text(length=np.random.randint(5, 15))

    # Generate timing
    timing_calc = TimingCalculator(qrm_wpm, timing_variance)
    elements = morse.text_to_elements(text)
    timing_sequence = timing_calc.get_timing_sequence(elements)

    # Generate audio
    qrm_audio = generate_cw_audio(
        timing_sequence=timing_sequence,
        frequency=qrm_frequency,
        sample_rate=sample_rate,
        envelope_type=sample_envelope_type()
    )

    # Pad or trim to match duration
    target_samples = int(duration * sample_rate)
    if len(qrm_audio) < target_samples:
        # Pad with silence
        qrm_audio = np.pad(qrm_audio, (0, target_samples - len(qrm_audio)))
    else:
        # Trim
        qrm_audio = qrm_audio[:target_samples]

    metadata = {
        'frequency': qrm_frequency,
        'wpm': qrm_wpm,
        'text': text,
    }

    return qrm_audio, metadata


def add_qrm(audio: np.ndarray, main_frequency: float,
           sample_rate: int = 16000) -> np.ndarray:
    """
    Add QRM (interfering CW signals) to audio.

    Args:
        audio: Input audio signal
        main_frequency: Frequency of main signal
        sample_rate: Sample rate in Hz

    Returns:
        Audio with added QRM
    """
    if np.random.random() >= 0.25:  # 25% of samples get QRM
        return audio

    duration = len(audio) / sample_rate

    # Number of interfering signals (85% one, 15% two)
    n_signals = 1 if np.random.random() < 0.85 else 2

    for _ in range(n_signals):
        qrm_audio, _ = generate_qrm_signal(main_frequency, duration, sample_rate)

        # Random strength relative to main signal
        # 50%: -15 to -5 dB (weaker)
        # 30%: -5 to 0 dB (comparable)
        # 20%: 0 to +5 dB (stronger)
        choice = np.random.random()
        if choice < 0.50:
            strength_db = np.random.uniform(-15, -5)
        elif choice < 0.80:
            strength_db = np.random.uniform(-5, 0)
        else:
            strength_db = np.random.uniform(0, 5)

        # Apply strength scaling
        strength_linear = 10 ** (strength_db / 20)
        audio = audio + qrm_audio * strength_linear

    return audio


def generate_fading_envelope(duration: float, sample_rate: int,
                            fading_type: str = 'slow') -> np.ndarray:
    """
    Generate fading envelope (QSB - ionospheric propagation).

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        fading_type: 'slow' or 'fast'

    Returns:
        Fading envelope (multiplicative, 0.0 to 1.0+)
    """
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate

    if fading_type == 'slow':
        # Slow fading: 0.1-1 Hz modulation, 6-20 dB depth
        fading_rate = np.random.uniform(0.1, 1.0)  # Hz
        fade_depth_db = np.random.uniform(6, 20)
    else:
        # Fast fading: 1-5 Hz modulation, 3-10 dB depth
        fading_rate = np.random.uniform(1.0, 5.0)
        fade_depth_db = np.random.uniform(3, 10)

    # Sinusoidal fading with random phase
    phase = np.random.uniform(0, 2 * np.pi)
    fade_db = fade_depth_db * np.sin(2 * np.pi * fading_rate * t + phase)

    # Convert dB to linear
    fade_linear = 10 ** (fade_db / 20)

    # Ensure envelope is positive and centered around 1.0
    fade_linear = fade_linear * 0.5 + 0.5

    return fade_linear


def apply_fading(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Apply fading to audio.

    Args:
        audio: Input audio signal
        sample_rate: Sample rate in Hz

    Returns:
        Audio with applied fading
    """
    if np.random.random() >= 0.30:  # 30% of samples get fading
        return audio

    duration = len(audio) / sample_rate

    # Select fading type (80% slow, 20% fast)
    fading_type = 'slow' if np.random.random() < 0.80 else 'fast'

    # Generate fading envelope
    envelope = generate_fading_envelope(duration, sample_rate, fading_type)

    return audio * envelope


def apply_agc_pumping(audio: np.ndarray, sample_rate: int = 16000,
                     attack_time: float = 0.030, release_time: float = 0.300) -> np.ndarray:
    """
    Simulate AGC (Automatic Gain Control) pumping.

    Args:
        audio: Input audio signal
        sample_rate: Sample rate in Hz
        attack_time: Attack time in seconds (10-50ms)
        release_time: Release time in seconds (100-500ms)

    Returns:
        Audio with AGC pumping effect
    """
    if np.random.random() >= 0.15:  # 15% of samples get AGC pumping
        return audio

    # Random AGC parameters
    attack_time = np.random.uniform(0.010, 0.050)
    release_time = np.random.uniform(0.100, 0.500)

    # Simple AGC simulation using envelope following
    envelope = np.abs(audio)

    # Smooth envelope with attack/release
    attack_coeff = np.exp(-1.0 / (attack_time * sample_rate))
    release_coeff = np.exp(-1.0 / (release_time * sample_rate))

    smoothed_envelope = np.zeros_like(envelope)
    smoothed_envelope[0] = envelope[0]

    for i in range(1, len(envelope)):
        if envelope[i] > smoothed_envelope[i-1]:
            # Attack
            smoothed_envelope[i] = attack_coeff * smoothed_envelope[i-1] + \
                                 (1 - attack_coeff) * envelope[i]
        else:
            # Release
            smoothed_envelope[i] = release_coeff * smoothed_envelope[i-1] + \
                                 (1 - release_coeff) * envelope[i]

    # Apply gain reduction (inverse of envelope)
    target_level = 0.5
    gain = np.where(smoothed_envelope > 0.01,
                   target_level / (smoothed_envelope + 0.01),
                   1.0)

    # Limit gain to reasonable range
    gain = np.clip(gain, 0.3, 3.0)

    return audio * gain


def apply_clipping(audio: np.ndarray, clip_percentage: float = 0.10) -> np.ndarray:
    """
    Apply clipping to simulate overdriven audio.

    Args:
        audio: Input audio signal
        clip_percentage: Percentage of samples to clip (0-0.20)

    Returns:
        Clipped audio
    """
    if np.random.random() >= 0.10:  # 10% of samples get clipping
        return audio

    clip_percentage = np.random.uniform(0.0, 0.20)

    # Find clip level that affects desired percentage of samples
    abs_audio = np.abs(audio)
    clip_level = np.percentile(abs_audio, (1 - clip_percentage) * 100)

    # Apply hard clipping
    clipped = np.clip(audio, -clip_level, clip_level)

    return clipped


def apply_bandpass_filter(audio: np.ndarray, center_freq: float,
                         q_factor: float, sample_rate: int = 16000) -> np.ndarray:
    """
    Apply narrow bandpass filter (simulates CW filter with ringing).

    Args:
        audio: Input audio signal
        center_freq: Center frequency in Hz
        q_factor: Q factor (10-50 for narrow CW filters)
        sample_rate: Sample rate in Hz

    Returns:
        Filtered audio
    """
    if np.random.random() >= 0.20:  # 20% of samples get filter ringing
        return audio

    import scipy.signal

    q_factor = np.random.uniform(10, 50)

    # Calculate bandwidth from Q factor
    bandwidth = center_freq / q_factor

    # Design bandpass filter
    low_freq = max(1, center_freq - bandwidth / 2)
    high_freq = min(sample_rate / 2 - 1, center_freq + bandwidth / 2)

    # Higher order filter for more ringing
    sos = scipy.signal.butter(6, [low_freq, high_freq], btype='band',
                            fs=sample_rate, output='sos')

    filtered = scipy.signal.sosfilt(sos, audio)

    return filtered
