"""
Audio tone synthesis for CW signals.
"""

import numpy as np


def generate_envelope(
    n_samples: int,
    rise_time: float,
    fall_time: float,
    sample_rate: int,
    envelope_type: str = "linear",
) -> np.ndarray:
    """
    Generate amplitude envelope for keying.

    Args:
        n_samples: Total number of samples
        rise_time: Rise time in seconds
        fall_time: Fall time in seconds
        sample_rate: Sample rate in Hz
        envelope_type: 'linear', 'cosine', or 'adsr'

    Returns:
        Envelope array (values 0.0 to 1.0)
    """
    envelope = np.ones(n_samples)

    rise_samples = int(rise_time * sample_rate)
    fall_samples = int(fall_time * sample_rate)

    if rise_samples + fall_samples >= n_samples:
        # Very short signal, just use triangular envelope
        rise_samples = n_samples // 2
        fall_samples = n_samples - rise_samples

    if envelope_type == "linear":
        # Linear rise and fall
        if rise_samples > 0:
            envelope[:rise_samples] = np.linspace(0, 1, rise_samples)
        if fall_samples > 0:
            envelope[-fall_samples:] = np.linspace(1, 0, fall_samples)

    elif envelope_type == "cosine":
        # Raised cosine (smoother)
        if rise_samples > 0:
            envelope[:rise_samples] = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, rise_samples)))
        if fall_samples > 0:
            envelope[-fall_samples:] = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, fall_samples)))

    elif envelope_type == "adsr":
        # ADSR: Attack-Decay-Sustain-Release
        attack_samples = rise_samples
        decay_samples = min(rise_samples, n_samples // 10)
        release_samples = fall_samples
        sustain_level = 0.8

        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

        if decay_samples > 0 and attack_samples + decay_samples < n_samples:
            decay_start = attack_samples
            decay_end = attack_samples + decay_samples
            envelope[decay_start:decay_end] = np.linspace(1, sustain_level, decay_samples)

        # Sustain region
        sustain_start = attack_samples + decay_samples
        sustain_end = n_samples - release_samples
        if sustain_end > sustain_start:
            envelope[sustain_start:sustain_end] = sustain_level

        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(sustain_level, 0, release_samples)

    return envelope


def generate_tone(
    frequency: float,
    duration: float,
    sample_rate: int,
    rise_time: float = 0.003,
    fall_time: float = 0.003,
    envelope_type: str = "linear",
    phase: float = 0.0,
) -> np.ndarray:
    """
    Generate a single CW tone with envelope.

    Args:
        frequency: Tone frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        rise_time: Rise time in seconds (1-15ms typical)
        fall_time: Fall time in seconds
        envelope_type: 'linear', 'cosine', or 'adsr'
        phase: Starting phase in radians

    Returns:
        Audio waveform (mono)
    """
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate

    # Generate sinusoidal tone
    tone = np.sin(2 * np.pi * frequency * t + phase)

    # Apply envelope
    envelope = generate_envelope(n_samples, rise_time, fall_time, sample_rate, envelope_type)
    tone = tone * envelope

    return tone


def generate_cw_audio(
    timing_sequence: list[tuple[float, bool]],
    frequency: float,
    sample_rate: int = 16000,
    rise_time: float | None = None,
    fall_time: float | None = None,
    envelope_type: str = "linear",
    frequency_drift: float = 0.0,
    chirp_amount: float = 0.0,
) -> np.ndarray:
    """
    Generate complete CW audio from timing sequence.

    Args:
        timing_sequence: List of (duration, is_tone_on) tuples
        frequency: Base frequency in Hz
        sample_rate: Sample rate in Hz
        rise_time: Rise time in seconds (None = random)
        fall_time: Fall time in seconds (None = random)
        envelope_type: Envelope type
        frequency_drift: Linear frequency drift in Hz over entire signal
        chirp_amount: Chirp amount in Hz on key-down

    Returns:
        Complete audio waveform
    """
    # Sample rise/fall times if not provided
    if rise_time is None:
        rise_time = sample_rise_fall_time()
    if fall_time is None:
        fall_time = rise_time

    audio_segments = []
    phase = 0.0  # Maintain phase continuity

    total_duration = sum(duration for duration, _ in timing_sequence)

    for i, (duration, is_tone_on) in enumerate(timing_sequence):
        if is_tone_on:
            # Calculate current frequency (with drift)
            progress = sum(d for d, _ in timing_sequence[:i]) / total_duration
            current_freq = frequency + (frequency_drift * progress)

            # Generate tone with chirp
            n_samples = int(duration * sample_rate)
            t = np.arange(n_samples) / sample_rate

            # Apply chirp on key-down (exponential rise at start)
            if chirp_amount != 0:
                chirp_envelope = 1 - np.exp(-t / 0.005)  # 5ms time constant
                freq_modulation = current_freq + chirp_amount * (1 - chirp_envelope)
            else:
                freq_modulation = current_freq

            # Generate tone with frequency modulation
            if chirp_amount != 0:
                # Instantaneous frequency varies
                phase_increment = 2 * np.pi * freq_modulation / sample_rate
                tone_phase = np.cumsum(phase_increment) + phase
                tone = np.sin(tone_phase)
                phase = tone_phase[-1]
            else:
                # Constant frequency
                tone = np.sin(2 * np.pi * current_freq * t + phase)
                phase = (phase + 2 * np.pi * current_freq * duration) % (2 * np.pi)

            # Apply envelope
            envelope = generate_envelope(
                n_samples, rise_time, fall_time, sample_rate, envelope_type
            )
            tone = tone * envelope

            audio_segments.append(tone)
        else:
            # Silence (gap)
            n_samples = int(duration * sample_rate)
            audio_segments.append(np.zeros(n_samples))

    # Concatenate all segments
    audio = (
        np.concatenate(audio_segments)
        if audio_segments
        else np.zeros(int(0.1 * sample_rate))  # Fallback for empty sequence
    )

    return audio


def sample_rise_fall_time() -> float:
    """
    Sample rise/fall time from realistic distribution.

    Returns:
        Rise/fall time in seconds
    """
    # Distribution: 40% fast (1-3ms), 40% medium (3-7ms), 20% soft (7-15ms)
    choice = np.random.random()
    if choice < 0.40:
        return np.random.uniform(0.001, 0.003)
    elif choice < 0.80:
        return np.random.uniform(0.003, 0.007)
    else:
        return np.random.uniform(0.007, 0.015)


def sample_envelope_type() -> str:
    """
    Sample envelope type from distribution.

    Returns:
        Envelope type string
    """
    return np.random.choice(["linear", "cosine", "adsr"], p=[0.60, 0.30, 0.10])


def sample_frequency(phase: int = 3) -> float:
    """
    Sample CW frequency from realistic distribution.

    Args:
        phase: Curriculum phase (1=foundation, 2=expansion, 3=mastery)

    Returns:
        Frequency in Hz
    """
    if phase == 1:
        # Phase 1: Sweet spot only (500-800 Hz)
        return np.random.uniform(500, 800)
    else:
        # Phase 2+: Full range (400-900 Hz)
        # Distribution: 60% sweet spot, 20% low, 20% high
        choice = np.random.random()
        if choice < 0.60:
            return np.random.uniform(500, 800)
        elif choice < 0.80:
            return np.random.uniform(400, 500)
        else:
            return np.random.uniform(800, 900)


def apply_frequency_drift(audio: np.ndarray, sample_rate: int, base_frequency: float) -> np.ndarray:
    """
    Apply frequency drift to existing audio (post-processing simulation).

    Note: This is a simplified version. In practice, drift is applied
    during tone generation (see generate_cw_audio).

    Args:
        audio: Input audio
        sample_rate: Sample rate in Hz
        base_frequency: Original frequency

    Returns:
        Audio with simulated drift effect
    """
    # This is a placeholder. Real implementation would resynthesize
    # with varying frequency. For now, just return original.
    return audio


def sample_frequency_drift_amount() -> float:
    """
    Sample frequency drift amount.

    Returns:
        Drift in Hz (±5 to ±20 Hz)
    """
    if np.random.random() < 0.30:  # 30% have drift
        return np.random.uniform(-20, 20)
    else:
        return 0.0


def sample_chirp_amount() -> float:
    """
    Sample key chirp amount.

    Returns:
        Chirp in Hz (10-30 Hz upward)
    """
    if np.random.random() < 0.15:  # 15% have chirp
        return np.random.uniform(10, 30)
    else:
        return 0.0
