"""
Main training sample generator - brings all components together.
"""

import numpy as np
from morse.morse_code import MorseCode
from morse.timing import TimingCalculator, sample_wpm, select_operator_style

from .audio_synthesis import (
    generate_cw_audio,
    sample_chirp_amount,
    sample_envelope_type,
    sample_frequency,
    sample_frequency_drift_amount,
    sample_rise_fall_time,
)
from .interference import (
    add_qrm,
    apply_agc_pumping,
    apply_bandpass_filter,
    apply_clipping,
    apply_fading,
)
from .noise import add_impulse_noise, add_noise, sample_snr
from .text_generator import generate_random_text, sample_text_length


def generate_training_sample(
    phase: int = 3, sample_rate: int = 16000, max_duration: float = 2.0
) -> tuple[np.ndarray, str, dict]:
    """
    Generate one complete training sample with all augmentations.

    This implements the full pipeline from the training plan.

    Args:
        phase: Curriculum phase (1=foundation, 2=expansion, 3=mastery)
        sample_rate: Audio sample rate in Hz
        max_duration: Maximum sample duration in seconds (for iOS deployment)

    Returns:
        Tuple of (audio_waveform, text_label, metadata)
    """
    morse = MorseCode()

    # 1. Generate content
    target_length = sample_text_length()
    text = generate_random_text(length=target_length)

    # 2. Select basic parameters
    wpm = sample_wpm(phase=phase)
    frequency = sample_frequency(phase=phase)

    # 3. Generate timing sequence
    timing_variance = select_operator_style()
    timing_calc = TimingCalculator(wpm, timing_variance)
    elements = morse.text_to_elements(text)
    timing_sequence = timing_calc.get_timing_sequence(elements)

    # Check duration and truncate if needed
    total_duration = sum(duration for duration, _ in timing_sequence)
    if total_duration > max_duration:
        # Truncate timing sequence
        current_duration = 0.0
        truncated_sequence = []
        truncated_text = []

        for i, (duration, is_on) in enumerate(timing_sequence):
            if current_duration + duration > max_duration:
                break
            truncated_sequence.append((duration, is_on))
            current_duration += duration

            # Track which character we're at
            if i < len(elements):
                element_type, char = elements[i]
                is_mark = element_type in ["dit", "dah"]
                is_new_char = not truncated_text or truncated_text[-1] != char
                if is_mark and is_new_char:
                    truncated_text.append(char)

        timing_sequence = truncated_sequence
        text = "".join(truncated_text)
        total_duration = current_duration

    # 4. Sample envelope parameters
    rise_time = sample_rise_fall_time()
    fall_time = rise_time
    envelope_type = sample_envelope_type()

    # 5. Sample frequency effects
    frequency_drift = sample_frequency_drift_amount()
    chirp_amount = sample_chirp_amount()

    # 6. Generate base audio tone
    audio = generate_cw_audio(
        timing_sequence=timing_sequence,
        frequency=frequency,
        sample_rate=sample_rate,
        rise_time=rise_time,
        fall_time=fall_time,
        envelope_type=envelope_type,
        frequency_drift=frequency_drift,
        chirp_amount=chirp_amount,
    )

    # 7. Select noise level
    snr_db = sample_snr(phase=phase)

    # 8. Add base noise
    audio = add_noise(audio, snr_db, frequency, sample_rate)

    # 9. Add interference (phase-dependent)
    if phase >= 2:
        # QRM (other CW signals)
        if phase == 2:
            # Phase 2: 15% QRM (lighter)
            if np.random.random() < 0.15:
                audio = add_qrm(audio, frequency, sample_rate)
        else:
            # Phase 3: 25% QRM (full)
            audio = add_qrm(audio, frequency, sample_rate)

        # QRN (impulse noise)
        if phase == 2:
            # Phase 2: 10% QRN
            if np.random.random() < 0.10:
                audio = add_impulse_noise(audio, sample_rate)
        else:
            # Phase 3: 20% QRN
            audio = add_impulse_noise(audio, sample_rate)

    # 10. Apply propagation effects (phase 3 only)
    if phase >= 3:
        audio = apply_fading(audio, sample_rate)

    # 11. Apply audio artifacts (phase 3 only)
    if phase >= 3:
        audio = apply_agc_pumping(audio, sample_rate)
        audio = apply_clipping(audio)
        audio = apply_bandpass_filter(audio, frequency, q_factor=20, sample_rate=sample_rate)

    # 12. Normalize to prevent clipping
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9  # Leave some headroom

    # 13. Package metadata
    metadata = {
        "text": text,
        "wpm": wpm,
        "frequency": frequency,
        "snr_db": snr_db,
        "timing_variance": timing_variance,
        "rise_time": rise_time,
        "envelope_type": envelope_type,
        "frequency_drift": frequency_drift,
        "chirp_amount": chirp_amount,
        "duration": len(audio) / sample_rate,
        "phase": phase,
    }

    return audio, text, metadata


def generate_sample_with_params(
    params: dict, sample_rate: int = 16000
) -> tuple[np.ndarray, str, dict]:
    """
    Generate a sample with specific parameters (for test set generation).

    Args:
        params: Dictionary of parameters to control generation
        sample_rate: Audio sample rate in Hz

    Returns:
        Tuple of (audio_waveform, text_label, metadata)
    """
    morse = MorseCode()

    # Extract or use defaults
    wpm = params.get("wpm", 20)
    frequency = params.get("frequency", 600)
    snr_db = params.get("snr_db", 15)
    text = params.get("text", generate_random_text())
    timing_variance = params.get(
        "timing_variance",
        {
            "dit_dah_ratio": 3.0,
            "element_gap_variance": 0.1,
            "char_gap_variance": 0.1,
            "word_gap_variance": 0.1,
        },
    )

    # Generate timing
    timing_calc = TimingCalculator(wpm, timing_variance)
    elements = morse.text_to_elements(text)
    timing_sequence = timing_calc.get_timing_sequence(elements)

    # Generate audio
    audio = generate_cw_audio(
        timing_sequence=timing_sequence,
        frequency=frequency,
        sample_rate=sample_rate,
        rise_time=params.get("rise_time", 0.003),
        fall_time=params.get("fall_time", 0.003),
        envelope_type=params.get("envelope_type", "linear"),
        frequency_drift=params.get("frequency_drift", 0.0),
        chirp_amount=params.get("chirp_amount", 0.0),
    )

    # Add noise
    audio = add_noise(audio, snr_db, frequency, sample_rate)

    # Optional effects
    if params.get("add_qrm", False):
        audio = add_qrm(audio, frequency, sample_rate)
    if params.get("add_qrn", False):
        audio = add_impulse_noise(audio, sample_rate)
    if params.get("add_fading", False):
        audio = apply_fading(audio, sample_rate)
    if params.get("add_agc", False):
        audio = apply_agc_pumping(audio, sample_rate)
    if params.get("add_clipping", False):
        audio = apply_clipping(audio)
    if params.get("add_filter", False):
        audio = apply_bandpass_filter(audio, frequency, q_factor=20, sample_rate=sample_rate)

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    metadata = {
        "text": text,
        "wpm": wpm,
        "frequency": frequency,
        "snr_db": snr_db,
        "duration": len(audio) / sample_rate,
    }

    return audio, text, metadata
