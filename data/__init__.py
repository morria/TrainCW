"""
Data generation and management for CW training.
"""

from .audio_synthesis import (
    generate_cw_audio,
    generate_tone,
    sample_envelope_type,
    sample_frequency,
    sample_rise_fall_time,
)
from .dataset import CWDataset, CWTestDataset, collate_fn
from .generator import generate_sample_with_params, generate_training_sample
from .interference import (
    add_qrm,
    apply_agc_pumping,
    apply_bandpass_filter,
    apply_clipping,
    apply_fading,
    generate_qrm_signal,
)
from .noise import (
    add_band_limited_noise,
    add_impulse_noise,
    add_noise,
    add_pink_noise,
    add_white_noise,
    sample_snr,
)
from .text_generator import (
    TextGenerator,
    generate_callsign,
    generate_random_text,
    sample_text_length,
)


__all__ = [
    # Datasets
    "CWDataset",
    "CWTestDataset",
    # Text generation
    "TextGenerator",
    "add_band_limited_noise",
    "add_impulse_noise",
    "add_noise",
    "add_pink_noise",
    "add_qrm",
    # Noise
    "add_white_noise",
    "apply_agc_pumping",
    "apply_bandpass_filter",
    "apply_clipping",
    "apply_fading",
    "collate_fn",
    "generate_callsign",
    # Audio synthesis
    "generate_cw_audio",
    # Interference
    "generate_qrm_signal",
    "generate_random_text",
    "generate_sample_with_params",
    "generate_tone",
    # Sample generation
    "generate_training_sample",
    "sample_envelope_type",
    "sample_frequency",
    "sample_rise_fall_time",
    "sample_snr",
    "sample_text_length",
]
