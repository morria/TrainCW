"""
Data generation and management for CW training.
"""

from .audio_synthesis import (
    generate_cw_audio,
    generate_tone,
    sample_frequency,
    sample_envelope_type,
    sample_rise_fall_time
)
from .noise import (
    add_white_noise,
    add_pink_noise,
    add_band_limited_noise,
    add_noise,
    add_impulse_noise,
    sample_snr
)
from .interference import (
    generate_qrm_signal,
    add_qrm,
    apply_fading,
    apply_agc_pumping,
    apply_clipping,
    apply_bandpass_filter
)
from .text_generator import (
    TextGenerator,
    generate_random_text,
    generate_callsign,
    sample_text_length
)
from .generator import (
    generate_training_sample,
    generate_sample_with_params
)
from .dataset import (
    CWDataset,
    CWTestDataset,
    collate_fn
)

__all__ = [
    # Audio synthesis
    'generate_cw_audio',
    'generate_tone',
    'sample_frequency',
    'sample_envelope_type',
    'sample_rise_fall_time',
    # Noise
    'add_white_noise',
    'add_pink_noise',
    'add_band_limited_noise',
    'add_noise',
    'add_impulse_noise',
    'sample_snr',
    # Interference
    'generate_qrm_signal',
    'add_qrm',
    'apply_fading',
    'apply_agc_pumping',
    'apply_clipping',
    'apply_bandpass_filter',
    # Text generation
    'TextGenerator',
    'generate_random_text',
    'generate_callsign',
    'sample_text_length',
    # Sample generation
    'generate_training_sample',
    'generate_sample_with_params',
    # Datasets
    'CWDataset',
    'CWTestDataset',
    'collate_fn',
]
