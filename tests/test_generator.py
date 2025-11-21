"""
Tests for data generation.
"""

import numpy as np
from data.audio_synthesis import generate_tone
from data.generator import generate_sample_with_params, generate_training_sample
from data.noise import add_white_noise
from data.text_generator import TextGenerator, generate_callsign


def test_generate_tone():
    """Test basic tone generation."""
    audio = generate_tone(frequency=600, duration=0.1, sample_rate=16000)

    assert len(audio) == 1600  # 0.1 * 16000
    assert audio.dtype == np.float64
    assert -1.0 <= audio.max() <= 1.0
    assert -1.0 <= audio.min() <= 1.0


def test_add_white_noise():
    """Test noise addition."""
    # Create simple tone
    audio = np.sin(2 * np.pi * 600 * np.arange(1600) / 16000)

    # Add noise at 10 dB SNR
    noisy = add_white_noise(audio, snr_db=10)

    # Check that noise was added (signals should be different)
    assert not np.allclose(audio, noisy)

    # Check length preserved
    assert len(noisy) == len(audio)


def test_text_generator():
    """Test text generation."""
    gen = TextGenerator()

    # Test random characters
    text = gen.generate_random_characters(10)
    assert len(text) == 10
    assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" for c in text)

    # Test callsign generation
    callsign = gen.generate_callsign()
    assert 3 <= len(callsign) <= 6  # Typical callsign length
    assert callsign[0].isalpha()  # Should start with letter

    # Test signal report
    report = gen.generate_signal_report()
    assert len(report) == 3
    assert report[0] in "345"
    assert report[1] in "3456789"
    assert report[2] in "789"


def test_generate_callsign():
    """Test standalone callsign generator."""
    callsign = generate_callsign()
    assert isinstance(callsign, str)
    assert len(callsign) >= 3


def test_generate_training_sample():
    """Test full training sample generation."""
    # Generate sample
    audio, text, metadata = generate_training_sample(phase=1, sample_rate=16000)

    # Check audio
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0
    assert audio.dtype in [np.float32, np.float64]

    # Check text
    assert isinstance(text, str)
    assert len(text) > 0

    # Check metadata
    assert "wpm" in metadata
    assert "frequency" in metadata
    assert "snr_db" in metadata
    assert 5 <= metadata["wpm"] <= 40
    assert 400 <= metadata["frequency"] <= 900


def test_generate_sample_with_params():
    """Test generation with specific parameters."""
    params = {
        "wpm": 20,
        "frequency": 600,
        "snr_db": 15,
        "text": "TEST",
    }

    _audio, text, metadata = generate_sample_with_params(params)

    # Check parameters were respected
    assert text == "TEST"
    assert metadata["wpm"] == 20
    assert metadata["frequency"] == 600
    assert metadata["snr_db"] == 15


def test_phase_progression():
    """Test that different phases produce different difficulty."""
    # Phase 1 should be easier (higher SNR)
    # Use 50 samples to ensure statistical significance
    samples_phase1 = [generate_training_sample(phase=1) for _ in range(50)]
    snr_phase1 = [m["snr_db"] for _, _, m in samples_phase1]

    # Phase 3 should include harder conditions
    samples_phase3 = [generate_training_sample(phase=3) for _ in range(50)]
    snr_phase3 = [m["snr_db"] for _, _, m in samples_phase3]

    # Phase 1 should have higher average SNR (20 dB vs ~14 dB expected)
    # With 50 samples, this should be reliable
    assert np.mean(snr_phase1) > np.mean(snr_phase3)
