"""Pytest configuration and shared fixtures for TrainCW tests."""

from pathlib import Path

import pytest


# Optional imports - only needed for integration tests
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@pytest.fixture
def sample_rate() -> int:
    """Standard sample rate for audio processing in TrainCW."""
    return 16000


@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test data.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Path to temporary test data directory
    """
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def random_seed() -> int:
    """Fixed random seed for reproducible tests."""
    seed = 42
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
    if NUMPY_AVAILABLE:
        # Use modern numpy random API
        _ = np.random.default_rng(seed)
    return seed


@pytest.fixture
def mock_audio_tensor(sample_rate: int):
    """Create a mock audio tensor for testing.

    Args:
        sample_rate: Audio sample rate

    Returns:
        Mock audio tensor (1 second of audio)

    Raises:
        pytest.skip: If torch is not available
    """
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")

    duration = 1.0  # seconds
    num_samples = int(sample_rate * duration)
    # Create a simple sine wave
    frequency = 600.0  # Hz
    t = torch.linspace(0, duration, num_samples)
    audio = torch.sin(2 * torch.pi * frequency * t)
    return audio


@pytest.fixture
def morse_test_text() -> str:
    """Sample Morse code text for testing."""
    return "HELLO WORLD"


@pytest.fixture
def morse_callsign() -> str:
    """Sample amateur radio callsign for testing."""
    return "W1ABC"


# Markers for test categorization
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests (e.g., training loops)")
