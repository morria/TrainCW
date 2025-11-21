"""Tests for the TrainCW command-line interface."""

import sys
from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_cli_import():
    """Test that the CLI module can be imported."""
    from traincw import cli

    assert cli is not None
    assert hasattr(cli, "main")


@pytest.mark.unit
def test_cli_no_args():
    """Test CLI with no arguments shows help."""
    from traincw.cli import main

    # Mock sys.argv to simulate no arguments
    with patch.object(sys, "argv", ["traincw"]):
        result = main()
        assert result == 0  # Should exit successfully


@pytest.mark.unit
def test_cli_version():
    """Test CLI --version flag."""
    from traincw.cli import main

    with patch.object(sys, "argv", ["traincw", "--version"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        # argparse raises SystemExit(0) for --version
        assert exc_info.value.code == 0


@pytest.mark.unit
def test_cli_train_command():
    """Test CLI train command."""
    from traincw.cli import main

    with patch.object(sys, "argv", ["traincw", "train", "--config", "config.yaml"]):
        result = main()
        assert result == 0  # Placeholder implementation returns 0


@pytest.mark.unit
def test_cli_evaluate_command():
    """Test CLI evaluate command."""
    from traincw.cli import main

    with patch.object(sys, "argv", ["traincw", "evaluate", "--checkpoint", "model.pt"]):
        result = main()
        assert result == 0


@pytest.mark.unit
def test_cli_export_command():
    """Test CLI export command."""
    from traincw.cli import main

    with patch.object(
        sys, "argv", ["traincw", "export", "--checkpoint", "model.pt", "--output", "model.mlmodel"]
    ):
        result = main()
        assert result == 0
