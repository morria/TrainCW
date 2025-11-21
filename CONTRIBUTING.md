# Contributing to TrainCW

Thank you for your interest in contributing to TrainCW! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) Virtual environment tool (venv, conda, etc.)

### Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/morria/TrainCW.git
   cd TrainCW
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   make install-dev
   # or
   pip install -e ".[dev]"
   pre-commit install
   ```

4. **Verify installation**
   ```bash
   make check
   # or run individual checks:
   make test
   make lint
   ```

## Development Workflow

### 1. Create a Branch

Create a feature branch from `main`:
```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-lstm-model`
- `fix/audio-synthesis-bug`
- `docs/update-readme`

### 2. Make Your Changes

- Write clean, readable code following PEP 8
- Add type hints to function signatures
- Write comprehensive docstrings (Google style)
- Add tests for new functionality
- Update documentation as needed

### 3. Code Quality Checks

Before committing, run:

```bash
# Format code
make format

# Run all checks
make check
```

Or use the pre-commit hooks (installed automatically):
```bash
git commit -m "Your commit message"
# Pre-commit hooks will run automatically
```

### 4. Write Tests

- Add unit tests for new functions/classes
- Add integration tests for complex features
- Aim for high test coverage (>80%)
- Use pytest fixtures for common test data

Example test structure:
```python
import pytest
from traincw.module import function

@pytest.mark.unit
def test_function_basic():
    """Test basic functionality."""
    result = function(input_data)
    assert result == expected_output

@pytest.mark.unit
def test_function_edge_case():
    """Test edge case handling."""
    with pytest.raises(ValueError):
        function(invalid_input)
```

### 5. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add CNN encoder for spectrogram processing

- Implement convolutional layers with batch normalization
- Add max pooling for temporal downsampling
- Include unit tests for encoder architecture
- Update documentation with architecture diagram
"
```

Commit message guidelines:
- Use present tense ("Add feature" not "Added feature")
- First line: concise summary (50 chars or less)
- Blank line after first line
- Detailed description if needed
- Reference issues: "Fixes #123" or "Relates to #456"

### 6. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title describing the change
- Detailed description of what and why
- Link to related issues
- Screenshots/examples if applicable

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (not 79)
- **Quotes**: Double quotes for strings
- **Imports**: Organized by ruff (isort)
- **Formatting**: Handled by ruff format

### Type Hints

Use type hints for all function signatures:

```python
from pathlib import Path
from typing import Optional, Tuple

def process_audio(
    audio_path: Path,
    sample_rate: int = 16000,
    normalize: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Process audio file and return tensor.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate in Hz
        normalize: Whether to normalize audio amplitude

    Returns:
        Tuple of (audio_tensor, actual_sample_rate)
    """
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def train_model(
    model: torch.nn.Module,
    dataset: Dataset,
    epochs: int,
    learning_rate: float = 1e-3,
) -> dict:
    """Train a neural network model.

    This function trains the provided model on the dataset using
    the Adam optimizer and CTC loss.

    Args:
        model: PyTorch model to train
        dataset: Training dataset
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer (default: 1e-3)

    Returns:
        Dictionary containing training metrics:
        - 'loss': Final training loss
        - 'cer': Character error rate
        - 'history': Loss history per epoch

    Raises:
        ValueError: If epochs < 1
        RuntimeError: If model training fails

    Example:
        >>> model = CNN_LSTM_CTC()
        >>> dataset = MorseDataset(...)
        >>> metrics = train_model(model, dataset, epochs=50)
        >>> print(f"Final CER: {metrics['cer']:.2%}")
    """
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_morse.py

# Run tests matching pattern
pytest -k "test_audio"

# Run fast tests only
pytest -m "not slow"

# Run with verbose output
pytest -v

# Run in parallel (faster)
make test-parallel
```

### Writing Tests

Test organization:
- `tests/test_module.py` for each module
- Use fixtures in `tests/conftest.py`
- Mark slow tests with `@pytest.mark.slow`
- Mark integration tests with `@pytest.mark.integration`

### Test Coverage

Aim for high coverage:
```bash
# Generate coverage report
pytest --cov=traincw --cov-report=html

# View in browser
open htmlcov/index.html
```

## Architecture Guidelines

### Project Structure

Follow the structure in `DESIGN.md`:

```
src/traincw/
├── morse/          # Morse code utilities
├── data/           # Data generation and datasets
├── models/         # Neural network architectures
├── training/       # Training loops and callbacks
├── evaluation/     # Metrics and evaluation
├── export/         # Model export (ONNX, Core ML)
└── utils/          # Shared utilities
```

### Adding New Modules

When adding a new module:

1. Create the module file: `src/traincw/module_name.py`
2. Add comprehensive docstrings
3. Create tests: `tests/test_module_name.py`
4. Update `__init__.py` if needed
5. Add to documentation

Use Claude Code's `/add-module` command for guidance!

## Pull Request Process

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] All tests pass: `make test`
- [ ] Linting passes: `make lint`
- [ ] Formatting is correct: `make format-check`
- [ ] Type checking passes (if applicable): `make typecheck`
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (for significant changes)

### PR Review

- Maintainers will review your PR
- Address any feedback or requested changes
- Keep the PR focused and atomic
- Be responsive to comments

### After Merge

- Delete your feature branch
- Pull the latest changes from main
- Celebrate! 🎉

## Common Tasks

### Add a New Model Architecture

1. Create `src/traincw/models/your_model.py`
2. Implement `torch.nn.Module` with forward pass
3. Add unit tests for architecture
4. Add integration test with dummy data
5. Document architecture in docstring
6. Update DESIGN.md if it's a major change

### Add a New Data Generator

1. Create generator in `src/traincw/data/`
2. Implement with proper randomization
3. Add visualization for debugging
4. Test with various parameters
5. Benchmark generation speed

### Add a New Metric

1. Create in `src/traincw/evaluation/metrics.py`
2. Follow sklearn-style API
3. Add comprehensive tests
4. Document metric calculation

## Getting Help

- **Documentation**: Check `DESIGN.md` and code docstrings
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions
- **Claude Code**: Use custom commands like `/explain-codebase`

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to TrainCW! 73 de AI 🎧
