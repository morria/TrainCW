# TrainCW

Neural network training system for real-time Morse code (CW) decoding. This project implements a CNN-LSTM-CTC model that learns to decode Morse code audio into text, designed for deployment on iOS devices via Core ML.

## Features

- **End-to-end CW decoder**: Converts audio spectrograms directly to text using CTC loss
- **Curriculum learning**: Progressive training from clean, slow signals to noisy, fast ones
- **Synthetic data generation**: Realistic training samples with configurable noise, interference, and timing variations
- **iOS deployment**: Export trained models to Core ML and ONNX formats
- **Comprehensive evaluation**: Character error rate (CER) and word error rate (WER) metrics

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0+ with CPU or CUDA support

### Setup

1. Clone the repository:
```bash
git clone https://github.com/morria/TrainCW.git
cd TrainCW
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e ".[dev,export,viz]"
```

3. Verify installation:
```bash
pytest
```

## Quick Start

### 1. Generate Demo Sample

Test the data generation system:

```bash
python scripts/demo_generation.py --output demo.wav --phase 3
```

This creates a synthetic Morse code audio file with accompanying text transcription.

### 2. Train a Model

Train using the default configuration:

```bash
python scripts/train.py --config configs/base_config.yaml
```

Or with a custom experiment name:

```bash
python scripts/train.py --config configs/base_config.yaml --experiment-name my_experiment
```

### 3. Evaluate the Model

Evaluate a trained checkpoint on test data:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/my_experiment/best_model.pt \
  --test-data path/to/test_data.pt \
  --output results.json
```

### 4. Export for Deployment

Export the trained model to ONNX and Core ML:

```bash
python scripts/export_model.py \
  --checkpoint checkpoints/my_experiment/best_model.pt \
  --format both \
  --output-dir exported_models \
  --quantize float16
```

## Full Pipeline Guide

### Training Pipeline

#### 1. Prepare Configuration

The training system uses YAML configuration files. Start with the base config:

```yaml
# configs/base_config.yaml
audio:
  sample_rate: 16000
  n_mels: 64

model:
  lstm_hidden_size: 256
  lstm_num_layers: 2

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001

curriculum:
  phase1_epochs: 30  # Easy: 12-25 WPM, high SNR
  phase2_epochs: 30  # Medium: 8-35 WPM, medium SNR
  # Phase 3 (full difficulty) runs automatically
```

Copy and modify for your experiment:

```bash
cp configs/base_config.yaml configs/my_config.yaml
# Edit configs/my_config.yaml as needed
```

#### 2. Start Training

Run the training script:

```bash
python scripts/train.py \
  --config configs/my_config.yaml \
  --experiment-name my_cw_decoder
```

The training process:
- Generates synthetic Morse code samples on-the-fly
- Uses curriculum learning (3 progressive phases)
- Automatically saves checkpoints every 5 epochs
- Validates after each epoch
- Implements early stopping based on validation CER

Training outputs:
```
checkpoints/my_cw_decoder/
├── config.yaml              # Saved configuration
├── training.log            # Training logs
├── best_model.pt           # Best checkpoint (lowest validation CER)
├── checkpoint_epoch_5.pt   # Periodic checkpoints
├── checkpoint_epoch_10.pt
└── ...
```

#### 3. Monitor Training

View training progress:

```bash
tail -f checkpoints/my_cw_decoder/training.log
```

Or use TensorBoard (if installed with `[viz]` extras):

```bash
tensorboard --logdir checkpoints/my_cw_decoder/logs
```

#### 4. Resume Training

Resume from a checkpoint if interrupted:

```bash
python scripts/train.py \
  --config configs/my_config.yaml \
  --resume checkpoints/my_cw_decoder/checkpoint_epoch_50.pt
```

### Evaluation Pipeline

#### 1. Generate Test Dataset

Create a test set with diverse conditions:

```bash
python scripts/generate_test_set.py \
  --output test_data.pt \
  --num-samples 1000 \
  --phase 3
```

#### 2. Run Evaluation

Evaluate your trained model:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/my_cw_decoder/best_model.pt \
  --test-data test_data.pt \
  --device cuda \
  --output evaluation_results.json
```

Evaluation metrics:
- **Character Error Rate (CER)**: Percentage of character-level errors
- **Word Error Rate (WER)**: Percentage of word-level errors
- **Sample predictions**: Shows reference vs. predicted text for inspection

Results are saved to JSON:

```json
{
  "checkpoint": "checkpoints/my_cw_decoder/best_model.pt",
  "test_data": "test_data.pt",
  "metrics": {
    "cer": 0.0234,
    "wer": 0.0891
  },
  "num_samples": 1000
}
```

### Export Pipeline

#### 1. Export to ONNX

For cross-platform deployment:

```bash
python scripts/export_model.py \
  --checkpoint checkpoints/my_cw_decoder/best_model.pt \
  --format onnx \
  --output-dir exported_models
```

#### 2. Export to Core ML

For iOS deployment:

```bash
python scripts/export_model.py \
  --checkpoint checkpoints/my_cw_decoder/best_model.pt \
  --format coreml \
  --output-dir exported_models \
  --quantize float16
```

Quantization options:
- `float32`: Full precision (larger file, highest accuracy)
- `float16`: Half precision (recommended, good balance)
- `int8`: 8-bit quantization (smallest file, slight accuracy loss)

#### 3. Export Both Formats

```bash
python scripts/export_model.py \
  --checkpoint checkpoints/my_cw_decoder/best_model.pt \
  --format both \
  --output-dir exported_models \
  --quantize float16
```

Output:
```
exported_models/
├── cw_decoder.onnx      # ONNX model
└── cw_decoder.mlmodel   # Core ML model
```

### Inference

#### Python Inference

Use the model for inference in Python:

```python
import torch
import torchaudio
from traincw.models.cnn_lstm_ctc import CWDecoder

# Load model
checkpoint = torch.load('checkpoints/my_cw_decoder/best_model.pt')
model = CWDecoder()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and preprocess audio
waveform, sample_rate = torchaudio.load('morse_audio.wav')

# Ensure correct sample rate (16kHz)
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = resampler(waveform)

# Convert to mel spectrogram
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=512,
    hop_length=160,
    win_length=400,
    n_mels=64,
    f_min=0.0,
    f_max=8000.0
)
spectrogram = mel_transform(waveform)
spectrogram = spectrogram.unsqueeze(0)  # Add batch dimension

# Decode
with torch.no_grad():
    predictions = model.decode(spectrogram, method='greedy')
    print(f"Decoded text: {predictions[0]}")

# Or use beam search for better accuracy
predictions = model.decode(spectrogram, method='beam_search', beam_width=10)
print(f"Decoded text (beam search): {predictions[0]}")
```

#### iOS Inference

Use the exported Core ML model in iOS:

```swift
import CoreML
import AVFoundation

// Load model
let model = try? cw_decoder()

// Process audio and get mel spectrogram
// (spectrogram preprocessing code here)

// Run inference
let prediction = try? model.prediction(spectrogram: spectrogramData)
let decodedText = prediction?.output
```

## Configuration Reference

### Audio Parameters

```yaml
audio:
  sample_rate: 16000      # Audio sample rate (Hz)
  n_fft: 512             # FFT window size
  hop_length: 160        # Hop length (10ms at 16kHz)
  win_length: 400        # Window length (25ms at 16kHz)
  n_mels: 64             # Number of mel bands
  f_min: 0.0             # Minimum frequency
  f_max: 8000.0          # Maximum frequency
```

### Model Architecture

```yaml
model:
  cnn_channels: [32, 64, 128, 256]  # CNN channel progression
  lstm_hidden_size: 256              # LSTM hidden units
  lstm_num_layers: 2                 # Number of LSTM layers
  bidirectional: true                # Use bidirectional LSTM
```

### Training Parameters

```yaml
training:
  batch_size: 32                    # Batch size
  num_epochs: 100                   # Total epochs
  learning_rate: 0.001              # Initial learning rate
  optimizer: adamw                   # Optimizer (adam/adamw)
  lr_schedule: cosine               # LR schedule (cosine/plateau/none)
  grad_clip_norm: 1.0               # Gradient clipping threshold
  early_stopping_patience: 15       # Patience for early stopping
  device: cpu                       # Device (cpu/cuda)
```

### Curriculum Learning

```yaml
curriculum:
  # Phase 1: Foundation (easy samples)
  phase1_epochs: 30
  phase1_wpm_range: [12.0, 25.0]
  phase1_snr_range: [15.0, 25.0]

  # Phase 2: Expansion (medium difficulty)
  phase2_epochs: 30
  phase2_wpm_range: [8.0, 35.0]
  phase2_snr_range: [10.0, 25.0]

  # Phase 3: Mastery (full difficulty range)
  # Runs automatically after phase 2
```

## Project Structure

```
TrainCW/
├── configs/              # Configuration files
│   └── base_config.yaml
├── data/                # Data generation modules
│   ├── generator.py     # Training sample generation
│   ├── audio_synthesis.py
│   ├── noise.py
│   └── interference.py
├── morse/               # Morse code utilities
│   ├── morse_code.py   # Morse encoding/decoding
│   └── timing.py       # Timing calculations
├── scripts/            # Command-line scripts
│   ├── train.py       # Training script
│   ├── evaluate.py    # Evaluation script
│   ├── export_model.py # Model export
│   ├── generate_test_set.py
│   └── demo_generation.py
├── src/traincw/       # Main package
│   ├── models/        # Neural network models
│   ├── training/      # Training logic
│   ├── evaluation/    # Evaluation metrics
│   ├── export/        # Model export utilities
│   └── utils/         # Shared utilities
└── tests/             # Test suite
```

## Performance Targets

- **Character Error Rate (CER)**: < 5% on clean signals, < 15% on noisy signals
- **Word Error Rate (WER)**: < 10% on clean signals, < 25% on noisy signals
- **Speed Range**: 5-40 WPM (words per minute)
- **SNR Range**: -5 to 25 dB
- **Inference Latency**: < 100ms on mobile devices

## Advanced Usage

### Custom Data Generation

Modify `data/generator.py` to customize training data:

```python
from data.generator import generate_training_sample

# Generate with specific parameters
audio, text, metadata = generate_training_sample(
    phase=3,
    wpm=25.0,
    snr_db=10.0,
    frequency=600.0
)
```

### Custom Model Architecture

Modify model parameters in config or create custom models in `src/traincw/models/`.

### Multi-GPU Training

Update config for distributed training:

```yaml
training:
  device: cuda
  distributed: true
  world_size: 4  # Number of GPUs
```

## Troubleshooting

### Out of Memory

Reduce batch size:
```yaml
training:
  batch_size: 16  # or 8
```

### Slow Training

- Use GPU: `device: cuda`
- Increase workers: `num_workers: 8`
- Enable mixed precision training (requires code modification)

### Poor Convergence

- Increase learning rate warmup
- Extend phase 1/2 duration
- Reduce learning rate
- Check data generation quality

## Development

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=traincw

# Specific test
pytest tests/test_model.py
```

### Code Quality

```bash
# Format code
ruff format .

# Lint
ruff check .

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Citation

If you use TrainCW in your research, please cite:

```bibtex
@software{traincw2024,
  title = {TrainCW: Neural Network Training System for Morse Code Decoding},
  author = {TrainCW Contributors},
  year = {2024},
  url = {https://github.com/morria/TrainCW}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

- Issues: https://github.com/morria/TrainCW/issues
- Documentation: See `DESIGN.md` and `IMPLEMENTATION.md`
