# Training System Implementation

This document describes the implementation of the training system for the CW (Morse code) decoder neural network. The implementation follows the design specifications in [DESIGN.md](DESIGN.md) and [TRAINING_DATA_PLAN.md](TRAINING_DATA_PLAN.md).

## ✅ Implemented Components

### 1. Model Architecture (`src/traincw/models/`)

- **CNN-LSTM-CTC Model** (`cnn_lstm_ctc.py`):
  - Complete implementation of the CNN-LSTM-CTC architecture
  - CNN encoder for feature extraction from spectrograms
  - Bidirectional LSTM for temporal modeling
  - CTC loss computation for training
  - Model size: ~2-3M parameters, <5MB

- **CNN Encoder** (`encoder.py`):
  - Configurable convolutional blocks
  - Batch normalization and max pooling
  - Automatic feature dimension calculation

- **CTC Decoder** (`decoder.py`):
  - Character vocabulary (A-Z, 0-9, punctuation, prosigns)
  - Greedy decoding
  - Beam search decoding
  - Text encoding/decoding utilities

### 2. Training Infrastructure (`src/traincw/training/`)

- **Trainer Class** (`trainer.py`):
  - Complete training loop with CTC loss
  - Validation with CER calculation
  - Checkpoint management (best N models)
  - TensorBoard logging
  - Early stopping
  - Learning rate scheduling (cosine annealing, plateau)
  - Gradient clipping

- **Curriculum Learning** (`curriculum.py`):
  - 3-phase curriculum scheduler
  - Phase 1: Foundation (easy samples, clean signals)
  - Phase 2: Expansion (medium difficulty, more noise)
  - Phase 3: Mastery (full difficulty, all conditions)
  - Automatic phase progression

### 3. Evaluation Metrics (`src/traincw/evaluation/`)

- **Character Error Rate (CER)**
- **Word Error Rate (WER)**
- **Edit Distance** (Levenshtein distance)
- **Confusion Matrix** generation

### 4. Audio Preprocessing (`src/traincw/utils/`)

- **Mel-Spectrogram Computation**:
  - Configurable STFT parameters
  - Log-scale (dB) conversion
  - Per-sample normalization
  - iOS-compatible parameters (16kHz, 64 mel bins)

- **Audio Normalization**:
  - RMS-based normalization to target dB level
  - Handles both NumPy and PyTorch tensors

- **Utility Functions**:
  - Padding/trimming spectrograms
  - Audio/spectrogram length conversions

### 5. Configuration Management (`src/traincw/utils/config.py`)

- Dataclass-based configuration system
- YAML file support
- Hierarchical config structure:
  - Audio preprocessing config
  - Model architecture config
  - Training config
  - Curriculum learning config
  - Data config
  - Export config

### 6. Export Functionality (`src/traincw/export/`)

- **ONNX Export** (`to_onnx.py`):
  - PyTorch → ONNX conversion
  - Dynamic batch and time dimensions
  - Automatic verification
  - Compatible with opset 14+

- **Core ML Export** (`to_coreml.py`):
  - ONNX → Core ML conversion
  - Quantization support (float32, float16, int8)
  - iOS 15+ deployment target
  - Neural Engine optimization
  - Model metadata

### 7. Scripts (`scripts/`)

- **Training Script** (`train.py`):
  - Complete training pipeline
  - Experiment management
  - Checkpoint resume support
  - Logging setup

- **Evaluation Script** (`evaluate.py`):
  - Model evaluation on test sets
  - Metrics reporting
  - Sample prediction visualization

- **Export Script** (`export_model.py`):
  - Model export to ONNX and/or Core ML
  - Quantization options
  - Format validation

### 8. Tests (`tests/`)

- **Model Tests** (`test_model.py`):
  - CNN encoder tests
  - CWDecoder model tests
  - CTC decoder tests
  - Loss computation tests

- **Metrics Tests** (`test_metrics.py`):
  - Edit distance tests
  - CER calculation tests
  - WER calculation tests
  - Confusion matrix tests

- **Audio Tests** (`test_audio.py`):
  - Spectrogram computation tests
  - Audio normalization tests
  - Padding/trimming tests
  - Length conversion tests

### 9. Configuration (`configs/`)

- **Base Configuration** (`base_config.yaml`):
  - Production-ready settings
  - Follows design specifications
  - Fully documented parameters

## 📋 What's NOT Implemented (Left for Data Generation Task)

The following components are **intentionally not implemented** as they are part of the data generation module:

### Data Generation Module (`src/traincw/data/`)

**Note**: This is a separate task and was explicitly excluded from this implementation.

Components needed for data generation:
- Morse code encoding utilities
- Audio tone synthesis
- Noise generation (white, pink, band-limited)
- Interference simulation (QRM, QRN)
- Fading effects
- Timing variance (operator "fist" simulation)
- Text generation (callsigns, words, random characters)
- PyTorch Dataset classes
- Data loaders for training and validation

## 🚀 Usage

### Installation

```bash
# Install base dependencies
pip install -e .

# Install with development tools
pip install -e ".[dev]"

# Install with export capabilities
pip install -e ".[export]"

# Install with visualization tools
pip install -e ".[viz]"

# Install everything
pip install -e ".[all]"
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v

# Run with coverage
pytest tests/ --cov=traincw --cov-report=html
```

### Training (Once Data Generation is Implemented)

```bash
# Train with default config
python scripts/train.py --config configs/base_config.yaml

# Resume from checkpoint
python scripts/train.py --config configs/base_config.yaml --resume checkpoints/best_model.pt

# Custom experiment name
python scripts/train.py --config configs/base_config.yaml --experiment-name my_experiment
```

### Evaluation

```bash
# Evaluate model
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --test-data path/to/test.pkl

# Save results
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --test-data path/to/test.pkl --output results.json
```

### Export

```bash
# Export to both ONNX and Core ML
python scripts/export_model.py --checkpoint checkpoints/best_model.pt --format both

# Export to ONNX only
python scripts/export_model.py --checkpoint checkpoints/best_model.pt --format onnx

# Export to Core ML with quantization
python scripts/export_model.py --checkpoint checkpoints/best_model.pt --format coreml --quantize float16
```

## 📦 Project Structure

```
TrainCW/
├── src/traincw/
│   ├── models/              ✅ Model architecture
│   │   ├── cnn_lstm_ctc.py  ✅ Complete CNN-LSTM-CTC model
│   │   ├── encoder.py       ✅ CNN encoder
│   │   └── decoder.py       ✅ CTC decoder utilities
│   ├── training/            ✅ Training infrastructure
│   │   ├── trainer.py       ✅ Trainer class
│   │   └── curriculum.py    ✅ Curriculum scheduler
│   ├── evaluation/          ✅ Evaluation metrics
│   │   └── metrics.py       ✅ CER, WER, edit distance
│   ├── export/              ✅ Model export
│   │   ├── to_onnx.py       ✅ ONNX export
│   │   └── to_coreml.py     ✅ Core ML export
│   ├── utils/               ✅ Utilities
│   │   ├── audio.py         ✅ Audio preprocessing
│   │   ├── config.py        ✅ Configuration management
│   │   └── logger.py        ✅ Logging utilities
│   └── data/                ❌ NOT IMPLEMENTED (separate task)
│       ├── generator.py     ❌ Data generation
│       ├── audio_synthesis.py ❌ Audio synthesis
│       ├── noise.py         ❌ Noise generation
│       ├── text_generator.py ❌ Text generation
│       └── dataset.py       ❌ PyTorch datasets
├── scripts/                 ✅ Command-line scripts
│   ├── train.py             ✅ Training script
│   ├── evaluate.py          ✅ Evaluation script
│   └── export_model.py      ✅ Export script
├── configs/                 ✅ Configuration files
│   └── base_config.yaml     ✅ Base configuration
├── tests/                   ✅ Unit tests
│   ├── test_model.py        ✅ Model tests
│   ├── test_metrics.py      ✅ Metrics tests
│   └── test_audio.py        ✅ Audio tests
├── DESIGN.md                📄 Design document
├── TRAINING_DATA_PLAN.md    📄 Data generation plan
└── IMPLEMENTATION.md        📄 This file
```

## 🎯 Key Features

### Model Features
- ✅ CNN-LSTM-CTC architecture as per design
- ✅ ~2-3M parameters, deployable to iOS (<5MB)
- ✅ Configurable architecture (channels, layers, hidden size)
- ✅ Bidirectional LSTM for better accuracy
- ✅ CTC loss for alignment-free training

### Training Features
- ✅ Curriculum learning (3 phases)
- ✅ CTC loss with gradient clipping
- ✅ Cosine annealing or plateau LR scheduling
- ✅ Early stopping with patience
- ✅ Checkpoint management (keep best N)
- ✅ TensorBoard logging
- ✅ Comprehensive validation with CER

### Evaluation Features
- ✅ Character Error Rate (CER)
- ✅ Word Error Rate (WER)
- ✅ Confusion matrix
- ✅ Sample prediction logging

### Export Features
- ✅ ONNX export with verification
- ✅ Core ML export with quantization
- ✅ iOS Neural Engine optimization
- ✅ Float16/Int8 quantization support

## 📝 Next Steps

### 1. Implement Data Generation Module ⏭️

This is the **next major task** and includes:

- **Morse code utilities** (`src/traincw/morse/`):
  - Morse code encoding/decoding
  - Timing calculations (WPM to milliseconds)
  - PARIS method implementation

- **Audio synthesis** (`src/traincw/data/`):
  - CW tone generation
  - Envelope shaping (rise/fall times)
  - Frequency drift and chirp
  - Timing variance (operator fist simulation)

- **Noise and interference**:
  - White, pink, band-limited noise
  - QRM (other CW signals)
  - QRN (atmospheric noise, impulses)
  - Fading (QSB simulation)
  - Audio artifacts (AGC, clipping, filter ringing)

- **Text generation**:
  - Random characters
  - Dictionary words
  - Callsign patterns (W1ABC, K2XYZ, etc.)
  - Q-codes and abbreviations
  - QSO phrases

- **Dataset classes**:
  - On-the-fly generation (infinite data)
  - Fixed validation/test sets
  - Curriculum-aware sampling
  - Efficient data loading

### 2. Streaming Audio Support 🎙️

**User Requirement**: Support streaming data from sound card with a few seconds of delay.

Implementation approach:
- Buffered audio input (e.g., 2-second sliding windows)
- Overlap-add for smooth streaming
- Real-time spectrogram computation
- Queue-based processing pipeline
- Integration with PyAudio or sounddevice

This will be implemented as part of a deployment/inference module:
- `src/traincw/streaming/` - Real-time audio streaming
- `scripts/stream_decode.py` - Live decoding script

### 3. Generate Test Sets

Once data generation is complete:
```bash
python scripts/generate_test_set.py --type validation --count 2000
python scripts/generate_test_set.py --type test_speed --count 800
python scripts/generate_test_set.py --type test_snr --count 800
# ... (see TRAINING_DATA_PLAN.md)
```

### 4. Train the Model

```bash
python scripts/train.py --config configs/base_config.yaml
```

Expected training time: ~35-42 hours on CPU for 100 epochs.

### 5. Evaluate and Export

```bash
# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --test-data data/test.pkl

# Export for iOS
python scripts/export_model.py --checkpoint checkpoints/best_model.pt --format coreml --quantize float16
```

## 🧪 Testing

The implementation includes comprehensive unit tests:

- **Model tests**: Test CNN encoder, LSTM, CTC decoder, forward pass
- **Metrics tests**: Test CER, WER, edit distance calculations
- **Audio tests**: Test spectrogram computation, normalization
- **Configuration tests**: Test config loading and saving
- **Export tests**: Test ONNX and Core ML export (when dependencies available)

All tests are designed to run quickly and don't require data generation.

## 📊 Expected Performance

Based on the design specifications:

### Model Performance Targets
- **CER < 5%** at SNR > 10dB, 15-25 WPM (typical conditions)
- **CER < 10%** at SNR > 5dB, 10-30 WPM (moderate conditions)
- **CER < 20%** at SNR 0-5dB (challenging conditions)

### iOS Deployment Targets
- **Model size**: < 5MB (float32) or < 2.5MB (float16)
- **Inference time**: < 100ms per 2-second window on iPhone 12+
- **Memory usage**: < 100MB
- **Battery**: < 10% per hour of continuous decoding

## 🔧 Configuration

The base configuration (`configs/base_config.yaml`) follows all design specifications:

- **Audio**: 16kHz, 64 mel bins, 10ms hop length
- **Model**: 4 CNN layers, 2 LSTM layers, 256 hidden size
- **Training**: 100 epochs, batch size 32, AdamW optimizer
- **Curriculum**: 3 phases (30 + 30 + 40 epochs)
- **Export**: Float16 quantization, Neural Engine support

See the config file for full details and customization options.

## 📖 Documentation

- **[DESIGN.md](DESIGN.md)**: Complete system design
- **[TRAINING_DATA_PLAN.md](TRAINING_DATA_PLAN.md)**: Detailed data generation plan
- **[README.md](README.md)**: Project overview
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)**: This file

## 🤝 Contributing

When implementing the data generation module:

1. Follow the specifications in `TRAINING_DATA_PLAN.md`
2. Maintain the same code structure and style
3. Add comprehensive tests
4. Update documentation
5. Ensure compatibility with the training pipeline

## 📜 License

MIT License (see LICENSE file)

---

**Status**: ✅ Training system implementation complete
**Next Task**: ⏭️ Implement data generation module
**Timeline**: Ready for data generation implementation and training
