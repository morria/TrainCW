# Training Data Generation System

This directory contains the complete synthetic data generation system for training the Morse code (CW) neural network decoder.

## Overview

The data generation system creates realistic CW audio with comprehensive augmentations including:
- Variable speeds (5-40 WPM)
- Multiple noise types (white, pink, band-limited)
- Interference (QRM - other CW signals, QRN - impulse noise)
- Propagation effects (fading, frequency drift, chirp)
- Audio artifacts (AGC pumping, clipping, filter ringing)
- Realistic timing variance (operator "fist" characteristics)

## Architecture

### Core Modules

#### `morse/` - Morse Code Utilities
- `morse_code.py`: Encoding/decoding morse patterns
- `timing.py`: WPM calculations and timing variance

#### `data/` - Data Generation
- `audio_synthesis.py`: CW tone generation with envelope shaping
- `noise.py`: White/pink/band-limited noise generation
- `interference.py`: QRM, QRN, fading, and audio artifacts
- `text_generator.py`: Content generation (callsigns, words, abbreviations)
- `generator.py`: Main sample generation pipeline
- `dataset.py`: PyTorch Dataset classes for training

## Usage

### Generate a Single Sample

```python
from data.generator import generate_training_sample

# Generate sample (phase 3 = full difficulty)
audio, text, metadata = generate_training_sample(phase=3)

print(f"Text: {text}")
print(f"WPM: {metadata['wpm']:.1f}")
print(f"SNR: {metadata['snr_db']:.1f} dB")
```

### Use in Training

```python
from data.dataset import CWDataset
from torch.utils.data import DataLoader

# Create dataset (10,000 samples per epoch)
dataset = CWDataset(
    samples_per_epoch=10000,
    phase=3,  # Curriculum phase
    sample_rate=16000
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4
)

# Training loop
for batch in dataloader:
    spectrograms = batch['spectrograms']
    text_indices = batch['text_indices']
    # ... train model
```

### Generate Test Sets

```bash
# Generate validation set (2000 samples, stratified)
python scripts/generate_test_set.py \
    --type validation \
    --count 2000 \
    --output data/validation.pkl

# Generate speed sweep test
python scripts/generate_test_set.py \
    --type test_speed \
    --count 800 \
    --output data/test_speed.pkl

# Generate SNR sweep test
python scripts/generate_test_set.py \
    --type test_snr \
    --count 800 \
    --output data/test_snr.pkl
```

### Demo Generation

```bash
# Generate demo sample
python scripts/demo_generation.py --output demo.wav --phase 3

# Generate multiple samples
python scripts/demo_generation.py --output demo.wav --count 5
```

## Curriculum Learning Phases

The system supports three curriculum phases:

### Phase 1: Foundation (Epochs 1-30)
- **Speed**: 12-25 WPM (moderate only)
- **SNR**: 15-25 dB (clean signals)
- **Interference**: None
- **Goal**: Learn basic alphabet and timing

### Phase 2: Expansion (Epochs 31-60)
- **Speed**: 8-35 WPM (wider range)
- **SNR**: 10-25 dB (moderate noise)
- **Interference**: 15% QRM, 10% QRN
- **Goal**: Handle noise and speed variance

### Phase 3: Mastery (Epochs 61-100)
- **Speed**: 5-40 WPM (full range)
- **SNR**: -5 to 30 dB (full range)
- **Interference**: 25% QRM, 20% QRN, 30% fading
- **Goal**: Master all real-world conditions

## Data Generation Parameters

### Frequency Range
- **Range**: 400-900 Hz
- **Distribution**: 60% in 500-800 Hz, 20% low, 20% high
- **Effects**: Frequency drift (±5-20 Hz), key chirp (10-30 Hz)

### Speed (WPM)
- **Range**: 5-40 WPM
- **Distribution**: Focus on 15-25 WPM (typical QSO speeds)
- **Variance**: ±2-5% speed drift over sample duration

### Timing Variance
- **Dit/Dah ratio**: 1:2.3 to 1:3.7 (ideal 1:3)
- **Element gaps**: ±0-35% variance
- **Operator styles**: Clean (50%), Typical (30%), Rushed (15%), Stretched (5%)

### Noise Levels (SNR)
- **Excellent**: 25-30 dB (10% of samples)
- **Good**: 15-20 dB (20%)
- **Fair**: 10-15 dB (25%)
- **Poor**: 5-10 dB (15%)
- **Very Poor**: 0-5 dB (10%)
- **Extreme**: -5-0 dB (5%)

### Interference
- **QRM** (other CW): 25% of samples, 1-2 signals at ±100-800 Hz offset
- **QRN** (impulse noise): 20% of samples, 0.5-3 crashes/second
- **Fading**: 30% of samples, slow (0.1-1 Hz) or fast (1-5 Hz)

### Content Types
- **Random characters**: 30% (alphabet learning)
- **Random words**: 25% (context)
- **Callsigns**: 20% (ham patterns)
- **Numbers/Reports**: 10% (signal reports)
- **Abbreviations**: 10% (CW shorthand)
- **QSO exchanges**: 5% (real conversations)

## Testing

Run tests to verify data generation:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_generator.py

# Run with coverage
pytest --cov=data --cov=morse tests/
```

## Performance

Expected generation performance:
- **Single sample**: 10-30 ms
- **Batch of 32**: 0.5-1 second
- **Training consumption**: 5-10 samples/sec
- **Generation rate**: 30-100 samples/sec

Generation is faster than training consumption, so it won't bottleneck training.

## References

See `TRAINING_DATA_PLAN.md` for complete specification and rationale.
