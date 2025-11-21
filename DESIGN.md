# Morse Code Neural Network Training System - Design Document

## Overview
This project trains a neural network to decode Morse code (CW) from audio input for deployment in an iOS application. The system generates synthetic training data with realistic variations and noise, then trains a model that can be exported to iOS Core ML format.

## 1. Technology Stack

### Training Environment
- **Language**: Python 3.10+
- **Deep Learning Framework**: PyTorch 2.0+
  - Rationale: Excellent flexibility, strong ecosystem, easy debugging
  - Good export options via ONNX → Core ML
- **Audio Processing**: librosa, scipy, soundfile
- **Numerical Computing**: NumPy, SciPy
- **Data Handling**: PyTorch DataLoader with custom Dataset classes
- **Visualization**: matplotlib, tensorboard
- **Model Export**: coremltools, onnx

### Key Libraries
```
torch>=2.0.0
torchaudio>=2.0.0
librosa>=0.10.0
scipy>=1.10.0
numpy>=1.24.0
soundfile>=0.12.0
coremltools>=7.0
onnx>=1.14.0
tensorboard>=2.13.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

## 2. Neural Network Architecture

### Approach: Spectrogram + CNN + Bi-LSTM + CTC

This is fundamentally a sequence-to-sequence problem with temporal alignment challenges.

#### Input Processing
```
Audio Waveform (16kHz)
    ↓
STFT / Mel-Spectrogram
    ↓
(Time × Frequency) Tensor
```

#### Model Architecture
```
Input: Spectrogram [Batch, Time, Freq]
    ↓
CNN Feature Extractor (3-4 layers)
    - Conv2D + BatchNorm + ReLU + MaxPool
    - Extract local time-frequency patterns
    - Output: [Batch, Time', Features]
    ↓
Bidirectional LSTM (2-3 layers)
    - Temporal modeling
    - Context from both directions
    - Output: [Batch, Time', Hidden]
    ↓
Fully Connected Layer
    - Map to character probabilities
    - Output: [Batch, Time', NumCharacters+1]
    ↓
CTC Loss / CTC Beam Search Decoder
    - Alignment-free training/inference
    - Output: Character sequence
```

#### Why This Architecture?

1. **Spectrogram Input**:
   - Better than raw waveform for frequency-domain signal (CW is essentially tone on/off)
   - Mel-spectrogram focuses on perceptually relevant frequencies
   - Reduces input dimensionality

2. **CNN Layers**:
   - Extract local time-frequency features
   - Detect tone presence, transitions
   - Learn robust representations across noise conditions

3. **Bidirectional LSTM**:
   - Model temporal dependencies (dot/dash sequences)
   - Bidirectional context helps with timing ambiguity
   - Handle variable-length sequences

4. **CTC (Connectionist Temporal Classification)**:
   - No need for frame-level alignment labels
   - Automatically learns timing
   - Outputs variable-length sequences
   - Standard for speech recognition

#### Alternative Architectures to Consider

1. **Transformer Encoder** (instead of LSTM)
   - Better parallelization
   - Self-attention for long-range dependencies
   - May be overkill for Morse code

2. **Wav2Vec2 Style** (future enhancement)
   - Self-supervised pre-training on unlabeled audio
   - Fine-tune for CW decoding

3. **Multi-task Learning**
   - Auxiliary tasks: predict speed (WPM), frequency, SNR
   - May improve robustness

## 3. Synthetic Data Generation Strategy

### Morse Code Fundamentals

**Timing Units** (based on WPM):
- Dit (dot): 1 unit
- Dah (dash): 3 units
- Intra-character gap: 1 unit
- Inter-character gap: 3 units
- Inter-word gap: 7 units

**WPM to time conversion**:
- Standard: "PARIS" method (50 units = 1 word)
- Time per unit (ms) = 1200 / WPM
- Example: 20 WPM → 60ms per unit → dit = 60ms, dah = 180ms

### Generation Parameters (Randomized)

#### 1. **Speed Variation**
- Range: 5-40 WPM
- Distribution: Emphasis on 15-25 WPM (common range)
- Per-sample: Random WPM from distribution

#### 2. **Frequency/Tone**
- Range: 400-900 Hz
- Typical CW frequencies: 500-800 Hz
- Some samples with frequency drift/chirp

#### 3. **Timing Variance (Fist Characteristics)**
Real operators have imperfect timing:
- **Dit/Dah ratio**: Ideal 1:3, vary to 1:2.5 to 1:3.5
- **Inter-element gap**: Vary ±20-30% from ideal
- **Inter-character gap**: Vary ±20-30%
- **Human patterns**:
  - Rushed dits, stretched dahs (or vice versa)
  - Inconsistent spacing
  - Slight speed drift over time

#### 4. **Envelope Shaping**
- **Rise/Fall time**: 1-10ms (soft key to hard key)
- **ADSR envelope**: Attack-Decay-Sustain-Release
- **Key clicks**: Sharp rise times create clicks
- Optional: Simulate relay chatter

#### 5. **Noise Types and Levels**

**Additive Noise:**
- **White noise**: SNR from -5dB to +25dB
- **Pink noise**: 1/f spectrum (more natural)
- **Band-limited noise**: Filtered around signal frequency

**Interference:**
- **QRM** (man-made interference): Other CW signals nearby
  - Random CW at different frequency (±100-500Hz offset)
  - Random speed and content
  - Varying strength
- **QRN** (atmospheric noise):
  - Impulse noise (static crashes)
  - Sporadic, random amplitude

**Propagation Effects:**
- **Fading**:
  - Slow fading (QSB): 0.1-1 Hz modulation
  - Fast fading: Rayleigh/Rician
  - Amplitude variations
- **Frequency drift**:
  - Slight frequency instability (±5-20Hz)
  - Chirp on key-down

**Audio Artifacts:**
- **AGC pumping**: Automatic Gain Control artifacts
- **Filter ringing**: Narrow bandpass filter effects
- **Clipping**: Overdriven audio

### Data Generation Pipeline

```python
def generate_sample():
    1. Select random text (characters/words)
    2. Choose speed (WPM) from distribution
    3. Choose frequency from range
    4. Generate ideal timing sequence
    5. Apply human timing variance
    6. Generate audio tone with envelope
    7. Add frequency drift/chirp (optional)
    8. Select noise/interference levels
    9. Add white/pink noise
    10. Add QRM (other CW) if selected
    11. Add QRN (impulse noise) if selected
    12. Apply fading envelope if selected
    13. Normalize and return (audio, text_label)
```

### Content Generation

**Character Set:**
- Letters: A-Z
- Numbers: 0-9
- Common prosigns: AR, SK, BT, etc.
- Punctuation: . , ? / (limited)

**Text Sources:**
- Random character sequences (for learning alphabet)
- Random words from dictionary
- Ham radio callsigns (W1ABC, K2XYZ patterns)
- Common CW abbreviations (TNX, 73, 599, etc.)
- QSO phrases (CQ, DE, RST, etc.)
- Mix of lengths: single chars to full sentences

**Data Distribution:**
- 40% random characters (alphabet learning)
- 30% real words
- 20% callsigns and numbers
- 10% CW abbreviations/prosigns

### Dataset Size Strategy

**On-the-fly Generation:**
- Generate samples during training (infinite data)
- Each epoch sees new variations
- No storage requirements
- Configurable samples per epoch (e.g., 10,000)

**Advantages:**
- Unlimited variety
- No storage overhead
- Easy to adjust difficulty (curriculum learning)
- Good regularization (never see exact same sample twice)

## 4. Training Strategy

### Preprocessing

**Audio:**
- Sample rate: 16kHz (sufficient for CW, reduces computation)
- Window length: 25ms
- Hop length: 10ms
- FFT size: 512
- Mel filters: 64 (or 80)
- Frequency range: 0-8kHz

**Normalization:**
- Per-sample spectrogram normalization
- Mean subtraction, variance scaling

**Augmentation:**
- Time stretching (±10%)
- Pitch shift (±50Hz)
- Volume scaling

### Training Configuration

**Batch Size:**
- 32-64 (depending on sequence length)
- Dynamic batching by length (pack similar lengths)

**Optimizer:**
- Adam or AdamW
- Learning rate: 1e-3 with cosine annealing or ReduceLROnPlateau
- Weight decay: 1e-5

**Loss Function:**
- CTC Loss (built into PyTorch)
- Blank label for no-character states

**Training Duration:**
- 50-100 epochs
- Early stopping based on validation CER

### Curriculum Learning

Start easy, gradually increase difficulty:

**Phase 1 (Epochs 1-20): Easy**
- Clean signals (SNR > 15dB)
- Moderate speeds (15-25 WPM)
- Low timing variance (±10%)
- Single frequency

**Phase 2 (Epochs 21-50): Medium**
- More noise (SNR > 5dB)
- Full speed range (5-40 WPM)
- Higher timing variance (±20%)
- Frequency variation

**Phase 3 (Epochs 51+): Hard**
- All noise types including QRM/QRN
- Full difficulty range
- Fading
- Maximum timing variance (±30%)

### Validation Strategy

**Validation Set:**
- Fixed set of 1000+ samples with known content
- Covers full range of conditions
- Pre-generated (not on-the-fly)

**Test Sets:**
- **By Speed**: 5, 10, 15, 20, 25, 30, 35, 40 WPM
- **By SNR**: -5, 0, 5, 10, 15, 20, 25 dB
- **By Noise Type**: Clean, White, QRM, QRN, Fading
- **Real recordings** (if available)

## 5. Evaluation Metrics

### Primary Metrics

**Character Error Rate (CER):**
```
CER = (Substitutions + Insertions + Deletions) / Total_Characters
```
- Most important metric
- Target: < 5% at moderate SNR (>10dB), moderate speed (15-25 WPM)
- Target: < 10% at low SNR (0-10dB)

**Word Error Rate (WER):**
```
WER = (Word_Errors) / Total_Words
```
- Secondary metric
- More human-interpretable

### Analysis

**Breakdown by Condition:**
- CER vs WPM (performance at different speeds)
- CER vs SNR (robustness to noise)
- CER vs Timing Variance (handling sloppy sending)
- Per-character confusion matrix

**Visualization:**
- Attention plots (if using attention)
- Spectrogram with predictions overlaid
- Error distribution histograms

## 6. Model Export for iOS

### Export Pipeline

```
PyTorch Model (.pt)
    ↓
ONNX Format (.onnx)
    ↓
Core ML Model (.mlmodel)
```

### Export Process

1. **Trace Model** with fixed input size
2. **Export to ONNX** with opset 14+
3. **Convert to Core ML** using coremltools
4. **Optimize** for iOS (quantization, pruning)
5. **Test** on sample inputs

### iOS Integration Considerations

**Input Pipeline:**
- iOS app captures audio from microphone
- Buffer audio (e.g., 2-second sliding window)
- Compute spectrogram (use Accelerate framework)
- Feed to Core ML model
- Apply CTC beam search decoder

**Performance:**
- Model size target: < 10MB
- Inference time: < 100ms per window
- Consider quantization (float16 or int8)

**Deployment:**
- Core ML model embedded in app bundle
- Metal GPU acceleration if available
- Fallback to CPU

## 7. Project Structure

```
TrainCW/
├── README.md                 # Project overview
├── DESIGN.md                 # This document
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
│
├── configs/
│   ├── base_config.yaml     # Base configuration
│   ├── small_model.yaml     # Smaller model for testing
│   └── production.yaml      # Production model config
│
├── morse/
│   ├── __init__.py
│   ├── morse_code.py        # Morse code utilities (encoding/decoding)
│   └── timing.py            # WPM, timing calculations
│
├── data/
│   ├── __init__.py
│   ├── generator.py         # Synthetic data generation
│   ├── audio_synthesis.py   # Audio tone generation
│   ├── noise.py             # Noise/interference generation
│   ├── text_generator.py    # Text content generation
│   └── dataset.py           # PyTorch Dataset classes
│
├── models/
│   ├── __init__.py
│   ├── cnn_lstm_ctc.py      # Main model architecture
│   ├── encoder.py           # CNN encoder
│   ├── decoder.py           # CTC decoder utilities
│   └── transformer.py       # Alternative transformer model
│
├── training/
│   ├── __init__.py
│   ├── train.py             # Main training script
│   ├── trainer.py           # Trainer class
│   ├── curriculum.py        # Curriculum learning schedule
│   └── callbacks.py         # Training callbacks
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py          # Evaluation script
│   ├── metrics.py           # CER, WER calculations
│   └── visualize.py         # Result visualization
│
├── export/
│   ├── __init__.py
│   ├── to_onnx.py          # PyTorch → ONNX
│   ├── to_coreml.py        # ONNX → Core ML
│   └── test_exported.py    # Test exported models
│
├── utils/
│   ├── __init__.py
│   ├── audio.py            # Audio processing utilities
│   ├── config.py           # Configuration management
│   └── logger.py           # Logging utilities
│
├── tests/
│   ├── test_morse.py
│   ├── test_generator.py
│   ├── test_model.py
│   └── test_export.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_experiments.ipynb
│   └── 03_results_analysis.ipynb
│
├── scripts/
│   ├── generate_test_set.py
│   ├── benchmark.py
│   └── demo.py
│
└── checkpoints/             # Saved models
    └── .gitkeep
```

## 8. Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Set up project structure
- [ ] Implement Morse code utilities
- [ ] Basic audio tone generation
- [ ] Simple noise addition
- [ ] Basic data generator with PyTorch Dataset

### Phase 2: Data Generation (Week 1-2)
- [ ] Complete noise/interference types
- [ ] Timing variance implementation
- [ ] Text generation (callsigns, words, etc.)
- [ ] Validate generated samples (listen, visualize)
- [ ] Create fixed validation/test sets

### Phase 3: Model Development (Week 2-3)
- [ ] Implement CNN-LSTM-CTC architecture
- [ ] Training loop with CTC loss
- [ ] Basic evaluation (CER calculation)
- [ ] Tensorboard logging
- [ ] Checkpoint saving/loading

### Phase 4: Training & Tuning (Week 3-4)
- [ ] Curriculum learning implementation
- [ ] Hyperparameter tuning
- [ ] Train production model
- [ ] Comprehensive evaluation
- [ ] Error analysis

### Phase 5: Export & Deployment (Week 4-5)
- [ ] ONNX export
- [ ] Core ML conversion
- [ ] Quantization/optimization
- [ ] iOS integration testing
- [ ] Documentation for iOS team

## 9. Success Criteria

### Model Performance
- **CER < 5%** at SNR > 10dB, 15-25 WPM
- **CER < 10%** at SNR > 5dB, 10-30 WPM
- **CER < 20%** at SNR 0-5dB (challenging conditions)
- Graceful degradation with difficulty

### Deployment
- Model size < 10MB
- Inference time < 100ms on iPhone (iPhone 12+)
- Successfully runs in Core ML on iOS

### Robustness
- Handles 5-40 WPM range
- Robust to ±30% timing variance
- Works with various noise types
- Handles frequency range 400-900 Hz

## 10. Future Enhancements

### Short-term
- Attention mechanism visualization
- Multi-task learning (speed/frequency prediction)
- Real-time streaming inference
- Language model integration (word-level correction)

### Long-term
- Transfer learning from real recordings
- Adaptive learning (personalization to operator's fist)
- Multi-signal decoding (multiple CW signals simultaneously)
- Prosign detection (AR, SK, BT, etc.)
- QRS/RST automated exchange recognition

## 11. References & Resources

### Morse Code
- [ARRL Learning Morse Code](http://www.arrl.org/learning-morse-code)
- ITU-R M.1677-1 (International Morse Code)
- PARIS method for WPM calculation

### Neural Networks for Audio
- "Connectionist Temporal Classification" (Graves et al., 2006)
- Wav2Vec 2.0 (Baevski et al., 2020)
- Deep Speech 2 (Amodei et al., 2015)

### PyTorch Resources
- PyTorch CTC Loss documentation
- Torchaudio transforms

### iOS Deployment
- Apple Core ML documentation
- coremltools documentation

## 12. Development Environment

### Hardware Requirements
- CPU: Modern multi-core (8+ cores recommended)
- RAM: 64GB available (will use ~8-16GB during training)
- Storage: ~50GB for code, checkpoints, logs
- GPU: Not required (CPU training specified)

### Software Requirements
- Python 3.10+
- PyTorch (CPU version)
- Standard ML stack

### Training Time Estimates
- Single epoch (10k samples): ~10-20 minutes on modern CPU
- Full training (100 epochs): ~1-2 days
- Can be interrupted and resumed (checkpoint saving)

---

## Getting Started

Once implementation begins:

1. Install dependencies: `pip install -r requirements.txt`
2. Run tests: `pytest tests/`
3. Generate sample data: `python scripts/generate_test_set.py`
4. Train model: `python training/train.py --config configs/base_config.yaml`
5. Evaluate: `python evaluation/evaluate.py --checkpoint checkpoints/best.pt`
6. Export: `python export/to_coreml.py --checkpoint checkpoints/best.pt`

Let's build a robust CW decoder! 73 de AI 🎧
