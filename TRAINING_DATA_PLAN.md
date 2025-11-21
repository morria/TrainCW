# Training Data Generation Plan for CW Neural Network

## Executive Summary

This document specifies the exact parameters, quantities, and procedures for generating synthetic training data for the Morse code (CW) neural network decoder. It provides concrete numbers for all training parameters and a detailed execution plan.

**Target Performance:**
- Character Error Rate (CER) < 5% at SNR > 10dB, 15-25 WPM
- CER < 10% at SNR > 5dB, 10-30 WPM
- Robust operation across real-world HF radio conditions

---

## 1. Training Data Parameters

### 1.1 Frequency Range

**Why frequency variation matters:** Real-world CW operates across amateur radio bands with receivers tuned to different beat frequencies. The model must recognize CW regardless of tone pitch.

**Configuration:**
- **Frequency Range:** 400-900 Hz
- **Distribution:**
  - 60% in sweet spot: 500-800 Hz
  - 20% low range: 400-500 Hz
  - 20% high range: 800-900 Hz
- **Selection Method:** Random sampling with weighted distribution

**Frequency Drift:**
- **Occurrence:** 30% of samples
- **Amount:** ±5 to ±20 Hz linear drift over sample duration
- **Purpose:** Simulates VFO instability, Doppler effects

**Key Chirp:**
- **Occurrence:** 15% of samples
- **Amount:** 10-30 Hz upward chirp on key-down (first 5-10ms)
- **Purpose:** Simulates transmitter instability on keying

**Frequency Stability Modes:**
- 70% stable frequency (modern rigs)
- 20% slow drift (older tube rigs, thermal drift)
- 10% chirp (unstable oscillators)

---

### 1.2 Speed (WPM) Range

**Why speed variation matters:** Operators send at vastly different speeds. Beginners use 5-10 WPM, typical QSOs are 15-25 WPM, and high-speed contest operators reach 35-40+ WPM.

**Configuration:**
- **Speed Range:** 5-40 WPM
- **Distribution (samples per epoch):**
  - 10% at 5-10 WPM (beginners)
  - 30% at 10-15 WPM (slow)
  - 40% at 15-25 WPM (typical) ← **Primary focus**
  - 15% at 25-35 WPM (fast)
  - 5% at 35-40 WPM (very fast)

**Speed Drift:**
- **Occurrence:** 25% of samples
- **Amount:** ±2-5% speed change over sample duration
- **Purpose:** Simulates operator fatigue, rhythm changes

**Farnsworth Timing:**
- **Not implemented initially** (can add later)
- Standard timing: Element spacing proportional to WPM
- Farnsworth: Characters sent fast, but with extra spacing

---

### 1.3 Timing Variance (Operator "Fist")

**Why timing variance matters:** Human operators don't send with perfect timing. Each operator has unique characteristics (their "fist"). The model must handle imperfect timing.

**Dit/Dah Ratio:**
- **Ideal:** 1:3
- **Actual range:** 1:2.3 to 1:3.7
- **Distribution:** Normal distribution centered at 1:3, σ=0.2

**Inter-element Gap (within characters):**
- **Ideal:** 1 time unit
- **Variance:** ±0% to ±35%
- **Distribution by sample:**
  - 40% tight timing: ±0-10%
  - 40% moderate variance: ±10-25%
  - 20% sloppy timing: ±25-35%

**Inter-character Gap:**
- **Ideal:** 3 time units
- **Variance:** ±0% to ±30%
- **Often compressed:** Rushed operators reduce this

**Inter-word Gap:**
- **Ideal:** 7 time units
- **Variance:** ±0% to ±25%
- **More stable:** Usually more consistent than other gaps

**Operator Style Modes:**
- 50% clean sender (±10% variance)
- 30% typical operator (±20% variance)
- 15% rushed/sloppy (±30% variance, compressed gaps)
- 5% stretched style (longer dahs, wider spacing)

---

### 1.4 Noise Levels

**Why noise variation matters:** HF propagation creates highly variable SNR conditions. Operators must decode through noise, QRM, QRN, and fading.

**Signal-to-Noise Ratio (SNR) Distribution:**

| SNR Range | Condition | % of Samples | Real-World Scenario |
|-----------|-----------|--------------|---------------------|
| 25-30 dB | Excellent | 10% | Strong local signal |
| 20-25 dB | Very Good | 15% | Good propagation |
| 15-20 dB | Good | 20% | Typical QSO |
| 10-15 dB | Fair | 25% | Moderate conditions |
| 5-10 dB | Poor | 15% | Weak signal |
| 0-5 dB | Very Poor | 10% | Near noise floor |
| -5-0 dB | Barely Readable | 5% | Extreme conditions |

**Noise Types and Probabilities:**

1. **White Noise (AWGN):**
   - **Present in:** 100% of samples
   - **Purpose:** Base noise floor
   - **Characteristics:** Flat frequency spectrum

2. **Pink Noise (1/f):**
   - **Present in:** 40% of samples
   - **Purpose:** More natural atmospheric noise
   - **Characteristics:** More energy at low frequencies

3. **Band-limited Noise:**
   - **Present in:** 30% of samples
   - **Bandwidth:** ±200-500 Hz around signal
   - **Purpose:** Filtered receiver noise

**Noise Application:**
- Each sample gets white noise at specified SNR
- Additional noise types added on top (further degrading SNR)
- Combined noise can reduce effective SNR by 3-6 dB

---

### 1.5 Interference Types

**Why interference matters:** Real HF bands are crowded. Multiple stations, atmospheric noise, and propagation effects create interference that the model must handle.

#### 1.5.1 QRM (Man-Made Interference)

**Other CW Signals:**
- **Occurrence:** 25% of training samples
- **Configuration:**
  - **Number of interfering signals:** 1-2 (85% one, 15% two)
  - **Frequency offset:** ±100 to ±800 Hz from main signal
  - **Minimum separation:** 100 Hz (too close = severe QRM)
  - **Strength:** -15 dB to +5 dB relative to main signal
    - 50%: -15 to -5 dB (weaker than main signal)
    - 30%: -5 to 0 dB (comparable strength)
    - 20%: 0 to +5 dB (stronger than main signal)
  - **Speed:** Random, independent of main signal (5-40 WPM)
  - **Content:** Random callsigns and text

**Purpose:** Teach model to focus on target frequency and ignore nearby signals

#### 1.5.2 QRN (Atmospheric/Natural Noise)

**Impulse Noise (Static Crashes):**
- **Occurrence:** 20% of training samples
- **Configuration:**
  - **Rate:** 0.5-3 impulses per second
  - **Duration:** 10-50 ms per impulse
  - **Amplitude:** 10-30 dB above signal level
  - **Shape:** Sharp attack, exponential decay
  - **Spectrum:** Broadband (simulates lightning static)

**Purpose:** Handle sudden interference bursts common on HF

#### 1.5.3 Propagation Effects

**Fading (QSB):**
- **Occurrence:** 30% of training samples
- **Types:**
  - **Slow Fading:** 80% of fading samples
    - Rate: 0.1-1 Hz modulation
    - Depth: 6-20 dB fade depth
    - Waveform: Sinusoidal or Rayleigh
  - **Fast Fading:** 20% of fading samples
    - Rate: 1-5 Hz modulation
    - Depth: 3-10 dB fade depth
    - Models: Rayleigh/Rician

**Purpose:** Simulate ionospheric propagation effects

#### 1.5.4 Audio Processing Artifacts

**AGC Pumping:**
- **Occurrence:** 15% of samples
- **Configuration:**
  - Attack time: 10-50 ms
  - Release time: 100-500 ms
  - Effect: Volume pumping between elements

**Clipping:**
- **Occurrence:** 10% of samples
- **Severity:** 0-20% of samples clipped
- **Purpose:** Overdriven audio stages

**Filter Ringing:**
- **Occurrence:** 20% of samples
- **Q factor:** 10-50 (narrow filters)
- **Purpose:** Narrow CW filter artifacts

---

### 1.6 Signal Envelope and Keying

**Why envelope matters:** Different transmitters and keys produce different signal shapes. Hard vs soft keying affects the audio signature.

**Rise/Fall Times:**
- **Distribution:**
  - 40%: Fast keying (1-3 ms) - modern rigs, bugs
  - 40%: Medium keying (3-7 ms) - typical hand keys
  - 20%: Soft keying (7-15 ms) - shaped keying to reduce clicks

**Envelope Shape:**
- 60%: Linear rise/fall
- 30%: Raised cosine (smoother)
- 10%: ADSR envelope (Attack-Decay-Sustain-Release)

**Key Clicks:**
- **Occurrence:** 10% of samples
- **Cause:** Very fast rise times (< 1ms)
- **Effect:** Spectral splatter, broadband clicks

---

## 2. Dataset Composition

### 2.1 Content Types

**Why varied content matters:** Model must decode everything from random practice code to real QSO conversations and callsigns.

**Character Set:**
- **Letters:** A-Z (26 characters)
- **Numbers:** 0-9 (10 characters)
- **Punctuation:** . , ? / (4 characters)
- **Prosigns:** AR SK BT KN (4 common prosigns)
- **Total vocabulary:** ~44 classes + blank (for CTC)

**Content Distribution (per epoch):**

| Content Type | Percentage | Purpose | Example |
|--------------|------------|---------|---------|
| Random Characters | 30% | Alphabet learning | "QRZT MFPW XJKL" |
| Random Words | 25% | Word context | "RADIO WORK SIGNAL GOING" |
| Callsigns | 20% | Ham patterns | "W1ABC K2XYZ DE N0CALL" |
| Numbers/Reports | 10% | Signal reports | "599 RST 73 5NN" |
| CW Abbreviations | 10% | Ham language | "TNX QSL QRZ OM" |
| QSO Phrases | 5% | Real exchanges | "CQ CQ DE W1ABC PSE K" |

**Length Distribution (constrained to 2-second windows for iOS deployment):**
- 10%: Very short (1-5 chars, < 0.5 sec)
- 30%: Short (6-10 chars, 0.5-1 sec)
- 40%: Medium (11-20 chars, 1-2 sec)
- 20%: Full length (21-30 chars, ~2 sec max)
- **Note:** Sample duration varies by WPM. At 20 WPM, 20 chars ≈ 2 seconds. No samples exceed 2 seconds to match iOS deployment window.

**Text Sources:**
- **Dictionary:** Common English words (5000 word list)
- **Callsign Generator:** Pattern-based (W/K/N/A prefix + number + suffix)
- **Q-code/Abbreviations:** Standard ham radio abbreviations
- **Random Generator:** Uniform character distribution for alphabet learning

---

### 2.2 Sample Counts

**Generation Strategy: On-the-fly (Infinite Data)**

Benefits:
- No storage requirements
- Unlimited variety
- Never see exact same sample twice (regularization)
- Easy curriculum learning (adjust difficulty per epoch)

**Samples Per Epoch:** 10,000
- Each epoch generates fresh samples
- Total unique combinations: ~10^20+ (effectively infinite)

**Training Duration:** 100 epochs
- **Total training samples:** ~1,000,000
- **Training time estimate:** 30-40 hours on modern CPU

---

### 2.3 Validation Set

**Purpose:** Fixed set to measure progress consistently across epochs

**Composition:**
- **Size:** 2,000 samples (pre-generated, fixed)
- **Storage:** ~500 MB (audio + labels)
- **Regeneration:** Never (same validation set throughout training)

**Content Distribution:**
- Matches training distribution
- Full range of speeds, SNR, interference
- Stratified sampling to ensure coverage

**Stratification:**
- 8 speed buckets × 7 SNR levels = 56 condition pairs
- ~35 samples per condition pair
- Plus edge cases (extreme fading, heavy QRM, etc.)

---

### 2.4 Test Sets

**Purpose:** Comprehensive evaluation under controlled conditions

#### Test Set 1: Speed Sweep (Clean Signals)
- **Speeds:** 5, 10, 15, 20, 25, 30, 35, 40 WPM
- **Samples per speed:** 100
- **Conditions:** Clean (SNR 25 dB, no interference, stable frequency)
- **Total:** 800 samples

#### Test Set 2: SNR Sweep (Moderate Speed)
- **SNR levels:** -5, 0, 5, 10, 15, 20, 25, 30 dB
- **Samples per SNR:** 100
- **Conditions:** 20 WPM, white noise only, clean timing
- **Total:** 800 samples

#### Test Set 3: Interference Types (Moderate Conditions)
- **Types:** Clean, White Noise, QRM, QRN, Fading, Combined
- **Samples per type:** 150
- **Conditions:** 20 WPM, 12 dB SNR
- **Total:** 900 samples

#### Test Set 4: Timing Variance
- **Variance levels:** Clean (±5%), Moderate (±15%), Sloppy (±30%)
- **Samples per level:** 100
- **Conditions:** 20 WPM, 15 dB SNR
- **Total:** 300 samples

#### Test Set 5: Combined Stress Test
- **Scenarios:** Realistic difficult combinations
  - Fast + Low SNR (35 WPM, 5 dB SNR)
  - Slow + Heavy QRM (10 WPM, +3dB QRM)
  - Fading + Sloppy timing
  - Etc. (20 scenarios × 50 samples each)
- **Total:** 1,000 samples

#### Test Set 6: Real Recordings (Future)
- **Source:** Actual ham radio recordings (when available)
- **Size:** TBD
- **Purpose:** Validate on real-world data

**Total Test Samples:** ~3,800 pre-generated samples

---

## 3. Data Generation Pipeline

### 3.1 Sample Generation Algorithm

```python
def generate_training_sample():
    """
    Generate one training sample with randomized parameters.
    Returns: (audio_waveform, text_label, metadata)
    """

    # 1. Generate content
    content_type = random.choice(CONTENT_TYPES, weights=CONTENT_WEIGHTS)
    text = generate_text(content_type, length=random_length())

    # 2. Select basic parameters
    wpm = sample_from_wpm_distribution()  # 5-40 WPM, weighted
    frequency = sample_from_frequency_distribution()  # 400-900 Hz
    sample_rate = 16000  # Fixed

    # 3. Generate timing sequence
    timing = morse_to_timing(text, wpm)

    # 4. Apply human timing variance
    timing_variance = select_operator_style()  # ±0-35%
    timing = apply_timing_variance(timing, timing_variance)

    # 5. Generate base audio tone
    audio = generate_cw_tone(
        timing=timing,
        frequency=frequency,
        sample_rate=sample_rate,
        envelope=select_envelope_type(),
        rise_time=sample_rise_time(),
        fall_time=sample_fall_time()
    )

    # 6. Apply frequency effects
    if random.random() < 0.30:  # 30% have drift
        audio = apply_frequency_drift(audio, amount=random_drift())
    if random.random() < 0.15:  # 15% have chirp
        audio = apply_key_chirp(audio, amount=random_chirp())

    # 7. Select noise level
    snr_db = sample_from_snr_distribution()  # -5 to 30 dB

    # 8. Add base noise
    noise_type = select_base_noise_type()  # White/Pink/Band-limited
    audio = add_noise(audio, snr_db, noise_type)

    # 9. Add interference (probabilistic)
    if random.random() < 0.25:  # 25% have QRM
        qrm_signal = generate_qrm(
            num_signals=1 or 2,
            frequency_offset=random_offset(),
            strength=random_strength()
        )
        audio = mix_signals(audio, qrm_signal)

    if random.random() < 0.20:  # 20% have QRN
        impulses = generate_qrn_impulses(
            duration=len(audio)/sample_rate,
            rate=random_impulse_rate()
        )
        audio = add_impulse_noise(audio, impulses)

    # 10. Apply propagation effects
    if random.random() < 0.30:  # 30% have fading
        fading_envelope = generate_fading(
            duration=len(audio)/sample_rate,
            fading_type=select_fading_type()
        )
        audio = apply_fading(audio, fading_envelope)

    # 11. Apply audio artifacts
    if random.random() < 0.15:  # 15% have AGC pumping
        audio = apply_agc_pumping(audio)

    if random.random() < 0.10:  # 10% have clipping
        audio = apply_clipping(audio, clip_level=random_clip())

    if random.random() < 0.20:  # 20% have filter ringing
        audio = apply_bandpass_filter(audio, q=random_q_factor())

    # 12. Normalize
    audio = normalize_audio(audio, target_rms=-20dB)

    # 13. Package metadata
    metadata = {
        'text': text,
        'wpm': wpm,
        'frequency': frequency,
        'snr_db': snr_db,
        'timing_variance': timing_variance,
        'has_qrm': has_qrm,
        'has_qrn': has_qrn,
        'has_fading': has_fading,
        # ... other parameters
    }

    return audio, text, metadata
```

### 3.2 Batch Generation

```python
class CWDataset(torch.utils.data.IterableDataset):
    """
    On-the-fly CW data generation.
    Infinite dataset - each epoch generates fresh samples.
    """

    def __init__(self, samples_per_epoch=10000, curriculum_phase=1):
        self.samples_per_epoch = samples_per_epoch
        self.curriculum_phase = curriculum_phase

    def __iter__(self):
        for _ in range(self.samples_per_epoch):
            # Generate sample with curriculum constraints
            audio, text, metadata = generate_training_sample()

            # Apply curriculum phase constraints
            if self.curriculum_phase == 1:  # Easy
                # Regenerate if too difficult
                while metadata['snr_db'] < 15 or metadata['wpm'] > 25:
                    audio, text, metadata = generate_training_sample()

            # Compute spectrogram
            spectrogram = compute_mel_spectrogram(audio)

            # Encode text as indices
            encoded_text = text_to_indices(text)

            yield spectrogram, encoded_text, len(encoded_text)
```

---

## 4. Curriculum Learning Strategy

**Why curriculum learning:** Starting with easy examples and gradually increasing difficulty helps the model learn more effectively and converge faster.

### 4.1 Phase 1: Foundation (Epochs 1-30)

**Goal:** Learn basic alphabet and timing patterns

**Constraints:**
- **Speed:** 12-25 WPM (moderate only)
- **SNR:** 15-25 dB (clean signals)
- **Timing variance:** ±0-15% (clean to moderate)
- **Frequency:** 500-800 Hz (sweet spot)
- **Interference:** None (pure signals)
- **Fading:** None
- **Content:** 50% random characters (alphabet focus)

**Expected CER by end of Phase 1:** < 10% on clean validation samples

### 4.2 Phase 2: Expansion (Epochs 31-60)

**Goal:** Handle wider range of speeds and moderate noise

**Constraints:**
- **Speed:** 8-35 WPM (wider range)
- **SNR:** 10-25 dB (introduce moderate noise)
- **Timing variance:** ±0-25% (include sloppy timing)
- **Frequency:** 400-900 Hz (full range)
- **Interference:**
  - 15% QRM (lighter than full training)
  - 10% QRN
  - No fading yet
- **Content:** Return to normal distribution

**Expected CER by end of Phase 2:** < 7% on moderate validation samples

### 4.3 Phase 3: Mastery (Epochs 61-100)

**Goal:** Handle all real-world conditions

**Constraints:**
- **All parameters at full range**
- **Full interference:** 25% QRM, 20% QRN, 30% fading
- **Extreme cases included:** Very low SNR, extreme speeds
- **Audio artifacts:** All types enabled

**Expected CER by end of Phase 3:**
- < 5% on good conditions (SNR > 10 dB, 15-25 WPM)
- < 10% on poor conditions (SNR 5-10 dB)
- < 20% on extreme conditions (SNR 0-5 dB)

---

## 5. Training Execution Plan

### 5.1 Hardware and Software Setup

**Hardware:**
- **CPU:** 8+ cores (modern x86_64)
- **RAM:** 16 GB minimum, 32 GB recommended
- **Storage:** 100 GB for code, checkpoints, logs, test sets
- **GPU:** Not required (CPU training specified)

**Software Stack:**
```
Python 3.10+
torch==2.1.0 (CPU version)
torchaudio==2.1.0
librosa==0.10.1
scipy==1.11.0
numpy==1.26.0
soundfile==0.12.1
matplotlib==3.8.0
tensorboard==2.15.0
tqdm==4.66.0
pyyaml==6.0
```

### 5.2 Training Configuration

**Model Architecture (Optimized for iOS):**
- **Input:** Mel-spectrogram (Time × 64 Mel bins)
- **CNN:** 4 Conv2D layers → [Batch, Time', 256]
  - Depthwise separable convolutions where possible (lower ops)
  - Channel count optimized for Metal GPU
- **LSTM:** 2-layer Bi-LSTM, hidden size 256 → [Batch, Time', 512]
  - Potentially replace with 1D Conv or GRU for faster iOS inference
  - Bi-LSTM may be replaced with uni-directional for streaming
- **FC:** Linear → [Batch, Time', 45] (44 chars + blank)
- **Total parameters:** ~2-3 million
  - **Target model size:** < 5 MB (float32) or < 2.5 MB (float16)
  - **iOS optimized:** Quantization-aware training considered

**Hyperparameters:**
- **Batch size:** 32
- **Optimizer:** AdamW
- **Learning rate:** 1e-3 with cosine annealing
- **Weight decay:** 1e-5
- **Gradient clipping:** Max norm 1.0
- **Loss:** CTC Loss

**Audio Preprocessing (iOS-Compatible):**
- **Sample rate:** 16 kHz (native iOS microphone rate)
- **Window:** 25 ms (400 samples)
- **Hop:** 10 ms (160 samples)
- **FFT size:** 512 (power of 2, efficient for Accelerate framework)
- **Mel bins:** 64 (balance between accuracy and computation)
- **Freq range:** 0-8000 Hz
- **Note:** Spectrogram computation on iOS uses vDSP/Accelerate framework

### 5.3 Training Schedule

**Total Duration:** 100 epochs

**Time Estimates:**
- **Per epoch:** 20-25 minutes (10k samples, batch 32, CPU)
- **Total training time:** ~35-42 hours (~1.5-2 days)

**Checkpointing:**
- Save every 5 epochs
- Save on validation improvement
- Keep best 3 checkpoints

**Validation:**
- Run every epoch
- Use fixed 2,000 sample validation set
- Monitor CER and loss

**Early Stopping:**
- Patience: 15 epochs
- Monitor: Validation CER
- Minimum epochs: 50 (ensure full curriculum)

### 5.4 Training Monitoring

**TensorBoard Logging:**
- Training loss (per batch, per epoch)
- Validation loss (per epoch)
- Validation CER (per epoch)
- Learning rate
- Sample spectrograms with predictions
- Character-wise confusion matrix
- CER breakdown by WPM, SNR

**Console Output:**
```
Epoch 1/100 [Phase 1: Foundation]
Train: 100% |████████████| 10000/10000 [20:15<00:00, 8.23 samples/s]
Train Loss: 45.23, Train CER: 35.2%
Val Loss: 42.11, Val CER: 32.5%
Best model saved! (CER improved from inf to 32.5%)
```

### 5.5 Evaluation Protocol

**During Training:**
- Validation CER every epoch
- Quick test on speed/SNR sweep every 10 epochs

**After Training:**
- Run full test suite (3,800 samples)
- Generate evaluation report:
  - CER vs WPM plot
  - CER vs SNR plot
  - Per-character confusion matrix
  - Error analysis by condition
  - Sample predictions with spectrograms

**Success Criteria:**
- ✓ Validation CER < 5%
- ✓ Test CER < 5% at SNR > 10dB, 15-25 WPM
- ✓ Test CER < 10% at SNR > 5dB, 10-30 WPM
- ✓ No catastrophic failures on any single character
- ✓ Graceful degradation with increasing difficulty

---

## 6. Data Generation Implementation

### 6.1 Pre-Generation vs On-the-Fly

**Decision: On-the-Fly Generation**

**Rationale:**
- **Storage:** 1M samples × 2 seconds × 16kHz × 2 bytes = ~64 GB
- **Variety:** Infinite variations prevent overfitting
- **Flexibility:** Easy to adjust curriculum mid-training
- **Augmentation:** New samples every epoch = natural augmentation

**Trade-offs:**
- Slower training (generation overhead)
- Non-reproducible training runs (different samples each time)
- Can't inspect full dataset upfront

**Mitigation:**
- Fixed validation/test sets for reproducibility
- Seed control for debugging
- Pre-generate small sample sets for inspection

### 6.2 Generation Performance

**Target:** Generate samples faster than model trains

**Bottlenecks:**
1. Audio tone synthesis
2. Noise generation
3. Spectrogram computation

**Optimizations:**
- NumPy vectorization
- Pre-computed lookup tables (sine waves)
- Cached noise buffers
- Efficient filtering (scipy)

**Expected Performance:**
- Single sample: 10-30 ms
- Batch of 32: 0.5-1 second
- Training consumes ~5-10 samples/sec
- Generation rate: ~30-100 samples/sec
- **Conclusion:** Generation won't bottleneck training

### 6.3 Validation Set Generation

**When:** Before training begins

**Process:**
```bash
python scripts/generate_test_set.py --type validation --count 2000 --output data/validation.pkl
```

**Stratification:**
```python
# Ensure coverage of all conditions
validation_samples = []
for wpm in [5, 10, 15, 20, 25, 30, 35, 40]:  # 8 speeds
    for snr in [-5, 0, 5, 10, 15, 20, 25]:   # 7 SNR levels
        # Generate ~35 samples per combination
        for _ in range(35):
            params = {
                'wpm': wpm + random.uniform(-2, 2),
                'snr_db': snr + random.uniform(-2, 2),
                # ... other randomized params
            }
            sample = generate_sample_with_params(params)
            validation_samples.append(sample)

# Total: 8 × 7 × 35 ≈ 1960 samples
# Plus ~40 edge cases = 2000 samples
```

### 6.4 Test Set Generation

**When:** Before training begins

**Process:**
```bash
python scripts/generate_test_set.py --type test_speed --count 800 --output data/test_speed.pkl
python scripts/generate_test_set.py --type test_snr --count 800 --output data/test_snr.pkl
python scripts/generate_test_set.py --type test_interference --count 900 --output data/test_interference.pkl
python scripts/generate_test_set.py --type test_timing --count 300 --output data/test_timing.pkl
python scripts/generate_test_set.py --type test_stress --count 1000 --output data/test_stress.pkl
```

**Storage:** ~1 GB total (compressed)

---

## 7. Expected Results

### 7.1 Learning Curves

**Phase 1 (Epochs 1-30):**
- Validation loss: Rapid decrease 50 → 10
- Validation CER: 50% → 10%
- Behavior: Learning alphabet, basic timing

**Phase 2 (Epochs 31-60):**
- Validation loss: Steady decrease 10 → 5
- Validation CER: 10% → 7%
- Behavior: Generalizing to noise, speed variance

**Phase 3 (Epochs 61-100):**
- Validation loss: Slow decrease 5 → 3
- Validation CER: 7% → 4-5%
- Behavior: Mastering difficult conditions

### 7.2 Final Performance Targets

**Clean Signals (SNR > 20 dB):**
- 15-25 WPM: CER < 2%
- 10-30 WPM: CER < 3%
- 5-35 WPM: CER < 5%

**Moderate Noise (SNR 10-20 dB):**
- 15-25 WPM: CER < 5%
- 10-30 WPM: CER < 8%

**Poor Conditions (SNR 5-10 dB):**
- 15-25 WPM: CER < 10%
- With QRM/QRN: CER < 15%

**Extreme Conditions (SNR 0-5 dB):**
- CER < 20% (readable but with errors)
- Human operators struggle here too

### 7.3 Failure Modes to Watch

**Common Errors:**
- E/T confusion (single dit vs dit-dit-dit)
- S/H confusion (similar patterns)
- Number confusion (timing-sensitive)
- Dropped characters in fades
- False detections in heavy QRM

**Mitigation:**
- More training on confused pairs
- Character-balanced training
- Fade-specific augmentation
- Better frequency discrimination (CNN depth)

---

## 8. Reproducibility and Logging

### 8.1 Random Seeds

**Problem:** On-the-fly generation is non-deterministic

**Solution:**
- Set global seed at training start
- Seed per epoch: `seed = base_seed + epoch`
- Validation/test sets: Fixed seeds

```python
# Training script
random.seed(42 + epoch)
np.random.seed(42 + epoch)
torch.manual_seed(42 + epoch)
```

### 8.2 Experiment Tracking

**Directory Structure:**
```
checkpoints/
└── experiment_20250121_143022/
    ├── config.yaml           # Full configuration
    ├── params.json           # Data generation params
    ├── model_epoch_050.pt    # Checkpoints
    ├── model_best.pt         # Best model
    ├── training_log.txt      # Console output
    ├── tensorboard/          # TensorBoard logs
    └── evaluation/
        ├── test_results.json
        ├── confusion_matrix.png
        └── cer_vs_snr.png
```

**Config Versioning:**
- All parameters saved to `config.yaml`
- Git commit hash embedded in checkpoint
- Data generation code frozen (or versioned)

---

## 9. Risk Mitigation

### 9.1 Potential Issues

**Issue 1: Model doesn't learn (CER stuck > 30%)**
- **Causes:** Architecture too small, learning rate wrong, CTC label issues
- **Diagnosis:** Check CTC decoder output, visualize predictions
- **Fix:** Increase model size, tune LR, verify label encoding

**Issue 2: Overfits to clean signals**
- **Causes:** Curriculum too aggressive, not enough noise
- **Diagnosis:** Train CER << Val CER, fails on noise
- **Fix:** Extend Phase 1, increase noise earlier

**Issue 3: Can't handle extreme speeds**
- **Causes:** Not enough samples, architecture limitations
- **Diagnosis:** CER high only at 5 or 40 WPM
- **Fix:** Increase training samples at extremes, longer sequences

**Issue 4: Confused by QRM**
- **Causes:** Frequency discrimination too weak
- **Diagnosis:** CER jumps with QRM present
- **Fix:** More CNN filters, narrower frequency resolution, more QRM training

**Issue 5: Training too slow**
- **Causes:** Data generation bottleneck
- **Diagnosis:** GPU/CPU idle while generating
- **Fix:** Multi-process data loading, pre-compute spectrograms, optimize synthesis

### 9.2 Contingency Plans

**Plan A: Baseline (Current Plan)**
- 100 epochs, full curriculum, on-the-fly generation

**Plan B: Pre-generated Dataset**
- If generation too slow: Pre-generate 100k samples
- Trade storage for speed
- Still do validation/test as planned

**Plan C: Reduced Complexity**
- If training stalls: Simplify data generation
- Remove rare effects (fading, chirp, etc.)
- Focus on core: speed, SNR, basic QRM

**Plan D: Transfer Learning**
- If from-scratch fails: Pre-train on speech dataset
- Fine-tune on CW (fewer parameters)
- Leverage existing audio models

---

## 10. Post-Training Validation

### 10.1 Comprehensive Test Suite

**Run all test sets:**
```bash
python evaluation/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --test-sets data/test_*.pkl \
    --output evaluation_report.pdf
```

**Generate report with:**
- Overall CER and WER
- CER breakdown by speed, SNR, interference type
- Per-character accuracy
- Confusion matrices
- Sample predictions with audio/spectrogram
- Error analysis

### 10.2 Qualitative Testing

**Manual listening test:**
- Generate 20-30 samples across conditions
- Listen to audio, read label, check prediction
- Verify "sounds reasonable" to human

**Edge case testing:**
- Extreme speeds (3 WPM, 50 WPM)
- Multiple prosigns in sequence
- Very long messages (100+ chars)
- Silence periods
- Frequency at band edges (400, 900 Hz)

### 10.3 Real-World Validation

**If available:**
- Test on real ham radio recordings
- Record from WebSDR, actual radios
- Compare to human copy and other decoders
- Collect failure cases for future training

---

## 11. Summary: Key Numbers

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Frequencies** | 400-900 Hz, focus 500-800 Hz | Covers typical CW receiver tones |
| **Speeds (WPM)** | 5-40, focus 15-25 | Beginner to expert range |
| **SNR Range** | -5 to +30 dB | Near noise floor to excellent |
| **Training samples/epoch** | 10,000 | Balance variety and training time |
| **Total epochs** | 100 | Enough for curriculum + convergence |
| **Validation set** | 2,000 samples | Fixed, stratified coverage |
| **Test sets** | 3,800 samples | Comprehensive condition sweep |
| **QRM occurrence** | 25% | Common on HF bands |
| **QRN occurrence** | 20% | Atmospheric noise |
| **Fading occurrence** | 30% | HF propagation |
| **Timing variance** | ±0-35% | Human operator range |
| **Training time** | 35-42 hours | ~2 days on modern CPU |
| **Model size** | 2-3M params, < 5 MB | Deployable to iOS (< 2.5 MB with float16) |
| **iOS inference target** | < 100ms per window | Real-time decoding on iPhone 12+ |
| **Sample duration** | Max 2 seconds | Matches iOS deployment window |

---

## 12. iOS Deployment Constraints and Optimization

### 12.1 iPhone Hardware Capabilities

**Target Devices:** iPhone 12 and newer (A14 Bionic+)

**Hardware Specs:**
- **CPU:** 6-core (2 performance + 4 efficiency)
- **GPU:** 4-core Apple GPU (Metal)
- **Neural Engine:** 16-core (11+ TOPS)
- **RAM:** 4-6 GB (shared with OS and apps)
- **Available memory:** ~2-3 GB for app peak usage

**Core ML Performance:**
- **Neural Engine:** Preferred for ML inference (fastest, most efficient)
- **Metal GPU:** Fallback for operations not supported by ANE
- **CPU:** Last resort (slowest, highest power consumption)

### 12.2 Model Size Constraints

**Disk Space:**
- **App bundle limit:** 4 GB uncompressed
- **Model target:** < 5 MB uncompressed (float32)
- **With quantization:** < 2.5 MB (float16) or < 1.5 MB (int8)
- **Multiple models:** Can ship several versions (< 20 MB total)

**Memory Constraints:**
- **Model weights in RAM:** 5 MB (float32) or 2.5 MB (float16)
- **Activation memory:** 10-50 MB depending on input size
- **Total budget:** < 100 MB for ML inference
- **Critical:** Avoid memory pressure (iOS will kill app)

**Impact on Training:**
- ✓ Keep model compact (2-3M params confirmed)
- ✓ Train with quantization-aware training (QAT)
- ✓ Prune unnecessary parameters
- ✗ Don't add more layers without justification

### 12.3 Inference Speed Requirements

**Real-Time Performance Goals:**
- **Latency:** < 100ms per inference window
- **Throughput:** Process 2-second windows every 500ms
- **User expectation:** Near-instant character decoding

**Metal Performance Estimates (A14+):**
- **CNN layers:** ~20-40ms (4 conv2d layers)
- **LSTM layers:** ~30-50ms (2-layer BiLSTM)
- **Total:** ~50-90ms per window (within budget)

**Neural Engine Optimization:**
- Core ML automatically uses Neural Engine when possible
- Operations must be ANE-compatible:
  - ✓ Conv2D, DepthwiseConv2D
  - ✓ LSTM (if implemented correctly)
  - ✓ MatMul, BatchNorm, ReLU
  - ✗ Some custom ops fall back to GPU/CPU
- Quantize to float16 or int8 for best ANE performance

**Impact on Training:**
- ✓ Test inference speed early (before full training)
- ✓ Profile Core ML model on actual iPhone
- ⚠ If too slow: Reduce model size, replace BiLSTM with faster alternative
- ⚠ Streaming mode: May need unidirectional LSTM for lower latency

### 12.4 Audio Input Pipeline on iOS

**Microphone Capture:**
```swift
// iOS captures at 44.1kHz or 48kHz natively
// Downsample to 16kHz for model input
let audioEngine = AVAudioEngine()
let inputNode = audioEngine.inputNode
let recordingFormat = inputNode.outputFormat(forBus: 0)
// Typical: 1 channel, 44.1kHz → convert to 16kHz
```

**Buffering Strategy:**
- **Capture:** 2-second sliding windows
- **Overlap:** 50% (1 second hop)
- **Real-time:** Process while capturing next window

**Spectrogram Computation:**
```swift
// Use Accelerate framework (vDSP)
import Accelerate

// FFT using vDSP_fft_zrip (highly optimized)
// Mel filterbank using matrix multiplication
// ~5-10ms to compute spectrogram on iPhone
```

**Total Latency Budget:**
- Microphone to audio buffer: ~50ms
- Spectrogram computation: ~10ms
- Model inference: ~70ms
- CTC decoding: ~10ms
- **Total:** ~140ms (acceptable for ham radio use)

**Impact on Training Data:**
- ✓ Train on 2-second windows (match deployment)
- ✓ Use 16kHz sample rate (no need to downsample on device)
- ✓ Test with actual iOS audio pipeline characteristics
- ⚠ iOS mic may have different noise profile than synthetic

### 12.5 Battery and Thermal Considerations

**Power Consumption:**
- **Neural Engine:** ~0.5W (very efficient)
- **Metal GPU:** ~1-2W (moderate)
- **CPU inference:** ~3-5W (battery drain)
- **Goal:** Stay on Neural Engine/GPU

**Thermal Throttling:**
- Extended use (30+ min) may cause thermal throttling
- CPU/GPU slow down ~30-50%
- Neural Engine less affected
- **Impact:** Model must run efficiently to avoid slowdown

**Background Operation:**
- iOS may suspend background audio processing
- Foreground operation preferred
- Limited CPU time in background

**Impact on Model Design:**
- ✓ Optimize for Neural Engine (quantized, ANE-compatible ops)
- ✓ Minimize inference frequency (don't run every 100ms if not needed)
- ✓ Test extended operation (60 min continuous decoding)

### 12.6 Core ML Export Considerations

**Export Pipeline:**
```
PyTorch → ONNX → Core ML
        ↓ (check compatibility)
        ↓ (quantize to float16/int8)
        ↓ (verify operations)
Core ML Model (.mlmodel)
```

**Known Issues:**
- **BiLSTM:** May not map perfectly to Core ML LSTM
  - Workaround: Implement as two separate unidirectional LSTMs
- **CTC Decoder:** Not built into Core ML
  - Solution: Implement CTC beam search in Swift (post-processing)
- **Dynamic sequence lengths:** Core ML prefers fixed shapes
  - Solution: Fixed 2-second windows, pad shorter sequences

**Quantization Strategy:**
- **Training:** Full precision (float32)
- **Post-training quantization:**
  - float16: Minimal accuracy loss (~0.5% CER increase)
  - int8: May lose 1-2% CER, but 4× smaller and faster
- **Recommendation:** Train → Test float16 → If good, try int8

**Impact on Training:**
- ✓ Design architecture for easy Core ML export
- ✓ Avoid exotic operations
- ✓ Test export early (prototype → train small → export → test on iPhone)
- ✓ Measure accuracy drop from quantization

### 12.7 Training Data Adjustments for iOS

**Realistic Input Length:**
- **Training window:** 2 seconds max (matches iOS buffer)
- **Character count:** Typically 3-15 characters per window
  - 20 WPM ≈ 10 chars/sec × 2 sec = ~20 chars max
  - 10 WPM ≈ 5 chars/sec × 2 sec = ~10 chars
- **Conclusion:** Limit training samples to 2-second windows

**Updated Length Distribution:**
- 10%: Very short (1-5 chars, < 0.5 sec)
- 30%: Short (6-10 chars, 0.5-1 sec)
- 40%: Medium (11-20 chars, 1-2 sec)
- 20%: Full length (21-30 chars, 2 sec)
- **No samples > 2 seconds** (trim to deployment reality)

**Microphone Noise Characteristics:**
- Add iPhone mic noise profile (if available)
- Test: Record silence on iPhone, analyze spectrum
- Include in noise generation (realistic frequency response)

**Audio Quality:**
- iPhone mic frequency response: ~80 Hz - 8 kHz
- May have low-frequency roll-off
- Train with realistic bandpass filtering

### 12.8 Validation on iOS

**Critical Pre-Deployment Tests:**

1. **Export Test:**
   ```bash
   python export/to_coreml.py --checkpoint best.pt --quantize float16
   # Verify model loads and runs
   ```

2. **iOS Inference Test:**
   ```swift
   // Load model
   let model = try MorseDecoder(configuration: .init())
   // Run on test samples
   let prediction = try model.prediction(spectrogram: testInput)
   // Measure latency
   ```

3. **Accuracy Test:**
   - Run same test set through:
     - PyTorch model (baseline)
     - Core ML float32
     - Core ML float16
     - Core ML int8 (if used)
   - Compare CER (should be within 1-2%)

4. **Performance Test:**
   - Measure inference time on actual iPhone
   - Test thermal performance (30 min continuous)
   - Monitor battery drain
   - Target: < 5% battery per hour of decoding

5. **Memory Test:**
   - Monitor peak memory usage
   - Should stay < 100 MB
   - No memory warnings or kills

**Acceptance Criteria for iOS:**
- ✓ Model exports to Core ML without errors
- ✓ All operations run on Neural Engine (or GPU as fallback)
- ✓ Inference < 100ms per window on iPhone 12
- ✓ CER within 2% of PyTorch version
- ✓ No memory issues during extended use
- ✓ Battery drain acceptable (< 10%/hour)

### 12.9 iOS-Specific Model Variants

**Consider Multiple Model Sizes:**

1. **Standard Model (Main):**
   - 2-3M params, float16
   - Target: iPhone 12+
   - Best accuracy

2. **Lite Model (Fallback):**
   - 1M params, int8
   - Target: Older devices (iPhone X, 11)
   - Slightly lower accuracy, faster

3. **Offline/Research Model:**
   - Larger model for offline processing
   - Not real-time constrained
   - Best possible accuracy

**Ship all three, select at runtime based on device.**

---

## 13. Next Steps

### Immediate Actions:
1. ✅ Document approved (this document)
2. Implement data generation pipeline
3. Implement model architecture
4. Generate validation and test sets
5. Run initial training experiment
6. Evaluate and iterate

### Success Metrics:
- Can generate 10,000 samples in < 5 minutes
- Validation set covers all conditions
- Training completes in < 48 hours
- Final CER < 5% on good conditions
- Model exports to Core ML successfully

---

**Document Version:** 1.0
**Date:** 2025-01-21
**Author:** AI Assistant
**Status:** Ready for Implementation

73 de AI 🎧
