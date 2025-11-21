---
description: Review synthetic data generation implementation
---

Review the Morse code data generation pipeline:

1. **Morse code generation** (`src/traincw/morse/`)
   - Verify timing calculations (WPM, dit/dah ratios)
   - Check character encoding accuracy
   - Validate prosign handling

2. **Audio synthesis** (`src/traincw/data/`)
   - Review tone generation with proper envelope
   - Check frequency range and stability
   - Verify audio sample rate and format

3. **Noise and interference**
   - Assess white/pink noise implementation
   - Review QRM (interference) and QRN (atmospheric) simulation
   - Check fading effects implementation
   - Validate SNR calculations

4. **Timing variance** (human "fist" characteristics)
   - Check dit/dah ratio variations
   - Review spacing randomization
   - Assess realism of timing patterns

5. **Dataset generation**
   - Verify PyTorch Dataset implementation
   - Check data augmentation pipeline
   - Validate text generation (callsigns, words, random)

Compare implementation against specifications in DESIGN.md and provide recommendations.
