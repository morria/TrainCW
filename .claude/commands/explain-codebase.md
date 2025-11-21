---
description: Provide a comprehensive overview of the codebase structure
---

Explain the TrainCW codebase architecture and organization:

1. **Project Overview**
   - Read and summarize README.md and DESIGN.md
   - Explain the main goal: training neural networks for Morse code decoding
   - Describe the pipeline: data generation → training → evaluation → export to Core ML

2. **Directory Structure**
   - `src/traincw/`: Main package
     - `morse/`: Morse code encoding/decoding utilities
     - `data/`: Synthetic data generation and PyTorch datasets
     - `models/`: Neural network architectures (CNN-LSTM-CTC)
     - `training/`: Training loops and curriculum learning
     - `evaluation/`: Metrics (CER, WER) and evaluation scripts
     - `export/`: Model export to ONNX and Core ML
     - `utils/`: Shared utilities
   - `tests/`: Test suite
   - `configs/`: Training configuration files
   - `scripts/`: Utility scripts
   - `.claude/`: Claude Code configuration

3. **Key Technologies**
   - PyTorch for deep learning
   - librosa/scipy for audio processing
   - CTC (Connectionist Temporal Classification) for sequence learning
   - Core ML for iOS deployment

4. **Development Workflow**
   - How to add new features
   - Testing strategy
   - CI/CD pipeline

5. **Entry Points**
   - CLI commands via `traincw`
   - Main training script
   - Evaluation and export scripts

Make it easy for new contributors to understand the project.
