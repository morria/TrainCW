#!/usr/bin/env python3
"""
Demo script to test CW data generation.

Usage:
    python scripts/demo_generation.py --output demo_sample.wav
"""

import argparse
import sys
from pathlib import Path


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import soundfile as sf
from data.generator import generate_training_sample


def main():
    parser = argparse.ArgumentParser(description="Generate demo CW sample")
    parser.add_argument("--output", default="demo_sample.wav", help="Output audio file")
    parser.add_argument(
        "--phase", type=int, default=3, choices=[1, 2, 3], help="Curriculum phase (1=easy, 3=hard)"
    )
    parser.add_argument("--count", type=int, default=1, help="Number of samples to generate")

    args = parser.parse_args()

    print(f"Generating {args.count} sample(s) with phase {args.phase}...")

    for i in range(args.count):
        # Generate sample
        audio, text, metadata = generate_training_sample(phase=args.phase)

        # Print info
        print(f"\nSample {i + 1}:")
        print(f"  Text: {text}")
        print(f"  WPM: {metadata['wpm']:.1f}")
        print(f"  Frequency: {metadata['frequency']:.1f} Hz")
        print(f"  SNR: {metadata['snr_db']:.1f} dB")
        print(f"  Duration: {metadata['duration']:.2f} seconds")
        print(f"  Timing variance: {metadata['timing_variance']}")

        # Save audio
        if args.count == 1:
            output_file = args.output
        else:
            output_file = args.output.replace(".wav", f"_{i + 1}.wav")

        sf.write(output_file, audio, 16000)
        print(f"  Saved to: {output_file}")


if __name__ == "__main__":
    main()
