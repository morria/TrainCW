#!/usr/bin/env python3
"""
Generate fixed test/validation sets for CW training.

Usage:
    python scripts/generate_test_set.py --type validation --count 2000 --output data/validation.pkl
    python scripts/generate_test_set.py --type test_speed --count 800 --output data/test_speed.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generator import generate_sample_with_params
from data.text_generator import generate_random_text


def generate_validation_set(count: int, sample_rate: int = 16000):
    """
    Generate stratified validation set covering all conditions.

    Args:
        count: Target number of samples (~2000)
        sample_rate: Audio sample rate

    Returns:
        List of (audio, text, metadata) tuples
    """
    samples = []

    # Stratify by speed and SNR
    speeds = [5, 10, 15, 20, 25, 30, 35, 40]
    snr_levels = [-5, 0, 5, 10, 15, 20, 25]

    samples_per_condition = count // (len(speeds) * len(snr_levels))

    print(f"Generating validation set: {len(speeds)} speeds x {len(snr_levels)} SNR levels")
    print(f"Samples per condition: {samples_per_condition}")

    for wpm in speeds:
        for snr in snr_levels:
            for _ in range(samples_per_condition):
                params = {
                    "wpm": wpm + np.random.uniform(-2, 2),
                    "snr_db": snr + np.random.uniform(-2, 2),
                    "frequency": np.random.uniform(500, 800),
                    "text": generate_random_text(),
                }
                audio, text, metadata = generate_sample_with_params(params, sample_rate)
                samples.append((audio, text, metadata))

    print(f"Generated {len(samples)} validation samples")
    return samples


def generate_speed_sweep_test(count: int = 800, sample_rate: int = 16000):
    """
    Generate test set sweeping speed (clean conditions).

    Args:
        count: Number of samples (100 per speed x 8 speeds = 800)
        sample_rate: Audio sample rate

    Returns:
        List of samples
    """
    samples = []
    speeds = [5, 10, 15, 20, 25, 30, 35, 40]
    samples_per_speed = count // len(speeds)

    print(f"Generating speed sweep test: {len(speeds)} speeds")

    for wpm in speeds:
        for _ in range(samples_per_speed):
            params = {
                "wpm": wpm,
                "snr_db": 25,  # Clean
                "frequency": 600,  # Standard frequency
                "text": generate_random_text(),
                "timing_variance": {
                    "dit_dah_ratio": 3.0,
                    "element_gap_variance": 0.05,
                    "char_gap_variance": 0.05,
                    "word_gap_variance": 0.05,
                },
            }
            audio, text, metadata = generate_sample_with_params(params, sample_rate)
            samples.append((audio, text, metadata))

    print(f"Generated {len(samples)} speed sweep samples")
    return samples


def generate_snr_sweep_test(count: int = 800, sample_rate: int = 16000):
    """
    Generate test set sweeping SNR (moderate speed).

    Args:
        count: Number of samples
        sample_rate: Audio sample rate

    Returns:
        List of samples
    """
    samples = []
    snr_levels = [-5, 0, 5, 10, 15, 20, 25, 30]
    samples_per_snr = count // len(snr_levels)

    print(f"Generating SNR sweep test: {len(snr_levels)} SNR levels")

    for snr in snr_levels:
        for _ in range(samples_per_snr):
            params = {
                "wpm": 20,
                "snr_db": snr,
                "frequency": 600,
                "text": generate_random_text(),
            }
            audio, text, metadata = generate_sample_with_params(params, sample_rate)
            samples.append((audio, text, metadata))

    print(f"Generated {len(samples)} SNR sweep samples")
    return samples


def generate_interference_test(count: int = 900, sample_rate: int = 16000):
    """
    Generate test set with different interference types.

    Args:
        count: Number of samples
        sample_rate: Audio sample rate

    Returns:
        List of samples
    """
    samples = []
    interference_types = ["clean", "qrm", "qrn", "fading", "combined"]
    samples_per_type = count // len(interference_types)

    print(f"Generating interference test: {len(interference_types)} types")

    for interference_type in interference_types:
        for _ in range(samples_per_type):
            params = {
                "wpm": 20,
                "snr_db": 12,
                "frequency": 600,
                "text": generate_random_text(),
                "add_qrm": interference_type in ["qrm", "combined"],
                "add_qrn": interference_type in ["qrn", "combined"],
                "add_fading": interference_type in ["fading", "combined"],
            }
            audio, text, metadata = generate_sample_with_params(params, sample_rate)
            samples.append((audio, text, metadata))

    print(f"Generated {len(samples)} interference test samples")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate CW test/validation sets")
    parser.add_argument(
        "--type",
        required=True,
        choices=["validation", "test_speed", "test_snr", "test_interference"],
        help="Type of test set to generate",
    )
    parser.add_argument("--count", type=int, required=True, help="Number of samples to generate")
    parser.add_argument("--output", required=True, help="Output file path (.pkl)")
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Audio sample rate (default: 16000)"
    )

    args = parser.parse_args()

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate samples
    print(f"Generating {args.type} set with {args.count} samples...")

    if args.type == "validation":
        samples = generate_validation_set(args.count, args.sample_rate)
    elif args.type == "test_speed":
        samples = generate_speed_sweep_test(args.count, args.sample_rate)
    elif args.type == "test_snr":
        samples = generate_snr_sweep_test(args.count, args.sample_rate)
    elif args.type == "test_interference":
        samples = generate_interference_test(args.count, args.sample_rate)

    # Save to file
    print(f"Saving to {args.output}...")
    with output_path.open("wb") as f:
        pickle.dump(samples, f)

    # Report statistics
    total_duration = sum(len(audio) / args.sample_rate for audio, _, _ in samples)
    print("\nDone!")
    print(f"  Samples: {len(samples)}")
    print(f"  Total duration: {total_duration:.1f} seconds ({total_duration / 60:.1f} minutes)")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
