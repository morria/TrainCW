#!/usr/bin/env python3
"""Evaluation script for CW decoder model."""

import argparse
import sys
from pathlib import Path

import torch


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from traincw.evaluation.metrics import batch_evaluate
from traincw.models.cnn_lstm_ctc import CWDecoder
from traincw.utils.logger import setup_logger


def load_model(checkpoint_path: str, device: str = "cpu") -> CWDecoder:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model (with default parameters if config not in checkpoint)
    model = CWDecoder()

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def evaluate_test_set(
    model: CWDecoder,
    test_loader,
    device: str = "cpu",
) -> dict:
    """
    Evaluate model on test set.

    Args:
        model: CWDecoder model
        test_loader: Test data loader
        device: Device to run evaluation on

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in test_loader:
            spectrograms = batch["spectrograms"].to(device)
            references = batch["texts"]

            # Decode predictions
            predictions = model.decode(spectrograms, method="greedy")

            all_predictions.extend(predictions)
            all_references.extend(references)

    # Calculate metrics
    metrics = batch_evaluate(all_predictions, all_references)

    return metrics, all_predictions, all_references


def main():
    parser = argparse.ArgumentParser(description="Evaluate CW decoder model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test data file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results",
    )
    args = parser.parse_args()

    # Set up logging
    logger = setup_logger("traincw")

    logger.info("=" * 80)
    logger.info("CW Decoder Evaluation")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {args.device}")
    logger.info("")

    # Load model
    logger.info("Loading model...")
    model = load_model(args.checkpoint, device=args.device)
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    logger.info(f"Model size: {model.get_model_size_mb():.2f} MB")
    logger.info("")

    # Load test data
    if args.test_data is None:
        logger.warning(
            "No test data provided. Evaluation cannot proceed.\n"
            "Specify test data with --test-data argument.\n"
            "Test data generation is part of the data module (not yet implemented)."
        )
        return

    # NOTE: Test data loading will be implemented when data generation is ready
    test_loader = None

    if test_loader is None:
        logger.warning("Test data loader not implemented yet. Exiting.")
        return

    # Evaluate
    logger.info("Evaluating model...")
    metrics, predictions, references = evaluate_test_set(model, test_loader, args.device)

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Results")
    logger.info("=" * 80)
    logger.info(f"Character Error Rate (CER): {metrics['cer']:.4f} ({metrics['cer'] * 100:.2f}%)")
    logger.info(f"Word Error Rate (WER): {metrics['wer']:.4f} ({metrics['wer'] * 100:.2f}%)")
    logger.info("")

    # Print sample predictions
    logger.info("Sample predictions:")
    for i in range(min(10, len(predictions))):
        logger.info(f"\nSample {i + 1}:")
        logger.info(f"  Reference:  {references[i]}")
        logger.info(f"  Prediction: {predictions[i]}")

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import json

        results = {
            "checkpoint": str(args.checkpoint),
            "test_data": str(args.test_data) if args.test_data else None,
            "metrics": metrics,
            "num_samples": len(predictions),
        }

        with Path.open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
