#!/usr/bin/env python3
"""Export trained model to ONNX and Core ML formats."""

import argparse
import sys
from pathlib import Path

import torch


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from traincw.export.to_coreml import export_pytorch_to_coreml, export_to_coreml
from traincw.export.to_onnx import export_to_onnx
from traincw.models.cnn_lstm_ctc import CWDecoder
from traincw.utils.logger import setup_logger


def load_model(checkpoint_path: str) -> CWDecoder:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Create model
    model = CWDecoder()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description="Export CW decoder model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["onnx", "coreml", "both"],
        default="both",
        help="Export format",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exported_models",
        help="Output directory for exported models",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["float32", "float16", "int8"],
        default="float16",
        help="Quantization mode for Core ML",
    )
    args = parser.parse_args()

    # Set up logging
    logger = setup_logger("traincw")

    logger.info("=" * 80)
    logger.info("CW Decoder Model Export")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Format: {args.format}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading model...")
    model = load_model(args.checkpoint)
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    logger.info(f"Model size: {model.get_model_size_mb():.2f} MB")
    logger.info("")

    # Export to ONNX
    if args.format in ["onnx", "both"]:
        logger.info("Exporting to ONNX...")
        onnx_path = output_dir / "cw_decoder.onnx"
        export_to_onnx(model, onnx_path)
        logger.info("")

    # Export to Core ML
    if args.format in ["coreml", "both"]:
        logger.info("Exporting to Core ML...")
        coreml_path = output_dir / "cw_decoder.mlmodel"

        if args.format == "coreml":
            # Export directly from PyTorch
            export_pytorch_to_coreml(
                model,
                coreml_path,
                quantize=args.quantize,
            )
        else:
            # Convert from ONNX
            onnx_path = output_dir / "cw_decoder.onnx"
            export_to_coreml(
                onnx_path,
                coreml_path,
                quantize=args.quantize,
            )
        logger.info("")

    logger.info("=" * 80)
    logger.info("Export completed successfully!")
    logger.info(f"Exported models saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
