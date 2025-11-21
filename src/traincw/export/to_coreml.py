"""Export ONNX model to Core ML format for iOS deployment."""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def export_to_coreml(
    onnx_path: str | Path,
    output_path: str | Path,
    quantize: str = "float16",
    compute_units: str = "ALL",
    verify: bool = True,
) -> None:
    """
    Export ONNX model to Core ML format.

    Args:
        onnx_path: Path to ONNX model file
        output_path: Path to save Core ML model (.mlmodel or .mlpackage)
        quantize: Quantization mode: "float32", "float16", or "int8" (default: "float16")
        compute_units: Compute units: "ALL", "CPU_ONLY", or "CPU_AND_GPU" (default: "ALL")
        verify: Whether to verify exported model (default: True)

    Example:
        >>> export_to_coreml("model.onnx", "model.mlmodel", quantize="float16")
    """
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError(
            "coremltools not installed. Install with: pip install coremltools"
        )

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting ONNX model to Core ML: {onnx_path}")
    logger.info(f"Quantization: {quantize}")
    logger.info(f"Compute units: {compute_units}")

    # Convert ONNX to Core ML
    try:
        # Load ONNX model
        model = ct.converters.onnx.convert(
            model=str(onnx_path),
            minimum_deployment_target=ct.target.iOS15,  # iOS 15+ for better support
        )

        # Apply quantization
        if quantize == "float16":
            logger.info("Applying float16 quantization...")
            model = ct.models.neural_network.quantization_utils.quantize_weights(
                model, nbits=16
            )
        elif quantize == "int8":
            logger.info("Applying int8 quantization...")
            model = ct.models.neural_network.quantization_utils.quantize_weights(
                model, nbits=8
            )
        elif quantize != "float32":
            logger.warning(f"Unknown quantization mode: {quantize}. Using float32.")

        # Set compute units
        compute_units_map = {
            "ALL": ct.ComputeUnit.ALL,
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        }

        if compute_units in compute_units_map:
            model.compute_unit = compute_units_map[compute_units]

        # Add metadata
        model.short_description = "CW Morse Code Decoder Neural Network"
        model.author = "TrainCW"
        model.license = "MIT"

        model.input_description["spectrogram"] = "Mel-spectrogram input (n_mels × time)"
        model.output_description["logits"] = (
            "Character logits (time × num_classes) for CTC decoding"
        )

        # Save Core ML model
        model.save(str(output_path))
        logger.info(f"Core ML model saved to {output_path}")

        # Print model info
        try:
            file_size_mb = output_path.stat().st_size / (1024**2)
            logger.info(f"Core ML model size: {file_size_mb:.2f} MB")
        except Exception:
            pass

        # Verify
        if verify:
            logger.info("Verifying Core ML model...")
            try:
                # Try to load the model
                loaded_model = ct.models.MLModel(str(output_path))
                logger.info("✓ Core ML model loaded successfully")

                # Print model spec
                spec = loaded_model.get_spec()
                logger.info(f"Core ML model inputs: {[i.name for i in spec.description.input]}")
                logger.info(f"Core ML model outputs: {[o.name for o in spec.description.output]}")

            except Exception as e:
                logger.error(f"Core ML verification failed: {e}")

    except Exception as e:
        logger.error(f"Failed to convert to Core ML: {e}")
        raise


def export_pytorch_to_coreml(
    pytorch_model: torch.nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int] = (1, 64, 200),
    quantize: str = "float16",
    compute_units: str = "ALL",
) -> None:
    """
    Export PyTorch model directly to Core ML (via ONNX intermediate).

    This is a convenience function that combines ONNX export and Core ML conversion.

    Args:
        pytorch_model: PyTorch model to export
        output_path: Path to save Core ML model
        input_shape: Input shape (batch, n_mels, time)
        quantize: Quantization mode
        compute_units: Compute units

    Example:
        >>> from traincw.models import CWDecoder
        >>> model = CWDecoder()
        >>> export_pytorch_to_coreml(model, "model.mlmodel")
    """
    from traincw.export.to_onnx import export_to_onnx

    output_path = Path(output_path)
    temp_onnx_path = output_path.with_suffix(".onnx")

    try:
        # Export to ONNX
        logger.info("Step 1/2: Exporting to ONNX...")
        export_to_onnx(pytorch_model, temp_onnx_path, input_shape=input_shape, verify=False)

        # Convert to Core ML
        logger.info("Step 2/2: Converting to Core ML...")
        export_to_coreml(
            temp_onnx_path,
            output_path,
            quantize=quantize,
            compute_units=compute_units,
        )

        logger.info("✓ PyTorch to Core ML export completed successfully")

    finally:
        # Clean up temporary ONNX file
        if temp_onnx_path.exists():
            temp_onnx_path.unlink()
            logger.info(f"Cleaned up temporary ONNX file: {temp_onnx_path}")
