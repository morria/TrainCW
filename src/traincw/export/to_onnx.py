"""Export PyTorch model to ONNX format."""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int] = (1, 64, 200),
    opset_version: int = 14,
    verify: bool = True,
) -> None:
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        input_shape: Input shape (batch, n_mels, time), default: (1, 64, 200)
        opset_version: ONNX opset version (default: 14)
        verify: Whether to verify exported model (default: True)

    Example:
        >>> from traincw.models import CWDecoder
        >>> model = CWDecoder()
        >>> export_to_onnx(model, "model.onnx")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set model to eval mode
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_shape)

    # Export to ONNX
    logger.info(f"Exporting model to ONNX with input shape {input_shape}")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["spectrogram"],
        output_names=["logits"],
        dynamic_axes={
            "spectrogram": {0: "batch", 2: "time"},  # Dynamic batch and time
            "logits": {0: "batch", 1: "time"},
        },
    )

    logger.info(f"Model exported to {output_path}")

    # Verify export
    if verify:
        try:
            import onnx
            import onnxruntime as ort

            # Load and check ONNX model
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model validation passed")

            # Test inference
            ort_session = ort.InferenceSession(str(output_path))
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)

            # Compare with PyTorch output
            with torch.no_grad():
                torch_output = model(dummy_input)

            diff = torch.abs(torch_output - torch.tensor(ort_outputs[0])).max().item()
            logger.info(f"Max difference between PyTorch and ONNX: {diff:.6f}")

            if diff < 1e-4:
                logger.info("✓ ONNX export verified successfully")
            else:
                logger.warning(
                    f"⚠ Large difference detected ({diff:.6f}). "
                    "ONNX model may not match PyTorch exactly."
                )

        except ImportError:
            logger.warning(
                "onnx or onnxruntime not installed. Skipping verification. "
                "Install with: pip install onnx onnxruntime"
            )
        except Exception as e:
            logger.error(f"ONNX verification failed: {e}")

    # Print model info
    try:
        file_size_mb = output_path.stat().st_size / (1024**2)
        logger.info(f"ONNX model size: {file_size_mb:.2f} MB")
    except Exception:
        pass
