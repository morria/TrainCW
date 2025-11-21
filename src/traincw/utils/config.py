"""Configuration management for TrainCW.

This module provides a flexible configuration system using dataclasses and YAML.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class AudioConfig:
    """Audio preprocessing configuration."""

    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160  # 10ms at 16kHz
    win_length: int = 400  # 25ms at 16kHz
    n_mels: int = 64
    f_min: float = 0.0
    f_max: float = 8000.0


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # CNN encoder
    cnn_channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    cnn_kernel_sizes: list[int] = field(default_factory=lambda: [3, 3, 3, 3])
    cnn_strides: list[int] = field(default_factory=lambda: [1, 1, 1, 1])
    cnn_pooling: list[int] = field(default_factory=lambda: [2, 2, 2, 2])

    # LSTM
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.1
    bidirectional: bool = True

    # Output
    num_classes: int = 45  # 44 characters + blank for CTC


@dataclass
class TrainingConfig:
    """Training configuration."""

    # General
    batch_size: int = 32
    num_epochs: int = 100
    samples_per_epoch: int = 10000

    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.999)

    # Learning rate schedule
    lr_schedule: str = "cosine"  # "cosine", "plateau", or "none"
    lr_warmup_epochs: int = 5
    lr_min: float = 1e-6

    # Gradient clipping
    grad_clip_norm: float = 1.0

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_epochs: int = 50

    # Checkpointing
    save_every_n_epochs: int = 5
    keep_best_n: int = 3

    # Validation
    val_every_n_epochs: int = 1

    # Device
    device: str = "cpu"  # "cpu" or "cuda"
    num_workers: int = 4


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""

    # Phase 1: Foundation (easy)
    phase1_epochs: int = 30
    phase1_wpm_range: tuple[float, float] = (12.0, 25.0)
    phase1_snr_range: tuple[float, float] = (15.0, 25.0)
    phase1_timing_variance: float = 0.15

    # Phase 2: Expansion (medium)
    phase2_epochs: int = 30
    phase2_wpm_range: tuple[float, float] = (8.0, 35.0)
    phase2_snr_range: tuple[float, float] = (10.0, 25.0)
    phase2_timing_variance: float = 0.25

    # Phase 3: Mastery (hard) - no constraints, full range


@dataclass
class DataConfig:
    """Data generation configuration (placeholder for future data generation)."""

    # Validation/test sets
    validation_path: Optional[str] = None
    test_paths: list[str] = field(default_factory=list)

    # Dataset parameters
    max_audio_length: float = 2.0  # seconds


@dataclass
class ExportConfig:
    """Model export configuration."""

    onnx_opset_version: int = 14
    coreml_quantize: str = "float16"  # "float32", "float16", or "int8"
    coreml_compute_units: str = "ALL"  # "ALL", "CPU_ONLY", or "CPU_AND_GPU"


@dataclass
class Config:
    """Main configuration class."""

    # Subconfigs
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    data: DataConfig = field(default_factory=DataConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # Paths
    output_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Experiment
    experiment_name: Optional[str] = None
    seed: int = 42

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        # Extract subconfig dictionaries
        audio_dict = config_dict.get("audio", {})
        model_dict = config_dict.get("model", {})
        training_dict = config_dict.get("training", {})
        curriculum_dict = config_dict.get("curriculum", {})
        data_dict = config_dict.get("data", {})
        export_dict = config_dict.get("export", {})

        # Create subconfigs
        audio = AudioConfig(**audio_dict)
        model = ModelConfig(**model_dict)
        training = TrainingConfig(**training_dict)
        curriculum = CurriculumConfig(**curriculum_dict)
        data = DataConfig(**data_dict)
        export = ExportConfig(**export_dict)

        # Extract top-level fields
        top_level = {
            k: v
            for k, v in config_dict.items()
            if k not in ["audio", "model", "training", "curriculum", "data", "export"]
        }

        return cls(
            audio=audio,
            model=model,
            training=training,
            curriculum=curriculum,
            data=data,
            export=export,
            **top_level,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "n_fft": self.audio.n_fft,
                "hop_length": self.audio.hop_length,
                "win_length": self.audio.win_length,
                "n_mels": self.audio.n_mels,
                "f_min": self.audio.f_min,
                "f_max": self.audio.f_max,
            },
            "model": {
                "cnn_channels": self.model.cnn_channels,
                "cnn_kernel_sizes": self.model.cnn_kernel_sizes,
                "cnn_strides": self.model.cnn_strides,
                "cnn_pooling": self.model.cnn_pooling,
                "lstm_hidden_size": self.model.lstm_hidden_size,
                "lstm_num_layers": self.model.lstm_num_layers,
                "lstm_dropout": self.model.lstm_dropout,
                "bidirectional": self.model.bidirectional,
                "num_classes": self.model.num_classes,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "num_epochs": self.training.num_epochs,
                "samples_per_epoch": self.training.samples_per_epoch,
                "optimizer": self.training.optimizer,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "betas": self.training.betas,
                "lr_schedule": self.training.lr_schedule,
                "lr_warmup_epochs": self.training.lr_warmup_epochs,
                "lr_min": self.training.lr_min,
                "grad_clip_norm": self.training.grad_clip_norm,
                "early_stopping_patience": self.training.early_stopping_patience,
                "early_stopping_min_epochs": self.training.early_stopping_min_epochs,
                "save_every_n_epochs": self.training.save_every_n_epochs,
                "keep_best_n": self.training.keep_best_n,
                "val_every_n_epochs": self.training.val_every_n_epochs,
                "device": self.training.device,
                "num_workers": self.training.num_workers,
            },
            "curriculum": {
                "phase1_epochs": self.curriculum.phase1_epochs,
                "phase1_wpm_range": self.curriculum.phase1_wpm_range,
                "phase1_snr_range": self.curriculum.phase1_snr_range,
                "phase1_timing_variance": self.curriculum.phase1_timing_variance,
                "phase2_epochs": self.curriculum.phase2_epochs,
                "phase2_wpm_range": self.curriculum.phase2_wpm_range,
                "phase2_snr_range": self.curriculum.phase2_snr_range,
                "phase2_timing_variance": self.curriculum.phase2_timing_variance,
            },
            "data": {
                "validation_path": self.data.validation_path,
                "test_paths": self.data.test_paths,
                "max_audio_length": self.data.max_audio_length,
            },
            "export": {
                "onnx_opset_version": self.export.onnx_opset_version,
                "coreml_quantize": self.export.coreml_quantize,
                "coreml_compute_units": self.export.coreml_compute_units,
            },
            "output_dir": self.output_dir,
            "log_dir": self.log_dir,
            "experiment_name": self.experiment_name,
            "seed": self.seed,
        }

    def save(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def load_config(path: str | Path) -> Config:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Config object

    Example:
        >>> config = load_config("configs/base_config.yaml")
        >>> print(config.model.lstm_hidden_size)
        256
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config_dict = yaml.safe_load(f)

    return Config.from_dict(config_dict)
