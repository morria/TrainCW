#!/usr/bin/env python3
"""Main training script for CW decoder model."""

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from traincw.models.cnn_lstm_ctc import create_model_from_config
from traincw.training.trainer import Trainer
from traincw.utils.config import load_config
from traincw.utils.logger import setup_logger


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train CW decoder model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (overrides config)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set experiment name
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    elif config.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment_name = f"cw_decoder_{timestamp}"

    # Create experiment directory
    experiment_dir = Path(config.output_dir) / config.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Update paths
    config.output_dir = str(experiment_dir)
    config.log_dir = str(experiment_dir / "logs")

    # Set up logging
    log_file = experiment_dir / "training.log"
    logger = setup_logger("traincw", log_file=log_file)

    logger.info("=" * 80)
    logger.info("CW Decoder Training")
    logger.info("=" * 80)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info("")

    # Save configuration
    config_save_path = experiment_dir / "config.yaml"
    config.save(config_save_path)
    logger.info(f"Configuration saved to {config_save_path}")

    # Set random seed
    set_seed(config.seed)
    logger.info(f"Random seed: {config.seed}")

    # Create model
    logger.info("\nCreating model...")
    model = create_model_from_config(config)
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    logger.info(f"Model size: {model.get_model_size_mb():.2f} MB")

    # Create data loaders
    # NOTE: Data loaders will be None until data generation is implemented
    # The Trainer class handles this gracefully
    train_loader = None
    val_loader = None

    if train_loader is None:
        logger.warning(
            "\n" + "=" * 80 + "\n"
            "WARNING: No training data loader available.\n"
            "This is expected because data generation has not been implemented yet.\n"
            "The training loop will run but will not train the model.\n"
            "Implement data generation in traincw.data module to enable training.\n"
            + "=" * 80 + "\n"
        )

    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Start training
    logger.info("\nStarting training...")
    logger.info("")

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint(trainer.current_epoch, is_best=False)

    logger.info("\nTraining completed!")
    logger.info(f"Best validation CER: {trainer.best_val_cer:.4f} ({trainer.best_val_cer*100:.2f}%)")
    logger.info(f"Checkpoints saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
