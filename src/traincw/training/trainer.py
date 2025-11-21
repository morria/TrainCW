"""Trainer class for CW decoder model."""

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from traincw.evaluation.metrics import calculate_cer
from traincw.models.cnn_lstm_ctc import compute_ctc_loss
from traincw.models.decoder import decode_predictions
from traincw.training.curriculum import CurriculumScheduler


class Trainer:
    """
    Trainer class for CW decoder model.

    Handles:
    - Training loop with CTC loss
    - Validation
    - Checkpointing
    - Learning rate scheduling
    - TensorBoard logging
    - Early stopping
    - Curriculum learning

    Args:
        model: CWDecoder model
        config: Training configuration
        train_loader: Training data loader (optional, for when data generation is ready)
        val_loader: Validation data loader (optional)
        logger: Logger instance (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        train_loader=None,
        val_loader=None,
        logger: logging.Logger | None = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger or logging.getLogger(__name__)

        # Device
        self.device = torch.device(config.training.device)
        self.model.to(self.device)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.lr_scheduler = self._create_lr_scheduler()

        # Curriculum scheduler
        self.curriculum = CurriculumScheduler(config.curriculum)

        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_cer = float("inf")
        self.epochs_without_improvement = 0

        # Checkpointing
        self.checkpoint_dir = Path(config.output_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoints: list[tuple[float, Path]] = []

        # TensorBoard
        self.writer: SummaryWriter | None = None
        if config.log_dir:
            log_dir = Path(config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from config."""
        if self.config.training.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                betas=self.config.training.betas,
                weight_decay=self.config.training.weight_decay,
            )
        elif self.config.training.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                betas=self.config.training.betas,
                weight_decay=self.config.training.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")

    def _create_lr_scheduler(self):
        """Create learning rate scheduler from config."""
        if self.config.training.lr_schedule == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.lr_min,
            )
        elif self.config.training.lr_schedule == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=self.config.training.lr_min,
            )
        elif self.config.training.lr_schedule == "none":
            return None
        else:
            raise ValueError(f"Unknown lr_schedule: {self.config.training.lr_schedule}")

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number (1-indexed)

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.current_epoch = epoch

        total_loss = 0.0
        num_batches = 0

        # Get curriculum constraints
        phase = self.curriculum.get_current_phase(epoch)
        self.logger.info(f"Training with curriculum: {phase.name}")

        # Progress bar
        if self.train_loader is not None:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

            for batch in pbar:
                loss = self.train_step(batch)
                total_loss += loss
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({"loss": f"{loss:.4f}"})

                # Log to TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar("train/batch_loss", loss, self.global_step)

                self.global_step += 1
        else:
            # No data loader yet - this is expected when data generation isn't implemented
            self.logger.warning(
                "No train_loader provided. Skipping training epoch. "
                "This is expected if data generation is not yet implemented."
            )
            return {"loss": 0.0}

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Log epoch metrics
        if self.writer is not None:
            self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)
            self.writer.add_scalar("train/learning_rate", self.get_lr(), epoch)

        return {"loss": avg_loss}

    def train_step(self, batch: dict[str, torch.Tensor]) -> float:
        """
        Perform one training step.

        Args:
            batch: Dictionary with keys:
                - "spectrograms": (batch, n_mels, time)
                - "targets": (batch, max_target_length)
                - "input_lengths": (batch,)
                - "target_lengths": (batch,)

        Returns:
            Loss value
        """
        # Move batch to device
        spectrograms = batch["spectrograms"].to(self.device)
        targets = batch["targets"].to(self.device)
        target_lengths = batch["target_lengths"].to(self.device)

        # Forward pass
        logits = self.model(spectrograms)

        # Compute input lengths after CNN/LSTM (time dimension may be reduced)
        actual_input_lengths = torch.full(
            (logits.size(0),), logits.size(1), dtype=torch.long, device=self.device
        )

        # Compute CTC loss
        loss = compute_ctc_loss(logits, targets, actual_input_lengths, target_lengths)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.training.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.grad_clip_norm
            )

        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        """
        Validate the model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics including CER
        """
        if self.val_loader is None:
            self.logger.warning("No validation loader provided. Skipping validation.")
            return {"cer": float("inf"), "loss": float("inf")}

        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_references = []

        pbar = tqdm(self.val_loader, desc="Validation")

        for batch in pbar:
            # Move to device
            spectrograms = batch["spectrograms"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)
            references = batch["texts"]  # Original text strings

            # Forward pass
            logits = self.model(spectrograms)

            # Compute loss
            actual_input_lengths = torch.full(
                (logits.size(0),), logits.size(1), dtype=torch.long, device=self.device
            )
            loss = compute_ctc_loss(logits, targets, actual_input_lengths, target_lengths)
            total_loss += loss.item()

            # Decode predictions
            predictions = decode_predictions(logits, method="greedy")

            all_predictions.extend(predictions)
            all_references.extend(references)

        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        cer = calculate_cer(all_predictions, all_references)

        # Log metrics
        self.logger.info(f"Validation - Loss: {avg_loss:.4f}, CER: {cer:.4f} ({cer * 100:.2f}%)")

        if self.writer is not None:
            self.writer.add_scalar("val/loss", avg_loss, epoch)
            self.writer.add_scalar("val/cer", cer, epoch)

            # Log sample predictions
            for i in range(min(5, len(all_predictions))):
                self.writer.add_text(
                    f"val/sample_{i}",
                    f"Pred: {all_predictions[i]}\nRef:  {all_references[i]}",
                    epoch,
                )

        return {"loss": avg_loss, "cer": cer}

    def train(self) -> None:
        """
        Main training loop.

        Trains for the configured number of epochs with validation,
        checkpointing, and early stopping.
        """
        self.logger.info(f"Starting training for {self.config.training.num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {self.model.get_num_parameters():,}")
        self.logger.info(f"Model size: {self.model.get_model_size_mb():.2f} MB")

        # Log curriculum phases
        self.logger.info("\n" + str(self.curriculum))

        start_time = time.time()

        for epoch in range(1, self.config.training.num_epochs + 1):
            epoch_start = time.time()

            # Training
            train_metrics = self.train_epoch(epoch)
            self.logger.info(
                f"Epoch {epoch}/{self.config.training.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}"
            )

            # Validation
            if epoch % self.config.training.val_every_n_epochs == 0:
                val_metrics = self.validate(epoch)

                # Update learning rate scheduler
                if self.lr_scheduler is not None:
                    if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step(val_metrics["cer"])
                    else:
                        self.lr_scheduler.step()

                # Check for improvement
                if val_metrics["cer"] < self.best_val_cer:
                    self.best_val_cer = val_metrics["cer"]
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(epoch, is_best=True)
                    self.logger.info(f"✓ New best model! CER: {self.best_val_cer:.4f}")
                else:
                    self.epochs_without_improvement += 1

                # Early stopping
                if (
                    self.epochs_without_improvement >= self.config.training.early_stopping_patience
                    and epoch >= self.config.training.early_stopping_min_epochs
                ):
                    self.logger.info(
                        f"Early stopping triggered after {epoch} epochs "
                        f"({self.epochs_without_improvement} epochs without improvement)"
                    )
                    break

            # Periodic checkpoint
            if epoch % self.config.training.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, is_best=False)

            epoch_time = time.time() - epoch_start
            self.logger.info(f"Epoch time: {epoch_time:.2f}s")

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time / 3600:.2f} hours")
        self.logger.info(f"Best validation CER: {self.best_val_cer:.4f}")

        if self.writer is not None:
            self.writer.close()

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_cer": self.best_val_cer,
            "config": self.config.to_dict(),
        }

        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved best model to {checkpoint_path}")

            # Manage best checkpoints
            self._manage_best_checkpoints(epoch, checkpoint)
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _manage_best_checkpoints(self, epoch: int, checkpoint: dict[str, Any]) -> None:
        """
        Manage keeping only the best N checkpoints.

        Args:
            epoch: Current epoch
            checkpoint: Checkpoint dictionary
        """
        cer = self.best_val_cer
        checkpoint_path = self.checkpoint_dir / f"best_epoch_{epoch:03d}_cer_{cer:.4f}.pt"

        torch.save(checkpoint, checkpoint_path)
        self.best_checkpoints.append((cer, checkpoint_path))

        # Sort by CER and keep only top N
        self.best_checkpoints.sort(key=lambda x: x[0])
        if len(self.best_checkpoints) > self.config.training.keep_best_n:
            # Remove worst checkpoint
            _, path_to_remove = self.best_checkpoints.pop()
            if path_to_remove.exists():
                path_to_remove.unlink()

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "lr_scheduler_state_dict" in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_cer = checkpoint.get("best_val_cer", float("inf"))

        self.logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {self.current_epoch})")

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
