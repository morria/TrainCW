"""Curriculum learning scheduler for progressive difficulty increase."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CurriculumPhase:
    """Configuration for a curriculum learning phase."""

    name: str
    start_epoch: int
    end_epoch: int
    wpm_range: Optional[tuple[float, float]] = None
    snr_range: Optional[tuple[float, float]] = None
    timing_variance: Optional[float] = None
    allow_qrm: bool = True
    allow_qrn: bool = True
    allow_fading: bool = True


class CurriculumScheduler:
    """
    Curriculum learning scheduler for CW training.

    Implements a 3-phase curriculum:
    - Phase 1 (Foundation): Easy samples (clean signals, moderate speeds)
    - Phase 2 (Expansion): Medium difficulty (more noise, wider speed range)
    - Phase 3 (Mastery): Full difficulty (all conditions)

    The scheduler adjusts training difficulty based on the current epoch.

    Args:
        config: Curriculum configuration

    Example:
        >>> from traincw.utils.config import Config
        >>> config = Config()
        >>> scheduler = CurriculumScheduler(config.curriculum)
        >>> constraints = scheduler.get_constraints(epoch=10)
        >>> print(constraints.name)
        'Phase 1: Foundation'
    """

    def __init__(self, config):
        self.config = config

        # Define phases
        self.phases = [
            CurriculumPhase(
                name="Phase 1: Foundation",
                start_epoch=1,
                end_epoch=config.phase1_epochs,
                wpm_range=config.phase1_wpm_range,
                snr_range=config.phase1_snr_range,
                timing_variance=config.phase1_timing_variance,
                allow_qrm=False,
                allow_qrn=False,
                allow_fading=False,
            ),
            CurriculumPhase(
                name="Phase 2: Expansion",
                start_epoch=config.phase1_epochs + 1,
                end_epoch=config.phase1_epochs + config.phase2_epochs,
                wpm_range=config.phase2_wpm_range,
                snr_range=config.phase2_snr_range,
                timing_variance=config.phase2_timing_variance,
                allow_qrm=True,  # Lighter QRM
                allow_qrn=True,  # Lighter QRN
                allow_fading=False,
            ),
            CurriculumPhase(
                name="Phase 3: Mastery",
                start_epoch=config.phase1_epochs + config.phase2_epochs + 1,
                end_epoch=float("inf"),
                wpm_range=None,  # Full range
                snr_range=None,  # Full range
                timing_variance=None,  # Full range
                allow_qrm=True,
                allow_qrn=True,
                allow_fading=True,
            ),
        ]

    def get_current_phase(self, epoch: int) -> CurriculumPhase:
        """
        Get the current curriculum phase for the given epoch.

        Args:
            epoch: Current training epoch (1-indexed)

        Returns:
            Current curriculum phase
        """
        for phase in self.phases:
            if phase.start_epoch <= epoch <= phase.end_epoch:
                return phase

        # Default to last phase if beyond all phases
        return self.phases[-1]

    def get_constraints(self, epoch: int) -> CurriculumPhase:
        """
        Get training constraints for the current epoch.

        This is an alias for get_current_phase for backward compatibility.

        Args:
            epoch: Current training epoch (1-indexed)

        Returns:
            Current curriculum phase with constraints
        """
        return self.get_current_phase(epoch)

    def should_use_constraint(self, epoch: int, constraint_name: str) -> bool:
        """
        Check if a specific constraint should be applied for the current epoch.

        Args:
            epoch: Current training epoch
            constraint_name: Name of constraint to check (e.g., "allow_qrm")

        Returns:
            True if constraint should be applied
        """
        phase = self.get_current_phase(epoch)
        return getattr(phase, constraint_name, True)

    def get_phase_progress(self, epoch: int) -> tuple[int, float]:
        """
        Get current phase number and progress within that phase.

        Args:
            epoch: Current training epoch

        Returns:
            Tuple of (phase_number, progress) where progress is 0.0 to 1.0
        """
        phase = self.get_current_phase(epoch)

        for i, p in enumerate(self.phases, start=1):
            if p.name == phase.name:
                phase_number = i

                # Calculate progress within phase
                if phase.end_epoch == float("inf"):
                    progress = 0.0  # Unknown duration
                else:
                    phase_duration = phase.end_epoch - phase.start_epoch + 1
                    epochs_into_phase = epoch - phase.start_epoch + 1
                    progress = epochs_into_phase / phase_duration

                return phase_number, progress

        return 0, 0.0

    def __repr__(self) -> str:
        """String representation of scheduler."""
        lines = ["CurriculumScheduler:"]
        for i, phase in enumerate(self.phases, start=1):
            lines.append(
                f"  Phase {i}: {phase.name} "
                f"(epochs {phase.start_epoch}-{phase.end_epoch})"
            )
        return "\n".join(lines)
