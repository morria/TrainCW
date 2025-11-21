"""Training infrastructure for CW decoder."""

from traincw.training.curriculum import CurriculumScheduler
from traincw.training.trainer import Trainer


__all__ = ["CurriculumScheduler", "Trainer"]
