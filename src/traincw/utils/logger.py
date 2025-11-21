"""Logging utilities for TrainCW."""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "traincw",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    format_string: str | None = None,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level (default: INFO)
        format_string: Custom format string (default: timestamp + level + message)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("traincw", "logs/training.log")
        >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = "[%(asctime)s] [%(levelname)s] %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
