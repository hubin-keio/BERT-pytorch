"""Logging utilities for BERT-pytorch"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name, typically __name__
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only add handlers if the logger doesn't already have them
    if not logger.handlers:
        logger.setLevel(level)

        # Console handler with formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Formatter
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def setup_logging(level: int = logging.INFO) -> None:
    """
    Setup logging for the entire application.

    Args:
        level: Logging level for all loggers
    """
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
