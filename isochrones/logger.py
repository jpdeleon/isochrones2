"""Logging utilities for the isochrones package.

This module provides logging configuration for the isochrones package,
including functions to initialize logging to both file and console outputs.
"""

from __future__ import annotations

__all__ = ["getLogger", "initLogging"]

import sys
import logging
from typing import Optional


def getLogger() -> logging.Logger:
    """Get the isochrones package logger.

    Returns:
        logging.Logger: Configured logger instance for the 'isochrones' namespace.
    """
    return logging.getLogger("isochrones")


def initLogging(filename: str, logger: Optional[logging.Logger] = None) -> logging.Logger:
    """Initialize logging for the isochrones package.

    Sets up logging to both a file and stdout with a consistent format.
    If a logger is provided, its handlers are cleared before adding new ones.

    Args:
        filename: Path to the log file where messages will be written.
        logger: Optional existing logger to configure. If None, retrieves
            the isochrones package logger. Defaults to None.

    Returns:
        logging.Logger: The configured logger instance.

    Examples:
        >>> logger = initLogging('isochrones.log')
        >>> logger.info('Starting stellar model fitting')
    """
    if logger is None:
        logger = getLogger()
    else:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s: %(message)s")

    fh = logging.FileHandler(filename)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    logger.addHandler(sh)
    return logger
