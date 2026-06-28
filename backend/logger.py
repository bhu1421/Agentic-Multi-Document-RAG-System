"""
Centralised logging configuration for the Agentic RAG backend.

Usage in any module:
    from backend.logger import get_logger
    logger = get_logger(__name__)
    logger.info("...")
"""
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a consistent format.

    Loggers are cached by name by Python's logging framework, so calling
    this multiple times with the same name is safe and cheap.
    """
    logger = logging.getLogger(name)

    # Only add a handler if none exists (prevents duplicate output on reloads)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Don't double-log via the root logger

    return logger
