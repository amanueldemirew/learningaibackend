import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    If no name is provided, returns the root logger.

    Args:
        name: Optional name for the logger. If None, returns root logger.

    Returns:
        logging.Logger: Configured logger instance
    """
    if name is None:
        return logging.getLogger()
    return logging.getLogger(name)
