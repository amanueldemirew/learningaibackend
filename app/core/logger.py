import logging
import sys
from typing import Any

# Configure logging format
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Create logger
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)


# Add custom method for structured logging
def log_info(message: str, **kwargs: Any) -> None:
    """Log info message with additional context"""
    if kwargs:
        logger.info(f"{message} - Context: {kwargs}")
    else:
        logger.info(message)


def log_error(message: str, error: Exception = None, **kwargs: Any) -> None:
    """Log error message with exception details and additional context"""
    if error:
        logger.error(f"{message} - Error: {str(error)} - Context: {kwargs}")
    else:
        logger.error(f"{message} - Context: {kwargs}")


def log_warning(message: str, **kwargs: Any) -> None:
    """Log warning message with additional context"""
    if kwargs:
        logger.warning(f"{message} - Context: {kwargs}")
    else:
        logger.warning(message)


# Add these methods to the logger object
logger.structured_info = log_info
logger.structured_error = log_error
logger.structured_warning = log_warning
