"""Centralized logger configuration for module-specific logging.

Following observability best practices from .claude/agents/observability.md:
- [LG1] Using loguru for beautiful, structured logs
- [LG5] Enable backtrace and diagnose for rich debugging
- [LG6] Human-readable timestamp format
- Module-specific log files in logs/ directory
"""

import sys
from pathlib import Path

from loguru import logger


def get_log_file_path(module_name: str) -> Path:
    """Convert module name to log file path.

    Examples:
        kaggle_map.llm.evaluator -> logs/llm.evaluator.log
        kaggle_map.dataloader.dataset -> logs/dataloader.dataset.log
        __main__ -> logs/main.log

    Args:
        module_name: Full module name from __name__

    Returns:
        Path to the log file
    """
    # Handle special cases
    if module_name == "__main__":
        relative_name = "main"
    elif module_name.startswith("kaggle_map."):
        # Strip the kaggle_map prefix
        relative_name = module_name[len("kaggle_map.") :]
    else:
        # Use as-is for other modules
        relative_name = module_name.replace(".", "_")

    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    return log_dir / f"{relative_name}.log"


def configure_logger(
    module_name: str,
    console_level: str = "DEBUG",
    file_level: str = "DEBUG",
    rotation: str = "10 MB",
    retention: int = 90,
    compression: str = "zip",
) -> None:
    """Configure module-specific logger with console and file handlers.

    Sets up:
    - Console handler (stderr) at INFO level with concise format
    - File handler at DEBUG level with detailed format and rotation

    Format for console: HH:mm:ss | LEVEL | filename:line | message
    Format for file: Full timestamp | LEVEL | module:line | message | extra context

    Args:
        module_name: The module's __name__ attribute
        console_level: Logging level for console output (default: INFO)
        file_level: Logging level for file output (default: DEBUG)
        rotation: When to rotate log files (default: "10 MB")
        retention: Number of rotated files to keep (default: 5)
        compression: Compression format for rotated files (default: "zip")
    """
    # Remove default handlers first
    logger.remove()

    # Get log file path
    log_file = get_log_file_path(module_name)

    # Extract just the filename from module for console format
    # e.g., kaggle_map.llm.evaluator -> evaluator
    module_name.split(".")[-1] if "." in module_name and module_name != "__main__" else module_name

    # Console handler - concise format
    logger.add(
        sys.stderr,
        level=console_level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}:{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        enqueue=False,  # Synchronous for console
    )

    # File handler - detailed format with rich debugging
    logger.add(
        str(log_file),
        level=file_level,
        format=("{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{line} | {message} | {extra}"),
        rotation=rotation,
        retention=retention,
        compression=compression,
        backtrace=True,  # [LG5] Full stack traces
        diagnose=True,  # [LG5] Variable values in stack traces
        enqueue=True,  # Async writing for performance
    )

    logger.debug(f"Logger configured for module: {module_name}")
    logger.debug(f"Log file: {log_file}")
    logger.debug(f"Console level: {console_level}, File level: {file_level}")


def setup_default_logging() -> None:
    """Set up a default logger configuration for modules that don't explicitly configure.

    This ensures all modules have at least console logging even if they
    don't call configure_logger explicitly.
    """
    # Only add if no handlers exist
    if not logger._core.handlers:  # type: ignore[attr-defined]
        logger.add(
            sys.stderr,
            level="INFO",
            format=(
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}:{line}</cyan> | "
                "<level>{message}</level>"
            ),
            colorize=True,
        )


# Example usage and testing
if __name__ == "__main__":
    # Configure logger for this module
    configure_logger(__name__)

    # Test different log levels
    logger.debug("This is a debug message - only in file")
    logger.info("This is an info message - console and file")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test with context binding
    bound_logger = logger.bind(user_id=123, request_id="abc-123")
    bound_logger.info("Processing request with context")

    # Test exception logging with backtrace
    try:
        x = 1 / 0
    except ZeroDivisionError:
        logger.exception("Division by zero occurred")

    logger.success("Logger configuration test completed")
