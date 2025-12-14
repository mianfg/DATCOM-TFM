"""Structured logging utilities with context management."""

import contextvars
from typing import Any

from loguru import logger


# Context variables for maintaining request/job context
job_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("job_context", default={})


class LoggingContext:
    """
    Context manager for structured logging with automatic context injection.
    
    Example:
        with LoggingContext(job_id=42, collection_id=1):
            logger.info("Processing job")  # Will include job_id and collection_id
    """

    def __init__(self, **context_data):
        """
        Initialize logging context.
        
        Args:
            **context_data: Key-value pairs to add to logging context
        """
        self.context_data = context_data
        self.token = None

    def __enter__(self):
        """Enter context and set context variables."""
        # Get current context and merge with new data
        current = job_context.get().copy()
        current.update(self.context_data)
        
        # Set context
        self.token = job_context.set(current)
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore previous context."""
        if self.token:
            job_context.reset(self.token)


def get_logging_context() -> dict[str, Any]:
    """
    Get current logging context.
    
    Returns:
        Dictionary of current context variables
    """
    return job_context.get().copy()


def configure_structured_logging():
    """
    Configure loguru to include context variables in all log messages.
    
    This should be called once at application startup.
    """
    
    def context_filter(record):
        """Add context variables to log record."""
        context = job_context.get()
        
        # Add context to record extras
        for key, value in context.items():
            record["extra"][key] = value
        
        return True
    
    # Remove default handler
    logger.remove()
    
    # Add new handler with context filter and structured format
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[job_id]}</cyan>:<cyan>{extra[consolidation_job_id]}</cyan> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        filter=context_filter,
        level="DEBUG",
        colorize=True,
    )
    
    # Also add JSON handler for production logs
    logger.add(
        sink="logs/app.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {extra} | {name}:{function}:{line} | {message}",
        filter=context_filter,
        level="INFO",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
        serialize=False,  # Use text format, not JSON (easier to read)
    )


def log_with_context(level: str, message: str, **extra_context):
    """
    Log a message with additional context.
    
    Args:
        level: Log level (debug, info, warning, error, critical)
        message: Log message
        **extra_context: Additional context to include in this log only
    """
    # Get current context
    context = get_logging_context()
    context.update(extra_context)
    
    # Log with context
    log_func = getattr(logger, level.lower())
    log_func(message, **context)

