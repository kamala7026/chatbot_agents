import logging
import os
from pathlib import Path

# Import the configurable log directory
from core.common.config import LOG_DIR

# This global flag ensures the setup process runs only once.
_logging_configured = False

def setup_logging():
    """
    Configures logging for the application.
    This function is now idempotent and will only run once per process.
    It configures the root logger to allow integration with libraries like Uvicorn.
    """
    global _logging_configured
    if _logging_configured:
        return

    # Get the root logger
    logger = logging.getLogger("aviator_chatbot")
    logger.setLevel(logging.DEBUG)

    # Uvicorn will add its own handlers. We clear them to ensure our file handler is primary.
    if logger.hasHandlers():
        logger.handlers.clear()

    # Use the configured log directory, ensuring it's a Path object for robustness
    log_dir = Path(LOG_DIR)
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_file = log_dir / "app.log"

        # Create file handler
        f_handler = logging.FileHandler(log_file, mode='a')
        f_handler.setLevel(logging.DEBUG)

        # Create formatter and add to handler
        f_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        f_handler.setFormatter(f_format)

        # Add file handler to the root logger
        logger.addHandler(f_handler)

        # Mark as configured
        _logging_configured = True
        
        # This initial log should confirm the handler is working.
        logger.info(f"Root logger configured. Logging to file: {log_file}")

    except PermissionError:
        # Fallback to console logging if file permissions fail.
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - LOGGING_ERROR: %(message)s'))
        logger.addHandler(console_handler)
        logger.error(f"Permission denied to write to log directory '{log_dir}'. Please check permissions.")
        _logging_configured = True # Mark as configured to avoid loops


# Get a named logger for the application. It will inherit from the root logger.
logger = logging.getLogger("aviator_chatbot")