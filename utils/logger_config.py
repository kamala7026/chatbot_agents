import logging
import os
from pathlib import Path

# Import the configurable log directory
from core.config import LOG_DIR

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

    # Prevent adding handlers multiple times by clearing any existing ones.
    if logger.hasHandlers():
        logger.handlers.clear()

    # Use the configured log directory, ensuring it's a Path object for robustness
    log_dir = Path(LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / "app.log"

    # Create file handler - Uvicorn will handle the console stream
    f_handler = logging.FileHandler(log_file)
    f_handler.setLevel(logging.DEBUG)

    # Create formatter and add to handler
    f_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    f_handler.setFormatter(f_format)

    # Add file handler to the logger
    logger.addHandler(f_handler)

    # Mark as configured
    _logging_configured = True
    
    # Log to confirm setup
    logging.getLogger("aviator_chatbot").info("File logging configured successfully.")


# Get the logger instance. The setup will be called from the app's entry point.
logger = logging.getLogger("aviator_chatbot")