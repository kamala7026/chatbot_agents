import logging
import os

def setup_logging():
    """Configures logging for the application."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "app.log")

    # Create logger
    logger = logging.getLogger("aviator_chatbot")
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to prevent duplicate logs in Streamlit reruns
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create console handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)

    # Create file handler
    f_handler = logging.FileHandler(log_file)
    f_handler.setLevel(logging.DEBUG) # Log warnings and errors to file

    # Create formatters and add to handlers
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    logger.info("Logging configured successfully.")
    return logger

# Initialize logger for the entire application
logger = setup_logging()