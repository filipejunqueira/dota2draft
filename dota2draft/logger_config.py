# dota2draft/logger_config.py

import logging
import os
from .config_loader import CONFIG

# Attempt to use Rich for prettier console logging
try:
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

LOG_LEVEL = logging.INFO # Default log level, can be made configurable later
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logger():
    """Configures the root logger for the application."""
    logger = logging.getLogger("dota2draft") # Get a logger specific to this application
    logger.setLevel(LOG_LEVEL)
    logger.handlers = [] # Clear existing handlers if any (e.g., during reloads)

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # --- Console Handler ---
    if RICH_AVAILABLE:
        console_handler = RichHandler(rich_tracebacks=True, show_path=False, markup=True)
    else:
        console_handler = logging.StreamHandler() # Default to basic stream handler
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File Handler ---
    log_file_path = CONFIG.get('log_file_path', 'dota2draft.log')
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
            # Use a basic print here as logger might not be fully set up or could recurse
            print(f"Error creating log directory {log_dir}: {e}. Logging to current directory.")
            log_file_path = os.path.basename(log_file_path) # Fallback to current dir

    try:
        file_handler = logging.FileHandler(log_file_path, mode='a') # Append mode
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except IOError as e:
        print(f"Error setting up file handler for {log_file_path}: {e}")

    # Prevent log messages from propagating to the root logger if it has default handlers
    logger.propagate = False 

    return logger

# Initialize and export the logger instance
logger = setup_logger()
