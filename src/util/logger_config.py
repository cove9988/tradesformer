import os
import datetime
import logging

def setup_logging(asset, console_level=logging.INFO, file_level=logging.DEBUG):
    """Set up logging configuration."""  
    log_file =f"./data/log/{asset}_{datetime.datetime.now():%Y%m%d}.log"
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    # Get root logger and configure it
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the root level to the lowest level you want to capture
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Optional: Avoid duplicate logs in libraries by disabling propagation
    logging.getLogger("matplotlib").setLevel(logging.ERROR)  # Suppress matplotlib logs
    logging.getLogger("torch").setLevel(logging.ERROR)  
    logging.getLogger("mplfinance").setLevel(logging.ERROR)  