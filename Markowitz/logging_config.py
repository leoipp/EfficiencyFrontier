import logging
import os
import colorlog
import sys  # Adicionado para redirecionar o logger para stdout

# Check if the logger already has handlers to avoid duplicate logs
logger = logging.getLogger("EfficiencyFrontier")
if not logger.hasHandlers():
    # Configure the logger
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()  # Allow log level to be set via environment variable
    logger.setLevel(log_level)

    # Add a StreamHandler with color formatting, using stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s: %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red, bg_white",
        },
        secondary_log_colors={},
        style='%'
    ))
    logger.addHandler(console_handler)
