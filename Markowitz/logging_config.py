import logging
import os

# Check if the logger already has handlers to avoid duplicate logs
logger = logging.getLogger("EfficiencyFrontier")
if not logger.hasHandlers():
    # Configure the logger
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()  # Allow log level to be set via environment variable
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s - %(name)s - %(message)s'
    )
    logger.setLevel(log_level)
