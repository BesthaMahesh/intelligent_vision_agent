import os
import sys
from pathlib import Path
from loguru import logger

# Ensure logs directory exists
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "vision_agent.log"

# Configure loguru logger
logger.remove()

# Console logger with clean formatting
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# Persistent file logger with rotation
logger.add(
    str(LOG_PATH),
    rotation="10 MB",
    retention="14 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    encoding="utf-8",
)

def get_logger(name: str = "VisionAgent"):
    return logger.bind(name=name)
