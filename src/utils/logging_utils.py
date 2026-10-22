"""Logging setup utilities."""
import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(log_file: Optional[Union[str, Path]] = None) -> logging.Logger:
    """Configures logging to output to console and optionally to a log file."""
    logger = logging.getLogger("sr_project")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
