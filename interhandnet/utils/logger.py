"""Console and file logging."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "interhandnet",
    log_file: Optional[PathLike] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a logger that writes to stdout and optionally to ``log_file``."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
