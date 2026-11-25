"""Loguru setup: coloured console output plus a `run.log` file in every run folder.

Usage anywhere in the codebase:

    from loguru import logger
    logger.info("rows={}", len(df))

Call `setup_logging(run_dir)` once at the entrypoint. Standard-library `logging` records
(MLflow, sklearn, Hydra, ...) are routed into Loguru so everything ends up in one file.
"""

from __future__ import annotations

import inspect
import logging
import sys
from pathlib import Path

from loguru import logger

CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"


class InterceptHandler(logging.Handler):
    """Forward stdlib logging records to Loguru (recipe from the Loguru docs)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(run_dir: str | Path | None = None, level: str = "INFO") -> Path | None:
    """Configure Loguru. Returns the log file path if `run_dir` is given."""
    logger.remove()
    logger.add(sys.stderr, level=level, format=CONSOLE_FORMAT, colorize=True)

    log_file = None
    if run_dir is not None:
        log_file = Path(run_dir) / "run.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # backtrace/diagnose: full tracebacks with variable values on crash.
        logger.add(log_file, level="DEBUG", format=FILE_FORMAT, backtrace=True, diagnose=True)

    # Route stdlib logging and `warnings.warn` into Loguru. Libraries that attach their own
    # handlers (MLflow does) are rewired explicitly so their output reaches run.log too.
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logging.captureWarnings(True)
    for name in ("mlflow",):
        lib_logger = logging.getLogger(name)
        lib_logger.handlers = [InterceptHandler()]
        lib_logger.propagate = False
    return log_file
