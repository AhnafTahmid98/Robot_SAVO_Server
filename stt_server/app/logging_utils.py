"""
Robot Savo — STT + LLM gateway logging utilities.

Centralized logging configuration so that:

- Local runs (python -m uvicorn app.main:app ...) get nice, readable logs.
- Docker runs integrate cleanly with `docker logs` and Uvicorn's access logs.
- All STT + LLM gateway code uses a consistent logger hierarchy.

Typical usage in app/main.py:

    from app.logging_utils import configure_logging, get_logger

    configure_logging()
    logger = get_logger(__name__)

    logger.info("STT gateway starting up...")

Notes
-----
We intentionally avoid over-engineered logging frameworks here. The goal is:

- One simple configuration function.
- Structured, timestamped logs.
- Respect LOG_LEVEL env var via settings.log_level.
"""

from __future__ import annotations

import logging
from logging import Logger
from typing import Optional

from .config import settings

# Name used as the root for this service's loggers.
ROOT_LOGGER_NAME = "stt_gateway"


class _LogFilter(logging.Filter):
    """
    Optional filter that can be extended to inject Robot Savo specific context
    (robot_id, service name, etc.) into log records.

    Currently passes through all records but is ready for future enrichment.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        # Example of context injection:
        # record.robot_id = settings.robot_id
        return True


def configure_logging(
    level: Optional[int | str] = None,
    force: bool = False,
) -> None:
    """
    Configure Python logging for the STT gateway.

    This should be called once at process startup (e.g. in app.main). If Uvicorn
    has already configured logging, we respect that unless `force=True` is used.

    Parameters
    ----------
    level : int | str | None
        Log level to apply. If None, uses settings.log_level.
        Examples: logging.INFO, "DEBUG", "WARNING".
    force : bool
        If True, existing handlers on the root logger are removed and replaced
        with our own. Use with care; normally the default (False) is fine when
        running under Uvicorn.
    """
    root_logger = logging.getLogger()

    if root_logger.handlers and not force:
        # Logging is already configured (likely by Uvicorn). We still ensure
        # our package logger inherits the correct level.
        _apply_level_to_root(level)
        return

    # Determine level from parameter or settings
    log_level = _resolve_level(level or settings.log_level)

    # Clear existing handlers if forcing
    if force:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

    root_logger.setLevel(log_level)

    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.addFilter(_LogFilter())

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)


def _apply_level_to_root(level: Optional[int | str]) -> None:
    """
    Apply a new level to the root logger if provided.
    """
    if level is None:
        # Fall back to settings.log_level
        level = settings.log_level

    log_level = _resolve_level(level)
    logging.getLogger().setLevel(log_level)


def _resolve_level(level: int | str) -> int:
    """
    Convert a level name or numeric level to a valid logging level int.

    Parameters
    ----------
    level : int | str
        Either a numeric log level (e.g. logging.INFO) or a string such as
        "INFO", "DEBUG", "WARNING".

    Returns
    -------
    int
        Numeric log level.

    Raises
    ------
    ValueError
        If the provided level cannot be resolved.
    """
    if isinstance(level, int):
        return level

    if isinstance(level, str):
        normalized = level.strip().upper()
        if normalized in logging._nameToLevel:  # type: ignore[attr-defined]
            return logging._nameToLevel[normalized]  # type: ignore[attr-defined]

    raise ValueError(f"Invalid log level: {level!r}")


def get_logger(name: Optional[str] = None) -> Logger:
    """
    Get a logger for use in modules.

    Usage:
        logger = get_logger(__name__)

    Parameters
    ----------
    name : str | None
        Logger name. If None, returns the root service logger "stt_gateway".

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if not name:
        name = ROOT_LOGGER_NAME
    else:
        # Nest under the root name for cleaner hierarchy:
        #   stt_gateway.app.main
        #   stt_gateway.app.stt_engine
        name = f"{ROOT_LOGGER_NAME}.{name}"

    return logging.getLogger(name)
