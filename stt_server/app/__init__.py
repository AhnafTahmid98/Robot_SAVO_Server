"""
Robot Savo — STT + LLM gateway package.

This package exposes shared configuration and logging utilities used by the
FastAPI application. Import from this package instead of individual modules
when you need global settings or logging setup, e.g.:

    from app import settings, init_logging
"""

from __future__ import annotations

import logging
from logging import Logger

from .config import settings as _settings


# Re-export the settings instance so other modules can do:
#   from app import settings
settings = _settings


def init_logging(level: int | str = logging.INFO) -> Logger:
    """
    Initialize a sane default logging configuration for the STT gateway.

    This is meant to integrate nicely with Uvicorn / FastAPI logs, while still
    being explicit enough for debugging Robot Savo speech issues.

    Call this once at process startup (e.g. at module import time in main.py).
    """
    # If logging has already been configured (e.g. by Uvicorn), do not override.
    if logging.getLogger().handlers:
        return logging.getLogger("stt_server")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)

    return logging.getLogger("stt_server")
