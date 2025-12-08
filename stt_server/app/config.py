"""
Robot Savo — STT + LLM gateway configuration.

This module exposes a single immutable `settings` instance that reads its values
from environment variables. It is intentionally lightweight and does not depend
on pydantic-settings, so it works cleanly inside Docker and in local venvs.

Environment variables (suggested defaults in brackets):

  STT_MODELS_DIR      [/models]          Directory where faster-whisper models live
  STT_MODEL_NAME      [small]            Model size or path (e.g. small, small.en, medium)
  STT_DEVICE          [cpu]              "cpu", "cuda", or "auto"
  STT_LANGUAGE        [auto]             Language hint for faster-whisper
                                         - "auto" or empty -> automatic detection
                                         - "en", "fi", "it", ... -> fixed language code

  LLM_SERVER_URL      [http://llm_server:8000]
                                         Base URL for Robot Savo LLM server inside Docker
  LLM_TIMEOUT_S       [15.0]             HTTP timeout (seconds) when calling LLM /chat

  ROBOT_ID            [robot_savo_pi]    Identifier passed through to LLM /chat
  LOG_LEVEL           [INFO]             Root log level for the STT gateway
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional
from urllib.parse import urlparse


@dataclass(frozen=True)
class Settings:
    # STT config
    stt_models_dir: str
    stt_model_name: str
    stt_device: str
    stt_language: Optional[str]  # None = let faster-whisper auto-detect

    # LLM config
    llm_server_url: str
    llm_timeout_s: float

    # Robot identity / logging
    robot_id: str
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        """
        Create a Settings instance from environment variables with sane defaults.
        Minimal validation is applied (e.g. LLM_SERVER_URL must be a valid URL).
        """
        stt_models_dir = os.getenv("STT_MODELS_DIR", "/models")

        # Multilingual model by default
        stt_model_name = os.getenv("STT_MODEL_NAME", "small")

        stt_device = os.getenv("STT_DEVICE", "cpu").lower()

        # "auto" or empty string → None → automatic language detection
        stt_language_raw = os.getenv("STT_LANGUAGE", "auto")
        if not stt_language_raw or stt_language_raw.lower() == "auto":
            stt_language: Optional[str] = None
        else:
            stt_language = stt_language_raw

        llm_server_url = os.getenv("LLM_SERVER_URL", "http://llm_server:8000")
        _validate_url(llm_server_url, env_name="LLM_SERVER_URL")

        llm_timeout_s_str = os.getenv("LLM_TIMEOUT_S", "15.0")
        try:
            llm_timeout_s = float(llm_timeout_s_str)
        except ValueError:
            raise ValueError(f"Invalid LLM_TIMEOUT_S value: {llm_timeout_s_str!r}")

        robot_id = os.getenv("ROBOT_ID", "robot_savo_pi")
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

        return cls(
            stt_models_dir=stt_models_dir,
            stt_model_name=stt_model_name,
            stt_device=stt_device,
            stt_language=stt_language,
            llm_server_url=llm_server_url,
            llm_timeout_s=llm_timeout_s,
            robot_id=robot_id,
            log_level=log_level,
        )


def _validate_url(value: str, env_name: str) -> None:
    """
    Very small safety check to catch obviously broken URLs early.
    """
    parsed = urlparse(value)
    if not (parsed.scheme and parsed.netloc):
        raise ValueError(
            f"{env_name} must be a valid URL, got {value!r}"
        )


# Global, immutable settings instance used by the rest of the package.
settings = Settings.from_env()

__all__ = ["Settings", "settings"]
