# app/core/config.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — Configuration
-------------------------------------
Upgraded for GPT-4o-mini Tier1 while keeping OpenRouter
disabled but available for future use.

This configuration supports:
- Tier1: OpenAI GPT-4o-mini (default)
- Tier1 (optional future): OpenRouter (kept in comments)
- Tier2: Local Ollama llama3.x
- Tier3: Template fallback
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------

APP_DIR: Path = Path(__file__).resolve().parents[1]
ROOT_DIR: Path = APP_DIR.parent

PROMPTS_DIR: Path = APP_DIR / "prompts"
MAP_DATA_DIR: Path = APP_DIR / "map_data"
LOGS_DIR: Path = ROOT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Global configuration for the LLM server.
    """

    # Where .env is loaded
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------------------- Server basics -----------------------------
    app_name: str = "Robot Savo LLM Server"
    environment: Literal["development", "production", "test"] = "development"
    debug: bool = True

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ---------------------------- Paths ------------------------------------
    prompts_dir: Path = PROMPTS_DIR
    map_data_dir: Path = MAP_DATA_DIR
    logs_dir: Path = LOGS_DIR

    nav_state_path: Path = MAP_DATA_DIR / "nav_state.json"
    robot_status_path: Path = MAP_DATA_DIR / "robot_status.json"
    known_locations_path: Path = MAP_DATA_DIR / "known_locations.json"

    # ---------------------------- Tier toggles ------------------------------
    tier1_enabled: bool = True
    tier2_enabled: bool = True
    tier3_enabled: bool = True

    # -----------------------------------------------------------------------
    # TIER 1 — ONLINE LLM (GPT-4o-mini by default)
    # -----------------------------------------------------------------------

    # Choose provider: "openai" or "openrouter"
    # Default = openai (GPT-4o-mini)
    tier1_provider: Literal["openai", "openrouter"] = "openai"

    # ======================== OPENAI GPT-4o-mini ==========================
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (env: OPENAI_API_KEY)",
    )

    # GPT-4o-mini model name
    tier1_openai_model: str = "gpt-4o-mini"

    # OpenAI REST endpoint
    tier1_openai_base_url: str = "https://api.openai.com/v1/chat/completions"


    # ======================== OPENROUTER (kept for future) =================
    # NOTE: disabled by default. Kept for reference.
    tier1_openrouter_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    tier1_openrouter_api_key: Optional[str] = None

    # You can keep previous model list for OpenRouter
    tier1_openrouter_model_candidates: list[str] = [
        "tngtech/deepseek-r1t2-chimera:free",
        "qwen/qwen3-coder:free",
        "meta-llama/llama-3.3-70b-instruct:free",
    ]

    # Timeout for Tier1
    tier1_timeout_s: float = 18.0


    # -----------------------------------------------------------------------
    # TIER 2 — LOCAL OLLAMA
    # -----------------------------------------------------------------------

    tier2_ollama_url: Optional[str] = Field(
        default=None,
        description="Local Ollama endpoint, e.g. http://localhost:11434/api/chat",
    )
    tier2_ollama_model: Optional[str] = Field(
        default=None,
        description="Ollama model name, e.g. llama3.2:latest",
    )
    tier2_temperature: float = 0.7
    tier2_max_tokens: int = 256


    # -----------------------------------------------------------------------
    # TIER 3 — TEMPLATE FALLBACK
    # -----------------------------------------------------------------------

    tier3_language: str = "en"
    tier3_enable_status_mode: bool = True


    # -----------------------------------------------------------------------
    # SAFETY LIMITS
    # -----------------------------------------------------------------------

    max_reply_chars: int = 512
    max_history_turns: int = 8


# Global instance
settings = Settings()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Robot Savo — Settings self-test")
    print(f"ROOT_DIR        : {ROOT_DIR}")
    print(f"APP_DIR         : {APP_DIR}")
    print(f"PROMPTS_DIR     : {settings.prompts_dir}")
    print(f"MAP_DATA_DIR    : {settings.map_data_dir}")
    print(f"LOGS_DIR        : {settings.logs_dir}")
    print(f"NavState path   : {settings.nav_state_path}")
    print(f"RobotStatus path: {settings.robot_status_path}")
    print(f"Locations path  : {settings.known_locations_path}")

    print("\n--- Tier1 config ---")
    print(f"Tier1 enabled   : {settings.tier1_enabled}")
    print(f"Tier1 provider  : {settings.tier1_provider}")
    print(f"OpenAI key set  : {bool(settings.openai_api_key)}")
    print(f"OpenAI model    : {settings.tier1_openai_model}")
    print(f"OpenRouter key? : {bool(settings.tier1_openrouter_api_key)}")

    print("\n--- Tier2 config ---")
    print(f"Ollama enabled  : {settings.tier2_enabled}")
    print(f"Ollama URL      : {settings.tier2_ollama_url}")
    print(f"Ollama model    : {settings.tier2_ollama_model}")

    print("\n--- Tier3 config ---")
    print(f"Tier3 enabled   : {settings.tier3_enabled}, language: {settings.tier3_language}")
