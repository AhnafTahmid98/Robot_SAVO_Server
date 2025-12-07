# app/providers/tier1_online.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — Tier1 Online Provider (OpenAI GPT-4o-mini)
-------------------------------------------------------------------
This module now supports *two* conceptual Tier1 engines:

1) OpenAI GPT-4o-mini  (DEFAULT, ACTIVE)
2) OpenRouter models   (LEGACY, kept in comments for future use)

ACTIVE behavior:
- Uses the official OpenAI Python client (no raw `requests`).
- Reads config from app/core/config.py (settings.*).
- Raises Tier1Error on failure so generate.py can fall back to Tier2/Tier3.

LEGACY behavior (OpenRouter):
- Entire implementation is preserved below in comments.
- To re-enable someday, you would:
    - uncomment the OpenRouter section (including `import requests`)
    - set TIER1_PROVIDER=openrouter in .env
    - provide TIER1_OPENROUTER_API_KEY, etc.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI  # Official OpenAI client

from app.core.config import settings

logger = logging.getLogger(__name__)


class Tier1Error(Exception):
    """Raised when Tier1 (online) fails in a recoverable way."""


# ============================================================================
# OPENAI (GPT-4o-mini) — Primary Tier1 Provider (NO `requests`)
# ============================================================================


def _build_openai_client() -> OpenAI:
    """
    Build an OpenAI client from settings.openai_api_key.

    The API key normally comes from the OPENAI_API_KEY environment variable,
    mapped into settings by pydantic-settings.
    """
    api_key = getattr(settings, "openai_api_key", None)
    if not api_key:
        raise Tier1Error("OpenAI API key missing (settings.openai_api_key is empty).")

    return OpenAI(api_key=api_key)


def _call_openai_chat_completions(
    messages: List[Dict[str, str]],
    model_name: Optional[str] = None,
) -> str:
    """
    Call OpenAI Chat Completions (GPT-4o-mini by default) and return assistant text.

    Parameters
    ----------
    messages:
        List of {"role": "system"|"user"|"assistant", "content": "..."} dicts.
    model_name:
        Name of the OpenAI model (defaults to settings.tier1_openai_model).

    Returns
    -------
    str
        Assistant reply content (trimmed).

    Raises
    ------
    Tier1Error
        On any client / API / response error.
    """
    client = _build_openai_client()
    model = model_name or getattr(settings, "tier1_openai_model", "gpt-4o-mini")

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            # You can wire these to config later if you want:
            # max_tokens=getattr(settings, "tier1_openai_max_tokens", 350),
            # temperature=getattr(settings, "tier1_openai_temperature", 0.6),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("OpenAI ChatCompletion failed: %s", exc)
        raise Tier1Error(f"OpenAI ChatCompletion error: {exc}") from exc

    try:
        choice0 = resp.choices[0]
        content = choice0.message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise Tier1Error(
            "OpenAI response missing choices[0].message.content"
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise Tier1Error("OpenAI returned empty content.")

    return content.strip()


# ============================================================================
# OPENROUTER (Legacy Tier1 Provider) — KEPT *ONLY* IN COMMENTS
# ============================================================================

# NOTE:
#   - This whole section is commented out.
#   - If you ever want to use OpenRouter again, you can:
#       1) Uncomment the import and functions below.
#       2) Set TIER1_PROVIDER=openrouter in your .env
#       3) Provide TIER1_OPENROUTER_API_KEY, etc., in .env / config.

# import requests
#
#
# def _build_openrouter_payload(
#     messages: List[Dict[str, str]],
#     model_name: str,
# ) -> Dict[str, Any]:
#     """Original payload builder (kept for future use)."""
#     payload: Dict[str, Any] = {
#         "model": model_name,
#         "messages": messages,
#     }
#     # Example: enable Grok reasoning
#     if model_name.startswith("x-ai/grok"):
#         payload["extra_body"] = {"reasoning": {"enabled": True}}
#     return payload
#
#
# def _call_openrouter_chat(
#     messages: List[Dict[str, str]],
#     model_name: str,
# ) -> str:
#     """
#     Call OpenRouter chat completions and return assistant text.
#     """
#     api_key = getattr(settings, "tier1_openrouter_api_key", None)
#     base_url = getattr(
#         settings,
#         "tier1_openrouter_base_url",
#         "https://openrouter.ai/api/v1/chat/completions",
#     )
#     if not api_key:
#         raise Tier1Error("OpenRouter API key is missing.")
#
#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json",
#     }
#     payload = _build_openrouter_payload(messages, model_name)
#
#     try:
#         resp = requests.post(
#             base_url,
#             headers=headers,
#             json=payload,
#             timeout=settings.tier1_timeout_s,
#         )
#     except requests.RequestException as exc:
#         raise Tier1Error(f"OpenRouter HTTP error: {exc}") from exc
#
#     if resp.status_code != 200:
#         text_preview = resp.text[:200].replace("\n", " ")
#         raise Tier1Error(f"OpenRouter HTTP {resp.status_code}: {text_preview}")
#
#     try:
#         data = resp.json()
#         content = data["choices"][0]["message"]["content"]
#     except Exception as exc:  # noqa: BLE001
#         raise Tier1Error(
#             "OpenRouter response missing choices[0].message.content"
#         ) from exc
#
#     if not isinstance(content, str) or not content.strip():
#         raise Tier1Error("OpenRouter returned empty content.")
#
#     return content.strip()


# ============================================================================
# AUTO-SELECT Tier1 Provider
# ============================================================================


def call_tier1_model(
    messages: List[Dict[str, str]],
) -> str:
    """
    Auto-select Tier1 provider based on settings.

    ACTIVE:
        - If settings.tier1_provider == "openai"  → use GPT-4o-mini via OpenAI client.

    LEGACY (commented):
        - If settings.tier1_provider == "openrouter" → would use OpenRouter,
          but that path is currently disabled / commented out.

    Raises
    ------
    Tier1Error
        If Tier1 is disabled, misconfigured, or provider is unsupported.
    """
    if not settings.tier1_enabled:
        raise Tier1Error("Tier1 disabled in config.")

    provider = str(getattr(settings, "tier1_provider", "openai")).lower()

    # -------------- OpenAI GPT-4o-mini (default, active) ----------------
    if provider == "openai":
        model_name = getattr(settings, "tier1_openai_model", "gpt-4o-mini")
        return _call_openai_chat_completions(messages, model_name=model_name)

    # -------------- OpenRouter (legacy, currently disabled) --------------
    # if provider == "openrouter":
    #     model = getattr(settings, "tier1_openrouter_model", "deepseek-chat")
    #     return _call_openrouter_chat(messages, model_name=model)

    raise Tier1Error(f"Unsupported or disabled Tier1 provider: {provider!r}")


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":
    print("Robot Savo — Tier1 provider self-test\n")
    print(f"Tier1 enabled : {settings.tier1_enabled}")
    print(f"Tier1 provider: {getattr(settings, 'tier1_provider', 'openai')}")
    print(f"OpenAI key set: {bool(getattr(settings, 'openai_api_key', None))}")
    print(f"OpenAI model  : {getattr(settings, 'tier1_openai_model', 'gpt-4o-mini')}")

    demo_messages = [
        {"role": "system", "content": "You are Robot Savo."},
        {"role": "user", "content": "Hello!"},
    ]

    print("\nNOTE: This self-test does NOT call the live API by default.")
    # If you really want to test a live call, uncomment:
    # if settings.tier1_enabled and getattr(settings, 'openai_api_key', None):
    #     try:
    #         reply = _call_openai_chat_completions(demo_messages)
    #         print("Live reply (first 200 chars):")
    #         print(reply[:200], "...")
    #     except Tier1Error as exc:
#         print("Tier1Error during live call:", exc)
