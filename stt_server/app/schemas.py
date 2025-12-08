"""
Robot Savo — STT + LLM gateway schemas.

This module defines the Pydantic models used by the HTTP API. The goal is to
keep the external contract between Pi ↔ Robot_Savo_Server explicit and stable.

High-level flow for /speech:

  Request:
    - Audio is sent as multipart/form-data (file field), so we do not define
      a Pydantic request body here.

  Response (SpeechResponse):
    - transcript : what the user said (STT result)
    - reply_text: what Robot Savo should say back (LLM result)
    - intent    : high-level intent (NAVIGATE / FOLLOW / STOP / STATUS / CHATBOT)
    - nav_goal  : canonical navigation goal, e.g. "A201" or "Info Desk"
    - tier_used : which tier the LLM pipeline used (TIER1 / TIER2 / TIER3)
    - llm_ok    : False if LLM call failed and only transcript is available
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """
    Simple health check response for /health.
    """
    status: str = Field(
        default="ok",
        description="Overall health status of the STT gateway.",
    )


class SpeechResponse(BaseModel):
    """
    Combined STT + LLM response for the /speech endpoint.

    This is what the Pi receives after sending audio. It is intentionally
    aligned with the Robot Savo LLM ChatResponse structure, with additional
    fields for the raw transcript, detected language and LLM call status.
    """

    transcript: str = Field(
        ...,
        description="Recognized user speech in plain text (STT result).",
    )

    language: Optional[str] = Field(
        default=None,
        description="Detected language code for the transcript (e.g. 'en', 'fi', 'it').",
    )

    reply_text: str = Field(
        "",
        description="What Robot Savo should say back to the user (LLM result).",
    )

    intent: Optional[str] = Field(
        default=None,
        description="High-level intent decided by the LLM (e.g. NAVIGATE, CHATBOT).",
    )

    nav_goal: Optional[str] = Field(
        default=None,
        description="Canonical navigation goal (e.g. 'A201', 'Info Desk') if intent is NAVIGATE.",
    )

    tier_used: Optional[str] = Field(
        default=None,
        description="Which LLM tier was used (e.g. TIER1, TIER2, TIER3).",
    )

    llm_ok: bool = Field(
        default=True,
        description="True if the LLM call succeeded; False if only transcript is available.",
    )
