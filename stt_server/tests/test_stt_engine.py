"""
Robot Savo — STT + LLM gateway tests: stt_engine.

These tests focus on the orchestration logic in stt_engine:

- We do NOT require a real faster-whisper model.
- We do NOT make real HTTP requests to the LLM server.

Instead, we:
  - Monkeypatch `transcribe_audio_bytes()` to return a deterministic transcript.
  - Monkeypatch `call_llm()` to return a fake LLM response or raise errors.
  - Verify that `run_speech_pipeline()` builds the correct result dict.

Run from stt_server root:

    cd ~/Robot_Savo_Server/stt_server
    python -m pytest -q
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from app import stt_engine


@pytest.fixture
def dummy_audio() -> bytes:
    """
    Provide a minimal non-empty audio payload.

    The actual content does not matter in these tests because we monkeypatch
    `transcribe_audio_bytes()`, but an empty payload would trigger early
    validation errors, so we keep it non-empty.
    """
    return b"\x00\x01\x02\x03\x04"


def test_run_speech_pipeline_success(monkeypatch: pytest.MonkeyPatch, dummy_audio: bytes) -> None:
    """
    When STT and LLM both succeed, run_speech_pipeline should:

      - use the transcript returned by transcribe_audio_bytes()
      - merge in reply_text, intent, nav_goal, tier_used from call_llm()
      - set llm_ok=True
    """

    # --- Arrange ---

    fake_transcript = "take me to A201 please"

    def fake_transcribe_audio_bytes(audio_bytes: bytes) -> str:
        assert audio_bytes == dummy_audio  # sanity check
        return fake_transcript

    fake_llm_response: Dict[str, Any] = {
        "reply_text": "Okay, I will guide you to A201.",
        "intent": "NAVIGATE",
        "nav_goal": "A201",
        "tier_used": "TIER1",
    }

    def fake_call_llm(transcript: str) -> Dict[str, Any]:
        assert transcript == fake_transcript
        return fake_llm_response

    monkeypatch.setattr(stt_engine, "transcribe_audio_bytes", fake_transcribe_audio_bytes)
    monkeypatch.setattr(stt_engine, "call_llm", fake_call_llm)

    # --- Act ---

    result = stt_engine.run_speech_pipeline(dummy_audio)

    # --- Assert ---

    assert result["transcript"] == fake_transcript
    assert result["reply_text"] == fake_llm_response["reply_text"]
    assert result["intent"] == fake_llm_response["intent"]
    assert result["nav_goal"] == fake_llm_response["nav_goal"]
    assert result["tier_used"] == fake_llm_response["tier_used"]
    assert result["llm_ok"] is True


def test_run_speech_pipeline_llm_failure(monkeypatch: pytest.MonkeyPatch, dummy_audio: bytes) -> None:
    """
    When STT succeeds but LLM fails (raises RuntimeError), run_speech_pipeline
    should:

      - still return the transcript
      - set reply_text="" and intent/nav_goal/tier_used=None
      - set llm_ok=False
      - NOT raise; the caller (FastAPI /speech) handles this gracefully
    """

    fake_transcript = "what are you doing now?"

    def fake_transcribe_audio_bytes(audio_bytes: bytes) -> str:
        assert audio_bytes == dummy_audio
        return fake_transcript

    def fake_call_llm(_transcript: str) -> Dict[str, Any]:
        raise RuntimeError("simulated LLM outage")

    monkeypatch.setattr(stt_engine, "transcribe_audio_bytes", fake_transcribe_audio_bytes)
    monkeypatch.setattr(stt_engine, "call_llm", fake_call_llm)

    # Act: pipeline should not raise, even though LLM failed.
    result = stt_engine.run_speech_pipeline(dummy_audio)

    # Assert: transcript preserved, LLM fields cleared, llm_ok=False.
    assert result["transcript"] == fake_transcript
    assert result["reply_text"] == ""
    assert result["intent"] is None
    assert result["nav_goal"] is None
    assert result["tier_used"] is None
    assert result["llm_ok"] is False


def test_transcribe_audio_bytes_raises_on_empty() -> None:
    """
    transcribe_audio_bytes should raise a RuntimeError when given empty audio.
    This protects the pipeline from meaningless requests.
    """
    with pytest.raises(RuntimeError):
        stt_engine.transcribe_audio_bytes(b"")
