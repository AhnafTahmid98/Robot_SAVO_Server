"""
Robot Savo — STT + LLM orchestration engine.

This module:
  - Loads a single shared faster-whisper model (lazy singleton).
  - Provides transcribe_audio_bytes() to turn raw audio bytes into text.
  - Calls the Robot Savo LLM server /chat endpoint with the transcript.
  - Exposes run_speech_pipeline() as the main "audio → JSON" function.

We deliberately:
  - Keep the public API simple for FastAPI: just call run_speech_pipeline().
  - Cache the STT model in-process so we only pay load cost once.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple  

import threading

import httpx
import numpy as np
from faster_whisper import WhisperModel

from .config import Settings
from .logging_utils import get_logger


logger = get_logger(__name__)

# Global model cache (lazy-loaded, thread-safe)
_MODEL_LOCK = threading.Lock()
_model: Optional[WhisperModel] = None


def _load_model() -> WhisperModel:
    """
    Lazily load the global faster-whisper model.

    IMPORTANT:
      - We pass the *model name* (e.g. "small.en") as repo_id.
      - We pass STT_MODELS_DIR as `download_root`, so the model is cached
        under that directory (e.g. "models/").
      - We let faster-whisper talk to HuggingFace on first use
        (local_files_only=False) so it can download the model if needed.

    After the first download, subsequent runs will reuse the local cache.
    """
    global _model

    if _model is not None:
        return _model

    with _MODEL_LOCK:
        if _model is not None:
            return _model

        settings = Settings.from_env()
        model_name = settings.stt_model_name
        download_root = settings.stt_models_dir or None

        logger.info(
            "Loading faster-whisper model '%s' (download_root='%s', device=%s)",
            model_name,
            download_root,
            settings.stt_device,
        )

        try:
            _model = WhisperModel(
                model_name,
                device=settings.stt_device,
                compute_type="int8",
                download_root=download_root,
                local_files_only=False,  # allow initial download if needed
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to load STT model: {exc}"
            logger.exception(msg)
            raise RuntimeError(msg) from exc

        logger.info("STT model '%s' loaded successfully", model_name)
        return _model


def transcribe_audio_bytes(audio_bytes: bytes) -> Tuple[str, Optional[str]]:  # <<< CHANGED (returns transcript + lang)
    """
    Run STT on raw audio bytes and return (transcript, detected_language).

    The input is assumed to be 16-bit PCM in a container (WAV, OGG, etc.).
    The decoding is handled by audio_utils before calling this function
    at the pipeline level; here we only care about the NumPy waveform.

    Returns:
      transcript: str
      detected_language: Optional[str]  (BCP-47-ish code like "en", "fi", "it")
    """
    if not audio_bytes:
        raise RuntimeError("Empty audio payload; nothing to transcribe")

    from .audio_utils import decode_audio_to_mono_16k  # local import to avoid cycles

    settings = Settings.from_env()
    model = _load_model()

    # Decode to mono 16 kHz float32
    samples, sample_rate = decode_audio_to_mono_16k(audio_bytes)

    if samples.size == 0:
        raise RuntimeError("Decoded audio is empty after processing")

    logger.debug(
        "Running STT on audio: shape=%s, dtype=%s, sr=%s",
        samples.shape,
        samples.dtype,
        sample_rate,
    )

    # faster-whisper expects a 1D float32 NumPy array and the sample rate.
    # Multilingual logic:
    #   - If STT_LANGUAGE is "auto" or empty -> let the model detect language.
    #   - If STT_LANGUAGE is a code (e.g. "en", "fi") -> force that language.
    raw_lang = (settings.stt_language or "").strip().lower()
    if raw_lang in ("", "auto", "automatic"):
        language_arg = None  # auto-detect
    else:
        language_arg = raw_lang  # fixed language from config

    segments, info = model.transcribe(  # <<< CHANGED (capture info, allow auto language)
        samples,
        beam_size=1,
        language=language_arg,
    )

    # Concatenate segments into a single transcript
    transcript_parts = [seg.text for seg in segments]
    transcript = " ".join(part.strip() for part in transcript_parts).strip()

    # Detected language from the model (if we did auto-detect)
    detected_language: Optional[str]
    if language_arg is None:
        # info.language is e.g. "en", "fi", "it"
        detected_language = getattr(info, "language", None)
    else:
        # We forced a language; report that.
        detected_language = language_arg

    logger.info("STT transcript: %r (lang=%r)", transcript, detected_language)
    return transcript, detected_language


def call_llm(transcript: str, language: Optional[str] = None) -> Dict[str, Any]:  # <<< CHANGED (added language)
    """
    Call the Robot Savo LLM /chat endpoint with the transcript.

    Returns a dict with keys:
      - reply_text
      - intent
      - nav_goal
      - tier_used

    Raises RuntimeError if the LLM call fails or returns an invalid payload.
    """
    settings = Settings.from_env()
    llm_url = settings.llm_server_url.rstrip("/") + "/chat"

    payload: Dict[str, Any] = {
        "source": "mic",
        "robot_id": settings.robot_id,
        "user_text": transcript,
        # Minimal fields required by the LLM server's ChatRequest.
        # If you later extend ChatRequest with new required fields,
        # update this payload accordingly.
    }

    # Pass BCP-47 language code to LLM server if we have one.
    # ChatRequest.language defaults to "en", so we only override when known.
    if language:
        payload["language"] = language  # <<< CHANGED (send language to LLM)

    logger.info("Calling LLM server at %s (lang=%r)", llm_url, language)

    try:
        with httpx.Client(timeout=settings.llm_timeout_s) as client:
            resp = client.post(llm_url, json=payload)
    except Exception as exc:  # noqa: BLE001
        msg = f"LLM HTTP request failed: {exc}"
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    if resp.status_code != 200:
        msg = f"LLM HTTP {resp.status_code}: {resp.text}"
        logger.error(msg)
        raise RuntimeError(msg)

    data = resp.json()
    logger.debug("LLM raw response JSON: %s", data)

    # We expect the LLM server to use ChatResponse schema:
    # {
    #   "reply_text": "...",
    #   "intent": "NAVIGATE",
    #   "nav_goal": "Campus Heart",
    #   "tier_used": "TIER1",
    #   ...
    # }
    reply_text = data.get("reply_text", "") or ""
    intent = data.get("intent")
    nav_goal = data.get("nav_goal")
    tier_used = data.get("tier_used")

    if not isinstance(reply_text, str):
        raise RuntimeError("LLM response missing valid 'reply_text'")

    return {
        "reply_text": reply_text,
        "intent": intent,
        "nav_goal": nav_goal,
        "tier_used": tier_used,
    }


def run_speech_pipeline(audio_bytes: bytes) -> Dict[str, Any]:
    """
    High-level pipeline:

      audio bytes -> transcript -> LLM -> combined result dict

    Returns a dict with keys:

      - transcript: str
      - reply_text: str
      - intent: Optional[str]
      - nav_goal: Optional[str]
      - tier_used: Optional[str]
      - llm_ok: bool
    """
    settings = Settings.from_env()

    # 1) STT
    transcript, detected_language = transcribe_audio_bytes(audio_bytes)  # <<< CHANGED

    # 2) Call LLM (best-effort)
    try:
        llm_result = call_llm(transcript, detected_language)  # <<< CHANGED
        llm_ok = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM call failed (continuing with transcript only): %s", exc)
        llm_result = {
            "reply_text": "",
            "intent": None,
            "nav_goal": None,
            "tier_used": None,
        }
        llm_ok = False

    result: Dict[str, Any] = {
        "transcript": transcript,
        "reply_text": llm_result["reply_text"],
        "intent": llm_result["intent"],
        "nav_goal": llm_result["nav_goal"],
        "tier_used": llm_result["tier_used"],
        "llm_ok": llm_ok,
        "language": detected_language,  # <<< CHANGED (include detected language)       
    }

    logger.info(
        "Speech pipeline result: intent=%r, nav_goal=%r, llm_ok=%s",
        result["intent"],
        result["nav_goal"],
        result["llm_ok"],
    )

    return result
