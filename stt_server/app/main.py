"""
Robot Savo — STT + LLM gateway FastAPI app.

Public contract (Pi ↔ Robot_Savo_Server):

  GET  /health
      → HealthResponse { status: "ok" }

  POST /speech
      Content-Type: multipart/form-data
      Body:
        file: <audio file>   # e.g. WAV/OGG/FLAC recorded on Pi

      → 200 OK, SpeechResponse JSON:
        {
          "transcript": "...",
          "reply_text": "...",
          "intent": "NAVIGATE" | "FOLLOW" | "STOP" | "STATUS" | "CHATBOT" | null,
          "nav_goal": "A201" | "Info Desk" | null,
          "tier_used": "TIER1" | "TIER2" | "TIER3" | null,
          "llm_ok": true | false
        }

The Pi only needs to know this API; the internal details (faster-whisper model,
LLM server URL, etc.) are handled inside this service.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .schemas import HealthResponse, SpeechResponse
from . import stt_engine
from .logging_utils import configure_logging, get_logger
from .config import settings

# Configure logging as early as possible so all imports use the same setup.
configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Robot Savo STT + LLM Gateway",
    description=(
        "Gateway service for Robot Savo speech pipeline:\n\n"
        "- Accepts audio from Pi at /speech\n"
        "- Runs STT with faster-whisper\n"
        "- Calls Robot Savo LLM /chat\n"
        "- Returns transcript + reply + intent + nav_goal\n"
    ),
    version="1.0.0",
)

# CORS: allow everything for now; you can tighten this later if needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    """
    Application startup hook.

    We do minimal work here:
      - log configuration summary
      - (optionally) warm up the STT model later if needed
    """
    logger.info("Robot Savo STT gateway starting up...")
    logger.info(
        "Config: models_dir=%s, model_name=%s, device=%s, lang=%s, llm_url=%s, robot_id=%s",
        settings.stt_models_dir,
        settings.stt_model_name,
        settings.stt_device,
        settings.stt_language,
        settings.llm_server_url,
        settings.robot_id,
    )
    # Optional warm-up (disabled for now because load can be heavy on first start):
    # from .stt_engine import _load_model
    # _load_model()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """
    Application shutdown hook.
    """
    logger.info("Robot Savo STT gateway shutting down...")


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """
    Lightweight health-check endpoint.

    This is safe to call frequently from monitoring, or a quick curl from Pi:
        curl http://PC-IP:9000/health
    """
    return HealthResponse(status="ok")


@app.post("/speech", response_model=SpeechResponse, tags=["speech"])
async def speech(file: UploadFile = File(...)) -> SpeechResponse:
    """
    Main speech endpoint for Robot Savo.

    Expected usage from Pi:
      - Send a short audio clip (e.g. WAV/OGG) in a multipart/form-data request
      - Field name: "file"
      - The gateway will:
          1) Decode the audio
          2) Run STT → transcript
          3) Call Robot Savo LLM /chat with the transcript
          4) Return transcript + reply_text + intent + nav_goal + tier_used

    Example (manual) curl:

        curl -X POST \\
          -F "file=@chunk.wav" \\
          http://PC-IP:9000/speech
    """
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty audio file received.",
            )

        result = stt_engine.run_speech_pipeline(audio_bytes)
        return SpeechResponse(**result)

    except HTTPException:
        # Already has an appropriate status code and message.
        raise

    except RuntimeError as exc:
        # Known pipeline error (STT/LLM issues that we want to expose as 400/500).
        logger.warning("Speech pipeline runtime error: %s", exc)
        # If STT failed, it is effectively a bad request from the caller side
        # (e.g. unsupported format, zero audio, etc.).
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:  # noqa: BLE001
        # Unknown/unexpected internal error.
        logger.exception("Unexpected error in /speech: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error in speech pipeline.",
        ) from exc
