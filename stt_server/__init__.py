"""
Robot Savo — Server stack top-level package.

This package groups together the STT gateway service and, in the future,
any shared tooling that belongs to the *server side* of Robot Savo
(running on the PC/Mac, not on the Pi).

Current layout (inside this package):

    stt_server/
        __init__.py          ← you are here
        app/                 ← FastAPI STT + LLM gateway
            __init__.py
            main.py
            config.py
            schemas.py
            audio_utils.py
            stt_engine.py
            logging_utils.py
        models/              ← faster-whisper models (not imported as code)
        tests/               ← optional unit tests

This file is intentionally minimal: it only marks `stt_server` as a Python
package and provides a conventional place to store package metadata in case
you later decide to publish or reuse it as a library.
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

# Bump this if you make breaking changes to the STT gateway API.
__version__ = "1.0.0"
