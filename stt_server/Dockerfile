# Robot Savo — STT + LLM gateway Dockerfile
#
# This image:
#   - Runs a FastAPI service exposing /health and /speech
#   - Uses faster-whisper for STT (audio → text)
#   - Calls the Robot Savo LLM server /chat internally
#
# Expected runtime:
#   - Models are mounted at /models from the host via docker-compose
#   - Service listens on port 9000 inside the container

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---- System dependencies ----
# ffmpeg is required by faster-whisper / soundfile for audio handling.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Python dependencies ----
# Install requirements first to leverage Docker layer caching.
COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Application code ----
COPY . .

# Directory where models will be mounted (do not bake models into image)
ENV STT_MODELS_DIR=/models

# Expose FastAPI/uvicorn port
EXPOSE 9000

# Default command: run the FastAPI app via uvicorn
# Note: if you change app.main:app location, update this line accordingly.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
