# Use python-slim for a smaller image
FROM python:3.9-slim

# Install system dependencies (FFmpeg and Node.js for yt-dlp)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements from the backend folder
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade yt-dlp

# Copy backend and frontend folders
COPY backend ./backend
COPY frontend ./frontend

# Directorios para cache de modelos Whisper y descargas
ENV WHISPER_CACHE_DIR=/app/model
RUN mkdir -p /app/model /app/backend/downloads

EXPOSE 10000

# Run uvicorn pointing to the main.py in the backend folder
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
