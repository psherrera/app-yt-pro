FROM python:3.11-slim

# Instalar dependencias del sistema (FFmpeg y Node.js para yt-dlp)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar (Optimizado para CPU en PC)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade yt-dlp

# Copiar el resto del código
COPY . .

# Directorio para cache de modelos Whisper (Persistente)
ENV WHISPER_CACHE_DIR=/app/model
RUN mkdir -p /app/model /app/downloads

EXPOSE 5000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-5000}"]
