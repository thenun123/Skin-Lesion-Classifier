# syntax=docker/dockerfile:1
# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Install dependencies into an isolated prefix — not carried into the runtime stage.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Add src/ to the module search path so `uvicorn api:app` resolves correctly.
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/models \
    LOG_DIR=/app/logs \
    APP_ENV=production

# libgl1/libglib2.0-0 required by some TensorFlow image codecs on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from the builder stage.
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy only application source — .dockerignore excludes .venv, tests, caches, etc.
COPY src/ ./src/

# models/ is normally mounted as a volume in production (see docker-compose.yml).
# COPY models/ ./models/ is intentionally omitted — use the volume mount instead.
RUN mkdir -p /app/models /app/logs

# Run as a non-root user — security best practice for production containers.
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" \
        || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
