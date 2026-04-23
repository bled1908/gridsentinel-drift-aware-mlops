# ──────────────────────────────────────────────
# Stage 1: builder — install all dependencies
# ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools needed for some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt

# ──────────────────────────────────────────────
# Stage 2: runtime — lean final image
# ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source and model artefact
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

# Set working directory to src so relative imports resolve
WORKDIR /app/src

# Runtime environment
ENV MODEL_PATH=/app/models/xgboost_baseline.json
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Healthcheck — Docker will poll /health every 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
