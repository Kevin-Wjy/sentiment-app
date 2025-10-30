# Dockerfile (slim, optimized)
FROM python:3.10-slim

LABEL maintainer="you@example.com"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# --- Install minimal system deps (install build deps temporarily) ---
# NOTE: keep packages minimal; we will purge build deps after pip install
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# copy only requirements first for layer caching
COPY requirements.txt /app/requirements.txt

# upgrade pip and install python deps, then purge build deps and clean caches
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    # remove build dependencies and apt lists to shrink image
    apt-get purge -y --auto-remove build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# copy app source (but NOT the large model folder — see .dockerignore)
COPY . /app

# create non-root user and fix permissions
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# single worker to avoid duplicate memory usage of model
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
