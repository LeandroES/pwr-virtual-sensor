# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.12

# ─── Stage 1: dependency builder ────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /build

# Install uv for fast dependency resolution and installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Set environment variables so uv knows where the venv is
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy only the manifest first to leverage layer caching [cite: 1]
COPY pyproject.toml .

# Create the venv and install dependencies using the standard path
RUN uv venv $VIRTUAL_ENV && \
    uv pip install --no-cache -r pyproject.toml

# ─── Stage 2: runtime image ─────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL maintainer="pwr-twin-team"
LABEL description="PWR Digital Twin – API / Worker runtime"

# Non-root user for security
RUN groupadd --system appgroup && useradd --system --gid appgroup appuser

WORKDIR /app

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Place venv binaries first in PATH
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

# Copy application source
COPY --chown=appuser:appgroup . .

USER appuser

EXPOSE 8000

# Default command runs the API; override in docker-compose for the worker
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
