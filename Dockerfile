# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/root/.local/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

FROM base AS deps

COPY pyproject.toml ./
COPY uv.lock* ./

RUN uv sync --frozen

FROM base AS runtime

COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY . .

#création des repertoire de stockage locaux 
RUN useradd -m appuser \
    && mkdir -p /app/data/raw /app/data/processed \
    && chown -R appuser:appuser /app

    USER appuser

EXPOSE 8000 8501 5000

CMD ["python", "--version"]