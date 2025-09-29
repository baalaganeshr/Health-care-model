# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

FROM base AS builder
ENV PIP_NO_CACHE_DIR=1
WORKDIR /tmp/app
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip wheel --wheel-dir /wheels -r requirements.txt

FROM base AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels
COPY . .
RUN chmod +x scripts/entrypoint.sh || true
RUN chown -R appuser:appuser /app
USER appuser
ENV PATH="/home/appuser/.local/bin:$PATH"
ENTRYPOINT ["/bin/bash", "/app/scripts/entrypoint.sh"]
CMD ["python", "run.py", "--help"]
