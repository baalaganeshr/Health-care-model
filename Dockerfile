# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS builder
WORKDIR /wheels
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /req/requirements.txt
RUN pip install --upgrade pip wheel && pip wheel -r /req/requirements.txt

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*
COPY . /app
RUN mkdir -p /app/data /app/artifacts /app/models && chown -R appuser:appuser /app
USER appuser
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import torch, pandas, sklearn; print('ok')" || exit 1
ENTRYPOINT ["bash","scripts/entrypoint.sh"]
CMD ["python","run.py","--help"]