FROM python:3.11.15-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/models/huggingface
ENV TRANSFORMERS_CACHE=/models/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/models/sentence-transformers
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --upgrade pip \
  && pip install . \
  && pip install rapidocr pypdfium2

COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
COPY docker ./docker

RUN chmod +x /app/docker/entrypoint.sh \
  && useradd --create-home --uid 1001 appuser \
  && mkdir -p /models/huggingface /models/sentence-transformers \
  && chown -R appuser:appuser /app /models

USER appuser

EXPOSE 8001

ENTRYPOINT ["/app/docker/entrypoint.sh"]
