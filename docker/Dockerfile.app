# The Chess PGN web app: Python + Tesseract.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

ENV CHESSPGN_DATA=/data \
    OLLAMA_URL=http://ollama:11434
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "--factory", "chesspgn.web.app:app_from_env", "--host", "0.0.0.0", "--port", "8000"]
