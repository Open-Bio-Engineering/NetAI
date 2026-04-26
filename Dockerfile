FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src/ src/
COPY tests/ tests/
COPY demo.py ./
COPY netai.yaml ./

RUN pip install --no-cache-dir -e ".[dev]"

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8001/api/status || exit 1

CMD ["uvicorn", "netai.api.app:create_app", "--host", "0.0.0.0", "--port", "8001", "--factory"]