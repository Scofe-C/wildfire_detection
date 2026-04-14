FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgdal-dev gdal-bin libgeos-dev libproj-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY tests/ tests/
COPY scripts/ scripts/

RUN mkdir -p data/static/fema_nri models/ignition \
    reports/validation reports/bias_gate reports/visualizations

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "from src.data.schema import load_schema; load_schema()" || exit 1

CMD ["pytest", "tests/", "-v", "--tb=short"]

# ── Serve target — Cloud Run inference API ────────────────────────────────────
FROM base AS serve
RUN pip install --no-cache-dir uvicorn[standard] fastapi
CMD ["python", "-m", "uvicorn", "src.api.inference_api:app", "--host", "0.0.0.0", "--port", "8080"]

# ── API target — OBJ-3 disaster report server ────────────────────────────────
FROM base AS dashboard
RUN pip install --no-cache-dir uvicorn[standard] fastapi python-multipart
COPY templates/ templates/
COPY corpus/ corpus/
RUN mkdir -p reports/disaster_reports
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
