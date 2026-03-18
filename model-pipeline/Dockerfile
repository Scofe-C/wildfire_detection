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

RUN mkdir -p data/static/fema_nri models/ignition \
    reports/validation reports/bias_gate reports/visualizations

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "from src.data.schema import load_schema; load_schema()" || exit 1

CMD ["pytest", "tests/", "-v", "--tb=short"]
