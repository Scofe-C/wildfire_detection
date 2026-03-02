# =============================================================================
# Wildfire MLOps Pipeline - Docker Image (multi-target, runtime install)
# =============================================================================

FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.4 AS airflow-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-venv python3-dev build-essential \
    curl git \
    libeccodes-dev libeccodes0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

ARG AIRFLOW_VERSION=2.8.1
ARG PYTHON_VERSION=3.10
ARG CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

WORKDIR /opt/airflow

COPY requirements.txt /tmp/requirements.txt

# Install in the SAME runtime environment (no /install prefix)
RUN python3 -m pip install --no-cache-dir \
    -r /tmp/requirements.txt \
    --constraint "${CONSTRAINT_URL}"

# Ensure site-packages is visible (safe)
RUN PYVER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" && \
    mkdir -p "/usr/local/lib/python${PYVER}/dist-packages" && \
    echo "/usr/local/lib/python${PYVER}/site-packages" > "/usr/local/lib/python${PYVER}/dist-packages/zzz_local_site_packages.pth" || true

ENV AIRFLOW_HOME=/opt/airflow
ENV PYTHONPATH="/opt/airflow:${PYTHONPATH}"

COPY dags/ /opt/airflow/dags/
COPY scripts/ /opt/airflow/scripts/
COPY configs/ /opt/airflow/configs/
COPY tests/ /opt/airflow/tests/

RUN mkdir -p /opt/airflow/data/raw/firms \
             /opt/airflow/data/raw/weather \
             /opt/airflow/data/processed/firms \
             /opt/airflow/data/processed/weather \
             /opt/airflow/data/processed/fused \
             /opt/airflow/data/static \
             /opt/airflow/logs

# sanity
RUN python3 -c "import airflow; print('airflow OK', airflow.__version__)"

# --- Targets ---
FROM airflow-base AS airflow-webserver
EXPOSE 8080
CMD ["airflow", "webserver", "--port", "8080"]

FROM airflow-base AS airflow-scheduler
CMD ["airflow", "scheduler"]

FROM airflow-base AS airflow-init
CMD ["airflow", "version"]

# --- CI test target (no CMD — used by GitHub Actions only) ---
# Inherits all installed packages from airflow-base.
# Runs: pytest tests/ and ruff check scripts/ dags/ tests/
FROM airflow-base AS test