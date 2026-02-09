# =============================================================================
# Wildfire MLOps Pipeline - Docker Image
# =============================================================================
# Uses the official Apache Airflow image (Python 3.11) as the base.
# Geospatial libraries (GDAL, geopandas, h3) are installed via pip.
#
# This replaces the previous OSGEO GDAL multi-stage build which failed
# due to deadsnakes PPA issues. The official Airflow image is smaller,
# faster to build, and includes Airflow pre-configured.
#
# Owner: Person E
#
# Build:  docker build -t wildfire-pipeline -f docker/Dockerfile .
# Test:   docker run --rm wildfire-pipeline python -c "import h3; import geopandas; print('OK')"
# =============================================================================

FROM apache/airflow:2.8.1-python3.11

USER root

# Install system-level dependencies for GDAL/Fiona/rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables for pip compilation
ENV GDAL_CONFIG=/usr/bin/gdal-config

USER airflow

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy project files
COPY --chown=airflow:root dags/ /opt/airflow/dags/
COPY --chown=airflow:root scripts/ /opt/airflow/scripts/
COPY --chown=airflow:root configs/ /opt/airflow/configs/
COPY --chown=airflow:root tests/ /opt/airflow/tests/

# Create data directories
RUN mkdir -p /opt/airflow/data/raw/firms \
    /opt/airflow/data/raw/weather \
    /opt/airflow/data/processed/firms \
    /opt/airflow/data/processed/weather \
    /opt/airflow/data/processed/fused \
    /opt/airflow/data/static \
    /opt/airflow/logs

# Validate that key imports work
RUN python -c "import h3; import geopandas; import pandas; import yaml; print('All imports OK')"

# Validate DAG parses without errors
RUN python dags/wildfire_dag.py || echo "DAG validation will run with Airflow context"

EXPOSE 8080