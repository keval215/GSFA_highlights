# GSFA Highlights — single image for api + worker (command differs per service).
# T4 VM: CUDA 12.4 runtime, torch cu124.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Python 3.11 + ODBC Driver 18 for SQL Server (pyodbc) + git (sports pkg)
# libgl1/libglib2.0-0/libxcb1/... : OpenCV native deps (full opencv-python is
# pulled in transitively by ultralytics and needs these even on a headless box).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip python3.11-dev \
        curl gnupg2 git ca-certificates unixodbc-dev \
        libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && echo "deb [arch=amd64,armhf,arm64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/ubuntu/22.04/prod jammy main" \
        > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --no-cache-dir --upgrade pip

# Torch first (large layer, changes rarely), then the rest.
RUN python3.11 -m pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cu124

WORKDIR /app
COPY requirements-service.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements-service.txt

COPY detectors/ detectors/
COPY team_classifier/ team_classifier/
COPY tracking/ tracking/
COPY video_analysis/ video_analysis/
COPY service/ service/
COPY sql/ sql/

ENV PYTHONPATH=/app

# Default command is the worker; api overrides in docker-compose.yml.
CMD ["python3.11", "-m", "service.worker"]
