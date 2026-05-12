# GSFA Highlights — single-image deployment.
# Base: CUDA 12.4 runtime (matches torch cu124 index in requirements-web.txt).
# Drop the `-cudnn` variant if you don't need cuDNN — easyocr doesn't.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps: Python, ffmpeg (for clip_extractor), and a few libs OpenCV expects.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        ffmpeg \
        git \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

WORKDIR /app

# Install Python deps first (cache layer)
COPY requirements-web.txt /app/requirements-web.txt
RUN python -m pip install --upgrade pip \
 && python -m pip install -r requirements-web.txt

# Pre-download EasyOCR English model so the first job doesn't cold-start.
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False); print('easyocr models cached')"

# App code (.dockerignore excludes data/, clips/, paddle/, etc.)
COPY video_highlight/ /app/video_highlight/
COPY webapp/          /app/webapp/
COPY scripts/         /app/scripts/

# Working dir for jobs (mounted as a volume on the host for disk space)
RUN mkdir -p /tmp/jobs
ENV WORK_DIR=/tmp/jobs

EXPOSE 8000

# uvicorn binds 0.0.0.0; nginx in front handles TLS on the host.
CMD ["uvicorn", "webapp.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
