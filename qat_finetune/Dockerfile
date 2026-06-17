# qat-finetune container — Python 3.11 + CUDA 12.1 + pinned QAT stack
#
# Base image: python:3.11-slim (CUDA torch installed via pip index below).
# All dep versions are pinned ==; bump only after re-running the spike suite
# (see CON-01 in knowledge/prds/174-yolov8-pose-quantization-pipeline.md).

FROM python:3.11-slim

WORKDIR /app

# System libraries required by OpenCV, PyTorch, and litert-torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy project files first (layer-cache friendly)
COPY pyproject.toml poetry.lock* ./

# Copy source
COPY app/ ./app/

# Install main deps (Poetry-managed) — no virtualenv inside container
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-interaction --no-ansi

# Install pinned QAT stack via pip (CON-01 — exact versions, no ranges).
# Versions are locked in requirements-cuda.txt and requirements-qat.txt.
# Bump only after re-running the spike suite (see docs/qat-finetune-version-policy.md).
COPY requirements-cuda.txt requirements-qat.txt ./
RUN pip install --no-cache-dir -r requirements-cuda.txt
RUN pip install --no-cache-dir -r requirements-qat.txt

# Build-time import verification (CPU — no GPU required).
# Catches broken .so links, missing symbols, and version conflicts at build time.
RUN python -c "\
import torch; \
import torchvision; \
import torchao; \
import litert_torch; \
from ultralytics import YOLO; \
print('torch', torch.__version__); \
print('torchao', torchao.__version__); \
print('litert_torch', getattr(litert_torch, '__version__', 'ok')); \
print('import verification passed')"

# GPU smoke-test (FR-M-09): run once inside the production image before final lock.
# Requires --gpus all; NOT run automatically during docker build.
# Run manually: docker run --gpus all io-qat-finetune python tools/qat_gpu_smoketest.py
COPY tools/ ./tools/

# Usage:
#   docker run --gpus all io-qat-finetune run \
#     --fp32-checkpoint-path s3://bucket/checkpoints/best.pt \
#     --source-mlflow-run-id <run_id> \
#     --dataset-dir /data/dataset \
#     --output-dir /artifacts/qat \
#     --output-bucket mlops-artifacts \
#     --output-prefix qat/exp-001 \
#     --experiment-name object-pose-v1-qat
ENTRYPOINT ["qat-finetune"]
