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

# Install pinned QAT stack via pip (CUDA cu121 wheel index).
# These are not in Poetry because torch CUDA wheels require a custom index.
# Versions must match pyproject.toml [tool.poetry.dependencies] pins exactly.
RUN pip install --no-cache-dir \
    torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    torchao==0.17.0 \
    litert-torch==0.9.1 \
    ultralytics==8.4.19

# CUDA smoke-test (FR-M-09): verify torch + torchao + litert_torch import
# and CUDA is visible at container build time (requires --gpus all during build).
# Uncomment in final production image validation:
# COPY tools/qat_gpu_smoketest.py /tmp/
# RUN python /tmp/qat_gpu_smoketest.py

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
