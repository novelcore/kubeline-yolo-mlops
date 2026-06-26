FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required by OpenCV, PyTorch, and pynvml
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy poetry files and README
COPY pyproject.toml poetry.lock* README.md ./

# Copy source code
COPY app/ ./app/

# Configure poetry: no virtualenv inside container, install main deps only
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-interaction --no-ansi

# GPU runtime: this image is python:3.12-slim (no CUDA base), so it does NOT put
# the host NVIDIA driver libs on the library path. GKE mounts the driver into the
# container at /usr/local/nvidia/lib64 (libcuda.so etc.), but without it on
# LD_LIBRARY_PATH torch (cu128 build) cannot find libcuda.so → torch.cuda.is_available()
# returns False and training silently falls back to CPU even on a T4 node (observed
# yolo-green3, 2026-06-26 — nvidia-smi worked in a cuda-base sibling pod but this
# slim image trained on cpu). Add the driver lib path + standard NVIDIA container env
# so the GPU is usable.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PATH=/usr/local/nvidia/bin:${PATH}

# Usage:
#   docker run io-model-training run \
#     --model-variant yolov8n-pose.pt \
#     --experiment-name object-pose-v1-yolov8n \
#     --dataset-dir /data/dataset \
#     --output-dir /artifacts/runs \
#     --source local
ENTRYPOINT ["model-training"]
