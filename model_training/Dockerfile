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

# Usage:
#   docker run io-model-training run \
#     --model-variant yolov8n-pose.pt \
#     --experiment-name spacecraft-pose-v1-yolov8n \
#     --dataset-dir /data/dataset \
#     --output-dir /artifacts/runs \
#     --source local
ENTRYPOINT ["model-training"]
