FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock* ./
COPY app/ ./app/

RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-interaction --no-ansi

# Usage:
#   docker run io-model-quantization run \
#     --mode ptq \
#     --fp32-checkpoint-path s3://bucket/checkpoints/best.pt \
#     --source-mlflow-run-id <run_id> \
#     --dataset-dir /data/dataset \
#     --output-dir /artifacts/quant \
#     --output-bucket mlops-artifacts \
#     --output-prefix quant/exp-001 \
#     --experiment-name object-pose-v1-quant
ENTRYPOINT ["model-quantization"]
