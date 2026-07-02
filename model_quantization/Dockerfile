FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

# README.md is required by pyproject's `readme = "README.md"`; poetry install fails without it.
COPY pyproject.toml poetry.lock* README.md ./
COPY app/ ./app/

RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-interaction --no-ansi

# Ultralytics TFLite INT8 export toolchain (PTQ path). ultralytics 8.4.19's
# model.export(format="tflite", int8=True) runs check_requirements([...]) that
# pip-installs these AT RUNTIME — which fails in an offline pod. Bake them in at
# build time. Versions per the v8.4.19 check_requirements window + onnx2tf 1.28's
# tested stack (tensorflow 2.19 <-> tf-keras 2.19 <-> onnx2tf <1.29, protobuf 5).
# NOTE: onnx_graphsurgeon is published on NVIDIA's index (Ultralytics passes the
# same --extra-index-url), NOT on PyPI.
RUN pip install --no-cache-dir --extra-index-url https://pypi.ngc.nvidia.com \
    "tensorflow==2.19.0" \
    "tf-keras<=2.19.0" \
    "onnx2tf>=1.26.3,<1.29.0" \
    "onnx>=1.12.0,<2.0.0" \
    "onnxslim>=0.1.82" \
    "onnxruntime>=1.18.1" \
    "onnx-graphsurgeon>=0.3.26" \
    "sng4onnx>=1.0.1" \
    "protobuf>=5,<6" \
    "ai-edge-litert>=1.2.0"

# Build-time verification: the export toolchain must import (fails the build in
# minutes if an install broke or a version is incompatible, instead of at runtime).
RUN python -c "\
import tensorflow, tf_keras, onnx, onnx2tf, onnxslim, onnxruntime, sng4onnx, onnx_graphsurgeon; \
from ultralytics import YOLO; \
print('tensorflow', tensorflow.__version__, '| onnx2tf ok | export toolchain import OK')"

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
