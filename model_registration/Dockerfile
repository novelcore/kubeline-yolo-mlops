FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for scikit-learn and other ML libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy poetry files and README
COPY pyproject.toml poetry.lock* README.md ./

# Copy source code
COPY app/ ./app/

# Configure poetry
RUN poetry config virtualenvs.create false

# Install dependencies and package
RUN poetry install --only=main

# Example usage: docker run io-model-registration run --model-name bert-finetuned --checkpoint-path /artifacts/checkpoints/model.pt --registry-url https://registry.example.com
ENTRYPOINT ["model-registration"]
