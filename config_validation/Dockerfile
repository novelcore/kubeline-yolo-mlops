FROM python:3.12-alpine

WORKDIR /app

# Install system dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    openssl-dev

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

# Example usage: docker run io-config-validation run --config-path /data/pipeline_config.yaml
ENTRYPOINT ["config-validation"]
