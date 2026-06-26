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

# Configure poetry
RUN poetry config virtualenvs.create false

# Install ONLY third-party dependencies first (no app code yet) so this
# expensive layer is cached on poetry.lock and is NOT invalidated when only
# app/ source changes. --no-root skips installing the local package itself.
RUN poetry install --only=main --no-root --no-interaction --no-ansi

# Copy source code (cheap layer — changes often, installs fast below)
COPY app/ ./app/

# Install the local package itself (fast — deps already present above)
RUN poetry install --only=main --no-interaction --no-ansi

# Example usage: docker run io-dataset-loading run --source-path /data/raw --output-dir /data/processed --format parquet
ENTRYPOINT ["dataset-loading"]
