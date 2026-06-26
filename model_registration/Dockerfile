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

# Example usage: docker run io-model-registration run --model-name bert-finetuned --checkpoint-path /artifacts/checkpoints/model.pt --registry-url https://registry.example.com
ENTRYPOINT ["model-registration"]
