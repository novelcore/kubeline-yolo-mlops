# KAOS YOLO MLOps Pipeline Template

An Argo Workflows-based MLOps pipeline for the KAOS YOLO project, built with the Kubestep Python Template pattern.

## Pipeline Steps

The pipeline consists of six sequential steps, each packaged as an independent containerized application. Steps 4–5 are the quantization steps (PRD-174) and are driven by the `quantization-mode` parameter (`none` / `ptq` / `qat`):

1. **config_validation** — Validates the pipeline configuration (hyperparameters, paths, model settings) before any work begins.
2. **dataset_loading** — Loads and preprocesses the dataset from the configured source, applies splits, and persists artifacts.
3. **model_training** — Trains the model using the validated configuration and loaded dataset, tracks metrics, and saves checkpoints.
4. **qat_finetune** — (GPU) Quantization-Aware Training fine-tune of the FP32 checkpoint (PT2E / torchao) → INT8 TFLite. Runs the QAT path.
5. **model_quantization** — Post-Training Quantization (Ultralytics INT8 TFLite export) **or** QAT pass-through, plus the FP32-vs-INT8 parity check (FR-M-03).
6. **model_registration** — Registers the trained model to the model registry with metadata, versioning, and promotion tags.

## Project Structure

```
kubeline-yolo-mlops/
├── config_validation/      # Step 1: Validate pipeline config
│   ├── app/
│   ├── tests/
│   ├── Dockerfile
│   └── pyproject.toml
├── dataset_loading/        # Step 2: Load & preprocess data
│   ├── app/
│   ├── tests/
│   ├── Dockerfile
│   └── pyproject.toml
├── model_training/         # Step 3: Train the model
│   ├── app/
│   ├── tests/
│   ├── Dockerfile
│   └── pyproject.toml
├── qat_finetune/           # Step 4: QAT fine-tune → INT8 TFLite (GPU)
│   ├── app/
│   ├── tests/
│   ├── Dockerfile
│   └── pyproject.toml
├── model_quantization/     # Step 5: PTQ / QAT pass-through + parity
│   ├── app/
│   ├── tests/
│   ├── Dockerfile
│   └── pyproject.toml
├── model_registration/     # Step 6: Register the model
│   ├── app/
│   ├── tests/
│   ├── Dockerfile
│   └── pyproject.toml
└── README.md
```

Each step follows the same internal structure:

- `app/cli.py` — CLI entry point using Typer
- `app/manager.py` — Orchestrator that wires config, models, and services
- `app/models/config.py` — Pydantic Settings for environment-based configuration
- `app/models/` — Domain-specific Pydantic models
- `app/services/` — Business logic
- `tests/` — Unit tests

## Quick Start

Each step can be developed and tested independently:

```bash
cd config_validation  # or any step directory
poetry install
poetry run app run [OPTIONS]
poetry run pytest
```

## Docker

Build and run any step:

```bash
docker build -t io-config-validation ./config_validation
docker run io-config-validation run --config-path /data/pipeline_config.yaml
```

## Configuration

All steps use Pydantic BaseSettings and can be configured via environment variables or `.env` files. See `env.example` in each step directory.

Cross-step pipeline parameters (dataset source, hyperparameters, checkpoint paths, etc.) live in `pipeline_config.yaml`. See `pipeline_config.example.yaml` for the full schema.

## Streaming Modes

The `dataset_loading` step supports three mutually exclusive read strategies, configured under `dataset:` in `pipeline_config.yaml`:

| Mode              | `labels_only` | `manifest_only` | What is downloaded to the pod                               | What the trainer reads              |
|-------------------|:-------------:|:---------------:|--------------------------------------------------------------|-------------------------------------|
| **Full download** |     `false`   |     `false`     | `data.yaml` + all images + all labels                        | Everything local                    |
| **Labels-only**   |     `true`    |     `false`     | `data.yaml` + all labels + `dataset_manifest.json`           | Labels local, images streamed       |
| **Manifest-only** |     `false`   |     `true`      | `data.yaml` + `dataset_manifest.json` (with `label_keys`)    | Both images and labels streamed     |

Setting both to `true` is rejected by `config_validation`.

**How it works end-to-end**

- `dataset_loading` writes `dataset_manifest.json` into the shared dataset directory whenever a streaming mode is active. The manifest records the S3 `bucket`, `prefix`, image keys per split, and (manifest-only) per-split label keys.
- `model_training` auto-detects the manifest on startup and switches `source=s3`. When `label_keys` are present it also sets `s3_stream_labels=True`. The step then injects an `S3PoseTrainer` that builds `S3YoloDataset` instances — images are fetched on demand into a bounded LRU disk cache (`--disk-cache-gb`, default 2 GiB). Cached images are reused across mosaic partners and epochs.
- No configuration is needed on the training side; the manifest drives the mode.

Streaming is the right choice when the dataset does not fit on the training pod's ephemeral disk, or when iteration costs are dominated by download time rather than epoch count. Full download is cheapest for small datasets that fit comfortably in local storage.

## Quantization and Export

Post-training export is configured under `export:` in `pipeline_config.yaml`:

```yaml
export:
  enabled: true
  formats: ["engine", "onnx"]   # TensorRT engine and/or ONNX
  precisions: ["fp16", "int8"]  # FP16 and/or INT8
```

After training finishes successfully, `model_training` exports the Cartesian product of `formats × precisions`, uploads each artefact to `s3://<checkpoint_bucket>/<checkpoint_prefix>/<experiment_name>/`, and logs it as an MLflow artifact under `exports/`. Exports are **non-fatal** — a failure in one combination is logged and the pipeline continues.

Constraints:

- `format` must be `engine` (TensorRT) or `onnx`.
- `precision` must be `fp16` or `int8`.
- **`onnx` + `int8` is not supported** and is skipped with a warning — Ultralytics' ONNX export does not quantize to INT8 natively. Use TensorRT for INT8.
- INT8 export uses the training `data.yaml` for TensorRT calibration, so the dataset (or at least a calibration subset) must remain accessible during export.

Set `export.enabled: false` to skip export entirely; the default is off.

## Train / Validation / Test Behavior

The pipeline follows the Ultralytics YOLO convention for splits. The `data.yaml` read from the dataset source declares `train`, `val`, and `test` keys, each pointing at an `images/<split>/` subdirectory.

- **`train`** — required. Used by `model.train()` every epoch for forward/backward passes and loss computation. All augmentations configured under `augmentation:` apply only to this split.
- **`val`** — required. Used by Ultralytics at the end of every epoch to compute validation metrics (`val/precision`, `val/recall`, `val/mAP50`, `val/mAP50_95`). These metrics drive the early-stopping patience counter and select the `best.pt` checkpoint.
- **`test`** — optional. Never seen during training; consumed only by the post-training evaluation phase.

**Post-training test evaluation**

As the final phase of `model_training` (step 9, after checkpoint upload and export), the service evaluates both `best.pt` and `last.pt` on the test split:

- Ultralytics `model.val(split="test")` is invoked once per checkpoint with `plots=True`.
- Metrics are logged to the same MLflow run under `test/best/{precision,recall,mAP50,mAP50_95}` and `test/last/...`.
- Every file Ultralytics writes to its validator `save_dir` (PR / P / R / F1 curves, confusion matrix, sample prediction overlays, `results.csv`) is logged as MLflow artifacts under `eval/test/best/` and `eval/test/last/`.
- The evaluation honours the active streaming mode: in S3 streaming modes the service constructs an `S3PoseValidator` (parallel to `S3PoseTrainer`) so the test split is streamed with the same bucket/prefix/cache configuration used during training.
- Test eval is **reporting only** — failures never fail the pipeline.
- If the test split is absent (`images/test/` missing locally, or no `splits.test` entry in the manifest), the phase is skipped with a single warning log.

There is no pipeline-config toggle for this behaviour — it is always on, and always safe to run.
