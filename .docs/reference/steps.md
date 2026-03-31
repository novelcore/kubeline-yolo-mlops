# Pipeline Steps

Detailed reference for each of the four pipeline steps.

## Resource Requirements

| Step | CPU | Memory | GPU | Storage | Typical Duration |
|---|---|---|---|---|---|
| Config Validation | 1 core | 512 MB | None | Minimal | ~5 seconds |
| Dataset Loading | 4 cores | 8 GB | None | Up to 20 GB | 1–30 minutes |
| Model Training | 32 cores | 128 GB | 2x A100-40GB | 10 TB | 2–48 hours |
| Model Registration | 1 core | 1 GB | None | Minimal | ~10 seconds |

---

## Step 1: Config Validation

**Purpose:** Validates `pipeline_config.yaml` and verifies that all external services are reachable before committing to compute-intensive steps.

**Compute:** CPU · **Typical duration:** 10–30 seconds

**Inputs:**

- `pipeline_config.yaml` (passed as a CLI parameter by Argo)

**Outputs:**

- Validated config as JSON (passed to Step 2 via Argo artifact)
- Liveness check results (logged)

**What it checks:**

- All required fields are present with correct types
- Field values are within allowed ranges (e.g., `image_size` is a multiple of 32)
- MLflow tracking server is reachable
- S3 / LakeFS endpoint is reachable
- If `dataset.source: "lakefs"`: LakeFS repo and branch exist

**Fails if:**

- Any field is invalid or out of range
- MLflow is unreachable (unless `SKIP_LIVENESS_CHECKS=true`)
- LakeFS repo does not exist

---

## Step 2: Dataset Loading

**Purpose:** Downloads and validates the training dataset from S3 or LakeFS, and optionally samples a subset.

**Compute:** CPU · **Typical duration:** 2–20 minutes (depends on dataset size and network speed)

**Inputs:**

- Validated config from Step 1
- Dataset from S3 / LakeFS

**Outputs:**

- Dataset stored at `/artifacts/dataset/` (Argo artifact volume)
- Dataset statistics report (image count, label validation results)

**What it does:**

1. Connects to S3 or LakeFS using credentials from environment variables
2. Downloads dataset images and label files
3. Validates YOLO Pose label format (keypoint count, coordinate ranges)
4. If `sample_size` is set: selects a deterministic random subset using `dataset.seed`
5. Writes the dataset to the shared artifact volume for Step 3

**Fails if:**

- S3 / LakeFS credentials are invalid
- Dataset path does not exist
- Label files are in an unrecognized format
- Zero images found after sampling

---

## Step 3: Model Training

**Purpose:** Trains a YOLO pose estimation model using Ultralytics, with full MLflow experiment tracking.

**Compute:** **GPU** · **Typical duration:** 2–48 hours (depends on dataset size, epochs, and GPU type)

**Inputs:**

- Dataset from Step 2 (artifact volume)
- Validated config from Step 1
- Optional: checkpoint from `checkpointing.resume_from`

**Outputs:**

- `best.pt` — best checkpoint by `val/mAP50`
- `last.pt` — final epoch checkpoint
- Training plots (`results.png`, confusion matrix, PR curves)
- MLflow run with all metrics, parameters, and artifacts logged

**What it does:**

1. Creates or retrieves the MLflow experiment
2. Initializes YOLO model from `model.variant` (or `pretrained_weights` if set)
3. Registers custom callbacks for:
   - Per-epoch validation metrics (mAP50, precision, recall)
   - System resource monitoring (GPU VRAM, GPU utilization, RAM)
   - Periodic S3 checkpoint saves every `checkpointing.interval_epochs` epochs
4. Calls `model.train()` — Ultralytics handles all MLflow logging automatically
5. Tags the completed run with `KUBECORE_*` metadata
6. Passes `best.pt` S3 path to Step 4

**Fails if:**

- GPU not available or CUDA not initialized
- MLflow server unreachable
- Out of GPU memory (reduce `batch_size` or `image_size`)
- Dataset corrupt or empty

---

## Step 4: Model Registration

**Purpose:** Registers the trained model checkpoints in the MLflow Model Registry with full lineage tags.

**Compute:** CPU · **Typical duration:** 10–30 seconds

**Inputs:**

- `best.pt` and `last.pt` S3 paths from Step 3
- Training run ID and metadata from Step 3
- Validated config from Step 1

**Outputs:**

- New version in MLflow Model Registry for `registration.registered_model_name`
- Lineage tags set on each version
- Stage promotion if `registration.promote_to` is set

**What it does:**

1. Registers `best.pt` as a new model version under `registered_model_name`
2. Registers `last.pt` as a second version
3. Sets lineage tags on both versions:
   - `training_run_id`, `dataset_version`, `dataset_sample_size`
   - `config_hash`, `git_commit`, `model_variant`
   - `best_mAP50`, `checkpoint_type`
4. If `promote_to` is set: transitions both versions to the specified stage
5. Retries MLflow API calls up to `MAX_RETRIES` times with exponential backoff

**Fails if:**

- MLflow server unreachable
- `best.pt` S3 URI is not accessible
- All retries exhausted (MLflow timeout)
