# Pipeline Steps

Detailed reference for each of the six pipeline steps.

## Compute Allocation

There is no per-step memory or core knob — memory follows the node size. Each step declares a compute class via `{step}-class` (e.g. `cpu-standard`, `gpu-t4`), which picks the node pool. Setting `{step}-gpu > 0` routes the step to a GPU node. Nodes scale from zero: they are provisioned when a step starts and released when it finishes.

| Step | Compute Class | GPU | Typical Duration |
|---|---|---|---|
| Config Validation | `cpu-standard` | No | ~5 seconds |
| Dataset Loading | `cpu-standard` | No | 1–30 minutes |
| Model Training | `gpu-t4` | Yes | 2–48 hours |
| QAT Finetune | `gpu-t4` | Yes | minutes–hours (only when `quantization-mode=qat`) |
| Model Quantization | `cpu-standard` | No | minutes |
| Model Registration | `cpu-standard` | No | ~10 seconds |

---

## Step 1: Config Validation

**Purpose:** Validates the submitted workflow parameters and verifies that all external services are reachable before committing to compute-intensive steps.

**Compute:** CPU · **Typical duration:** 10–30 seconds

**Inputs:**

- Argo submission parameters (passed as CLI arguments by Argo)

**Outputs:**

- Validated config as JSON (passed to Step 2 via Argo artifact)
- Liveness check results (logged)

**What it checks:**

- All required fields are present with correct types
- Field values are within allowed ranges (e.g., `image-size` is a multiple of 32)
- MLflow tracking server is reachable
- S3 / LakeFS endpoint is reachable
- If `dataset-source` is `"lakefs"`: LakeFS repo and branch exist

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
4. If `dataset-sample-size` is set: selects a deterministic random subset using `dataset-seed`
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
- Optional: checkpoint from the `checkpoint-resume-from` parameter

**Outputs:**

- `best.pt` — best checkpoint by `val/mAP50`
- `last.pt` — final epoch checkpoint
- Training plots (`results.png`, confusion matrix, PR curves)
- MLflow run with all metrics, parameters, and artifacts logged

**What it does:**

1. Creates or retrieves the MLflow experiment
2. Initializes YOLO model from `model-variant` (or `pretrained-weights` if set)
3. Registers custom callbacks for:
   - Per-epoch validation metrics (mAP50, precision, recall)
   - System resource monitoring (GPU VRAM, GPU utilization, RAM)
   - Periodic S3 checkpoint saves every `checkpointing-interval-epochs` epochs
4. Calls `model.train()` — Ultralytics handles all MLflow logging automatically
5. Tags the completed run with `KUBECORE_*` metadata
6. Passes `best.pt` S3 path to the next step

**Fails if:**

- GPU not available or CUDA not initialized
- MLflow server unreachable
- Out of GPU memory (reduce `batch-size` or `image-size`)
- Dataset corrupt or empty

---

## Step 4: QAT Finetune

**Purpose:** Runs quantization-aware fine-tuning to produce an INT8 model that recovers accuracy lost to naive quantization. **This step only runs when `quantization-mode=qat`** — for `none` or `ptq` it is skipped.

**Compute:** **GPU** · **Typical duration:** minutes to hours

**Inputs:**

- `best.pt` FP32 checkpoint from Step 3
- Dataset from Step 2
- Validated config from Step 1 (`qat-epochs`, `qat-lr`, `quantization-*`)

**Outputs:**

- A headless INT8 model (`model_int8.tflite`) whose detection head runs in host software
- A separate MLflow run logging `qat_finetune_loss`, linked to the training run via a `source_run_id` tag

**What it does:**

1. Loads the FP32 `best.pt` and prepares it for quantization-aware training (built with torchao + litert — Ultralytics does **not** support QAT natively)
2. Fine-tunes for `qat-epochs` at `qat-lr`, logging `qat_finetune_loss`
3. Exports a headless INT8 `.tflite` model
4. Hands the INT8 model to Step 5 for parity checking

**Fails if:**

- GPU not available
- QAT export fails

---

## Step 5: Model Quantization

**Purpose:** Produces the INT8 model and verifies it against the FP32 model. For `ptq` it does the post-training INT8 export via Ultralytics; for `qat` it passes through the model from Step 4. In both cases it runs an FP32-vs-INT8 parity check.

**Compute:** CPU · **Typical duration:** minutes

**Inputs:**

- FP32 `best.pt` from Step 3 (and, for QAT, the INT8 model from Step 4)
- Dataset from Step 2 (for calibration and parity frames)
- Validated config from Step 1 (`quantization-mode`, `quantization-calibration-frames`, `quantization-parity-frames`, etc.)

**Outputs:**

- INT8 `.tflite` artifact logged on the quantization MLflow run — `best_int8.tflite` for PTQ, `model_int8.tflite` for QAT
- Parity metrics: `parity_max_abs_error`, and `fp32_` / `int8_` / `delta_` mAP
- A separate MLflow run linked to the training run via a `source_run_id` tag

**What it does:**

1. For `ptq`: calibrates on `quantization-calibration-frames` frames and exports an INT8 `.tflite` via Ultralytics. For `qat`: takes the INT8 model from Step 4.
2. Runs FP32 vs INT8 inference on `quantization-parity-frames` frames and computes `parity_max_abs_error` and the FP32/INT8 mAP delta
3. Logs the INT8 `.tflite` as an artifact on this run (it is **not** placed in the model registry)

**Fails if:**

- `quantization-calibration-frames` is outside `[100, 10000]` or `quantization-parity-frames` `< 1`
- Parity exceeds `quantization-parity-max-abs-error`
- Export or calibration fails

---

## Step 6: Model Registration

**Purpose:** Registers the trained model checkpoints in the MLflow Model Registry with full lineage tags.

**Compute:** CPU · **Typical duration:** 10–30 seconds

**Inputs:**

- `best.pt` and `last.pt` S3 paths from Step 3
- Training run ID and metadata from Step 3
- Validated config from Step 1

**Outputs:**

- New version in MLflow Model Registry for the `registered-model-name` parameter
- Lineage tags set on each version
- Stage promotion if `promote-to` is set

**What it does:**

1. Registers `best.pt` as a new model version under `registered-model-name`
2. Registers `last.pt` as a second version
3. Sets lineage tags on both versions:
   - `training_run_id`, `dataset_version`, `dataset_sample_size`
   - `config_hash`, `git_commit`, `model_variant`
   - `best_mAP50`, `checkpoint_type`
4. If `promote-to` is set: transitions both versions to the specified stage
5. Retries MLflow API calls up to `MAX_RETRIES` times with exponential backoff

**Fails if:**

- MLflow server unreachable
- `best.pt` S3 URI is not accessible
- All retries exhausted (MLflow timeout)

---

## Quantization Modes at a Glance

The `quantization-mode` parameter controls which of the two quantization steps run:

| Mode | QAT Finetune (Step 4) | Model Quantization (Step 5) |
|---|---|---|
| `none` | Skipped | Skipped — only the FP32 model is produced |
| `ptq` | Skipped | Post-training INT8 export via Ultralytics + parity |
| `qat` | Runs (GPU) — QAT fine-tune, headless INT8 export | Pass-through of the QAT INT8 model + parity |
