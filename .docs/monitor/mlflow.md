# MLflow Dashboard

MLflow is where your experiment results are stored. Think of it as version control for your training runs — it records what you trained, how you trained it, and what the results were.

## Core Concepts

**Experiments** — A named group of related training runs. All runs for an experiment appear together in the MLflow UI.

**Runs** — The pipeline creates **one MLflow run per step**, not one run per pipeline. The training, qat-finetune, quantization, and registration steps each log their own separate run within the same experiment, linked back to the training run by a `source_run_id` tag. Each run has:

- **Parameters** — Inputs (learning rate, batch size, epochs, optimizer, etc.)
- **Metrics** — Output measurements (loss, mAP50, precision, recall), optionally logged per epoch
- **Artifacts** — Files produced (model checkpoints, plots, config files)
- **Tags** — Key-value metadata (dataset version, git commit, pipeline ID)

**Registered Models** — A named entry in the Model Registry. Each registration creates a new version:

```
Registered Model: "spacecraft-pose-yolo"
  ├── Version 1  (best.pt from run abc123)
  ├── Version 2  (best.pt from run def456)
  └── Version 3  (best.pt from run ghi789)  ← latest
```

**Model Stages** — Versions move through lifecycle stages:

```
None → Staging → Production → Archived
```

MLflow has four main components. This pipeline uses two:

| Component | What It Does | Used by Pipeline? |
|---|---|---|
| **Tracking** | Logs parameters, metrics, and artifacts per run | Yes |
| **Model Registry** | Versions and stages trained models | Yes |
| **Projects** | Packages ML code in a reproducible format | No |
| **Models** | Standard format for deploying models | No |

---

## How MLflow Is Deployed

Your MLflow instance runs as a remote tracking server:

```
┌─────────────────────┐
│   MLflow UI + API    │  ← Your MLFLOW_TRACKING_URI
│   (Tracking Server)  │
├─────────────────────┤
│  Backend Store:      │  ← PostgreSQL (params, metrics, tags, metadata)
│  Artifact Store:     │  ← S3 bucket (model files, plots, large outputs)
└─────────────────────┘
```

- **Backend store** (PostgreSQL) holds structured data: run metadata, parameters, metrics, and tags.
- **Artifact store** (S3) holds binary files: model checkpoints (`best.pt`, `last.pt`), training plots, and config files.
- **Proxy artifact storage** — Artifact uploads go *through* the tracking server to S3. You do not need direct S3 credentials for artifact logging.

!!! info "Why uploads go through the server"
    This is by design — it avoids giving the training container direct S3 write credentials. Large files like `best.pt` (~25 MB for YOLOv8n) pass through the server before reaching S3. If uploads feel slow, this is normal.

---

## Opening the Dashboard

Open your MLflow URL in a browser (ask your administrator if you do not have it). If prompted, enter your username and password.

You land on the MLflow home page.

## Dashboard Layout

The MLflow UI has two main sections, accessible from the top navigation bar:

- **Experiments** — lists all training runs, grouped by experiment name
- **Models** — the Model Registry, where trained models are stored and versioned

## Finding Your Experiment

In the left sidebar under **Experiments**, find the experiment name matching the `experiment-name` parameter you submitted with your workflow (e.g., `spacecraft-pose-v1-yolov8n`).

Click it to see all runs in that experiment.

## The Runs Table

Each row in the runs table is one step run (training, qat-finetune, quantization, or registration). Runs from the same pipeline share a `source_run_id` tag pointing at the training run. Key columns:

| Column | What It Shows |
| --- | --- |
| Run Name | Pipeline workflow name (e.g., `spacecraft-pose-v1-20260401`) |
| Created | When the run started |
| Duration | Wall-clock training time |
| Status | `RUNNING`, `FINISHED`, or `FAILED` |
| `val/mAP50` | Final validation mean average precision (the primary quality metric) |
| `train/pose_loss` | Final pose estimation loss |

!!! tip "Show the metrics you care about"
    Click the **Columns** button above the table to add or remove columns.
    Add `val/mAP50` and `val/mAP50_95` as default columns to spot the best runs at a glance.

## Opening a Run

Click any run name to open its detail page. The detail page has three key tabs:

### Overview Tab

Shows a summary including:

- All **parameters** (hyperparameters from your config file)
- All **tags** including `kubecore.project_name` and `kubecore.workflow_name` — these link the run back to its pipeline execution
- The run ID (a long hex string like `abc123def456`) — useful for SDK queries

### Metrics Tab

Shows all metrics logged during training as line charts over epochs.

**Key metrics to watch:**

| Metric | What It Means |
| --- | --- |
| `train/box_loss` | Bounding box regression loss |
| `train/pose_loss` | Keypoint pose estimation loss — should decrease steadily |
| `train/kobj_loss` | Keypoint objectness loss |
| `train/cls_loss` | Classification loss |
| `train/dfl_loss` | Distribution focal loss |
| `val/precision` | Validation precision |
| `val/recall` | Validation recall |
| `val/mAP50` | Mean average precision at IoU=0.50 — the primary quality indicator |
| `val/mAP50_95` | mAP averaged over IoU thresholds 0.50–0.95 — stricter quality measure |
| `system/ram_used_gb` | System RAM in use (GB) |
| `system/ram_percent` | System RAM usage as a percentage |
| `system/cpu_percent` | CPU utilization percentage |
| `system/gpu_vram_used_gb` | GPU VRAM in use (GB) |
| `system/gpu_vram_total_gb` | Total GPU VRAM (GB) |
| `system/gpu_utilization_pct` | GPU compute utilization (%) |

All system metrics are logged per epoch with `step=current_epoch`, so they render as line charts.

**Quantization-step metrics** (logged on the separate quantization run, not the training run):

| Metric | What It Means |
| --- | --- |
| `parity_max_abs_error` | Largest absolute difference between FP32 and INT8 model outputs |
| `fp32_mAP50` / `int8_mAP50` | mAP50 of the FP32 and INT8 models |
| `delta_mAP50` | The INT8-vs-FP32 mAP50 drop (accuracy lost to quantization) |

**QAT-finetune metric** (logged on the qat-finetune run when `quantization-mode=qat`):

| Metric | What It Means |
| --- | --- |
| `qat_finetune_loss` | Loss during quantization-aware fine-tuning |

### Visualizing Metric Charts

1. Select one or more metrics from the list on the left
2. The chart area renders a line plot with epoch on the x-axis
3. Use the controls to:
    - Switch between **linear** and **log** scale (useful for loss curves)
    - **Zoom** into specific epoch ranges by clicking and dragging
    - **Download** the data as CSV for external analysis
    - Toggle **smoothing** for noisy metrics

### Useful Chart Combinations

| Combination | What It Tells You |
|---|---|
| `val/mAP50` + `val/mAP50_95` | Track overall model quality improvement over epochs |
| `train/box_loss` + `train/pose_loss` | Watch for convergence or divergence between loss terms |
| `system/gpu_vram_used_gb` + `system/gpu_utilization_pct` | Check if GPU is being fully utilized |
| `system/ram_used_gb` | Detect memory leaks during long training runs |

!!! tip "Low GPU utilization?"
    If `system/gpu_utilization_pct` is consistently low, the bottleneck is likely data loading, not compute. Try increasing the number of data loader workers or reducing image size.

### Artifacts Tab

Shows files uploaded during the run:

| Artifact | Description |
| --- | --- |
| `best.pt` | Model checkpoint with the highest validation mAP50 |
| `last.pt` | Model checkpoint from the final epoch |
| `args.yaml` | Full Ultralytics configuration used for the run |
| `results.png` | Training metrics plotted across all epochs |
| `confusion_matrix.png` | Confusion matrix on validation data |
| `P_curve.png` | Precision-confidence curve |
| `R_curve.png` | Recall-confidence curve |
| `PR_curve.png` | Precision-recall curve |

Click any image to preview it inline. Click a `.pt` file to download it.

!!! info "INT8 model lives on the quantization run"
    When quantization runs, the INT8 `.tflite` is logged as an artifact on the **model-quantization run** — `best_int8.tflite` for PTQ, `model_int8.tflite` for QAT. It is **not** placed in the Model Registry; the registry holds the FP32 `best.pt` (and optionally `last.pt`).

## Comparing Runs

To compare two or more runs side-by-side:

1. In the runs table, tick the checkboxes next to the runs you want to compare.
2. Click **Compare** above the table.

The comparison view shows:

- **Parameter diff** — highlights which parameters differ between runs
- **Metric charts** — overlaid training curves for direct visual comparison
- **Scatter plots** — plot any metric against any parameter (e.g., `lr0` vs `val/mAP50`)

This is the fastest way to answer *"what changed between my best and worst run?"*

---

## Key Terminology

| Term | Definition |
|---|---|
| **Tracking URI** | The URL of the MLflow server (e.g., `http://mlflow.example.com:5000`) |
| **Experiment** | A named collection of runs (e.g., `spacecraft-pose-toy`) |
| **Run** | One training execution — has params, metrics, artifacts, and tags |
| **Parameter** | A training input logged once per run (e.g., `lr0=0.01`) |
| **Metric** | A training output, optionally logged per step/epoch (e.g., `val/mAP50=0.82`) |
| **Artifact** | A file attached to a run (e.g., `best.pt`, `results.png`) |
| **Tag** | A key-value label on a run or model version (e.g., `dataset_version=v1`) |
| **Registered Model** | A named model entry in the registry with one or more versions |
| **Model Version** | A specific checkpoint registered under a model name |
| **Stage** | A lifecycle label on a model version: `None`, `Staging`, `Production`, `Archived` |
| **Backend Store** | Database (PostgreSQL) that holds run metadata, params, metrics |
| **Artifact Store** | Object storage (S3) that holds model files and other binary artifacts |
| **Proxy Artifacts** | Artifact uploads that go through the tracking server instead of directly to S3 |

---

Continue to reading and understanding your metrics in detail:

[:octicons-arrow-right-24: Reading Metrics](../results/metrics.md)
