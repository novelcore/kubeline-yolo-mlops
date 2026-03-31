# MLflow Beginner's Guide

A practical introduction to MLflow and how it fits into our YOLO MLOps pipeline.

---

## Table of Contents

1. [What Is MLflow?](#1-what-is-mlflow)
2. [Core Concepts](#2-core-concepts)
3. [How MLflow Is Deployed](#3-how-mlflow-is-deployed)
4. [How Our Pipeline Uses MLflow](#4-how-our-pipeline-uses-mlflow)
5. [Using the MLflow Dashboard](#5-using-the-mlflow-dashboard)
6. [Using the MLflow Python SDK](#6-using-the-mlflow-python-sdk)
7. [Common Workflows](#7-common-workflows)
8. [Troubleshooting](#8-troubleshooting)
9. [Key Terminology Cheat Sheet](#9-key-terminology-cheat-sheet)

---

## 1. What Is MLflow?

MLflow is an open-source platform for managing the machine learning lifecycle. Think of it as **version control for your experiments** — it records what you trained, how you trained it, and what the results were, so you never lose track of a run.

MLflow has four main components. We use two of them:

| Component              | What It Does                                    | We Use It? |
|------------------------|-------------------------------------------------|------------|
| **Tracking**           | Logs parameters, metrics, and artifacts per run | Yes        |
| **Model Registry**     | Versions and stages trained models              | Yes        |
| **Projects**           | Packages ML code in a reproducible format       | No         |
| **Models**             | Standard format for deploying models            | No         |

---

## 2. Core Concepts

### Experiments

An **experiment** is a named group of related training runs. In our pipeline, the experiment name is set in `pipeline_config.yaml`:

```yaml
experiment:
  name: "spacecraft-pose-toy"
```

All runs for this experiment appear together in the MLflow UI, making it easy to compare them.

### Runs

A **run** is a single execution of the training step. Each run has:

- **Parameters** — Inputs to training (learning rate, batch size, epochs, optimizer, etc.)
- **Metrics** — Output measurements (loss, mAP50, precision, recall, etc.), optionally logged at each epoch (step)
- **Artifacts** — Files produced by the run (model checkpoints, plots, config files)
- **Tags** — Key-value metadata (dataset version, git commit, pipeline ID)

### Registered Models

A **registered model** is a named entry in the Model Registry. Each registration creates a new version:

```
Registered Model: "spacecraft-pose-yolo"
  ├── Version 1  (best.pt from run abc123)
  ├── Version 2  (best.pt from run def456)
  └── Version 3  (best.pt from run ghi789)  ← latest
```

### Model Stages

Registered model versions can be promoted through stages to control deployment:

```
None → Staging → Production → Archived
```

---

## 3. How MLflow Is Deployed

Our MLflow instance runs as a remote tracking server with the following architecture:

```
┌─────────────────────┐
│   MLflow UI + API    │  ← http://mlflow.example.com:5000
│   (Tracking Server)  │
├─────────────────────┤
│  Backend Store:      │  ← PostgreSQL (stores params, metrics, tags, metadata)
│  Artifact Store:     │  ← S3 bucket (stores model files, plots, large outputs)
└─────────────────────┘
```

- **Backend store** (PostgreSQL) holds structured data: run metadata, parameters, metrics, and tags.
- **Artifact store** (S3) holds binary files: model checkpoints (`best.pt`, `last.pt`), training plots, and config files.
- **Proxy artifact storage** — Our pipeline creates experiments with `artifact_location="mlflow-artifacts:"`, which means artifact uploads go *through* the tracking server. The client does not need direct S3 credentials for artifact logging.

### Connection Settings

MLflow access is configured via environment variables in each step's `.env` file:

```bash
MLFLOW_TRACKING_URI=http://mlflow.example.com:5000   # Required — server address
MLFLOW_TRACKING_USERNAME=...                        # Optional — basic auth
MLFLOW_TRACKING_PASSWORD=...                     # Optional — basic auth
```

In Kubernetes, these are injected from Secrets — they never appear in workflow YAML or source code.

---

## 4. How Our Pipeline Uses MLflow

MLflow is used in two pipeline steps: `model_training` and `model_registration`.

### Step 3: model_training — Experiment Tracking

The training step uses **Ultralytics' built-in MLflow callback** to automatically log everything. The training service sets two environment variables before calling `model.train()`:

```python
os.environ["MLFLOW_TRACKING_URI"] = self._mlflow_tracking_uri
os.environ["MLFLOW_EXPERIMENT_NAME"] = params.experiment_name
```

The Ultralytics library detects these and handles all MLflow communication automatically.

#### What Gets Logged Automatically (by Ultralytics)

| Category    | Examples                                                                 |
|-------------|--------------------------------------------------------------------------|
| Parameters  | `epochs`, `batch_size`, `lr0`, `optimizer`, `imgsz`, `model_variant`     |
| Metrics     | `train/box_loss`, `train/pose_loss`, `val/mAP50`, `val/mAP50-95`        |
| Artifacts   | `best.pt`, `last.pt`, training plots, `args.yaml`                        |

#### What Gets Logged by Our Custom Callbacks

On top of the built-in logging, our pipeline adds:

| Callback               | What It Logs                                           |
|------------------------|--------------------------------------------------------|
| `on_fit_epoch_end`     | Validation metrics (`val/precision`, `val/recall`, `val/mAP50`, `val/mAP50_95`) and system resources (`gpu/memory_used_mb`, `gpu/utilization_pct`, `system/ram_used_mb`) — all logged per epoch with a step number |
| `on_train_batch_end`   | Captures per-batch training losses (`train/box_loss`, `train/pose_loss`, etc.) into a local dict |
| `on_train_epoch_end`   | Uploads periodic checkpoints to S3 every N epochs      |
| `on_train_end`         | Logs a completion summary to the console                |

#### Kubecore Pipeline Metadata

After training, the service tags the MLflow run with `KUBECORE_*` environment variables injected by the Kubeline platform:

```
kubecore.project_name = ...
kubecore.workflow_name = ...
kubecore.app_name = ...
```

This makes every run traceable back to its pipeline execution.

### Step 4: model_registration — Model Registry

After training completes, the registration step:

1. **Registers `best.pt`** under the configured model name (e.g., `spacecraft-pose-yolo`)
2. **Registers `last.pt`** as a separate version (derived from the best.pt path)
3. **Sets lineage tags** on each version for traceability:

| Tag                   | Example Value                   |
|-----------------------|---------------------------------|
| `checkpoint_type`     | `best` or `last`                |
| `training_run_id`     | `abc123def456`                  |
| `dataset_version`     | `v1`                            |
| `dataset_sample_size` | `1000`                          |
| `config_hash`         | `sha256:...`                    |
| `git_commit`          | `a1b2c3d`                       |
| `model_variant`       | `yolov8n-pose.pt`               |
| `best_mAP50`          | `0.82`                          |

4. **Promotes the version** to a stage (`Staging` or `Production`) if configured:

```yaml
registration:
  registered_model_name: "spacecraft-pose-yolo"
  promote_to: "Staging"
```

---

## 5. Using the MLflow Dashboard

Open your browser to the `MLFLOW_TRACKING_URI` (e.g., `http://mlflow.example.com:5000`). If basic auth is enabled, you will be prompted for the username and password configured via `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`.

### Dashboard Layout

The MLflow UI has three main areas:

```
┌──────────────────────────────────────────────────────────┐
│  [Experiments]   [Models]   [⚙ Settings]                 │  ← Top navigation bar
├────────────┬─────────────────────────────────────────────┤
│            │                                             │
│ Experiment │         Runs Table / Run Detail             │
│   List     │                                             │
│  (sidebar) │                                             │
│            │                                             │
└────────────┴─────────────────────────────────────────────┘
```

- **Top navigation bar** — Switch between Experiments (tracking) and Models (registry).
- **Sidebar** — Lists all experiments. Click one to load its runs.
- **Main pane** — Shows the runs table, or a single run's detail view when you click into one.

### Experiments Page

The sidebar lists all experiments by name. Our pipeline creates one experiment per Argo Workflows entry:

```yaml
experiment:
  name: "spacecraft-pose-toy"
```

Click an experiment to see all runs that belong to it.

### Runs Table

Each row is a training run. The table columns include:

| Column       | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| **Run Name** | Set by the pipeline to the Kubeline workflow name (e.g., `spacecraft-pose-toy-20260324-143022`) |
| **Created**  | Timestamp when the run started                                              |
| **Duration** | Wall-clock time of the run                                                  |
| **Status**   | `RUNNING`, `FINISHED`, or `FAILED`                                          |
| **Params**   | Expandable — shows all hyperparameters (`lr0`, `epochs`, `batch`, etc.)      |
| **Metrics**  | Final values for each logged metric (`val/mAP50`, `val/mAP50_95`, etc.)     |
| **Tags**     | Metadata like `kubecore.project_name` and `kubecore.workflow_name`          |

You can customize visible columns by clicking the **Columns** button above the table.

#### Filtering and Searching Runs

Use the search bar above the runs table to filter runs by parameters, metrics, or tags:

```
# Runs with mAP50 above 0.5
metrics.`val/mAP50` > 0.5

# Runs that used a specific learning rate
params.lr0 = "0.01"

# Runs from a specific pipeline execution
tags.`kubecore.workflow_name` = "spacecraft-pose-toy-20260324-143022"
```

You can also sort by any column — click the column header to sort ascending/descending.

### Run Detail View

Click a run name to open its detail page. The detail view has several tabs:

#### Overview Tab

Shows a summary of the run including:
- **Run ID** — The unique identifier (e.g., `abc123def456`)
- **Status** and **Duration**
- **Parameters** — Full list of all logged hyperparameters
- **Tags** — All metadata tags including `kubecore.*` pipeline tags and `mlflow.runName`

#### Metrics Tab

Displays all metrics logged during training. Our pipeline logs two categories of metrics per epoch:

**Model Metrics** (from our `on_fit_epoch_end` callback):

| Metric            | Description                              |
|-------------------|------------------------------------------|
| `val/precision`   | Validation precision                     |
| `val/recall`      | Validation recall                        |
| `val/mAP50`       | Mean average precision at IoU=0.50       |
| `val/mAP50_95`    | Mean average precision at IoU=0.50:0.95  |

**Training Losses** (from the Ultralytics built-in callback):

| Metric              | Description                                |
|----------------------|--------------------------------------------|
| `train/box_loss`     | Bounding box regression loss               |
| `train/pose_loss`    | Keypoint pose estimation loss              |
| `train/kobj_loss`    | Keypoint objectness loss                   |
| `train/cls_loss`     | Classification loss                        |
| `train/dfl_loss`     | Distribution focal loss                    |

**System Resource Metrics** (from our `on_fit_epoch_end` callback via `ResourceMonitor`):

| Metric                       | Description                               |
|------------------------------|-------------------------------------------|
| `system/ram_used_gb`         | System RAM in use (GB)                    |
| `system/ram_percent`         | System RAM usage as a percentage          |
| `system/cpu_percent`         | CPU utilization percentage                |
| `system/gpu_vram_used_gb`    | GPU VRAM in use (GB) — requires GPU       |
| `system/gpu_vram_total_gb`   | Total GPU VRAM (GB) — requires GPU        |
| `system/gpu_utilization_pct` | GPU compute utilization (%) — requires GPU|

All metrics are logged with `step=current_epoch`, which means the dashboard renders them as line charts over epochs.

**Visualizing metric charts:**

1. Select one or more metrics from the list on the left
2. The chart area renders a line plot with epoch on the x-axis
3. Use the controls to:
   - Switch between **linear** and **log** scale (useful for loss curves)
   - **Zoom** into specific epoch ranges by clicking and dragging
   - **Download** the data as CSV for external analysis
   - Toggle **smoothing** for noisy metrics

**Useful chart combinations:**

- `val/mAP50` + `val/mAP50_95` — track overall model quality improvement
- `train/box_loss` + `train/pose_loss` — watch for convergence or divergence
- `system/gpu_vram_used_gb` + `system/gpu_utilization_pct` — check if GPU is being fully utilized
- `system/ram_used_gb` — detect memory leaks during long training runs

#### Artifacts Tab

Shows all files uploaded during the run. Our pipeline produces:

| Artifact        | Description                                                      |
|-----------------|------------------------------------------------------------------|
| `best.pt`       | Model checkpoint with the highest validation mAP50               |
| `last.pt`       | Model checkpoint from the final epoch                            |
| `args.yaml`     | Full Ultralytics configuration used for the run                  |
| `results.png`   | Training metrics plotted across all epochs                       |
| `confusion_matrix.png` | Confusion matrix on validation data                       |
| `P_curve.png`   | Precision-confidence curve                                       |
| `R_curve.png`   | Recall-confidence curve                                          |
| `PR_curve.png`  | Precision-recall curve                                           |

Click any file to preview it (images render inline) or download it.

### Comparing Runs

1. Go back to the runs table for an experiment
2. Select two or more runs using the checkboxes on the left
3. Click the **Compare** button

The comparison view shows:

- **Parameter diff** — Side-by-side table highlighting parameters that differ between runs (e.g., different learning rates or batch sizes)
- **Metric charts** — Overlaid line charts so you can compare training curves visually
- **Scatter plots** — Plot any metric against any parameter to find correlations (e.g., `lr0` vs `val/mAP50`)

This is the fastest way to answer "what changed between run A and run B?" and "which hyperparameters led to better results?"

### Model Registry Page

Click **Models** in the top navigation bar to access the Model Registry.

#### Registered Models List

Shows all registered model names (e.g., `spacecraft-pose-yolo`). Each entry displays:
- **Latest version number**
- **Last updated timestamp**
- **Version counts per stage** (None / Staging / Production / Archived)

#### Model Version Detail

Click a model name and then a version number to see:
- **Source run** — Link back to the training run that produced this checkpoint
- **Lineage tags** — `training_run_id`, `dataset_version`, `dataset_sample_size`, `config_hash`, `git_commit`, `model_variant`, `best_mAP50`, `checkpoint_type`
- **Current stage** — `None`, `Staging`, `Production`, or `Archived`
- **Stage transition history** — Who promoted the version and when

#### Promoting a Model Version

To move a model version to a new stage via the UI:
1. Open the model version detail page
2. Click the **Stage** dropdown
3. Select the target stage (`Staging`, `Production`, or `Archived`)
4. Add an optional comment explaining why
5. Click **Transition**

### Dashboard Tips

- **Bookmark experiment URLs** — Each experiment has a stable URL like `http://mlflow.example.com:5000/#/experiments/1` that you can bookmark or share with teammates.
- **Use tags for filtering** — Our pipeline sets `kubecore.*` tags on every run. Use `tags.kubecore.project_name = ...` in the search bar to filter by project.
- **Pin important metrics** — In the runs table, click the column settings to show `val/mAP50` as a default column so you can quickly spot the best runs.
- **Check system metrics** — If a run is slower than expected, check the `system/gpu_utilization_pct` chart. Low GPU utilization often means the bottleneck is data loading, not compute.
- **Watch for OOM risk** — If `system/gpu_vram_used_gb` is close to `system/gpu_vram_total_gb`, reduce `batch_size` or `imgsz` to avoid out-of-memory crashes.

---

## 6. Using the MLflow Python SDK

### Installation

```bash
pip install mlflow
```

### Connecting to the Server

```python
import mlflow

mlflow.set_tracking_uri("http://mlflow.example.com:5000")
```

### Listing Experiments

```python
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"{exp.name} (id={exp.experiment_id})")
```

### Searching Runs

```python
# Find all runs in an experiment
runs = mlflow.search_runs(experiment_names=["spacecraft-pose-toy"])

# Filter by metric
best_runs = mlflow.search_runs(
    experiment_names=["spacecraft-pose-toy"],
    filter_string="metrics.`val/mAP50` > 0.5",
    order_by=["metrics.`val/mAP50` DESC"],
)
print(best_runs[["run_id", "params.epochs", "metrics.val/mAP50"]])
```

### Retrieving a Specific Run

```python
client = mlflow.tracking.MlflowClient()
run = client.get_run("abc123def456")

print(f"Status: {run.info.status}")
print(f"Parameters: {run.data.params}")
print(f"Metrics: {run.data.metrics}")
print(f"Tags: {run.data.tags}")
```

### Downloading Artifacts

```python
# Download best.pt from a specific run
local_path = client.download_artifacts("abc123def456", "best.pt", dst_path="/tmp")
print(f"Downloaded to: {local_path}")
```

### Listing Model Versions

```python
versions = client.search_model_versions("name='spacecraft-pose-yolo'")
for v in versions:
    print(f"Version {v.version} | Stage: {v.current_stage} | Run: {v.run_id}")
```

---

## 7. Common Workflows

### "Which run produced this model version?"

```python
client = mlflow.tracking.MlflowClient()
version = client.get_model_version("spacecraft-pose-yolo", "3")
run_id = version.tags.get("training_run_id")
run = client.get_run(run_id)
print(run.data.params)  # Full hyperparameters for that model
```

### "What dataset was used to train model version 3?"

```python
version = client.get_model_version("spacecraft-pose-yolo", "3")
print(version.tags.get("dataset_version"))      # e.g., "v1"
print(version.tags.get("dataset_sample_size"))   # e.g., "1000"
```

### "Compare all runs that used learning rate 0.01"

```python
runs = mlflow.search_runs(
    experiment_names=["spacecraft-pose-toy"],
    filter_string="params.lr0 = '0.01'",
    order_by=["metrics.`val/mAP50` DESC"],
)
```

### "Promote a model to Production"

```python
client.transition_model_version_stage(
    name="spacecraft-pose-yolo",
    version="3",
    stage="Production",
)
```

### "Find the best run in an experiment"

```python
best = mlflow.search_runs(
    experiment_names=["spacecraft-pose-toy"],
    order_by=["metrics.`val/mAP50` DESC"],
    max_results=1,
)
print(best.iloc[0]["run_id"])
```

---

## 8. Troubleshooting

### "No active MLflow run" warnings in training logs

The Ultralytics MLflow callback did not activate. Check that:
- `MLFLOW_TRACKING_URI` is set and reachable
- The `mlflow` Python package is installed in the training container
- The Ultralytics version supports the MLflow callback (v8.0.196+)

### "Failed to ensure MLflow experiment"

The training service could not create or find the experiment. Check:
- Network connectivity to the MLflow tracking server
- Authentication credentials (`MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`)
- The server is running: `curl http://mlflow.example.com:5000/api/2.0/mlflow/experiments/list`

### Metrics not appearing per epoch

If metrics appear as a single point instead of a line chart, they were logged without a `step` parameter. Our custom callbacks log with `step=current_epoch` — if you see flat metrics, the Ultralytics callback may be overriding with step-less logging. Check the Ultralytics MLflow integration version.

### Model registration fails with retries

The `model_registration` step retries MLflow calls up to 3 times with exponential backoff (1s, 2s, 4s). If all retries fail, check:
- MLflow server health
- The `best.pt` artifact URI is valid and accessible
- Network timeouts (`TIMEOUT` env var, default 60s)

### Artifact upload is slow

Artifacts are uploaded through the MLflow tracking server (proxy mode). Large files like `best.pt` (~25 MB for YOLOv8n) go through the server before reaching S3. This is by design — it avoids giving the training container direct S3 write credentials. If this is too slow, consider increasing server resources.

---

## 9. Key Terminology Cheat Sheet

| Term                    | Definition                                                                          |
|-------------------------|-------------------------------------------------------------------------------------|
| **Tracking URI**        | The URL of the MLflow server (`http://mlflow.example.com:5000`)                     |
| **Experiment**          | A named collection of runs (e.g., `spacecraft-pose-toy`)                            |
| **Run**                 | One training execution — has params, metrics, artifacts, and tags                   |
| **Parameter**           | A training input logged once per run (e.g., `lr0=0.01`)                             |
| **Metric**              | A training output, optionally logged per step/epoch (e.g., `val/mAP50=0.82`)       |
| **Artifact**            | A file attached to a run (e.g., `best.pt`, `results.png`)                           |
| **Tag**                 | A key-value label on a run or model version (e.g., `dataset_version=v1`)            |
| **Registered Model**    | A named model entry in the registry with one or more versions                       |
| **Model Version**       | A specific checkpoint registered under a model name                                 |
| **Stage**               | A lifecycle label on a model version: `None`, `Staging`, `Production`, `Archived`   |
| **Backend Store**       | Database (PostgreSQL) that holds run metadata, params, metrics                      |
| **Artifact Store**      | Object storage (S3) that holds model files and other binary artifacts               |
| **Proxy Artifacts**     | Artifact uploads that go through the tracking server instead of directly to S3      |
