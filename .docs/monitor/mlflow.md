# MLflow Dashboard

MLflow is where your experiment results are stored. Think of it as version control for your training runs — it records what you trained, how you trained it, and what the results were.

## Opening the Dashboard

Open your MLflow URL in a browser (ask your administrator if you do not have it). If prompted, enter your username and password.

You land on the MLflow home page.

## Dashboard Layout

The MLflow UI has two main sections, accessible from the top navigation bar:

- **Experiments** — lists all training runs, grouped by experiment name
- **Models** — the Model Registry, where trained models are stored and versioned

## Finding Your Experiment

In the left sidebar under **Experiments**, find the experiment name matching your `pipeline_config.yaml`:

```yaml
experiment:
  name: "spacecraft-pose-v1-yolov8n"
```

Click it to see all runs in that experiment.

## The Runs Table

Each row in the runs table is one pipeline execution. Key columns:

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
| `val/mAP50` | Mean average precision at IoU=0.50 — the primary quality indicator |
| `val/mAP50_95` | mAP averaged over IoU thresholds 0.50–0.95 — stricter quality measure |
| `train/pose_loss` | Keypoint regression loss — should decrease steadily |
| `train/box_loss` | Bounding box loss |
| `system/gpu_utilization_pct` | GPU compute usage — should be consistently high (>70%) |
| `system/gpu_vram_used_gb` | GPU memory in use |

### Artifacts Tab

Shows files uploaded during the run:

| Artifact | Description |
| --- | --- |
| `best.pt` | Model checkpoint with the highest `val/mAP50` |
| `last.pt` | Model checkpoint from the final epoch |
| `results.png` | Training metrics plotted over all epochs |
| `args.yaml` | Full configuration used for the run |
| `confusion_matrix.png` | Validation confusion matrix |

Click any image to preview it inline. Click a `.pt` file to download it.

## Comparing Runs

To compare two or more runs side-by-side:

1. In the runs table, tick the checkboxes next to the runs you want to compare.
2. Click **Compare** above the table.

The comparison view shows:

- **Parameter diff** — highlights which parameters differ between runs
- **Metric charts** — overlaid training curves for direct visual comparison
- **Scatter plots** — plot any metric against any parameter (e.g., `learning_rate` vs `val/mAP50`)

This is the fastest way to answer *"what changed between my best and worst run?"*

---

Continue to reading and understanding your metrics in detail:

[:octicons-arrow-right-24: Reading Metrics](../results/metrics.md)
