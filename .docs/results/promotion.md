# Promoting a Model

After a run completes, the trained model is automatically registered in the MLflow Model Registry.
Promotion moves a registered model version through lifecycle stages: `None → Staging → Production`.

## What "Promotion" Means

The Model Registry gives each trained checkpoint a **stage** label:

| Stage | Meaning |
| --- | --- |
| `None` | Just registered, not yet evaluated |
| `Staging` | Validated by the ML team, ready for testing in a pre-production environment |
| `Production` | Approved for production workloads |
| `Archived` | Superseded by a newer version, no longer active |

Stages are metadata — they do not deploy the model anywhere. They signal to downstream systems and team members which version to use.

## Automatic Promotion via Parameters

Set the `promote-to` and `registered-model-name` parameters at Argo submission time to automatically promote the model when the pipeline finishes:

| Parameter | Value |
| --- | --- |
| `registered-model-name` | `spacecraft-pose-yolo` |
| `promote-to` | `Staging` |

With these parameters, after every successful run the newly registered `best.pt` is automatically promoted to `Staging`. Set `promote-to` to empty (the default) for no automatic promotion, or to `Production` to promote directly.

## Manual Promotion via MLflow UI

To promote a version manually:

1. Go to the MLflow dashboard and click **Models** in the top navigation bar.
2. Click your model name (e.g., `spacecraft-pose-yolo`).
3. Find the version you want to promote and click it.
4. On the version detail page, click the **Stage** dropdown.
5. Select `Staging`, `Production`, or `Archived`.
6. Add an optional comment (e.g., *"Validated on test set, mAP50=0.78"*).
7. Click **Transition**.

The stage change is immediate and recorded in the version history.

## Manual Promotion via Python SDK

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://mlflow.example.com:5000")

# Promote version 3 of spacecraft-pose-yolo to Production
client.transition_model_version_stage(
    name="spacecraft-pose-yolo",
    version="3",
    stage="Production",
    archive_existing_versions=True,  # Move previous Production version to Archived
)
```

## Finding Which Run Produced a Model Version

Every registered version has full lineage tags set by the pipeline:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://mlflow.example.com:5000")
version = client.get_model_version("spacecraft-pose-yolo", "3")

# Key lineage tags
print(version.tags["training_run_id"])      # MLflow run ID
print(version.tags["dataset_version"])       # e.g., "v1"
print(version.tags["dataset_sample_size"])   # e.g., "5000"
print(version.tags["model_variant"])         # e.g., "yolov8n-pose.pt"
print(version.tags["best_mAP50"])            # e.g., "0.82"
print(version.tags["git_commit"])            # Git commit SHA
print(version.tags["checkpoint_type"])       # "best" or "last"
```

This lets you answer *"exactly what data and config produced this model?"* for any version.
