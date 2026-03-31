# Argo Logs

How to navigate logs for a running or completed pipeline in Argo Workflows.

## Finding Your Workflow

1. Open the Argo Workflows UI and select your project namespace (`ml-<project-name>`).
2. Click **Workflows** in the sidebar.
3. Find your run in the list — workflows are named with a timestamp, e.g., `kubeline-yolo-mlops-20260401-143022`.
4. Click the workflow name to open the DAG view.

## Reading the DAG View

The DAG view shows the four pipeline steps as connected nodes:

```
[Config Validation] → [Dataset Loading] → [Model Training] → [Model Registration]
```

**Node colours:**

| Colour | Status |
| --- | --- |
| Grey | Pending |
| Blue / pulsing | Running |
| Green | Succeeded |
| Red | Failed |
| Orange | Skipped or error |

## Viewing Step Logs

1. Click on any step node to open its detail panel on the right.
2. Click the **Logs** tab.
3. Logs stream in real time for running steps. For completed steps, the full log is shown.

## What Each Step Logs

### Config Validation

A successful run shows:

```
INFO  Validating pipeline config...
INFO  ✓ experiment section valid
INFO  ✓ dataset section valid
INFO  ✓ model section valid
INFO  ✓ training section valid
INFO  ✓ checkpointing section valid
INFO  ✓ registration section valid
INFO  ✓ MLflow server reachable at http://mlflow.example.com:5000
INFO  ✓ S3/LakeFS endpoint reachable
INFO  Config validation passed.
```

If you see `ERROR` lines here, your `pipeline_config.yaml` has invalid values. The error message will indicate which field failed.

### Dataset Loading

A successful run shows:

```
INFO  Connecting to LakeFS at https://lakefs.example.com...
INFO  Downloading dataset v1 from io-data/main...
INFO  Downloaded 5000 images (train: 4000, val: 1000)
INFO  Validated label format: YOLO Pose (11 keypoints)
INFO  Dataset ready at /artifacts/dataset/
```

If you see `No images found` or `Label format invalid`, check your [dataset configuration](../configure/datasets.md).

### Model Training

Training logs are the most verbose. Key lines to look for:

```
# Per-epoch summary (logged every epoch):
Epoch 10/100: box_loss=0.842 pose_loss=3.21 val/mAP50=0.43

# Checkpoint saves:
INFO  Checkpoint saved to s3://your-bucket/checkpoints/epoch_10.pt

# Final summary:
INFO  Training complete. Best mAP50: 0.82 at epoch 87
```

!!! tip "Monitor GPU utilization"
    If you see `gpu/utilization_pct` staying below 30% in the logs, the bottleneck is likely data loading rather than compute. Consider increasing `batch_size` or reducing `image_size`.

### Model Registration

A successful run shows:

```
INFO  Registering model: spacecraft-pose-yolo
INFO  Registered best.pt as version 3
INFO  Registered last.pt as version 4
INFO  Lineage tags set on versions 3 and 4
INFO  Promoted version 3 to Staging
INFO  Registration complete.
```

## Debugging a Failed Step

If a step turns red:

1. Click the failed step node.
2. Open the **Logs** tab and scroll to the bottom — the error is usually the last few lines.
3. Check the **Summary** tab for the exit code and error message.

Common failure causes per step are listed in [Troubleshooting](../reference/troubleshooting.md).
