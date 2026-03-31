# Reading Metrics

After a training run completes, here is how to read and act on the results in MLflow.

## The Primary Metric: mAP50

The most important number after training is `val/mAP50` — **Mean Average Precision at IoU threshold 0.50**.

This is a single number between 0 and 1 that measures how accurately the model detects and localizes objects in validation images.

| mAP50 Range | Interpretation |
| --- | --- |
| < 0.3 | Poor — model is not learning effectively |
| 0.3–0.5 | Developing — may need more epochs or different hyperparameters |
| 0.5–0.7 | Good — viable baseline for further tuning |
| 0.7–0.85 | Strong — production-quality for many use cases |
| > 0.85 | Excellent — typically near the ceiling for this dataset |

For spacecraft pose estimation specifically, `mAP50 > 0.7` is a reasonable target for Phase 1.

## Reading the Metric Charts

In the MLflow run detail page, go to the **Metrics** tab and select metrics to chart.

### Understanding a Healthy Training Curve

A well-behaved training run looks like this:

- `val/mAP50` rises steadily across epochs, then plateaus
- `train/pose_loss` decreases steadily and flattens
- `train/box_loss` decreases in parallel with `pose_loss`

**Signs of problems:**

| Pattern | Likely Cause |
| --- | --- |
| Loss spikes repeatedly | Learning rate too high — reduce `learning_rate` |
| mAP50 plateaus very early (< 20 epochs) | Model may be underfitting — try a larger variant or more epochs |
| Loss decreases but mAP50 stays flat | Validation set mismatch — check dataset distribution |
| mAP50 rises then drops | Overfitting — add `dropout`, reduce `epochs`, or increase `weight_decay` |
| `gpu_utilization_pct` < 30% throughout | Data loading bottleneck — consider increasing `batch_size` |

### Checking System Resource Usage

The **system metrics** charts tell you how efficiently the GPU was used:

- `system/gpu_utilization_pct` should stay above 70% during training steps
- `system/gpu_vram_used_gb` should be comfortably below `system/gpu_vram_total_gb`
  - If they are close, reduce `batch_size` or `image_size` to avoid out-of-memory crashes on future runs

## Comparing Runs

To find which hyperparameter change made the biggest difference:

1. Select the runs you want to compare (checkboxes in the runs table)
2. Click **Compare**
3. Use the **Scatter Plot** view: set X-axis to a parameter (e.g., `learning_rate`) and Y-axis to `val/mAP50`

This quickly reveals which parameter values correlate with higher accuracy.

## Filtering Runs

Use the search bar above the runs table to filter runs programmatically:

```
# Only runs with mAP50 above 0.6
metrics.`val/mAP50` > 0.6

# Only runs that used learning rate 0.001
params.learning_rate = "0.001"

# Only runs from a specific pipeline execution
tags.`kubecore.workflow_name` = "spacecraft-pose-v1-20260401-143022"
```

## Using the Python SDK

You can also query results programmatically:

```python
import mlflow

mlflow.set_tracking_uri("http://mlflow.example.com:5000")

# Find the best run in an experiment
best = mlflow.search_runs(
    experiment_names=["spacecraft-pose-v1-yolov8n"],
    order_by=["metrics.`val/mAP50` DESC"],
    max_results=1,
)
print(best.iloc[0][["run_id", "metrics.val/mAP50", "params.learning_rate"]])
```

---

Ready to move a model to production?

[:octicons-arrow-right-24: Promoting a Model](promotion.md)
