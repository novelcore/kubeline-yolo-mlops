# Troubleshooting

Common failure modes and how to resolve them.

---

## Config Validation failures

### `pipeline_config.yaml validation failed: image_size must be a multiple of 32`

`image_size` must be 320, 352, 384, ..., 640, 672, ... etc.

**Fix:** Change `training.image_size` to the nearest multiple of 32.

### `Failed to reach MLflow at http://...`

The config validation step checks that the MLflow server is reachable before proceeding.

**Fix:** Verify the `MLFLOW_TRACKING_URI` environment variable is set correctly.
Ask your platform administrator for the correct MLflow URL for your project.

If you are running locally and do not have MLflow available, set:

```bash
SKIP_LIVENESS_CHECKS=true
```

### `LakeFS repository 'my-repo' not found`

The `dataset.lakefs_repo` value does not match any existing LakeFS repository.

**Fix:** Check the repository name in the LakeFS UI. Repository names are case-sensitive.

---

## Dataset Loading failures

### `No images found at s3://...`

The S3 path does not contain any `.jpg`, `.jpeg`, or `.png` files in an `images/` subdirectory.

**Fix:** Verify the path exists and your dataset is in the expected [YOLO Pose format](../configure/datasets.md).

### `Label format invalid: expected 11 keypoints, found 7`

Your label files have a different number of keypoints than the pipeline expects.

**Fix:** The pipeline is configured for 11-keypoint spacecraft pose annotations (SPEED+ format).
If your dataset has a different keypoint count, contact your platform team to update the pipeline configuration.

### `Access denied to s3://...`

The pipeline does not have permission to read from the specified S3 bucket.

**Fix:** Contact your platform administrator. The S3 bucket policy needs to grant read access to the pipeline's service account.

---

## Model Training failures

### GPU pod stays in Pending state

The training step cannot schedule because no GPU nodes are available.

**Fix:**

1. Check that `spec.operators.nvidiaGpu.enabled: true` is set on your KubePool.
2. Verify your KubeApp `spec.mlPipeline.gpu.maxResources` allows at least 1 GPU.
3. Check AWS EC2 service quotas for the requested GPU instance family.

### `CUDA out of memory`

The model and batch do not fit in GPU VRAM.

**Fix:** Reduce one or more of these in `pipeline_config.yaml`:

```yaml
training:
  batch_size: 8          # Halve the batch size
  image_size: 320        # Reduce image resolution
  amp: true              # Ensure AMP is enabled (halves memory)
```

### `No active MLflow run` warnings

The Ultralytics MLflow callback did not activate — metrics may not be logged.

**Fix:** Confirm `MLFLOW_TRACKING_URI` is set and reachable from within the training pod:

```bash
# From a debug pod in the ml-<project> namespace:
curl http://mlflow.example.com:5000/api/2.0/mlflow/experiments/list
```

### Training loss is NaN after a few epochs

Numerical instability during training.

**Fix:**

1. Ensure `training.amp: true` (mixed precision helps stabilize gradients).
2. Reduce `training.learning_rate` (try dividing by 10).
3. Check that the dataset does not contain corrupt images.

---

## Model Registration failures

### `Model registration failed after 3 retries`

The registration step could not connect to MLflow reliably.

**Fix:** Check MLflow server health and network connectivity from within the cluster.
Increase `MAX_RETRIES` and `TIMEOUT` environment variables if the server is slow but reachable.

### `best.pt artifact URI is not accessible`

The S3 path for `best.pt` could not be read by the registration step.

**Fix:** Verify the `checkpointing.storage_path` bucket is accessible from both the training and registration pods.
Both steps need read/write access to the same S3 bucket.

---

## Argo Workflows issues

### `WorkflowTemplate not found in namespace ml-<project>`

The Argo WorkflowTemplate was not yet synced from the GitOps repository.

**Fix:**

1. Check that the KubeApp is in `Ready` state.
2. Wait 2–3 minutes for ArgoCD to sync the WorkflowTemplate.
3. Verify the template exists: go to Argo UI → Workflow Templates in your namespace.

### Workflow stuck in `Pending` for more than 5 minutes

The first step (Config Validation) is not starting.

**Fix:**

1. Check CPU node availability in your KubePool.
2. Verify Karpenter is running: check KubePool status for `operators.karpenter` readiness.
3. Check Argo controller logs for scheduling errors.
