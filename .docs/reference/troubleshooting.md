# Troubleshooting

Common failure modes and how to resolve them.

---

## Config Validation failures

### `image_size must be a multiple of 32`

`image-size` must be 320, 352, 384, ..., 640, 672, ... etc.

**Fix:** Change the `image-size` parameter to the nearest multiple of 32 when resubmitting the workflow.

### `Failed to reach MLflow at http://...`

The config validation step checks that the MLflow server is reachable before proceeding.

**Fix:** Verify the `MLFLOW_TRACKING_URI` environment variable is set correctly.
Ask your platform administrator for the correct MLflow URL for your project.

If you are running locally and do not have MLflow available, set:

```bash
SKIP_LIVENESS_CHECKS=true
```

### `LakeFS repository 'my-repo' not found`

The `lakefs-repo` parameter value does not match any existing LakeFS repository. Note that `lakefs-repo` is a platform-injected parameter — it is pre-filled by the KAOS platform, not set by the user.

**Fix:** Verify the repository name in the LakeFS UI (names are case-sensitive). If it is incorrect, contact your platform administrator to update the platform configuration.

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

**Fix:** Reduce one or more of these Argo submission parameters when resubmitting:

| Parameter | Suggested Value | Why |
| --- | --- | --- |
| `batch-size` | `8` | Halve the batch size |
| `image-size` | `320` | Reduce image resolution |
| `amp` | `true` | Ensure AMP is enabled (halves memory) |

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

1. Ensure `amp` is set to `true` (mixed precision helps stabilize gradients).
2. Reduce `learning-rate` (try dividing by 10).
3. Check that the dataset does not contain corrupt images.

---

## Model Quantization & QAT failures

### `calibration_frames Input should be greater than or equal to 100`

The `model-quantization` step (and `qat-finetune`, for QAT) rejects a
`quantization-calibration-frames` value below 100 — INT8 calibration needs at
least ~100 representative frames to learn activation ranges reliably.

**Fix:** Set `quantization-calibration-frames` to a value in **100–10000**. The
dataset must actually contain at least that many frames — on a 100-image dataset
`calibration-frames` can be at most `100`. Use a larger dataset if you need more.

### `qat-finetune` sits in `Pending` or fails to start

`qat-finetune` runs on a GPU node (`gpu-t4`); like training it waits for GPU
capacity and is not "failed" while `Pending`.

**Fix:** Confirm GPU capacity is available (see *Model Training → GPU pod stays in
Pending*). Note `qat-finetune` only runs when `quantization-mode=qat`; for `ptq`
or `none` it is skipped.

### Parity check reports a large error / `parity_passed=false`

`parity_max_abs_error` is an **absolute** difference between the FP32 and INT8
outputs, so it scales with the model's output magnitude — a well-trained model
can show a large number and still be fine. The parity check is **non-fatal**: the
INT8 artifact is still produced and the run continues.

**Fix:** Read parity alongside the quantization-run mAP metrics
(`fp32_mAP50` vs `int8_mAP50`), not in isolation.

### INT8 mAP is much lower than the FP32 model

Naive per-tensor INT8 quantization is coarse and can sharply reduce the accuracy
of a YOLO pose model — the FP32 model may be strong while the INT8 mAP drops.

**Fix:** This is a known limitation of per-tensor quantization. Prefer **QAT**
(`quantization-mode=qat`), which trains the model to tolerate the rounding, or
move to per-channel quantization for deployable INT8 accuracy.

---

## Model Registration failures

### `Model registration failed after 3 retries`

The registration step could not connect to MLflow reliably.

**Fix:** Check MLflow server health and network connectivity from within the cluster.
Increase `MAX_RETRIES` and `TIMEOUT` environment variables if the server is slow but reachable.

### `best.pt artifact URI is not accessible`

The S3 path for `best.pt` could not be read by the registration step.

**Fix:** Verify the `checkpoint-bucket` (a platform-injected parameter) is accessible from both the training and registration pods.
Both steps need read/write access to the same S3 bucket.

---

## MLflow issues

### Metrics appear as single points instead of line charts

If metrics show as a single dot rather than a curve over epochs, they were logged without a `step` parameter. The pipeline's custom callbacks log with `step=current_epoch`. If you see flat metrics, the Ultralytics callback may be overriding with step-less logging — check the Ultralytics MLflow integration version (v8.0.196+ required).

### `Failed to ensure MLflow experiment` error

The training service could not create or find the experiment. Check:

- Network connectivity to the MLflow tracking server
- Authentication credentials (`MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`)
- Server health: `curl <MLFLOW_TRACKING_URI>/api/2.0/mlflow/experiments/list`

### Artifact upload is slow

Artifacts are uploaded through the MLflow tracking server (proxy mode), not directly to S3. Large files like `best.pt` (~25 MB) go through the server first. This is by design to avoid giving training containers direct S3 credentials. If this is too slow, consider increasing MLflow server resources.

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
