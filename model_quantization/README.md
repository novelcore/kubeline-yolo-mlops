# model-quantization

Quantization step of the Kubeline YOLO MLOps pipeline (PRD-174).

Produces an INT8 model from the trained FP32 YOLOv8-pose checkpoint. Supports:
- **PTQ** — post-training INT8 quantization (Ultralytics TFLite int8 export with a
  calibration set), and
- **QAT pass-through** — packages/validates the INT8 TFLite produced by `qat-finetune`.

Also runs the FP32-vs-INT8 parity check (FR-M-03). CPU step.

## Run

```bash
model-quantization run \
  --mode ptq \
  --fp32-checkpoint-path s3://bucket/checkpoints/best.pt \
  --source-mlflow-run-id <run_id> \
  --dataset-dir /work/dataset \
  --output-dir /work/quant_output \
  --output-bucket <bucket> \
  --output-prefix quant/<exp> \
  --experiment-name <name>
```
