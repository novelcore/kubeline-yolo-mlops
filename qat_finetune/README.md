# qat-finetune

Quantization-Aware Training (QAT) step of the Kubeline YOLO MLOps pipeline (PRD-174).

Takes the FP32 YOLOv8-pose checkpoint produced by `model_training`, runs a short
QAT fine-tune (PT2E / torchao), and exports an INT8 TFLite model. Runs on a GPU
(CUDA) node.

## Run

```bash
qat-finetune run \
  --fp32-checkpoint-path s3://bucket/checkpoints/best.pt \
  --source-mlflow-run-id <run_id> \
  --dataset-dir /work/dataset \
  --output-dir /work/qat_output \
  --output-bucket <bucket> \
  --output-prefix qat/<exp> \
  --experiment-name <name>
```

Dependency versions for the CUDA/QAT stack are pinned in `requirements-cuda.txt`
and `requirements-qat.txt` (see `docs/qat-finetune-version-policy.md`).
