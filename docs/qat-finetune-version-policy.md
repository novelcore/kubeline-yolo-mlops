# qat-finetune version policy (CON-01)

The `qat_finetune` step pins **exact** versions of the CUDA / QAT stack because the
PT2E QAT → INT8 TFLite path is sensitive to torch / torchao / litert-torch
compatibility. Versions live in two pip lockfiles (installed in the Dockerfile,
after the Poetry deps):

- `qat_finetune/requirements-cuda.txt` — PyTorch CUDA wheels (from the pinned index)
- `qat_finetune/requirements-qat.txt` — the QAT extension stack

## Current pins

| Package | Version | Source |
|---|---|---|
| torch | `2.11.0` | `--index-url https://download.pytorch.org/whl/cu126` |
| torchvision | `0.26.0` | cu126 index |
| torchao | `0.17.0` | PyPI |
| litert-torch | `0.9.1` | PyPI |
| ultralytics | `8.4.19` | PyPI |

**CUDA index note:** `torch 2.11.0` / `torchvision 0.26.0` are **not** published on the
`cu121` index (which tops out at torch 2.5.1) — they exist on **cu126+**. The index is
therefore `cu126`. The GKE T4 node driver supports it (the `model-training` step already
runs the PyPI-default cu126 torch wheel on that node).

## How to bump versions

These wheels are **Linux/CUDA-only** — they cannot be installed or validated on macOS/arm.
Do not bump casually. Procedure:

1. Update `requirements-cuda.txt` and/or `requirements-qat.txt` (keep `==` pins; keep the
   index matching the torch CUDA build).
2. The Dockerfile runs a **build-time import verification** (`torch`, `torchvision`,
   `torchao`, `litert_torch`, `ultralytics`) — a failed build here catches broken .so
   links / version conflicts before deploy.
3. Run the **GPU smoke-test** (FR-M-09) inside the built image on a CUDA host:
   `docker run --gpus all io-qat-finetune python tools/qat_gpu_smoketest.py`
   (or run the qat-finetune step on the T4 node) — confirms `torch.export` on the
   YOLOv8-pose model, PT2E prepare/convert, and the litert INT8 TFLite export all work.
4. Only then commit the updated lockfiles.

## Validation gates (in order)
1. `docker build` — deps resolve + import check (CI kaniko build).
2. GPU smoke-test / a real qat-finetune run on the T4 — the QAT logic itself.
