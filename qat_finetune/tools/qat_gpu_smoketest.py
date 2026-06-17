#!/usr/bin/env python3
"""GPU smoke-test for the qat-finetune production image (FR-M-09).

Verifies the pinned CUDA stack is functional inside the actual container:
  - torch CUDA is available and usable
  - torchao C++ extensions load and execute on GPU
  - litert_torch imports cleanly
  - ultralytics YOLO imports cleanly

Run inside the built image:
    docker run --gpus all io-qat-finetune python tools/qat_gpu_smoketest.py

Completes in ~30 s on a T4. No dataset or checkpoint required.
"""

import sys
import time


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def check_torch_cuda() -> None:
    import torch

    print(f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        _fail(
            "torch.cuda.is_available() is False — ensure the container was "
            "started with --gpus all and the NVIDIA Container Toolkit is installed."
        )
    device_name = torch.cuda.get_device_name(0)
    print(f"  CUDA device: {device_name}")

    # Basic CUDA matmul to confirm execution path works
    device = torch.device("cuda")
    a = torch.randn(512, 512, device=device)
    b = torch.randn(512, 512, device=device)
    c = torch.mm(a, b)
    assert c.shape == (512, 512), f"unexpected shape {c.shape}"
    print("  Matmul (512×512): OK")


def check_torchao() -> None:
    import torch
    import torchao

    print(f"torchao {torchao.__version__}")

    # Exercise the PT2E QAT quantizer C++ path on GPU
    from torchao.quantization import int8_dynamic_activation_int8_weight, quantize_

    device = torch.device("cuda")
    model = torch.nn.Linear(64, 64).to(device).eval()
    quantize_(model, int8_dynamic_activation_int8_weight())
    x = torch.randn(4, 64, device=device)
    out = model(x)
    assert out.shape == (4, 64), f"unexpected shape {out.shape}"
    print("  INT8 dynamic quant (CUDA): OK")


def check_litert_torch() -> None:
    import litert_torch  # noqa: F401

    version = getattr(litert_torch, "__version__", "unknown")
    print(f"litert_torch {version}")
    print("  Import: OK")


def check_ultralytics() -> None:
    import ultralytics
    from ultralytics import YOLO  # noqa: F401

    print(f"ultralytics {ultralytics.__version__}")
    print("  Import: OK")


def main() -> None:
    print("=== qat-finetune GPU smoke-test (FR-M-09) ===\n")
    t0 = time.monotonic()

    check_torch_cuda()
    check_torchao()
    check_litert_torch()
    check_ultralytics()

    elapsed = time.monotonic() - t0
    print(f"\nAll checks passed in {elapsed:.1f}s — image is production-ready.")


if __name__ == "__main__":
    main()
