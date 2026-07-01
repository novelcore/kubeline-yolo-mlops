"""Test configuration for qat_finetune.

Sets up sys.modules stubs for torch, torchao, litert_torch, and ultralytics
before any test imports the service. This allows tests to run without the
pinned QAT stack (torch==2.11.0 + CUDA) being installed in the test env.
"""

import sys
from unittest.mock import MagicMock

# ── torch ──────────────────────────────────────────────────────────────────────
_mock_torch = MagicMock(name="torch")
_mock_torch.cuda.is_available.return_value = False
_mock_torch.zeros.return_value = MagicMock(name="torch.zeros_result")

_mock_torch_nn = MagicMock(name="torch.nn")
_mock_torch_export = MagicMock(name="torch.export")
_mock_torch_optim = MagicMock(name="torch.optim")
_mock_torch_utils = MagicMock(name="torch.utils")
_mock_torch_utils_data = MagicMock(name="torch.utils.data")
_mock_torch_no_grad = MagicMock(name="torch.no_grad")
_mock_torch_no_grad.return_value.__enter__ = MagicMock(return_value=None)
_mock_torch_no_grad.return_value.__exit__ = MagicMock(return_value=False)
_mock_torch.no_grad = _mock_torch_no_grad

# ── torchvision ────────────────────────────────────────────────────────────────
_mock_torchvision = MagicMock(name="torchvision")
_mock_torchvision_transforms = MagicMock(name="torchvision.transforms")

# ── torchao ────────────────────────────────────────────────────────────────────
_mock_torchao = MagicMock(name="torchao")
_mock_torchao_quantization = MagicMock(name="torchao.quantization")
_mock_torchao_pt2e = MagicMock(name="torchao.quantization.pt2e")

# ── litert_torch ───────────────────────────────────────────────────────────────
_mock_litert = MagicMock(name="litert_torch")
_mock_litert_quantization = MagicMock(name="litert_torch.quantization")
_mock_litert_pt2e = MagicMock(name="litert_torch.quantization.pt2e")

# ── ultralytics ────────────────────────────────────────────────────────────────
_mock_ultralytics = MagicMock(name="ultralytics")

# ── PIL ────────────────────────────────────────────────────────────────────────
_mock_pil = MagicMock(name="PIL")
_mock_pil_image = MagicMock(name="PIL.Image")

sys.modules.update(
    {
        "torch": _mock_torch,
        "torch.nn": _mock_torch_nn,
        "torch.export": _mock_torch_export,
        "torch.optim": _mock_torch_optim,
        "torch.utils": _mock_torch_utils,
        "torch.utils.data": _mock_torch_utils_data,
        "torchvision": _mock_torchvision,
        "torchvision.transforms": _mock_torchvision_transforms,
        "torchao": _mock_torchao,
        "torchao.quantization": _mock_torchao_quantization,
        "torchao.quantization.pt2e": _mock_torchao_pt2e,
        "litert_torch": _mock_litert,
        "litert_torch.quantization": _mock_litert_quantization,
        "litert_torch.quantization.pt2e": _mock_litert_pt2e,
        "ultralytics": _mock_ultralytics,
        "PIL": _mock_pil,
        "PIL.Image": _mock_pil_image,
    }
)
