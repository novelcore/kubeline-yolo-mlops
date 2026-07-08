"""Parity test service — FR-M-03.

Compares INT8 TFLite outputs against FP32 YOLO reference outputs on a sample
of calibration frames. Reports max-absolute-error and pass/fail against a
configurable threshold.

The comparison is **mode-aware** (headless flag) so that BOTH sides emit the
same tensor as the TFLite under test:

* PTQ  (``headless=False``) — Ultralytics ``model.export(int8=True)`` exports
  the FULL model (detection head included), emitting the pre-NMS
  ``(1, 38, 8400)`` prediction tensor. BOTH the FP32 ``.pt`` reference and the
  INT8 ``.tflite`` are run through ``ultralytics.nn.autobackend.AutoBackend``
  so that identical preprocessing, int8 (de)quantization AND **coordinate
  denormalization** are applied, landing both sides in the same pixel-coord
  space (see ``_run_autobackend_inference``).
* QAT  (``headless=True``)  — ``qat_service`` exports the backbone+neck only
  (CON-03 head exclusion), emitting raw multi-scale features. The FP32
  reference forwards through every layer except the detection head, and the
  headless litert TFLite is dequantized by hand (AutoBackend does not apply to
  the headless export — it has no detection head / task metadata).

INT8 I/O (de)quantization on the headless path mirrors Ultralytics'
AutoBackend TFLite path (``ultralytics/nn/autobackend.py``,
``AutoBackend.forward``, v8.3.x): inputs are quantized with
``im / scale + zero_point`` and outputs dequantized with
``(x - zero_point) * scale`` using each tensor's own quantization params.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

from app.models.quantization import ParityReport

logger = logging.getLogger(__name__)


class ParityTestError(Exception):
    """Raised when the parity test cannot run (not when it fails)."""


class ParityTestService:
    """Runs FP32 vs INT8 TFLite parity check."""

    def run(
        self,
        tflite_path: str,
        fp32_checkpoint_path: str,
        dataset_dir: str,
        image_size: int,
        parity_frames: int,
        seed: int,
        max_abs_error_threshold: float,
        headless: bool = True,
    ) -> ParityReport:
        """Run the FP32-vs-INT8 parity check.

        Parameters
        ----------
        headless:
            Selects which comparison to run so that both sides emit the SAME
            tensor space. ``True`` for the QAT path (backbone+neck only,
            CON-03) — a hand-rolled headless forward vs a hand-dequantized
            litert output. ``False`` for the PTQ path (full model,
            ``(1, 38, 8400)``) — both ``.pt`` and ``.tflite`` are run through
            AutoBackend so int8 dequant AND coordinate denorm land both sides
            in pixel-coord space. Comparing a headless FP32 reference against a
            full-model TFLite (or vice versa) compares different tensors and
            yields a meaningless error (historically a spurious shape-mismatch
            → 1.0, or — for PTQ without denorm — a ~640x coordinate error).
        """
        logger.info(
            "Parity test | tflite=%s checkpoint=%s frames=%d seed=%d "
            "threshold=%.4f headless=%s",
            tflite_path,
            fp32_checkpoint_path,
            parity_frames,
            seed,
            max_abs_error_threshold,
            headless,
        )

        frames = self._load_frames(dataset_dir, image_size, parity_frames, seed)
        logger.info("Loaded %d frames for parity test", len(frames))

        if headless:
            # QAT: the INT8 tflite is HEADLESS (backbone+neck, CON-03). Reattach
            # the FP32 detection head to its feature maps — exactly as the edge
            # target runs it — so BOTH sides are DECODED detections in pixel
            # space. That's a task-level comparison (like PTQ), not an
            # uninterpretable raw-feature MSE.
            fp32_outputs = self._run_fp32_inference(
                fp32_checkpoint_path, frames, headless=False
            )
            tflite_outputs = self._run_head_reattach_inference(
                tflite_path, fp32_checkpoint_path, frames
            )
        else:
            # PTQ: run BOTH the .pt and the full-model .tflite through
            # AutoBackend for identical preprocessing, int8 dequant and
            # coordinate denormalization (pixel-coord space on both sides).
            fp32_outputs = self._run_autobackend_inference(fp32_checkpoint_path, frames)
            tflite_outputs = self._run_autobackend_inference(tflite_path, frames)

        max_err = self._compute_max_abs_error(fp32_outputs, tflite_outputs)
        passed = max_err <= max_abs_error_threshold

        logger.info(
            "Parity result | max_abs_error=%.6f threshold=%.4f passed=%s",
            max_err,
            max_abs_error_threshold,
            passed,
        )

        return ParityReport(
            parity_passed=passed,
            max_abs_error=max_err,
            threshold=max_abs_error_threshold,
            frames_tested=len(frames),
        )

    def save_report(self, report: ParityReport, output_dir: str) -> str:
        """Write parity_report.json to output_dir. Returns the file path."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "parity_report.json")
        with open(path, "w") as f:
            json.dump(report.model_dump(), f, indent=2)
        logger.info("Parity report written: %s", path)
        return path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_frames(
        self,
        dataset_dir: str,
        image_size: int,
        count: int,
        seed: int,
    ) -> list[np.ndarray]:
        """Return a seeded random sample of preprocessed image arrays."""
        from PIL import Image

        images_dir = self._find_images_dir(dataset_dir)
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        all_paths = sorted(
            p for p in Path(images_dir).rglob("*") if p.suffix.lower() in exts
        )

        if not all_paths:
            raise ParityTestError(f"No images found in {images_dir!r}")

        rng = random.Random(seed)
        selected = rng.sample(all_paths, min(count, len(all_paths)))

        frames = []
        for p in selected:
            img = Image.open(p).convert("RGB").resize((image_size, image_size))
            arr = np.array(img, dtype=np.float32) / 255.0  # HWC, [0,1]
            arr = np.transpose(arr, (2, 0, 1))  # CHW
            arr = np.expand_dims(arr, axis=0)  # NCHW
            frames.append(arr)

        return frames

    def _find_images_dir(self, dataset_dir: str) -> str:
        """Locate the images directory within the YOLO dataset layout."""
        candidates = [
            os.path.join(dataset_dir, "images", "val"),
            os.path.join(dataset_dir, "images", "train"),
            os.path.join(dataset_dir, "images"),
        ]
        for candidate in candidates:
            if os.path.isdir(candidate):
                return candidate
        raise ParityTestError(
            f"No images directory found in {dataset_dir!r}. " f"Searched: {candidates}"
        )

    def _run_autobackend_inference(
        self,
        weights_path: str,
        frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Run inference on a ``.pt`` or ``.tflite`` via Ultralytics AutoBackend.

        AutoBackend applies identical preprocessing for both weight types and,
        for the INT8 TFLite, both int8 dequantization (``(x - zp) * scale``)
        AND coordinate denormalization (box channels ``[0,2]*=w`` / ``[1,3]*=h``
        and, for pose, keypoint channels ``5::3*=w`` / ``6::3*=h`` on the
        ``(1, 38, 8400)`` tensor) — see ``ultralytics/nn/autobackend.py``
        ``AutoBackend.forward`` lines 807-836 (v8.3.x). Running the FP32 ``.pt``
        through the SAME entry point (line 657-658 → native model forward, which
        is already in pixel coords) puts both sides in the same pixel-coord
        space, so the parity error reflects only quantization noise — not the
        ~640x coordinate-scale artefact of comparing normalized vs pixel coords.
        """
        from ultralytics.nn.autobackend import AutoBackend

        # First positional arg (Ultralytics renamed the kwarg ``weights`` → ``model``
        # in v8.4.x; passing positionally is robust to either name).
        backend = AutoBackend(
            weights_path, device=torch.device("cpu"), fp16=False
        )
        backend.eval() if hasattr(backend, "eval") else None

        outputs = []
        with torch.no_grad():
            for frame in frames:
                # frame: NCHW float32 [0,1] — AutoBackend handles NHWC transpose,
                # int8 (de)quant, and coordinate denorm internally.
                tensor = torch.from_numpy(frame).to(torch.float32)
                out = backend(tensor)
                pred = self._extract_prediction(out)
                outputs.append(pred.cpu().numpy().astype(np.float32).flatten())

        return outputs

    def _run_fp32_inference(
        self,
        checkpoint_path: str,
        frames: list[np.ndarray],
        headless: bool = True,
    ) -> list[np.ndarray]:
        """Run the FP32 YOLO reference forward on each frame.

        ``headless=True``  → backbone+neck only (QAT parity, matches CON-03).
        ``headless=False`` → full model forward (PTQ parity, matches the
        Ultralytics ``export(int8=True)`` full-model TFLite). NOTE: the PTQ path
        normally routes through :meth:`_run_autobackend_inference` instead; this
        branch is retained for direct/unit use.
        """
        yolo = YOLO(checkpoint_path)
        module = yolo.model.eval()

        if headless:
            module.model[-1].training = True  # CON-03: head exclusion

        outputs = []
        with torch.no_grad():
            for frame in frames:
                # frame: NCHW float32 ndarray
                tensor = torch.from_numpy(frame)
                if headless:
                    out = self._forward_headless(module, tensor)
                else:
                    out = self._forward_full(module, tensor)
                outputs.append(out.cpu().numpy())

        return outputs

    def _forward_full(self, module: Any, tensor: Any) -> Any:
        """Full-model forward — returns the raw pre-NMS prediction tensor.

        In eval mode (and not export mode), the YOLOv8 Detect/Pose head returns
        ``(prediction, extras)`` where ``prediction`` is the decoded
        ``(1, no, 8400)`` tensor (``no = 38`` for the pose model under test).
        This matches the FULL-model TFLite produced by
        ``model.export(format='tflite', int8=True)``.
        """
        out = module(tensor)
        return self._extract_prediction(out)

    def _extract_prediction(self, out: Any) -> "torch.Tensor":
        """Extract the primary ``(1, no, 8400)`` prediction tensor.

        The head/backend may return a bare tensor, a ``(tensor, extras)`` tuple,
        or — for segment/pose — a ``((tensor, proto), preds)`` nested tuple.
        Peel the leading element(s) until a torch.Tensor is reached.
        """
        while isinstance(out, (tuple, list)):
            if not out:
                raise ParityTestError("FP32 full-model forward returned empty output")
            out = out[0]
        if not isinstance(out, torch.Tensor):
            raise ParityTestError(f"Unexpected FP32 forward output type: {type(out)!r}")
        return out

    def _forward_headless(self, module: Any, tensor: Any) -> Any:
        """Forward pass up to (not including) the detection head."""
        # Collect all sub-modules in forward order
        layers = list(module.model.children())
        head = layers[-1]  # detection head — excluded
        x = tensor
        saved: dict[int, Any] = {}

        for i, layer in enumerate(layers[:-1]):
            # Ultralytics layers track their 'f' (from-index) attribute
            f = getattr(layer, "f", -1)
            if isinstance(f, int):
                x_in = saved[f] if f != -1 else x
            else:
                x_in = [saved[j] if j != -1 else x for j in f]
            x = layer(x_in)
            saved[i] = x

        # Flatten and concatenate multi-scale outputs
        if isinstance(x, (list, tuple)):
            x = torch.cat([o.flatten(1) for o in x], dim=1)
        return x

    def _run_tflite_inference(
        self,
        tflite_path: str,
        frames: list[np.ndarray],
        image_size: int,
    ) -> list[np.ndarray]:
        """Run INT8 TFLite inference via ai-edge-litert (headless/QAT path).

        Mirrors Ultralytics AutoBackend (``nn/autobackend.py``,
        ``AutoBackend.forward``, v8.3.x):

        * Input in NHWC float [0, 1]. If the input tensor is int-typed,
          quantize with ``im / scale + zero_point`` using the input tensor's
          own quantization params.
        * If an output tensor is int-typed, dequantize with
          ``(x - zero_point) * scale`` using that output's own params.

        No coordinate denormalization is applied here — the headless litert
        export has no detection head, so its outputs are raw feature maps (not
        box/keypoint coords) and the FP32 headless reference is likewise raw.
        The full-model PTQ path uses :meth:`_run_autobackend_inference`, which
        DOES denormalize.
        """
        from ai_edge_litert.interpreter import Interpreter

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        in_detail = input_details[0]
        input_dtype = in_detail["dtype"]
        input_is_int = input_dtype in (np.int8, np.int16, np.uint8)

        outputs = []
        for frame in frames:
            # frame: NCHW float32 [0,1] — convert to NHWC to match the TFLite.
            nhwc = np.transpose(frame[0], (1, 2, 0))[np.newaxis].astype(np.float32)

            if input_is_int:
                scale, zero_point = in_detail["quantization"]
                nhwc = (nhwc / scale + zero_point).astype(input_dtype)

            interpreter.set_tensor(in_detail["index"], nhwc)
            interpreter.invoke()

            out_parts = []
            for d in output_details:
                x = interpreter.get_tensor(d["index"])
                if d["dtype"] in (np.int8, np.int16, np.uint8):
                    scale, zero_point = d["quantization"]
                    x = (x.astype(np.float32) - zero_point) * scale
                out_parts.append(x.astype(np.float32).flatten())
            outputs.append(np.concatenate(out_parts))

        return outputs

    def _run_head_reattach_inference(
        self,
        tflite_path: str,
        fp32_checkpoint_path: str,
        frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Run the headless INT8 tflite, then reattach the FP32 detection head.

        The QAT tflite emits raw backbone+neck feature maps (no head, CON-03).
        This runs them through the model's own FP32 head + decode — the same path
        the edge device uses in host software — yielding decoded ``(1, no, 8400)``
        detections in pixel space, directly comparable to the FP32 full-model
        reference. This replaces the uninterpretable raw-feature parity.
        """
        from ai_edge_litert.interpreter import Interpreter

        yolo = YOLO(fp32_checkpoint_path)
        module = yolo.model.eval()
        head = module.model[-1]
        head.training = False  # decode mode

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        in_detail = interpreter.get_input_details()[0]
        out_details = interpreter.get_output_details()
        input_is_int = in_detail["dtype"] in (np.int8, np.int16, np.uint8)
        # litert (torch->tflite) preserves torch NCHW layout; Ultralytics tflites
        # are NHWC. Detect from the declared input shape (channels dim == 3).
        in_shape = list(in_detail["shape"])
        input_is_nchw = len(in_shape) == 4 and in_shape[1] == 3

        outputs = []
        with torch.no_grad():
            for frame in frames:
                # frame is NCHW (1,3,H,W). NCHW tflite: pass as-is; NHWC: transpose.
                if input_is_nchw:
                    inp = frame.astype(np.float32)
                else:
                    inp = np.transpose(frame[0], (1, 2, 0))[np.newaxis].astype(
                        np.float32
                    )
                if input_is_int:
                    scale, zero_point = in_detail["quantization"]
                    inp = (inp / scale + zero_point).astype(in_detail["dtype"])
                interpreter.set_tensor(in_detail["index"], inp)
                interpreter.invoke()

                feats = []
                for d in out_details:
                    x = interpreter.get_tensor(d["index"])
                    if d["dtype"] in (np.int8, np.int16, np.uint8):
                        scale, zero_point = d["quantization"]
                        x = (x.astype(np.float32) - zero_point) * scale
                    t = torch.from_numpy(np.ascontiguousarray(x)).float()
                    # NCHW-input tflites emit NCHW feature maps (the torch head's
                    # native layout); only NHWC tflites need the channel-last swap.
                    if t.dim() == 4 and not input_is_nchw:
                        t = t.permute(0, 3, 1, 2).contiguous()
                    feats.append(t)

                feats = [f for f in feats if f.dim() == 4]
                # head expects [P3, P4, P5] — largest spatial map first.
                feats = sorted(feats, key=lambda f: -(f.shape[-1] * f.shape[-2]))
                decoded = self._extract_prediction(head(list(feats)))
                outputs.append(decoded.cpu().numpy().astype(np.float32).flatten())

        return outputs

    def _compute_max_abs_error(
        self,
        fp32_outputs: list[np.ndarray],
        tflite_outputs: list[np.ndarray],
    ) -> float:
        """Return the frame-wise max absolute error across all frames.

        Both sides are compared as flat vectors of the same length. A genuine
        length mismatch (which, once the parity mode is correct, indicates a
        real model/tflite mismatch rather than a headless-vs-full artefact) is
        reported as the maximum possible error (1.0).
        """
        if len(fp32_outputs) != len(tflite_outputs):
            raise ParityTestError(
                f"Output count mismatch: FP32={len(fp32_outputs)}, "
                f"TFLite={len(tflite_outputs)}"
            )

        max_err = 0.0
        for i, (fp32, tfl) in enumerate(zip(fp32_outputs, tflite_outputs)):
            fp32 = np.asarray(fp32).flatten()
            tfl = np.asarray(tfl).flatten()
            if fp32.shape != tfl.shape:
                logger.warning(
                    "Length mismatch at frame %d: FP32=%s TFLite=%s — "
                    "returning max error 1.0",
                    i,
                    fp32.shape,
                    tfl.shape,
                )
                return 1.0
            err = float(np.max(np.abs(fp32 - tfl)))
            max_err = max(max_err, err)

        return max_err
