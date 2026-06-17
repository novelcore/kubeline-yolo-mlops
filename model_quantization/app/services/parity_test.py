"""Parity test service — FR-M-03.

Compares INT8 TFLite outputs against FP32 YOLO reference outputs on a sample
of calibration frames. Reports max-absolute-error and pass/fail against a
configurable threshold.
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
    ) -> ParityReport:
        logger.info(
            "Parity test | tflite=%s checkpoint=%s frames=%d seed=%d threshold=%.4f",
            tflite_path,
            fp32_checkpoint_path,
            parity_frames,
            seed,
            max_abs_error_threshold,
        )

        frames = self._load_frames(dataset_dir, image_size, parity_frames, seed)
        logger.info("Loaded %d frames for parity test", len(frames))

        fp32_outputs = self._run_fp32_inference(fp32_checkpoint_path, frames)
        tflite_outputs = self._run_tflite_inference(tflite_path, frames, image_size)

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
            f"No images directory found in {dataset_dir!r}. "
            f"Searched: {candidates}"
        )

    def _run_fp32_inference(
        self,
        checkpoint_path: str,
        frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Run YOLO FP32 headless inference, return raw backbone+neck outputs."""
        yolo = YOLO(checkpoint_path)
        module = yolo.model.eval()
        module.model[-1].training = True  # CON-03: head exclusion

        outputs = []
        with torch.no_grad():
            for frame in frames:
                # frame: NCHW float32 ndarray
                tensor = torch.from_numpy(frame)
                # Run forward through all layers except the final detection head
                out = self._forward_headless(module, tensor)
                outputs.append(out.numpy())

        return outputs

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
        """Run INT8 TFLite inference via ai-edge-litert."""
        from ai_edge_litert.interpreter import Interpreter

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # INT8 TFLite expects NHWC uint8 or float32 depending on quantization
        input_dtype = input_details[0]["dtype"]
        input_shape = input_details[0]["shape"]  # [1, H, W, C] for NHWC

        outputs = []
        for frame in frames:
            # frame: NCHW float32 — convert to NHWC
            nhwc = np.transpose(frame[0], (1, 2, 0))[np.newaxis]  # NHWC

            if input_dtype == np.uint8:
                nhwc = (nhwc * 255).clip(0, 255).astype(np.uint8)
            elif input_dtype == np.int8:
                nhwc = (nhwc * 255 - 128).clip(-128, 127).astype(np.int8)
            else:
                nhwc = nhwc.astype(np.float32)

            interpreter.set_tensor(input_details[0]["index"], nhwc)
            interpreter.invoke()

            # Concatenate all output tensors into a single flat vector
            out_parts = [
                interpreter.get_tensor(d["index"]).astype(np.float32).flatten()
                for d in output_details
            ]
            outputs.append(np.concatenate(out_parts))

        return outputs

    def _compute_max_abs_error(
        self,
        fp32_outputs: list[np.ndarray],
        tflite_outputs: list[np.ndarray],
    ) -> float:
        """Return the frame-wise max absolute error across all frames."""
        if len(fp32_outputs) != len(tflite_outputs):
            raise ParityTestError(
                f"Output count mismatch: FP32={len(fp32_outputs)}, "
                f"TFLite={len(tflite_outputs)}"
            )

        max_err = 0.0
        for i, (fp32, tfl) in enumerate(zip(fp32_outputs, tflite_outputs)):
            if fp32.shape != tfl.shape:
                logger.warning(
                    "Shape mismatch at frame %d: FP32=%s TFLite=%s — "
                    "returning max error 1.0",
                    i,
                    fp32.shape,
                    tfl.shape,
                )
                return 1.0
            err = float(np.max(np.abs(fp32 - tfl)))
            max_err = max(max_err, err)

        return max_err
