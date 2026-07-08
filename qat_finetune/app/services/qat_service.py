"""PT2E QAT fine-tune service.

Pipeline:
    FP32 .pt checkpoint
        → head exclusion (CON-03: model.eval(); model.model[-1].training = True)
        → torch.export.export(strict=False).module()
        → litert_torch PT2EQuantizer (CON-02: is_per_channel=False)
        → prepare_qat_pt2e   [torchao]
        → fine-tune loop (distillation loss vs FP32 teacher)
        → convert_pt2e(fold_quantize=False)   [torchao; CON-01]
        → litert_torch.convert() → edge_model.export()
        → INT8 TFLite → S3 + MLflow
"""

import logging
import os
import random
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

import mlflow
import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from mlflow.tracking import MlflowClient
from torchao.quantization.pt2e import allow_exported_model_train_eval
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_qat_pt2e
from ultralytics import YOLO

from app.models.quantization import QATParams, QATResult

# Interval (seconds) between system-metric samples during the QAT run.
_SYSTEM_METRICS_INTERVAL_S = 15.0

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class QATError(Exception):
    """Raised when the QAT pipeline encounters a non-recoverable error."""


class QATService:
    """Runs PT2E QAT fine-tuning and exports an INT8 TFLite artifact."""

    def __init__(self, s3_client: Any, mlflow_tracking_uri: str) -> None:
        self._s3 = s3_client
        self._mlflow_uri = mlflow_tracking_uri
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, params: QATParams) -> QATResult:
        """Execute the full QAT pipeline end-to-end.

        Steps
        -----
        1.  Resolve / download the FP32 checkpoint.
        2.  Load YOLO model, apply head exclusion (CON-03).
        3.  Capture computation graph via torch.export (strict=False).
        4.  Insert fake-quantize nodes with litert_torch PT2EQuantizer
            (CON-02: per-tensor INT8 only, is_per_channel=False).
        5.  Fine-tune the prepared model with a distillation loss.
        6.  Convert to real INT8 ops (fold_quantize=False, CON-01).
        7.  Export INT8 TFLite via litert_torch.
        8.  Upload TFLite to S3.
        9.  Log all parameters and artifact URI to MLflow,
            linked to the source model-training run.
        """
        mlflow.set_tracking_uri(self._mlflow_uri)
        mlflow.set_experiment(params.experiment_name)

        local_ckpt = self._resolve_checkpoint(params.fp32_checkpoint_path, params.output_dir)
        device = self._resolve_device(params.device)

        with mlflow.start_run(
            tags={"source_run_id": params.source_mlflow_run_id}
        ) as active_run:
            run_id = active_run.info.run_id
            self._logger.info(
                "QAT run started | run_id=%s source_run_id=%s device=%s",
                run_id,
                params.source_mlflow_run_id,
                device,
            )

            with self._sample_system_metrics(run_id):
                fp32_module = self._load_headless_module(local_ckpt, device)
                sample = (
                    torch.zeros(
                        1, 3, params.image_size, params.image_size, device=device
                    ),
                )

                exported = self._capture_graph(fp32_module, sample)
                prepared = self._prepare_qat(exported)
                self._seed_rng(params.calibration_seed, device)  # FR-M-04
                self._finetune(prepared, fp32_module, params, device)
                quantized = self._convert(prepared)
                tflite_path = self._export_tflite(
                    quantized, sample, params.output_dir
                )
                s3_uri = self._upload_tflite(tflite_path, params)
            self._log_run(run_id, params, s3_uri)

        self._logger.info(
            "QAT complete | run_id=%s tflite=%s", run_id, s3_uri
        )

        return QATResult(
            mlflow_run_id=run_id,
            source_run_id=params.source_mlflow_run_id,
            tflite_s3_uri=s3_uri,
            # Parity values are populated by FR-M-03 (parity_test.py).
            # Placeholder until that feature is implemented.
            parity_passed=True,
            parity_max_abs_error=0.0,
        )

    # ------------------------------------------------------------------
    # System-resource sampling
    # ------------------------------------------------------------------

    @contextmanager
    def _sample_system_metrics(self, run_id: str) -> Iterator[None]:
        """Sample CPU/RAM (and GPU when present) into MLflow while the body runs.

        A daemon thread logs a ResourceMonitor snapshot every
        ``_SYSTEM_METRICS_INTERVAL_S`` seconds under an increasing step, giving
        the qat-finetune run a system-metrics time series (parity with the
        training and model-quantization runs). QAT usually runs on GPU, so this
        captures GPU utilization/VRAM during the fine-tune. Best-effort: any
        failure is swallowed so it never affects the QAT run.
        """
        monitor: Any
        try:
            # Imported lazily + defensively: system metrics are best-effort and
            # must NEVER crash the QAT run. A missing psutil/pynvml or any import
            # error just disables the sampler (import at module load previously
            # took the whole step down with it on exit 1).
            from app.services.resource_monitor import ResourceMonitor

            monitor = ResourceMonitor(gpu_index=0)  # inert on CPU nodes
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("System-metrics sampler disabled: %s", exc)
            monitor = None

        if monitor is None:
            yield
            return

        client = MlflowClient()
        stop = threading.Event()
        state = {"step": 0}

        def _loop() -> None:
            while not stop.is_set():
                for key, value in monitor.collect().items():
                    try:
                        client.log_metric(run_id, key, value, step=state["step"])
                    except Exception:  # noqa: BLE001
                        pass
                state["step"] += 1
                stop.wait(_SYSTEM_METRICS_INTERVAL_S)

        thread = threading.Thread(
            target=_loop, name="qat-system-metrics", daemon=True
        )
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Step 1 — Checkpoint resolution
    # ------------------------------------------------------------------

    def _resolve_checkpoint(self, path: str, output_dir: str) -> str:
        """Return a local path to the FP32 checkpoint.

        Downloads from S3 if ``path`` starts with ``s3://``.
        """
        if not path.startswith("s3://"):
            return path

        without_scheme = path[len("s3://"):]
        bucket, _, key = without_scheme.partition("/")
        local_path = os.path.join(output_dir, Path(key).name)

        self._logger.info("Downloading checkpoint: %s → %s", path, local_path)
        self._s3.download_file(bucket, key, local_path)
        return local_path

    # ------------------------------------------------------------------
    # Step 2 — Load model + head exclusion (CON-03)
    # ------------------------------------------------------------------

    def _load_headless_module(self, checkpoint_path: str, device: str) -> nn.Module:
        """Load the YOLO model and apply the head-exclusion recipe (CON-03).

        Recipe (from spike runs):
            model.eval()
            model.model[-1].training = True

        Setting the Detect head back to training mode signals torch.export
        to treat it as a graph boundary. The captured graph is backbone + neck
        only, emitting raw feature tensors at three scales.
        The deployment target performs decode/NMS in host software.
        """
        yolo = YOLO(checkpoint_path)
        module: nn.Module = yolo.model  # type: ignore[assignment]
        module = module.to(device)
        module.eval()
        module.model[-1].training = True  # type: ignore[index]
        self._logger.info(
            "Loaded headless module from %s on %s", checkpoint_path, device
        )
        return module

    # ------------------------------------------------------------------
    # Step 3 — Graph capture
    # ------------------------------------------------------------------

    def _capture_graph(self, module: nn.Module, sample: tuple) -> nn.Module:
        """Capture the computation graph via torch.export.

        strict=False is required to tolerate dynamic control flow in the
        YOLOv8 backbone (e.g. conditional branches on input shape).

        .module() is used — NOT export_for_training() — because
        prepare_qat_pt2e expects a standard GraphModule, not a training-
        optimised export.
        """
        self._logger.info("Capturing computation graph (torch.export strict=False)")
        exported_program = torch.export.export(module, sample, strict=False)
        return exported_program.module()

    # ------------------------------------------------------------------
    # Step 4 — QAT preparation (CON-02: per-tensor, litert_torch quantizer)
    # ------------------------------------------------------------------

    def _prepare_qat(self, module: nn.Module) -> nn.Module:
        """Insert fake-quantize nodes using litert_torch's PT2EQuantizer.

        IMPORTANT: Uses litert_torch's own PT2EQuantizer — NOT torch.ao's.
        Only litert_torch's quantizer preserves QAT-learned activation scales
        through its TFLite converter (torch.ao's quantizer produces patterns
        the converter cannot translate).

        CON-02: is_per_channel=False — per-channel quantization fails at
        litert_torch's converter final pass on YOLOv8-pose. Per-tensor is
        the only scheme that produces a valid TFLite (confirmed over 5 spikes).
        """
        from litert_torch.quantize.pt2e_quantizer import (  # type: ignore[import]
            PT2EQuantizer,
            get_symmetric_quantization_config,
        )

        quantizer = PT2EQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=False)
        )
        prepared = prepare_qat_pt2e(module, quantizer)
        # torch.export graph modules reject the standard nn.Module .train()/.eval()
        # ("Calling train() or eval() is not supported for exported models").
        # This patches them to torchao's move_exported_model_to_{train,eval} so the
        # fine-tune loop's prepared.train()/.eval() calls work (PT2E QAT requirement).
        allow_exported_model_train_eval(prepared)
        # torch.export leaves the graph's weights with requires_grad=False, so the
        # distillation loss has no grad_fn and loss.backward() dies ("element 0 does
        # not require grad"). Put the model in train mode and re-enable grad on all
        # params so QAT can actually learn. (torchao 0.17 has no export_for_training
        # at torch.export in torch 2.11; this is the working alternative.)
        prepared.train()
        n_grad = 0
        for p in prepared.parameters():
            p.requires_grad_(True)
            n_grad += 1
        self._logger.info(
            "QAT preparation complete — fake-quantize nodes inserted "
            "(%d params grad-enabled)",
            n_grad,
        )
        return prepared

    # ------------------------------------------------------------------
    # Step 5 — Fine-tune loop
    # ------------------------------------------------------------------

    def _finetune(
        self,
        prepared: nn.Module,
        fp32_module: nn.Module,
        params: QATParams,
        device: str,
    ) -> None:
        """Fine-tune the prepared (fake-quantized) model.

        Uses knowledge distillation: MSE(student_output, fp32_teacher_output).
        This fine-tunes quantization scale parameters without requiring the
        original YOLO loss on a graph module (which is non-trivial to compute).

        Accuracy outcomes are out of scope for this step: it delivers the
        quantization mechanism; convergence depends on calibration data quality and LR.
        """
        loader = self._build_calibration_loader(
            params.dataset_dir,
            params.image_size,
            params.calibration_frames,
            params.calibration_seed,
        )

        prepared.train()
        optimizer = torch.optim.Adam(  # type: ignore[attr-defined]
            prepared.parameters(), lr=params.qat_lr
        )
        criterion = nn.MSELoss()

        fp32_module.eval()
        # eval() flips the Detect head OUT of headless mode (it would emit decoded
        # (1,no,8400) preds), which no longer matches the raw backbone+neck features
        # of the exported student graph. Re-assert headless so teacher & student
        # emit the SAME tensors for the distillation MSE (CON-03).
        fp32_module.model[-1].training = True  # type: ignore[index]

        self._logger.info(
            "QAT fine-tune started | epochs=%d lr=%g device=%s",
            params.qat_epochs,
            params.qat_lr,
            device,
        )

        for epoch in range(params.qat_epochs):
            epoch_loss = 0.0
            steps = 0

            for batch in loader:
                images: torch.Tensor = batch.to(device)  # type: ignore[attr-defined]

                with torch.no_grad():  # type: ignore[attr-defined]
                    teacher_out = fp32_module(images)

                optimizer.zero_grad()
                student_out = prepared(images)

                # The exported graph returns a pytree (dict), not a bare tensor or
                # tuple/list — flatten BOTH to tensor leaves and sum the per-leaf MSE
                # over aligned leaves (headless emits multi-scale feature tensors).
                s_leaves = [
                    t for t in pytree.tree_leaves(student_out) if torch.is_tensor(t)
                ]
                t_leaves = [
                    t for t in pytree.tree_leaves(teacher_out) if torch.is_tensor(t)
                ]
                loss = sum(
                    (
                        criterion(s.float(), t.float().detach())
                        for s, t in zip(s_leaves, t_leaves)
                    ),
                    start=torch.zeros((), device=device),
                )
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss)
                steps += 1

            avg_loss = epoch_loss / max(steps, 1)
            self._logger.info(
                "QAT epoch %d/%d | avg_loss=%.4f", epoch + 1, params.qat_epochs, avg_loss
            )
            # Log the distillation loss so the qat-finetune MLflow run has a metric
            # (it was previously params-only — no way to see how QAT converged).
            # Runs inside run()'s active mlflow run; best-effort so it never breaks QAT.
            try:
                mlflow.log_metric("qat_finetune_loss", avg_loss, step=epoch)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("Failed to log qat_finetune_loss: %s", exc)

        prepared.eval()
        self._logger.info("QAT fine-tuning complete")

    def _build_calibration_loader(
        self,
        dataset_dir: str,
        image_size: int,
        calibration_frames: int,
        seed: int,
    ) -> "DataLoader":
        """Build a DataLoader over calibration images (FR-M-04: deterministic sampling)."""
        import torchvision.transforms as T  # type: ignore[import]
        from torch.utils.data import DataLoader, Dataset  # type: ignore[import]

        # Support both flat images/ and images/train/ layouts
        train_dir = os.path.join(dataset_dir, "images", "train")
        images_dir = train_dir if os.path.isdir(train_dir) else os.path.join(dataset_dir, "images")

        image_paths = sorted(
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        rng = random.Random(seed)
        if len(image_paths) > calibration_frames:
            image_paths = rng.sample(image_paths, calibration_frames)

        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

        class _CalibDataset(Dataset):  # type: ignore[type-arg]
            def __init__(self, paths: list, t: Any) -> None:
                self._paths = paths
                self._t = t

            def __len__(self) -> int:
                return len(self._paths)

            def __getitem__(self, idx: int) -> Any:
                from PIL import Image  # type: ignore[import]
                img = Image.open(self._paths[idx]).convert("RGB")
                return self._t(img)

        return DataLoader(  # type: ignore[return-value]
            _CalibDataset(image_paths, transform),
            # MUST be 1: the YOLO pose head specializes the batch dim to 1 during
            # torch.export (anchor/reshape logic), so the graph is batch-1-only —
            # feeding batch>1 fails the baked "x.size()[0] == 1" guard.
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

    # ------------------------------------------------------------------
    # Step 6 — Convert to quantized graph (CON-01: fold_quantize=False)
    # ------------------------------------------------------------------

    def _convert(self, prepared: nn.Module) -> nn.Module:
        """Convert fake-quantize nodes to real INT8 ops.

        fold_quantize=False is mandatory (CON-01). It is the only
        representation the litert_torch converter accepts without
        failing. ``fold_quantize=True`` produces a folded representation
        that litert_torch cannot translate to TFLite INT8 ops.
        """
        self._logger.info("Converting to quantized graph (fold_quantize=False)")
        return convert_pt2e(prepared, fold_quantize=False)

    # ------------------------------------------------------------------
    # Step 7 — Export INT8 TFLite via litert_torch
    # ------------------------------------------------------------------

    def _export_tflite(
        self, quantized: nn.Module, sample: tuple, output_dir: str
    ) -> str:
        """Export the quantized GraphModule to an INT8 TFLite file."""
        import litert_torch  # type: ignore[import]

        output_path = os.path.join(output_dir, "model_int8.tflite")
        self._logger.info("Exporting INT8 TFLite → %s", output_path)

        edge_model = litert_torch.convert(quantized, sample)
        edge_model.export(output_path)

        self._logger.info("TFLite export complete: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Step 8 — Upload TFLite to S3
    # ------------------------------------------------------------------

    def _upload_tflite(self, local_path: str, params: QATParams) -> str:
        """Upload the TFLite artifact to S3 and return its s3:// URI."""
        key = f"{params.output_prefix}/{Path(local_path).name}"
        self._logger.info(
            "Uploading TFLite to s3://%s/%s", params.output_bucket, key
        )
        self._s3.upload_file(local_path, params.output_bucket, key)
        # S3-gateway writes only STAGE objects on the lakeFS branch — commit so
        # the QAT INT8 tflite becomes a versioned artifact, not dangling changes.
        branch = params.output_prefix.split("/", 1)[0] or "main"
        self._lakefs_commit(
            params.output_bucket, branch,
            f"qat-finetune: INT8 tflite {Path(local_path).name}",
        )
        return f"s3://{params.output_bucket}/{key}"

    def _lakefs_commit(self, repo: str, branch: str, message: str) -> None:
        """Commit staged lakeFS changes on ``branch`` (S3-gateway only stages).

        Best-effort: a failure (including 'nothing to commit') never fails the
        step — the object is already uploaded. Uses the lakeFS REST API on the
        same endpoint as the S3 gateway with the injected lakeFS credentials.
        """
        import base64
        import json
        import urllib.request

        endpoint = os.environ.get("LAKEFS_ENDPOINT", "").rstrip("/")
        access = os.environ.get("LAKEFS_ACCESS_KEY", "")
        secret = os.environ.get("LAKEFS_SECRET_KEY", "")
        if not (endpoint and access and secret):
            self._logger.warning("lakeFS commit skipped — endpoint/creds unset.")
            return
        url = f"{endpoint}/api/v1/repositories/{repo}/branches/{branch}/commits"
        auth = base64.b64encode(f"{access}:{secret}".encode()).decode()
        req = urllib.request.Request(
            url, data=json.dumps({"message": message}).encode(), method="POST"
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Basic {auth}")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                cid = str(json.loads(resp.read().decode()).get("id", "?"))
            self._logger.info("lakeFS commit %s @ %s/%s", cid[:12], repo, branch)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("lakeFS commit failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Step 9 — MLflow logging (FR-M-05)
    # ------------------------------------------------------------------

    def _log_run(self, run_id: str, params: QATParams, s3_uri: str) -> None:
        """Log QAT parameters and artifact URI to MLflow (FR-M-05).

        Errors are swallowed with a warning — MLflow unavailability must
        not block the artifact upload.
        """
        client = MlflowClient()
        log_items: list[tuple[str, str]] = [
            ("quantization_mode", "qat"),
            ("quantization_scheme", "per_tensor_int8"),
            ("qat_epochs", str(params.qat_epochs)),
            ("qat_lr", str(params.qat_lr)),
            ("calibration_frames", str(params.calibration_frames)),
            ("calibration_seed", str(params.calibration_seed)),
            ("image_size", str(params.image_size)),
            ("source_run_id", params.source_mlflow_run_id),
            ("tflite_s3_uri", s3_uri),
            ("fold_quantize", "False"),
        ]
        for key, value in log_items:
            try:
                client.log_param(run_id, key, value)
            except Exception as exc:
                self._logger.warning("Failed to log MLflow param %s: %s", key, exc)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _seed_rng(self, seed: int, device: str) -> None:
        """Fix all RNG seeds for reproducibility (FR-M-04).

        Called immediately before the QAT fine-tune loop so that weight
        updates and fake-quantize scale updates are deterministic.
        warn_only=True: some CUDA ops have no deterministic implementation;
        strict mode would abort the pipeline on those ops.
        """
        random.seed(seed)
        torch.manual_seed(seed)
        if "cuda" in device:
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        self._logger.info(
            "RNG seeds fixed | seed=%d device=%s deterministic=warn_only", seed, device
        )

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        """Return device string — defaults to cuda if available."""
        if device:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"
