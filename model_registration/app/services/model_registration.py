import logging
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Optional

import mlflow
from mlflow.tracking import MlflowClient

from app.models.registration import RegistrationParams, RegistrationResult


class ModelRegistrationError(Exception):
    """Raised when model registration fails unrecoverably."""


class ModelRegistrationService:
    """Registers YOLO model checkpoints in the MLflow model registry.

    Parameters
    ----------
    mlflow_tracking_uri:
        URI of the MLflow tracking server.
    max_retries:
        Number of attempts before giving up on an MLflow call.
    """

    _RETRY_DELAYS = (1, 2, 4)

    def __init__(self, mlflow_tracking_uri: str, max_retries: int = 3) -> None:
        self._mlflow_tracking_uri = mlflow_tracking_uri
        self._max_retries = max_retries
        self._logger = logging.getLogger(__name__)

    def run(self, params: RegistrationParams) -> RegistrationResult:
        """Execute the full registration pipeline.

        1. Connect to MLflow.
        2. Register best.pt under params.registered_model_name.
        3. Set lineage tags on the best.pt version.
        4. Optionally register last.pt and tag it.
        5. Optionally assign a registry alias (champion/challenger) to the best.pt version.
        6. Return a RegistrationResult.
        """
        mlflow.set_tracking_uri(self._mlflow_tracking_uri)
        client = MlflowClient()

        last_checkpoint_path = self._resolve_last_checkpoint(params)

        self._logger.info(
            "Registering best.pt from %s under model '%s'",
            params.best_checkpoint_path,
            params.registered_model_name,
        )
        best_version = self._register_checkpoint(
            client=client,
            model_uri=params.best_checkpoint_path,
            registered_model_name=params.registered_model_name,
        )
        self._logger.info("Registered best.pt as version %s", best_version)

        self._set_lineage_tags(
            client=client,
            registered_model_name=params.registered_model_name,
            version=best_version,
            params=params,
            checkpoint_type="best",
        )

        last_version: Optional[int] = None
        if last_checkpoint_path is not None:
            self._logger.info("Registering last.pt from %s", last_checkpoint_path)
            last_version = self._register_checkpoint(
                client=client,
                model_uri=last_checkpoint_path,
                registered_model_name=params.registered_model_name,
            )
            self._logger.info("Registered last.pt as version %s", last_version)
            self._set_lineage_tags(
                client=client,
                registered_model_name=params.registered_model_name,
                version=last_version,
                params=params,
                checkpoint_type="last",
            )

        promoted_to: Optional[str] = None
        if params.promote_to:
            self._with_retry(
                lambda: client.set_registered_model_alias(
                    name=params.registered_model_name,
                    alias=params.promote_to,
                    version=str(best_version),
                )
            )
            promoted_to = params.promote_to
            self._logger.info(
                "Assigned alias '%s' to version %s", promoted_to, best_version
            )

        # Register exported variants (F-06); non-fatal per variant (CON-01)
        exported_versions = self._register_exported_variants(
            client=client,
            params=params,
            best_version=best_version,
        )

        return RegistrationResult(
            registered_model_name=params.registered_model_name,
            best_version=best_version,
            last_version=last_version,
            registered_at=datetime.now(timezone.utc),
            promoted_to=promoted_to,
            exported_versions=exported_versions,
        )

    def _register_exported_variants(
        self,
        client: MlflowClient,
        params: RegistrationParams,
        best_version: int,
    ) -> dict[str, int]:
        """Register each exported model as a separate MLflow model version (F-06, D-03).

        For each entry in ``params.exported_models``, registers the S3 URI under
        ``{registered_model_name}-{sanitised_label}`` and sets lineage tags that
        link back to the FP32 source model version.

        Failures on individual variants are non-fatal (CON-01): a warning is
        logged and the loop continues.  The returned dict maps each export label
        to the registered version number for all variants that succeeded.

        Name sanitisation (CON-04): labels are passed through
        ``re.sub(r"[^a-zA-Z0-9_-]", "-", label)`` before being used as a
        registry name suffix.
        """
        if not params.exported_models:
            return {}

        exported_versions: dict[str, int] = {}

        for label, s3_uri in params.exported_models.items():
            safe_label = re.sub(r"[^a-zA-Z0-9_\-]", "-", label)
            export_model_name = f"{params.registered_model_name}-{safe_label}"

            try:
                self._logger.info(
                    "Registering exported variant %s from %s under '%s'",
                    label,
                    s3_uri,
                    export_model_name,
                )
                version = self._register_checkpoint(
                    client=client,
                    model_uri=s3_uri,
                    registered_model_name=export_model_name,
                )
                self._logger.info(
                    "Registered exported variant %s as version %s", label, version
                )

                # Lineage tags linking this exported version back to the FP32 source
                lineage_tags: dict[str, str] = {
                    "checkpoint_type": "exported",
                    "export_label": label,
                    "source_model_name": params.registered_model_name,
                    "source_model_version": str(best_version),
                    "source_run_id": params.mlflow_run_id,
                }
                version_str = str(version)
                for key, value in lineage_tags.items():

                    def _set_tag(
                        k: str = key,
                        v: str = value,
                        n: str = export_model_name,
                        ver: str = version_str,
                    ) -> None:
                        client.set_model_version_tag(
                            name=n, version=ver, key=k, value=v
                        )

                    try:
                        self._with_retry(_set_tag)
                    except ModelRegistrationError as exc:
                        self._logger.warning(
                            "Failed to set tag '%s' on %s v%s: %s",
                            key,
                            export_model_name,
                            version_str,
                            exc,
                        )

                exported_versions[label] = version

            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Failed to register exported variant %s (non-fatal, CON-01): %s",
                    label,
                    exc,
                )

        return exported_versions

    def _resolve_last_checkpoint(self, params: RegistrationParams) -> Optional[str]:
        """Return the last.pt S3 URI, deriving it from best.pt path if not given."""
        if params.last_checkpoint_path is not None:
            return params.last_checkpoint_path

        if "best.pt" in params.best_checkpoint_path:
            derived = params.best_checkpoint_path.replace("best.pt", "last.pt")
            self._logger.debug("Derived last.pt path: %s", derived)
            return derived

        self._logger.warning(
            "Cannot derive last.pt path from '%s' (no 'best.pt' substring); "
            "skipping last.pt registration",
            params.best_checkpoint_path,
        )
        return None

    def _register_checkpoint(
        self,
        client: MlflowClient,
        model_uri: str,
        registered_model_name: str,
    ) -> int:
        """Register a single checkpoint and return the integer version number."""
        mv = self._with_retry(
            lambda: mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )
        )
        return int(mv.version)

    def _set_lineage_tags(
        self,
        client: MlflowClient,
        registered_model_name: str,
        version: int,
        params: RegistrationParams,
        checkpoint_type: str,
    ) -> None:
        """Apply lineage and checkpoint-type tags to a registered model version."""
        tags: dict[str, str] = {"checkpoint_type": checkpoint_type}

        if params.mlflow_run_id:
            tags["training_run_id"] = params.mlflow_run_id
        if params.dataset_version is not None:
            tags["dataset_version"] = params.dataset_version
        if params.dataset_sample_size is not None:
            tags["dataset_sample_size"] = str(params.dataset_sample_size)
        if params.config_hash is not None:
            tags["config_hash"] = params.config_hash
        if params.git_commit is not None:
            tags["git_commit"] = params.git_commit
        if params.model_variant is not None:
            tags["model_variant"] = params.model_variant
        if params.best_map50 is not None:
            tags["best_mAP50"] = str(params.best_map50)
        if params.exported_models:
            for label, uri in params.exported_models.items():
                tags[f"export_{label}"] = uri

        version_str = str(version)
        for key, value in tags.items():

            def _set_tag(k: str = key, v: str = value) -> None:
                client.set_model_version_tag(
                    name=registered_model_name,
                    version=version_str,
                    key=k,
                    value=v,
                )

            self._with_retry(_set_tag)

    def _with_retry(self, fn: Callable[[], Any]) -> Any:
        """Call fn with exponential backoff, raising ModelRegistrationError on exhaustion."""
        delays = list(self._RETRY_DELAYS[: self._max_retries - 1]) + [None]
        last_exc: Optional[Exception] = None

        for attempt, delay in enumerate(delays, start=1):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if delay is None:
                    break
                self._logger.warning(
                    "MLflow call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt,
                    self._max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise ModelRegistrationError(
            f"MLflow call failed after {self._max_retries} attempts: {last_exc}"
        ) from last_exc
