import logging
import hashlib
from pathlib import Path
from datetime import datetime, timezone

from app.models.registration import RegistrationParams, RegistrationResult


class ModelRegistrationError(Exception):
    """Custom exception for model registration errors."""
    pass


class ModelRegistrationService:
    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def run(self, params: RegistrationParams, token: str = "") -> RegistrationResult:
        """Register a trained model to the model registry."""
        try:
            self._logger.info(
                f"Registering model: {params.model_name} -> {params.registry_url}"
            )

            # Validate checkpoint
            self._validate_checkpoint(params.checkpoint_path)

            # Resolve version
            version = params.version or self._generate_version(params.checkpoint_path)

            # Simulate registration
            artifact_uri = self._register(params, version, token)

            # Promote if requested
            promoted_to = None
            if params.promote_to:
                self._promote(params.model_name, version, params.promote_to, token)
                promoted_to = params.promote_to

            result = RegistrationResult(
                model_name=params.model_name,
                version=version,
                registry_url=params.registry_url,
                registered_at=datetime.now(timezone.utc),
                promoted_to=promoted_to,
                artifact_uri=artifact_uri,
            )

            print(
                f"Model registered: {params.model_name} v{version} "
                f"at {params.registry_url}"
                + (f" [promoted to {promoted_to}]" if promoted_to else "")
            )
            return result

        except ModelRegistrationError:
            raise
        except Exception as e:
            self._logger.error(f"Unexpected error: {e}")
            raise ModelRegistrationError(f"Registration failed: {e}") from e

    def _validate_checkpoint(self, checkpoint_path: str) -> None:
        """Validate that the checkpoint file exists."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise ModelRegistrationError(
                f"Checkpoint not found: {checkpoint_path}"
            )
        if not path.is_file():
            raise ModelRegistrationError(
                f"Checkpoint path is not a file: {checkpoint_path}"
            )

    def _generate_version(self, checkpoint_path: str) -> str:
        """Generate a version string from the checkpoint file hash."""
        content = Path(checkpoint_path).read_bytes()
        short_hash = hashlib.sha256(content).hexdigest()[:8]
        return f"0.1.0+{short_hash}"

    def _register(
        self, params: RegistrationParams, version: str, token: str
    ) -> str:
        """Simulate registering the model and return the artifact URI."""
        artifact_uri = (
            f"{params.registry_url}/models/{params.model_name}/versions/{version}"
        )
        self._logger.info(f"Registered artifact: {artifact_uri}")
        return artifact_uri

    def _promote(
        self, model_name: str, version: str, stage: str, token: str
    ) -> None:
        """Simulate promoting a model version to a stage."""
        self._logger.info(f"Promoted {model_name} v{version} -> {stage}")
