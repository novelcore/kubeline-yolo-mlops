import pytest
from pathlib import Path

from app.models.registration import RegistrationParams
from app.services.model_registration import (
    ModelRegistrationService,
    ModelRegistrationError,
)


@pytest.fixture
def service() -> ModelRegistrationService:
    return ModelRegistrationService()


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    cp = tmp_path / "model.pt"
    cp.write_bytes(b"fake-checkpoint-data")
    return cp


def test_register_model(service, checkpoint, capfd):
    """Test that registration prints success and returns result."""
    params = RegistrationParams(
        model_name="test-model",
        checkpoint_path=str(checkpoint),
        registry_url="https://registry.example.com",
    )
    result = service.run(params)
    out, _ = capfd.readouterr()

    assert "Model registered" in out
    assert result.model_name == "test-model"
    assert "0.1.0+" in result.version
    assert result.promoted_to is None


def test_register_with_explicit_version(service, checkpoint, capfd):
    """Test registration with an explicit version string."""
    params = RegistrationParams(
        model_name="test-model",
        checkpoint_path=str(checkpoint),
        registry_url="https://registry.example.com",
        version="2.0.0",
    )
    result = service.run(params)
    assert result.version == "2.0.0"


def test_register_and_promote(service, checkpoint, capfd):
    """Test registration with promotion to staging."""
    params = RegistrationParams(
        model_name="test-model",
        checkpoint_path=str(checkpoint),
        registry_url="https://registry.example.com",
        promote_to="staging",
    )
    result = service.run(params)
    out, _ = capfd.readouterr()

    assert "promoted to staging" in out
    assert result.promoted_to == "staging"


def test_missing_checkpoint_raises_error(service):
    """Test that a missing checkpoint raises an error."""
    params = RegistrationParams(
        model_name="test-model",
        checkpoint_path="/nonexistent/model.pt",
        registry_url="https://registry.example.com",
    )
    with pytest.raises(ModelRegistrationError, match="Checkpoint not found"):
        service.run(params)


def test_checkpoint_is_directory_raises_error(service, tmp_path):
    """Test that a directory checkpoint path raises an error."""
    dir_path = tmp_path / "model_dir"
    dir_path.mkdir()
    params = RegistrationParams(
        model_name="test-model",
        checkpoint_path=str(dir_path),
        registry_url="https://registry.example.com",
    )
    with pytest.raises(ModelRegistrationError, match="not a file"):
        service.run(params)
