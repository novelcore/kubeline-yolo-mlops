import pytest
from pathlib import Path

from app.models.training import TrainingParams
from app.services.model_training import ModelTrainingService, ModelTrainingError


@pytest.fixture
def service() -> ModelTrainingService:
    return ModelTrainingService()


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    d = tmp_path / "dataset"
    d.mkdir()
    (d / "train.csv").write_text("col1\nval1\n")
    return d


def test_training_produces_checkpoint(service, dataset_dir, tmp_path, capfd):
    """Test that training creates a checkpoint and prints results."""
    params = TrainingParams(
        model_name="test-model",
        dataset_dir=str(dataset_dir),
        output_dir=str(tmp_path / "checkpoints"),
        epochs=3,
    )
    metrics = service.run(params)
    out, _ = capfd.readouterr()

    assert "Training complete" in out
    assert metrics.epochs_completed == 3
    assert metrics.final_train_loss > 0
    assert Path(metrics.checkpoint_path).exists()


def test_training_saves_metadata(service, dataset_dir, tmp_path):
    """Test that training saves metadata JSON alongside checkpoint."""
    params = TrainingParams(
        model_name="test-model",
        dataset_dir=str(dataset_dir),
        output_dir=str(tmp_path / "checkpoints"),
        epochs=2,
    )
    metrics = service.run(params)
    meta_path = Path(metrics.checkpoint_path).with_suffix(".meta.json")
    assert meta_path.exists()


def test_missing_dataset_raises_error(service, tmp_path):
    """Test that a missing dataset directory raises an error."""
    params = TrainingParams(
        model_name="test-model",
        dataset_dir="/nonexistent/dataset",
        output_dir=str(tmp_path / "checkpoints"),
    )
    with pytest.raises(ModelTrainingError, match="Dataset directory not found"):
        service.run(params)


def test_dataset_not_a_directory_raises_error(service, tmp_path):
    """Test that a file path (not dir) raises an error."""
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("data")
    params = TrainingParams(
        model_name="test-model",
        dataset_dir=str(file_path),
        output_dir=str(tmp_path / "checkpoints"),
    )
    with pytest.raises(ModelTrainingError, match="not a directory"):
        service.run(params)
