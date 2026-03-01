from pathlib import Path

from app.manager import Manager


def test_manager_runs_training(tmp_path, capfd):
    """Test that Manager.run completes training and prints output."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "train.csv").write_text("col1\nval1\n")

    manager = Manager()
    manager.run(
        model_name="test-model",
        dataset_dir=str(dataset_dir),
        output_dir=str(tmp_path / "checkpoints"),
        epochs=2,
    )
    out, _ = capfd.readouterr()
    assert "Training complete" in out
