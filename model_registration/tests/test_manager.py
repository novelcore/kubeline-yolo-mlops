from pathlib import Path

from app.manager import Manager


def test_manager_runs_registration(tmp_path, capfd):
    """Test that Manager.run registers a model and prints output."""
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fake-checkpoint")

    manager = Manager()
    manager.run(
        model_name="test-model",
        checkpoint_path=str(checkpoint),
    )
    out, _ = capfd.readouterr()
    assert "Model registered" in out
