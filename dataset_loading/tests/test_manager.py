from pathlib import Path

from app.manager import Manager


def test_manager_runs_dataset_loading(tmp_path, capfd):
    """Test that Manager.run loads and splits a dataset."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("col1,col2\na,1\nb,2\nc,3\n")

    manager = Manager()
    manager.run(
        source_path=str(csv_file),
        output_dir=str(tmp_path / "output"),
        format="csv",
    )
    out, _ = capfd.readouterr()
    assert "Dataset loaded" in out
