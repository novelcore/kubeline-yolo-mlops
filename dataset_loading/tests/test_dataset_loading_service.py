import pytest
from pathlib import Path

from app.models.dataset import DatasetParams
from app.services.dataset_loading import DatasetLoadingService, DatasetLoadingError


@pytest.fixture
def service() -> DatasetLoadingService:
    return DatasetLoadingService()


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    data = "col1,col2\na,1\nb,2\nc,3\nd,4\ne,5\n"
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(data)
    return csv_path


def test_load_csv_dataset(service, csv_file, tmp_path, capfd):
    """Test loading a CSV dataset and splitting it."""
    params = DatasetParams(
        source_path=str(csv_file),
        output_dir=str(tmp_path / "output"),
        format="csv",
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
    )
    result = service.run(params)
    out, _ = capfd.readouterr()
    assert "Dataset loaded" in out
    assert result.total_records == 5
    assert result.train_records + result.val_records + result.test_records == 5


def test_missing_source_raises_error(service, tmp_path):
    """Test that a missing source path raises an error."""
    params = DatasetParams(
        source_path="/nonexistent/path.csv",
        output_dir=str(tmp_path / "output"),
        format="csv",
    )
    with pytest.raises(DatasetLoadingError, match="Source path not found"):
        service.run(params)


def test_format_mismatch_raises_error(service, tmp_path):
    """Test that a mismatched file extension raises an error."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("col1\nval1")
    params = DatasetParams(
        source_path=str(csv_file),
        output_dir=str(tmp_path / "output"),
        format="parquet",
    )
    with pytest.raises(DatasetLoadingError, match="does not match format"):
        service.run(params)


def test_directory_source(service, tmp_path, capfd):
    """Test loading from a directory of files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "file1.jsonl").write_text('{"a":1}\n')
    (data_dir / "file2.jsonl").write_text('{"a":2}\n')

    params = DatasetParams(
        source_path=str(data_dir),
        output_dir=str(tmp_path / "output"),
        format="jsonl",
    )
    result = service.run(params)
    assert result.total_records == 2
