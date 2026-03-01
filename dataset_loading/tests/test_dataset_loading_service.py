"""Tests for DatasetLoadingService.

All S3 interactions are mocked — no real AWS calls are made.
The tests build a minimal YOLO pose dataset tree in pytest's ``tmp_path``
fixture and verify that the service behaves correctly.
"""

import json
import random
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from app.models.dataset import YoloDatasetParams
from app.services.dataset_loading import (
    SPLITS,
    DatasetLoadingError,
    DatasetLoadingService,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_label_line(tokens: int = 38) -> str:
    """Return a synthetic label line with the given number of tokens."""
    return " ".join(["0"] + [f"0.5"] * (tokens - 1))


def _build_yolo_tree(
    root: Path,
    *,
    n_train: int = 3,
    n_val: int = 2,
    n_test: int = 2,
    tokens_per_line: int = 38,
    missing_label_split: str = "",
    missing_label_stem: str = "",
    empty_label_split: str = "",
    empty_label_stem: str = "",
    include_data_yaml: bool = True,
) -> None:
    """Create a minimal YOLO dataset tree under *root*.

    Parameters
    ----------
    root:
        Dataset root directory (will be created if absent).
    n_train / n_val / n_test:
        Number of image+label pairs per split.
    tokens_per_line:
        Token count written on every label line (use != 38 to inject bad data).
    missing_label_split / missing_label_stem:
        If both are set, the label file for that image is omitted.
    empty_label_split / empty_label_stem:
        If both are set, the label file for that image is created but empty.
    include_data_yaml:
        Whether to write a ``data.yaml`` at *root*.
    """
    counts = {"train": n_train, "val": n_val, "test": n_test}
    for split, n in counts.items():
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(1, n + 1):
            stem = f"img{i:06d}"
            img_path = root / "images" / split / f"{stem}.jpg"
            img_path.write_bytes(b"\xff\xd8\xff")  # minimal JPEG magic bytes

            label_path = root / "labels" / split / f"{stem}.txt"
            if split == missing_label_split and stem == missing_label_stem:
                continue  # deliberately omit the label file
            if split == empty_label_split and stem == empty_label_stem:
                label_path.write_text("")
            else:
                label_path.write_text(_make_label_line(tokens_per_line) + "\n")

    if include_data_yaml:
        data = {
            "path": str(root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "kpt_shape": [11, 3],
            "flip_idx": [],
            "names": {0: "spacecraft"},
        }
        (root / "data.yaml").write_text(yaml.dump(data))


def _make_mock_s3(dataset_root: Path) -> MagicMock:
    """Return a mock boto3 S3 client whose ``download_file`` copies from *dataset_root*.

    ``list_objects_v2`` pages are built from the actual files in *dataset_root*
    so the download logic receives a realistic response structure.
    """
    mock_s3 = MagicMock()

    # Build the list of keys from the actual files on disk
    all_keys: list[str] = []
    for path in sorted(dataset_root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(dataset_root)
            all_keys.append(str(relative))

    prefix = "datasets/speedplus_yolo/v1/"

    def list_objects_v2_paginator() -> MagicMock:
        paginator = MagicMock()
        contents = [{"Key": prefix + k} for k in all_keys]
        paginator.paginate.return_value = [{"Contents": contents}]
        return paginator

    mock_s3.get_paginator.return_value = list_objects_v2_paginator()

    def download_file(bucket: str, key: str, dest: str) -> None:
        relative = key[len(prefix) :]
        src = dataset_root / relative
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(src.read_bytes())

    mock_s3.download_file.side_effect = download_file
    return mock_s3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_source(tmp_path: Path) -> Path:
    """A minimal but complete YOLO source tree (pre-download)."""
    src = tmp_path / "source"
    _build_yolo_tree(src)
    return src


@pytest.fixture()
def valid_params(tmp_path: Path) -> YoloDatasetParams:
    """Standard params for a happy-path run."""
    return YoloDatasetParams(
        version="v1",
        source="s3",
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture()
def service_with_valid_source(
    valid_source: Path, valid_params: YoloDatasetParams
) -> DatasetLoadingService:
    """Service whose mock S3 client serves files from ``valid_source``."""
    mock_s3 = _make_mock_s3(valid_source)
    return DatasetLoadingService(s3_client=mock_s3)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_run_returns_correct_stats(
        self,
        service_with_valid_source: DatasetLoadingService,
        valid_params: YoloDatasetParams,
    ) -> None:
        """Service.run() returns correct image counts for a valid dataset."""
        stats = service_with_valid_source.run(valid_params)

        assert stats.train_images == 3
        assert stats.val_images == 2
        assert stats.test_images == 2
        assert stats.sampled is False
        assert stats.sample_size is None
        assert stats.version == "v1"
        assert stats.source == "s3"

    def test_output_directory_created(
        self,
        service_with_valid_source: DatasetLoadingService,
        valid_params: YoloDatasetParams,
    ) -> None:
        """The output directory is created even if it did not exist."""
        service_with_valid_source.run(valid_params)
        assert Path(valid_params.output_dir).is_dir()

    def test_data_yaml_written(
        self,
        service_with_valid_source: DatasetLoadingService,
        valid_params: YoloDatasetParams,
    ) -> None:
        """data.yaml is written and contains required keys with correct path."""
        service_with_valid_source.run(valid_params)

        yaml_path = Path(valid_params.output_dir) / "data.yaml"
        assert yaml_path.exists()

        with yaml_path.open() as fh:
            content = yaml.safe_load(fh)

        required_keys = {"path", "train", "val", "test", "kpt_shape", "names"}
        assert required_keys.issubset(content.keys())
        assert content["path"] == str(Path(valid_params.output_dir).resolve())

    def test_dataset_stats_json_written(
        self,
        service_with_valid_source: DatasetLoadingService,
        valid_params: YoloDatasetParams,
    ) -> None:
        """dataset_stats.json is written with correct structure."""
        service_with_valid_source.run(valid_params)

        stats_path = Path(valid_params.output_dir) / "dataset_stats.json"
        assert stats_path.exists()

        with stats_path.open() as fh:
            data = json.load(fh)

        assert data["version"] == "v1"
        assert data["source"] == "s3"
        assert data["train_images"] == 3
        assert data["sampled"] is False

    def test_image_label_files_present(
        self,
        service_with_valid_source: DatasetLoadingService,
        valid_params: YoloDatasetParams,
    ) -> None:
        """Every downloaded image has a corresponding label file."""
        service_with_valid_source.run(valid_params)
        out = Path(valid_params.output_dir)
        for split in SPLITS:
            for img in (out / "images" / split).iterdir():
                label = out / "labels" / split / (img.stem + ".txt")
                assert label.exists(), f"Missing label for {img}"


# ---------------------------------------------------------------------------
# Source resolution tests
# ---------------------------------------------------------------------------


class TestSourceResolution:
    def test_path_override_used_when_provided(self, tmp_path: Path) -> None:
        """When path_override is set, the custom bucket/prefix is used."""
        src = tmp_path / "source"
        _build_yolo_tree(src)

        # Build a mock that responds to a custom bucket/prefix
        mock_s3 = MagicMock()
        custom_prefix = "custom/path/"

        all_keys: list[str] = []
        for path in sorted(src.rglob("*")):
            if path.is_file():
                all_keys.append(str(path.relative_to(src)))

        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Contents": [{"Key": custom_prefix + k} for k in all_keys]}
        ]
        mock_s3.get_paginator.return_value = paginator

        def download_file(bucket: str, key: str, dest: str) -> None:
            relative = key[len(custom_prefix) :]
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes((src / relative).read_bytes())

        mock_s3.download_file.side_effect = download_file

        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
            path_override="s3://custom-bucket/custom/path/",
        )

        service = DatasetLoadingService(s3_client=mock_s3)
        service.run(params)

        _, kwargs_list = (
            mock_s3.get_paginator.call_args_list[0],
            mock_s3.get_paginator.call_args_list,
        )
        # Verify the paginator was called with the right bucket
        mock_s3.get_paginator.assert_called_with("list_objects_v2")

    def test_invalid_path_override_raises(self, tmp_path: Path) -> None:
        """An invalid path_override (not s3://) raises DatasetLoadingError."""
        mock_s3 = MagicMock()
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
            path_override="not-a-valid-uri",
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        with pytest.raises(DatasetLoadingError, match="valid s3:// URI"):
            service.run(params)

    def test_empty_s3_prefix_raises(self, tmp_path: Path) -> None:
        """When S3 returns no objects, DatasetLoadingError is raised."""
        mock_s3 = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": []}]
        mock_s3.get_paginator.return_value = paginator

        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        with pytest.raises(DatasetLoadingError, match="No objects found"):
            service.run(params)


# ---------------------------------------------------------------------------
# Validation failure tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_split_directory_raises(self, tmp_path: Path) -> None:
        """A missing split directory under images/ causes DatasetLoadingError."""
        src = tmp_path / "source"
        _build_yolo_tree(src)
        # Remove the test split to simulate an incomplete download
        import shutil

        shutil.rmtree(src / "images" / "test")

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        with pytest.raises(DatasetLoadingError, match="Missing split directory"):
            service.run(params)

    def test_image_without_label_raises(self, tmp_path: Path) -> None:
        """An image file with no corresponding label file raises DatasetLoadingError."""
        src = tmp_path / "source"
        _build_yolo_tree(
            src,
            missing_label_split="train",
            missing_label_stem="img000001",
        )

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        with pytest.raises(DatasetLoadingError, match="Missing label file"):
            service.run(params)

    def test_empty_label_file_raises(self, tmp_path: Path) -> None:
        """An empty label file raises DatasetLoadingError."""
        src = tmp_path / "source"
        _build_yolo_tree(
            src,
            empty_label_split="val",
            empty_label_stem="img000001",
        )

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        with pytest.raises(DatasetLoadingError, match="empty"):
            service.run(params)

    def test_wrong_token_count_raises(self, tmp_path: Path) -> None:
        """A label line with wrong token count raises DatasetLoadingError."""
        src = tmp_path / "source"
        _build_yolo_tree(src, tokens_per_line=5)  # should be 38

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        with pytest.raises(DatasetLoadingError, match="Label format error"):
            service.run(params)

    def test_missing_data_yaml_key_raises(self, tmp_path: Path) -> None:
        """data.yaml missing a required key raises DatasetLoadingError."""
        src = tmp_path / "source"
        _build_yolo_tree(src)

        # Overwrite data.yaml with an incomplete version
        incomplete = {"path": str(src), "train": "images/train"}
        (src / "data.yaml").write_text(yaml.dump(incomplete))

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        with pytest.raises(DatasetLoadingError, match="missing required keys"):
            service.run(params)


# ---------------------------------------------------------------------------
# Sampling tests
# ---------------------------------------------------------------------------


class TestSampling:
    def test_sampling_keeps_exact_count(self, tmp_path: Path) -> None:
        """With sample_size=2 exactly 2 images remain in each split."""
        src = tmp_path / "source"
        _build_yolo_tree(src, n_train=5, n_val=5, n_test=5)

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
            sample_size=2,
            seed=0,
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        stats = service.run(params)

        assert stats.train_images == 2
        assert stats.val_images == 2
        assert stats.test_images == 2
        assert stats.sampled is True
        assert stats.sample_size == 2

    def test_sampling_label_files_match_images(self, tmp_path: Path) -> None:
        """After sampling every remaining image still has a label file."""
        src = tmp_path / "source"
        _build_yolo_tree(src, n_train=6, n_val=6, n_test=6)

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
            sample_size=3,
            seed=42,
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        service.run(params)

        out = Path(params.output_dir)
        for split in SPLITS:
            imgs = list((out / "images" / split).glob("*.jpg"))
            for img in imgs:
                label = out / "labels" / split / (img.stem + ".txt")
                assert label.exists(), f"No label for sampled image {img}"

    def test_sampling_exceeds_available_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When sample_size > available images, all are kept and a warning is logged."""
        import logging

        src = tmp_path / "source"
        _build_yolo_tree(src, n_train=2, n_val=2, n_test=2)

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
            sample_size=100,
            seed=7,
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        with caplog.at_level(logging.WARNING):
            stats = service.run(params)

        assert stats.train_images == 2
        assert stats.val_images == 2
        assert stats.test_images == 2
        assert any("keeping all" in record.message.lower() for record in caplog.records)

    def test_sampling_is_reproducible(self, tmp_path: Path) -> None:
        """Two runs with the same seed produce the same file set."""
        src = tmp_path / "source"
        _build_yolo_tree(src, n_train=10, n_val=10, n_test=10)

        def _run(out_dir: str) -> set[str]:
            mock_s3 = _make_mock_s3(src)
            params = YoloDatasetParams(
                version="v1",
                source="s3",
                output_dir=out_dir,
                sample_size=4,
                seed=99,
            )
            service = DatasetLoadingService(s3_client=mock_s3)
            service.run(params)
            return {
                f.name
                for split in SPLITS
                for f in (Path(out_dir) / "images" / split).glob("*.jpg")
            }

        result_a = _run(str(tmp_path / "output_a"))
        result_b = _run(str(tmp_path / "output_b"))
        assert result_a == result_b
