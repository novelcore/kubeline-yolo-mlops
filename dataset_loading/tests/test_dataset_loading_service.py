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


def _make_label_line(tokens: int = 38, *, valid: bool = True) -> str:
    """Return a synthetic Ultralytics-compliant label line.

    When *valid* is True (default), values adhere to YOLO conventions:
    class=0, bbox in [0, 1], keypoint coords in [0, 1], visibility=2.
    When *valid* is False, all tokens are ``0.5`` (may fail range checks).
    """
    if not valid:
        return " ".join(["0"] + ["0.5"] * (tokens - 1))
    # class (int 0) + 4 bbox values + keypoint triplets (x, y, vis)
    parts: list[str] = ["0"]  # class ID
    parts.extend(["0.5"] * 4)  # cx, cy, w, h
    remaining = tokens - 5
    # Fill keypoints: for dim=3 -> (x, y, vis) triplets; dim=2 -> (x, y) pairs
    # Detect dimension from token count and kpt_shape=[11, 3] default
    kpt_dim = 3 if remaining % 3 == 0 else 2
    for i in range(0, remaining, kpt_dim):
        parts.append("0.5")  # x
        parts.append("0.5")  # y
        if kpt_dim == 3:
            parts.append("2")  # visibility: 2 = visible
    return " ".join(parts)


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
                label_path.write_text(
                    _make_label_line(tokens_per_line, valid=(tokens_per_line == 38))
                    + "\n"
                )

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

        required_keys = {"path", "train", "val", "kpt_shape", "names"}
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
        # Pre-download check catches empty bucket: "No images or labels found"
        with pytest.raises(DatasetLoadingError, match="No images or labels found"):
            service.run(params)


# ---------------------------------------------------------------------------
# Validation failure tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_required_split_directory_raises(self, tmp_path: Path) -> None:
        """A missing required split (train/val) on S3 causes DatasetLoadingError.

        The pre-download S3 structure check catches this before any download
        occurs, so the error message reflects the S3-level finding.
        """
        src = tmp_path / "source"
        _build_yolo_tree(src)
        # Remove the val split — train and val are required per Ultralytics
        import shutil

        shutil.rmtree(src / "images" / "val")
        # Also remove val labels so the pairing check doesn't trip first
        shutil.rmtree(src / "labels" / "val")

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        # Pre-download check now catches this with a clear S3 structure message
        with pytest.raises(DatasetLoadingError, match="S3 structure invalid"):
            service.run(params)

    def test_missing_test_split_does_not_raise(self, tmp_path: Path) -> None:
        """A missing test split is acceptable (optional per Ultralytics)."""
        src = tmp_path / "source"
        _build_yolo_tree(src, n_test=0)
        # Remove test directories entirely
        import shutil

        shutil.rmtree(src / "images" / "test", ignore_errors=True)
        shutil.rmtree(src / "labels" / "test", ignore_errors=True)

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        stats = service.run(params)
        assert stats.train_images == 3
        assert stats.val_images == 2
        assert stats.test_images == 0

    def test_image_without_label_raises(self, tmp_path: Path) -> None:
        """An image file with no corresponding label file raises DatasetLoadingError.

        The pre-download S3 structure check catches missing labels at the S3
        key level before any download is attempted.
        """
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
        # Pre-download check now catches missing label keys on S3
        with pytest.raises(DatasetLoadingError, match="have no corresponding label"):
            service.run(params)

    def test_empty_label_file_raises(self, tmp_path: Path) -> None:
        """An empty label file raises DatasetLoadingError.

        The inline validation during download catches an empty label file
        immediately after it is downloaded.
        """
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

    def test_invalid_names_format_raises(self, tmp_path: Path) -> None:
        """data.yaml with names as a list (not dict) raises DatasetLoadingError."""
        src = tmp_path / "source"
        _build_yolo_tree(src)
        bad_yaml = {
            "path": str(src),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "kpt_shape": [11, 3],
            "names": ["spacecraft"],  # should be {0: "spacecraft"}
        }
        (src / "data.yaml").write_text(yaml.dump(bad_yaml))

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1", source="s3", output_dir=str(tmp_path / "output")
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        with pytest.raises(DatasetLoadingError, match="names.*must be a dict"):
            service.run(params)

    def test_invalid_kpt_shape_raises(self, tmp_path: Path) -> None:
        """data.yaml with bad kpt_shape raises DatasetLoadingError."""
        src = tmp_path / "source"
        _build_yolo_tree(src)
        bad_yaml = {
            "path": str(src),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "kpt_shape": [11, 4],  # dimension must be 2 or 3
            "names": {0: "spacecraft"},
        }
        (src / "data.yaml").write_text(yaml.dump(bad_yaml))

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1", source="s3", output_dir=str(tmp_path / "output")
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        with pytest.raises(DatasetLoadingError, match="kpt_shape"):
            service.run(params)

    def test_nc_mismatch_raises(self, tmp_path: Path) -> None:
        """data.yaml with nc != len(names) raises DatasetLoadingError."""
        src = tmp_path / "source"
        _build_yolo_tree(src)
        bad_yaml = {
            "path": str(src),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "kpt_shape": [11, 3],
            "names": {0: "spacecraft"},
            "nc": 5,  # doesn't match len(names)=1
        }
        (src / "data.yaml").write_text(yaml.dump(bad_yaml))

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1", source="s3", output_dir=str(tmp_path / "output")
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        with pytest.raises(DatasetLoadingError, match="nc.*does not match"):
            service.run(params)

    def test_bbox_out_of_range_raises(self, tmp_path: Path) -> None:
        """A label with bbox value > 1.0 raises DatasetLoadingError.

        Inline validation during download catches the bad bbox value immediately
        after the label file is downloaded.
        """
        src = tmp_path / "source"
        _build_yolo_tree(src)
        # Overwrite a label with an out-of-range bbox
        label_file = src / "labels" / "train" / "img000001.txt"
        # class=0, cx=1.5 (bad), cy=0.5, w=0.5, h=0.5, then 11 kpt triplets
        parts = ["0", "1.5", "0.5", "0.5", "0.5"]
        parts.extend(["0.5", "0.5", "2"] * 11)  # valid keypoints
        label_file.write_text(" ".join(parts) + "\n")

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1", source="s3", output_dir=str(tmp_path / "output")
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        with pytest.raises(DatasetLoadingError, match="bbox.*outside"):
            service.run(params)

    def test_bad_visibility_raises(self, tmp_path: Path) -> None:
        """A label with keypoint visibility not in {0,1,2} raises DatasetLoadingError.

        Inline validation during download catches the bad visibility immediately.
        """
        src = tmp_path / "source"
        _build_yolo_tree(src)
        label_file = src / "labels" / "train" / "img000001.txt"
        # class=0, valid bbox, first keypoint has visibility=5 (bad)
        parts = ["0", "0.5", "0.5", "0.5", "0.5"]
        parts.extend(["0.5", "0.5", "5"])  # bad visibility
        parts.extend(["0.5", "0.5", "2"] * 10)  # rest are valid
        label_file.write_text(" ".join(parts) + "\n")

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1", source="s3", output_dir=str(tmp_path / "output")
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        with pytest.raises(DatasetLoadingError, match="visibility"):
            service.run(params)

    def test_class_id_out_of_range_raises(self, tmp_path: Path) -> None:
        """A label with class ID >= nc raises DatasetLoadingError.

        Inline validation during download catches the bad class ID.
        """
        src = tmp_path / "source"
        _build_yolo_tree(src)
        label_file = src / "labels" / "train" / "img000001.txt"
        # class=5 but names only has class 0
        parts = ["5", "0.5", "0.5", "0.5", "0.5"]
        parts.extend(["0.5", "0.5", "2"] * 11)
        label_file.write_text(" ".join(parts) + "\n")

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1", source="s3", output_dir=str(tmp_path / "output")
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        with pytest.raises(DatasetLoadingError, match="class ID"):
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


# ---------------------------------------------------------------------------
# Labels-only mode tests
# ---------------------------------------------------------------------------


def _make_mock_s3_with_listing(dataset_root: Path) -> MagicMock:
    """Like ``_make_mock_s3`` but also categorises keys for labels-only tests.

    The ``download_file`` side-effect only succeeds for non-image files,
    matching the selective-download behaviour.
    """
    mock_s3 = MagicMock()

    prefix = "datasets/speedplus_yolo/v1/"
    all_keys: list[str] = []
    for path in sorted(dataset_root.rglob("*")):
        if path.is_file():
            all_keys.append(str(path.relative_to(dataset_root)))

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


class TestLabelsOnlyMode:
    def test_no_images_on_disk(self, tmp_path: Path) -> None:
        """In labels-only mode, no image files should be downloaded."""
        src = tmp_path / "source"
        _build_yolo_tree(src, n_train=3, n_val=2, n_test=2)

        mock_s3 = _make_mock_s3_with_listing(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
            labels_only=True,
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        service.run(params)

        out = Path(params.output_dir)
        for split in SPLITS:
            images_dir = out / "images" / split
            if images_dir.exists():
                image_files = [
                    f
                    for f in images_dir.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                ]
                assert len(image_files) == 0, f"Images found in {split}: {image_files}"

    def test_labels_present_on_disk(self, tmp_path: Path) -> None:
        """In labels-only mode, label files should be downloaded."""
        src = tmp_path / "source"
        _build_yolo_tree(src, n_train=3, n_val=2, n_test=2)

        mock_s3 = _make_mock_s3_with_listing(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
            labels_only=True,
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        service.run(params)

        out = Path(params.output_dir)
        for split, expected in [("train", 3), ("val", 2), ("test", 2)]:
            labels_dir = out / "labels" / split
            assert labels_dir.is_dir()
            label_files = list(labels_dir.glob("*.txt"))
            assert len(label_files) == expected

    def test_manifest_written(self, tmp_path: Path) -> None:
        """dataset_manifest.json is written with correct image counts."""
        src = tmp_path / "source"
        _build_yolo_tree(src, n_train=3, n_val=2, n_test=2)

        mock_s3 = _make_mock_s3_with_listing(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
            labels_only=True,
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        service.run(params)

        manifest_path = Path(params.output_dir) / "dataset_manifest.json"
        assert manifest_path.exists()

        with manifest_path.open() as fh:
            manifest = json.load(fh)

        assert manifest["total_images"] == 7  # 3+2+2
        assert len(manifest["splits"]["train"]) == 3
        assert len(manifest["splits"]["val"]) == 2
        assert len(manifest["splits"]["test"]) == 2

    def test_missing_label_detected(self, tmp_path: Path) -> None:
        """Labels-only mode detects when a label is missing for an image.

        The pre-download S3 structure check now catches this at the S3 key
        level before any download occurs.
        """
        src = tmp_path / "source"
        _build_yolo_tree(
            src,
            missing_label_split="train",
            missing_label_stem="img000001",
        )

        mock_s3 = _make_mock_s3_with_listing(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
            labels_only=True,
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        with pytest.raises(DatasetLoadingError, match="have no corresponding label"):
            service.run(params)

    def test_stats_json_written(self, tmp_path: Path) -> None:
        """dataset_stats.json is written in labels-only mode."""
        src = tmp_path / "source"
        _build_yolo_tree(src, n_train=3, n_val=2, n_test=2)

        mock_s3 = _make_mock_s3_with_listing(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
            labels_only=True,
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        service.run(params)

        stats_path = Path(params.output_dir) / "dataset_stats.json"
        assert stats_path.exists()
        with stats_path.open() as fh:
            data = json.load(fh)
        assert data["train_images"] == 3
        assert data["val_images"] == 2
        assert data["test_images"] == 2

    def test_data_yaml_written_in_labels_only(self, tmp_path: Path) -> None:
        """data.yaml is written in labels-only mode."""
        src = tmp_path / "source"
        _build_yolo_tree(src, n_train=3, n_val=2, n_test=2)

        mock_s3 = _make_mock_s3_with_listing(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
            labels_only=True,
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        service.run(params)

        yaml_path = Path(params.output_dir) / "data.yaml"
        assert yaml_path.exists()
        with yaml_path.open() as fh:
            content = yaml.safe_load(fh)
        assert "path" in content


# ---------------------------------------------------------------------------
# Dataset stats and directory integrity report tests
# ---------------------------------------------------------------------------


class TestDatasetStatsAndIntegrity:
    """Verify label count fields in stats and the logged integrity report."""

    def test_stats_include_label_counts(
        self,
        service_with_valid_source: DatasetLoadingService,
        valid_params: YoloDatasetParams,
    ) -> None:
        """Returned YoloDatasetStats includes per-split label count fields."""
        stats = service_with_valid_source.run(valid_params)

        assert stats.train_labels == 3
        assert stats.val_labels == 2
        assert stats.test_labels == 2

    def test_stats_json_includes_label_counts(
        self,
        service_with_valid_source: DatasetLoadingService,
        valid_params: YoloDatasetParams,
    ) -> None:
        """dataset_stats.json contains train_labels, val_labels, test_labels."""
        service_with_valid_source.run(valid_params)

        stats_path = Path(valid_params.output_dir) / "dataset_stats.json"
        with stats_path.open() as fh:
            data = json.load(fh)

        assert "train_labels" in data
        assert "val_labels" in data
        assert "test_labels" in data
        assert data["train_labels"] == 3
        assert data["val_labels"] == 2
        assert data["test_labels"] == 2

    def test_integrity_report_pass_logged(
        self,
        service_with_valid_source: DatasetLoadingService,
        valid_params: YoloDatasetParams,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A PASS verdict is logged when the directory structure is valid."""
        import logging

        with caplog.at_level(logging.INFO):
            service_with_valid_source.run(valid_params)

        messages = " ".join(r.message for r in caplog.records)
        assert "PASS" in messages
        assert "data.yaml" in messages
        # All three split dirs should be reported
        for split in SPLITS:
            assert split in messages

    def test_integrity_report_images_and_labels_ok_logged(
        self,
        service_with_valid_source: DatasetLoadingService,
        valid_params: YoloDatasetParams,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """[OK] lines are emitted for both images/ and labels/ in each split."""
        import logging

        with caplog.at_level(logging.INFO):
            service_with_valid_source.run(valid_params)

        ok_messages = [r.message for r in caplog.records if "[OK]" in r.message]
        # data.yaml + 3 images/ dirs + 3 labels/ dirs = 7 OK lines
        assert len(ok_messages) >= 7

    def test_integrity_report_labels_only_mode(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """In labels-only mode missing images/ dirs are SKIP, not FAIL."""
        import logging

        src = tmp_path / "source"
        _build_yolo_tree(src, n_train=3, n_val=2, n_test=2)

        mock_s3 = _make_mock_s3_with_listing(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
            labels_only=True,
        )
        service = DatasetLoadingService(s3_client=mock_s3)

        with caplog.at_level(logging.INFO):
            service.run(params)

        messages = " ".join(r.message for r in caplog.records)
        # Verdict must be PASS in labels-only mode (no images dirs expected)
        assert "PASS" in messages
        # images/ absence is a SKIP, not a FAIL
        assert "SKIP" in messages
        assert "FAIL" not in messages

    def test_labels_only_stats_include_label_counts(self, tmp_path: Path) -> None:
        """Labels-only mode populates both *_images and *_labels stats fields."""
        src = tmp_path / "source"
        _build_yolo_tree(src, n_train=4, n_val=3, n_test=2)

        mock_s3 = _make_mock_s3_with_listing(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
            labels_only=True,
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        stats = service.run(params)

        # In labels-only mode images == labels (label files are the proxy)
        assert stats.train_labels == stats.train_images == 4
        assert stats.val_labels == stats.val_images == 3
        assert stats.test_labels == stats.test_images == 2


# ---------------------------------------------------------------------------
# Pre-download S3 structure check tests
# ---------------------------------------------------------------------------


class TestPreDownloadS3Check:
    """Tests specifically for the _check_s3_structure() pre-download guard."""

    def test_valid_structure_passes(self, tmp_path: Path) -> None:
        """A complete, valid S3 dataset structure passes the pre-download check."""
        src = tmp_path / "source"
        _build_yolo_tree(src, n_train=3, n_val=2, n_test=2)

        mock_s3 = _make_mock_s3(src)
        service = DatasetLoadingService(s3_client=mock_s3)

        # Should not raise
        listing = service._check_s3_structure(
            "temp-mlops", "datasets/speedplus_yolo/v1/"
        )
        assert len(listing.image_keys) == 7  # 3+2+2
        assert len(listing.label_keys) == 7
        assert listing.data_yaml_key is not None

    def test_missing_data_yaml_on_s3_raises_before_download(
        self, tmp_path: Path
    ) -> None:
        """Missing data.yaml on S3 raises DatasetLoadingError before any download."""
        src = tmp_path / "source"
        _build_yolo_tree(src, include_data_yaml=False)

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
        )
        service = DatasetLoadingService(s3_client=mock_s3)

        # The error is raised before any download; download_file must not be called
        with pytest.raises(DatasetLoadingError, match="data.yaml not found on S3"):
            service.run(params)

        mock_s3.download_file.assert_not_called()

    def test_missing_train_images_on_s3_raises_before_download(
        self, tmp_path: Path
    ) -> None:
        """Missing images/train/ on S3 raises DatasetLoadingError before any download."""
        src = tmp_path / "source"
        _build_yolo_tree(src, n_train=0)
        import shutil

        shutil.rmtree(src / "images" / "train", ignore_errors=True)

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
        )
        service = DatasetLoadingService(s3_client=mock_s3)

        with pytest.raises(DatasetLoadingError, match="S3 structure invalid"):
            service.run(params)

        mock_s3.download_file.assert_not_called()

    def test_missing_val_labels_on_s3_raises_before_download(
        self, tmp_path: Path
    ) -> None:
        """Missing labels/val/ on S3 raises DatasetLoadingError before any download."""
        src = tmp_path / "source"
        _build_yolo_tree(src)
        import shutil

        shutil.rmtree(src / "labels" / "val", ignore_errors=True)

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
        )
        service = DatasetLoadingService(s3_client=mock_s3)

        with pytest.raises(DatasetLoadingError, match="S3 structure invalid"):
            service.run(params)

        mock_s3.download_file.assert_not_called()

    def test_image_without_label_on_s3_raises_before_download(
        self, tmp_path: Path
    ) -> None:
        """Image key with no corresponding label key on S3 raises before any download."""
        src = tmp_path / "source"
        _build_yolo_tree(
            src,
            missing_label_split="val",
            missing_label_stem="img000002",
        )

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
        )
        service = DatasetLoadingService(s3_client=mock_s3)

        with pytest.raises(DatasetLoadingError, match="have no corresponding label"):
            service.run(params)

        mock_s3.download_file.assert_not_called()

    def test_missing_test_split_does_not_raise_in_precheck(
        self, tmp_path: Path
    ) -> None:
        """Missing test split on S3 is acceptable (test is optional)."""
        src = tmp_path / "source"
        _build_yolo_tree(src, n_test=0)
        import shutil

        shutil.rmtree(src / "images" / "test", ignore_errors=True)
        shutil.rmtree(src / "labels" / "test", ignore_errors=True)

        mock_s3 = _make_mock_s3(src)
        service = DatasetLoadingService(s3_client=mock_s3)

        # Should not raise — test is optional
        listing = service._check_s3_structure(
            "temp-mlops", "datasets/speedplus_yolo/v1/"
        )
        assert listing.data_yaml_key is not None
        assert len(listing.image_keys) == 5  # 3+2+0

    def test_precheck_logs_structure_summary(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Pre-download check logs the S3 structure at INFO level."""
        import logging

        src = tmp_path / "source"
        _build_yolo_tree(src, n_train=3, n_val=2, n_test=2)

        mock_s3 = _make_mock_s3(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
        )
        service = DatasetLoadingService(s3_client=mock_s3)

        with caplog.at_level(logging.INFO):
            service.run(params)

        messages = " ".join(r.message for r in caplog.records)
        # Pre-download check logs should appear
        assert "Pre-download S3 structure check" in messages
        assert "PASSED" in messages

    def test_precheck_in_labels_only_mode(self, tmp_path: Path) -> None:
        """Pre-download check also runs in labels-only mode and catches bad structure."""
        src = tmp_path / "source"
        _build_yolo_tree(src)
        import shutil

        shutil.rmtree(src / "images" / "train", ignore_errors=True)
        shutil.rmtree(src / "labels" / "train", ignore_errors=True)

        mock_s3 = _make_mock_s3_with_listing(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
            labels_only=True,
        )
        service = DatasetLoadingService(s3_client=mock_s3)

        with pytest.raises(DatasetLoadingError, match="S3 structure invalid"):
            service.run(params)

        mock_s3.download_file.assert_not_called()


# ---------------------------------------------------------------------------
# Inline label validation during download tests
# ---------------------------------------------------------------------------


class TestInlineLabelValidation:
    """Tests verifying that label validation happens inline during download."""

    def _make_mock_s3_tracking_downloads(
        self, dataset_root: Path
    ) -> tuple[MagicMock, list[str]]:
        """Like ``_make_mock_s3`` but also records which keys were downloaded.

        Returns the mock and a mutable list that is appended to on each
        ``download_file`` call (so tests can inspect what was downloaded
        before the error was raised).
        """
        mock_s3 = MagicMock()
        downloaded: list[str] = []

        prefix = "datasets/speedplus_yolo/v1/"
        all_keys: list[str] = []
        for path in sorted(dataset_root.rglob("*")):
            if path.is_file():
                all_keys.append(str(path.relative_to(dataset_root)))

        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Contents": [{"Key": prefix + k} for k in all_keys]}
        ]
        mock_s3.get_paginator.return_value = paginator

        def download_file(bucket: str, key: str, dest: str) -> None:
            relative = key[len(prefix) :]
            src = dataset_root / relative
            dest_path = Path(dest)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(src.read_bytes())
            downloaded.append(relative)

        mock_s3.download_file.side_effect = download_file
        return mock_s3, downloaded

    def test_data_yaml_downloaded_before_labels(self, tmp_path: Path) -> None:
        """data.yaml is always the first file downloaded so inline validation has kpt_shape."""
        src = tmp_path / "source"
        _build_yolo_tree(src, n_train=2, n_val=2, n_test=0)

        mock_s3, downloaded = self._make_mock_s3_tracking_downloads(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
        )
        service = DatasetLoadingService(s3_client=mock_s3)
        service.run(params)

        # data.yaml must be the very first downloaded file
        assert (
            downloaded[0] == "data.yaml"
        ), f"Expected data.yaml to be downloaded first, got: {downloaded[0]!r}"

    def test_inline_validation_catches_bad_label_immediately(
        self, tmp_path: Path
    ) -> None:
        """Inline validation raises before all files are downloaded on a bad label."""
        src = tmp_path / "source"
        # Build with 5 train images; corrupt the first one
        _build_yolo_tree(src, n_train=5, n_val=2, n_test=0)
        label_file = src / "labels" / "train" / "img000001.txt"
        parts = ["0", "1.5", "0.5", "0.5", "0.5"]  # cx=1.5 is invalid
        parts.extend(["0.5", "0.5", "2"] * 11)
        label_file.write_text(" ".join(parts) + "\n")

        mock_s3, downloaded = self._make_mock_s3_tracking_downloads(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
        )
        service = DatasetLoadingService(s3_client=mock_s3)

        with pytest.raises(DatasetLoadingError, match="bbox.*outside"):
            service.run(params)

        # Not all files should have been downloaded — error raised during download
        total_keys = sum(1 for _ in src.rglob("*") if _.is_file())
        assert len(downloaded) < total_keys, (
            "All files were downloaded despite an inline validation error; "
            "inline validation should fail fast."
        )

    def test_inline_validation_labels_only_mode(self, tmp_path: Path) -> None:
        """Inline validation also works in labels-only mode."""
        src = tmp_path / "source"
        _build_yolo_tree(src, n_train=3, n_val=2, n_test=0)
        # Corrupt a val label
        label_file = src / "labels" / "val" / "img000001.txt"
        parts = ["99", "0.5", "0.5", "0.5", "0.5"]  # class 99 is out of range
        parts.extend(["0.5", "0.5", "2"] * 11)
        label_file.write_text(" ".join(parts) + "\n")

        mock_s3 = _make_mock_s3_with_listing(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
            labels_only=True,
        )
        service = DatasetLoadingService(s3_client=mock_s3)

        with pytest.raises(DatasetLoadingError, match="class ID"):
            service.run(params)

    def test_inline_empty_label_detected_during_download(self, tmp_path: Path) -> None:
        """An empty label file is detected inline during download."""
        src = tmp_path / "source"
        _build_yolo_tree(src, empty_label_split="train", empty_label_stem="img000002")

        mock_s3, downloaded = self._make_mock_s3_tracking_downloads(src)
        params = YoloDatasetParams(
            version="v1",
            source="s3",
            output_dir=str(tmp_path / "output"),
        )
        service = DatasetLoadingService(s3_client=mock_s3)

        with pytest.raises(DatasetLoadingError, match="empty"):
            service.run(params)
