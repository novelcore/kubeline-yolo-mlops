"""Validator tests — the CLI's accept/reject must mirror the pipeline's
dataset_loading contract (data.yaml keys, required splits, stem matching, pose
token count)."""
from __future__ import annotations

import pathlib

import pytest

from dataset_cli.validate import validate_dataset

POSE_ROW = "0 0.5 0.5 0.2 0.2 " + " ".join(["0.5"] * 33)  # 1+4+11*3 = 38 tokens
GOOD_YAML = "path: .\ntrain: images/train\nval: images/val\nkpt_shape: [11, 3]\nnames:\n  0: spacecraft\n"


def _make(root: pathlib.Path, *, yaml_text=GOOD_YAML, splits=("train", "val"),
          row=POSE_ROW, drop_label=False, extra_image=False):
    if yaml_text is not None:
        (root / "data.yaml").write_text(yaml_text)
    for split in splits:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "images" / split / "a.jpg").write_bytes(b"\xff\xd8\xff")
        if not drop_label:
            (root / "labels" / split / "a.txt").write_text(row + "\n")
    if extra_image:
        (root / "images" / "train" / "lonely.jpg").write_bytes(b"\xff\xd8\xff")
    return root


def test_good_dataset_passes(tmp_path):
    r = validate_dataset(_make(tmp_path))
    assert r.ok, r.report()
    assert r.stats["image-label pairs"] == 2


def test_missing_data_yaml_fails(tmp_path):
    _make(tmp_path, yaml_text=None)
    r = validate_dataset(tmp_path)
    assert not r.ok
    assert any("data.yaml" in e for e in r.errors)


def test_missing_val_split_fails(tmp_path):
    _make(tmp_path, splits=("train",))
    r = validate_dataset(tmp_path)
    assert not r.ok
    assert any("val" in e for e in r.errors)


def test_stem_mismatch_fails(tmp_path):
    _make(tmp_path, extra_image=True)
    r = validate_dataset(tmp_path)
    assert not r.ok
    assert any("no matching label" in e for e in r.errors)


def test_bad_token_count_fails(tmp_path):
    _make(tmp_path, row="0 0.5 0.5 0.2 0.2 0.1 0.1")  # 7 tokens
    r = validate_dataset(tmp_path)
    assert not r.ok
    assert any("tokens" in e for e in r.errors)


def test_bad_kpt_shape_fails(tmp_path):
    _make(tmp_path, yaml_text=GOOD_YAML.replace("[11, 3]", "[11, 5]"))
    r = validate_dataset(tmp_path)
    assert not r.ok
    assert any("kpt_shape" in e for e in r.errors)


def test_nc_mismatch_fails(tmp_path):
    _make(tmp_path, yaml_text=GOOD_YAML + "nc: 3\n")
    r = validate_dataset(tmp_path)
    assert not r.ok
    assert any("nc" in e for e in r.errors)


def test_not_a_directory(tmp_path):
    r = validate_dataset(tmp_path / "nope")
    assert not r.ok
