"""Per-app config discovery — the CLI must carry NO hardcoded instance values;
it reads lakefsUrl/repo from .kubecore/dataset-config.yaml, discovered by walking
up from the dataset dir or CWD."""
from __future__ import annotations

import pathlib

from dataset_cli import appconfig


def _write_config(root: pathlib.Path, **vals):
    d = root / ".kubecore"
    d.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}: {v}" for k, v in vals.items()]
    (d / "dataset-config.yaml").write_text("\n".join(lines) + "\n")


def test_discovers_config_walking_up(tmp_path, monkeypatch):
    appconfig.load_config.cache_clear()
    _write_config(tmp_path, lakefsUrl="https://lakefs-acme.example.com", repo="acme")
    nested = tmp_path / "some" / "dataset" / "dir"
    nested.mkdir(parents=True)
    assert appconfig.config_value("lakefsUrl", str(nested)) == "https://lakefs-acme.example.com"
    assert appconfig.config_value("repo", str(nested)) == "acme"


def test_no_config_returns_none(tmp_path, monkeypatch):
    appconfig.load_config.cache_clear()
    monkeypatch.delenv("KUBECORE_DATASET_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)  # empty dir, no .kubecore anywhere up to /
    # start at an isolated dir with no config
    assert appconfig.config_value("lakefsUrl", str(tmp_path)) is None


def test_env_override_points_at_config(tmp_path, monkeypatch):
    appconfig.load_config.cache_clear()
    cfg = tmp_path / "custom.yaml"
    cfg.write_text("lakefsUrl: https://from-env.example.com\nrepo: envrepo\n")
    monkeypatch.setenv("KUBECORE_DATASET_CONFIG", str(cfg))
    assert appconfig.config_value("lakefsUrl") == "https://from-env.example.com"
    assert appconfig.config_value("repo") == "envrepo"
