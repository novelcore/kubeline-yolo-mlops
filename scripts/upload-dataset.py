#!/usr/bin/env python3
"""Upload a local YOLO-pose dataset to lakeFS — the one-liner wrapper.

This is a thin convenience wrapper around the `kubecore-dataset` CLI
(`dataset_tools/dataset_cli`). It logs you in via the browser, validates the
dataset, and incrementally syncs it (uploads AND deletions) to a lakeFS branch
so it shows up in the pipeline's `dataset-ref` dropdown.

    ./scripts/upload-dataset.py <local-dataset-dir> <branch> [--url URL] [--repo REPO]

  <local-dataset-dir>  Directory with data.yaml + images/{train,val} +
                       labels/{train,val} at its ROOT (Ultralytics pose).
  <branch>             lakeFS branch to sync into (this is the value shown in
                       the dropdown). Created from the default branch if new.

Config (flags override env):
  --url  / LAKEFS_URL    lakeFS ingress, e.g. https://lakefs-<project>.<baseDns>
  --repo / LAKEFS_REPO   lakeFS repository name

Prefer the full CLI for anything beyond the happy path:
    kubecore-dataset --help
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make dataset_tools importable whether run from the repo root or elsewhere.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "dataset_tools"))

from dataset_cli.__main__ import main as cli_main  # noqa: E402


def main() -> int:
    argv = sys.argv[1:]
    if len(argv) < 2 or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv[:1] in (["-h"], ["--help"]) else 2
    dataset_dir, branch = argv[0], argv[1]
    passthrough = argv[2:]
    return cli_main(["upload", dataset_dir, "--branch", branch, *passthrough])


if __name__ == "__main__":
    sys.exit(main())
