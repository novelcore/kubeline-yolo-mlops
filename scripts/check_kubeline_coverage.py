#!/usr/bin/env python3
"""Fail if kubeline.yaml step shims drop config keys or CLI hyperparameters.

Guards the config -> CLI flag mapping inside kubeline.yaml v2 against drift:
every key in pipeline_config.example.yaml must be forwarded by some step shim
(or be on the explicit ALLOWLIST below with a reason), and every model-training
CLI hyperparameter flag must be emitted by the model-training shim (or
allowlisted). Run from the repo root:

    python3 scripts/check_kubeline_coverage.py
"""

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# Config keys intentionally NOT forwarded to any CLI, with the reason.
CONFIG_KEY_ALLOWLIST = {
    "experiment.tags": "no CLI flag exists in any step (tracked app-side gap)",
    "dataset.labels_only": "rejected in-cluster: requires shared filesystem",
    "dataset.manifest_only": "always forced true by the shims in-cluster",
    "resources.cpu_cores": "superseded by platform compute classes (cpu-class)",
    "resources.gpu_count": "superseded by platform compute classes (gpu-class)",
    "resources.gpu_type": "superseded by platform compute classes (gpu-class)",
    "resources.memory_gb": "superseded by platform compute classes",
}

# model-training CLI options intentionally NOT emitted by the shim.
CLI_FLAG_ALLOWLIST = {
    "--device": "GPU assignment is platform-controlled (compute class)",
    "--s3-bucket": "auto-detected from dataset_manifest.json",
    "--s3-prefix": "auto-detected from dataset_manifest.json",
    "--dataset-dir": "emitted with a fixed /work path",  # emitted, fixed value
    "--output-dir": "emitted with a fixed /work path",
    "--disk-cache-gb": "platform default, not user config",
    "--source": "always s3 in-cluster",
    "--export": "emitted from export.enabled",
}


def flatten(d: dict, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for k, v in d.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            keys |= flatten(v, f"{path}.")
        else:
            keys.add(path)
    return keys


def shim_text(kubeline: dict) -> str:
    parts: list[str] = []
    for step in kubeline.get("steps", []):
        for a in step.get("args") or []:
            parts.append(str(a))
    return "\n".join(parts)


def cli_option_flags(cli_path: Path) -> set[str]:
    """Derive kebab-case flags from typer.Option parameter names."""
    flags: set[str] = set()
    in_run = False
    for line in cli_path.read_text().splitlines():
        if re.match(r"\s+def run\(", line):
            in_run = True
            continue
        if in_run and re.match(r"\s+\) -> ", line):
            break
        m = re.match(r"\s+([a-z_0-9]+): .*typer\.Option", line)
        if in_run and m:
            flags.add("--" + m.group(1).replace("_", "-"))
    return flags


def main() -> int:
    kubeline = yaml.safe_load((ROOT / "kubeline.yaml").read_text())
    if kubeline.get("version") != 2:
        print("kubeline.yaml is not v2; nothing to check")
        return 0

    cfg = yaml.safe_load((ROOT / "pipeline_config.example.yaml").read_text())
    shims = shim_text(kubeline)
    failures: list[str] = []

    # 1. Every config key must appear in some shim (as a config lookup) or be allowlisted.
    for key in sorted(flatten(cfg)):
        if any(key == a or key.startswith(a + ".") for a in CONFIG_KEY_ALLOWLIST):
            continue
        leaf = key.split(".")[-1]
        section = key.split(".")[0]
        if not (re.search(rf"['\"]({leaf})['\"]", shims) and section in shims):
            failures.append(f"config key not forwarded by any shim: {key}")

    # 2. Every model-training CLI option must be emitted by the shim or allowlisted.
    mt_flags = cli_option_flags(ROOT / "model_training" / "app" / "cli.py")
    emitted = set(re.findall(r"--[a-z0-9-]+", shims))
    for flag in sorted(mt_flags):
        if flag in CLI_FLAG_ALLOWLIST:
            continue
        # bool flags may only appear as --no-<flag> in the negative branch
        if flag not in emitted and f"--no-{flag.lstrip('-')}" not in emitted:
            failures.append(f"model-training CLI flag not emitted by shim: {flag}")

    if failures:
        print("kubeline coverage check FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(
        f"kubeline coverage OK: {len(flatten(cfg)) - len(CONFIG_KEY_ALLOWLIST)} config keys forwarded, "
        f"{len(mt_flags)} model-training CLI flags accounted for"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
