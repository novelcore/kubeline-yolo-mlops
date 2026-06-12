#!/usr/bin/env python3
"""Fail if kubeline.yaml step shims drop config keys or CLI hyperparameters.

Guards the config -> CLI flag mapping inside kubeline.yaml v2 against drift:

Check 1 — every key in pipeline_config.example.yaml must be forwarded by some
          step shim (or be on the explicit ALLOWLIST below with a reason).

Check 2 — every model-training CLI hyperparameter flag must be emitted by the
          model-training shim (or allowlisted).

Check 3 — every non-allowlisted config key must have a corresponding entry in
          pipeline.parameters (so the Argo UI form exposes it). Each config key
          is mapped to a parameter name via PARAM_NAME_MAP (for renamed params)
          or the default kebab-case conversion of the leaf name. Keys explicitly
          not user-tunable are added to PARAM_COVERAGE_ALLOWLIST with a reason.

Run from the repo root:

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

# Config keys not required to have a pipeline.parameters entry because they are
# platform-injected, derived at runtime, or covered by the CONFIG_KEY_ALLOWLIST.
PARAM_COVERAGE_ALLOWLIST = {
    # These are already in CONFIG_KEY_ALLOWLIST (not user-configurable at all)
    "experiment.tags": "no CLI flag; also in CONFIG_KEY_ALLOWLIST",
    "dataset.labels_only": "rejected in-cluster; also in CONFIG_KEY_ALLOWLIST",
    "dataset.manifest_only": "always forced true; also in CONFIG_KEY_ALLOWLIST",
    "resources.cpu_cores": "platform compute class; also in CONFIG_KEY_ALLOWLIST",
    "resources.gpu_count": "platform compute class; also in CONFIG_KEY_ALLOWLIST",
    "resources.gpu_type": "platform compute class; also in CONFIG_KEY_ALLOWLIST",
    "resources.memory_gb": "platform compute class; also in CONFIG_KEY_ALLOWLIST",
    # These are derived from other parameters at shim runtime, not separate params
    "dataset.lakefs_repo": "platform-injected from KubeLine generator context",
    "dataset.lakefs_branch": "platform-injected from KubeLine generator context",
    "registration.registered_model_name": "covered by registered-model-name param",
    "registration.promote_to": "covered by promote-to param",
}

# Explicit mapping from config key (dotted path) to pipeline parameter name,
# for cases where the leaf name differs from the Argo parameter name.
# Convention: leaf_name -> kebab-case unless overridden here.
PARAM_NAME_MAP: dict[str, str] = {
    "experiment.name": "experiment-name",
    "experiment.description": "experiment-description",
    "dataset.version": "dataset-version",
    "dataset.source": "dataset-source",
    "dataset.path_override": "dataset-path-override",
    "dataset.sample_size": "dataset-sample-size",
    "dataset.seed": "dataset-seed",
    "model.variant": "model-variant",
    "model.pretrained_weights": "pretrained-weights",
    "training.epochs": "epochs",
    "training.batch_size": "batch-size",
    "training.image_size": "image-size",
    "training.learning_rate": "learning-rate",
    "training.cos_lr": "cos-lr",
    "training.lrf": "lrf",
    "training.optimizer": "optimizer",
    "training.momentum": "momentum",
    "training.weight_decay": "weight-decay",
    "training.warmup_epochs": "warmup-epochs",
    "training.warmup_momentum": "warmup-momentum",
    "training.dropout": "dropout",
    "training.label_smoothing": "label-smoothing",
    "training.nbs": "nbs",
    "training.freeze": "freeze",
    "training.amp": "amp",
    "training.close_mosaic": "close-mosaic",
    "training.seed": "training-seed",
    "training.deterministic": "deterministic",
    "training.pose": "pose",
    "training.kobj": "kobj",
    "training.box": "box",
    "training.cls": "cls",
    "training.dfl": "dfl",
    "checkpointing.interval_epochs": "checkpointing-interval-epochs",
    "checkpointing.storage_path": "checkpointing-storage-path",
    "checkpointing.resume_from": "checkpoint-resume-from",
    "early_stopping.patience": "early-stopping-patience",
    "augmentation.hsv_h": "aug-hsv-h",
    "augmentation.hsv_s": "aug-hsv-s",
    "augmentation.hsv_v": "aug-hsv-v",
    "augmentation.degrees": "aug-degrees",
    "augmentation.translate": "aug-translate",
    "augmentation.scale": "aug-scale",
    "augmentation.shear": "aug-shear",
    "augmentation.perspective": "aug-perspective",
    "augmentation.flipud": "aug-flipud",
    "augmentation.fliplr": "aug-fliplr",
    "augmentation.mosaic": "aug-mosaic",
    "augmentation.mixup": "aug-mixup",
    "augmentation.copy_paste": "aug-copy-paste",
    "augmentation.erasing": "aug-erasing",
    "augmentation.bgr": "aug-bgr",
    "export.enabled": "export-enabled",
    "export.formats": "export-formats",
    "export.precisions": "export-precisions",
    "registration.registered_model_name": "registered-model-name",
    "registration.promote_to": "promote-to",
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


def pipeline_param_names(kubeline: dict) -> set[str]:
    """Extract all parameter names declared in pipeline.parameters."""
    names: set[str] = set()
    for p in (kubeline.get("pipeline") or {}).get("parameters") or []:
        if isinstance(p, dict) and "name" in p:
            names.add(p["name"])
    return names


def config_key_to_param_name(key: str) -> str:
    """Derive the expected pipeline parameter name for a config key.

    Uses PARAM_NAME_MAP if an explicit mapping exists, otherwise converts
    the leaf segment to kebab-case.
    """
    if key in PARAM_NAME_MAP:
        return PARAM_NAME_MAP[key]
    leaf = key.split(".")[-1]
    return leaf.replace("_", "-")


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

    # 3. Every non-allowlisted config key must have a corresponding pipeline.parameters entry.
    param_names = pipeline_param_names(kubeline)
    for key in sorted(flatten(cfg)):
        if key in PARAM_COVERAGE_ALLOWLIST:
            continue
        if any(key == a or key.startswith(a + ".") for a in CONFIG_KEY_ALLOWLIST):
            continue
        expected_param = config_key_to_param_name(key)
        if expected_param not in param_names:
            failures.append(
                f"config key '{key}' has no pipeline.parameters entry "
                f"(expected param name: '{expected_param}')"
            )

    if failures:
        print("kubeline coverage check FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    n_config_covered = len(flatten(cfg)) - len(CONFIG_KEY_ALLOWLIST)
    n_params = len(param_names)
    print(
        f"kubeline coverage OK: {n_config_covered} config keys forwarded, "
        f"{len(mt_flags)} model-training CLI flags accounted for, "
        f"{n_params} pipeline parameters declared"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
