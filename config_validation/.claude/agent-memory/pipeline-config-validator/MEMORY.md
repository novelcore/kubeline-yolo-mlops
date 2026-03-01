# Pipeline Config Validator — Agent Memory

## Interface Convention (established 2026-03-01)

The config_validation step follows the Kubestep pattern: CLI accepts individual field flags
(not a YAML file path). The orchestrator parses the YAML and passes values as CLI flags.

### CLI -> Manager -> Service call chain
- `cli.py` builds a nested `config_dict` from flat Typer options and calls `manager.run(config_dict=..., output_path=...)`
- `manager.py` forwards to `service.run(config_dict=..., output_path=...)`
- `service.py` calls `_validate_schema(config_dict)` directly — no YAML loading in the service

### Key files
- `/home/thanos/Documents/kubeline-yolo-mlops/config_validation/app/cli.py` — flat Typer options, reconstructs nested dict
- `/home/thanos/Documents/kubeline-yolo-mlops/config_validation/app/manager.py` — Manager.run(config_dict, output_path)
- `/home/thanos/Documents/kubeline-yolo-mlops/config_validation/app/services/config_validation.py` — no _load_yaml; run(config_dict, output_path)
- `/home/thanos/Documents/kubeline-yolo-mlops/orchestrate.sh` — run_config_validation() passes individual CFG_* flags

### orchestrate.sh conventions
- Shell vars: `CFG_{section}_{key}` (e.g., `CFG_experiment_name`, `CFG_training_epochs`)
- Required fields use `:?` syntax: `${CFG_experiment_name:?Missing experiment.name in config}`
- Optional fields use `:-` with conditional flag: `[[ -n "$val" ]] && flags+=" --flag '${val}'"`
- Output paths: local=`${REPO_ROOT}/.tmp/validated_config.json`, docker=`/artifacts/validated_config.json`
- Log line after parse_yaml: `log_info "Experiment: ${CFG_experiment_name:-unknown}"`

### AugmentationConfig
Not exposed via CLI — uses model defaults. The dict passed to PipelineConfig simply omits the
`augmentation` key and AugmentationConfig.default_factory provides all defaults.

### Field that is NOT passed via CLI
- `tags` (dict[str, str] under experiment) — too complex for shell; always None from CLI path

## PipelineConfig Schema Summary
See `/home/thanos/Documents/kubeline-yolo-mlops/config_validation/app/models/pipeline_config.py`
for full validators. Key constraints:
- `experiment.name`: alphanumeric + hyphens, must not start with hyphen
- `dataset.source`: must be 's3' or 'lakefs'
- `model.variant`: must match `^yolov(8|9|10|11)[nsmlx]-pose\.pt$`
- `training.image_size`: must be multiple of 32
- `training.optimizer`: must be 'SGD', 'Adam', or 'AdamW'
- `checkpointing.storage_path`: must start with 's3://' or 'lakefs://'
- `checkpointing.resume_from`: null, 'auto', or S3 path starting with 's3://'
- `TrainingConfig`: has `extra="ignore"` — silently drops unknown fields like `scheduler`
- `PipelineConfig`: has `extra="ignore"` — silently drops `resources` section
