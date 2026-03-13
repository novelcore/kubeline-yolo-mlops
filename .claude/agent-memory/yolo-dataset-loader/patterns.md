# Implementation Patterns — YOLO Dataset Loader

## Kubestep Wiring Pattern
```
cli.py  →  Manager.__init__(config)  →  DatasetLoadingService(s3_client)
                                Manager.run(...)  →  service.run(YoloDatasetParams)
```
- Manager owns boto3 client construction (`_build_s3_client`).
- LakeFS: if `config.lakefs_endpoint` is set, use it as `endpoint_url`; credentials from `LAKEFS_ACCESS_KEY` / `LAKEFS_SECRET_KEY`.
- S3: standard AWS credentials via `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` or IAM role.

## Mock Pattern for Tests
```python
def _make_mock_s3(dataset_root: Path) -> MagicMock:
    # list_objects_v2 paginator returns Contents built from actual disk files
    # download_file copies bytes from dataset_root using relative path after prefix
```
- Never make real S3 calls in tests.
- `_build_yolo_tree(root, n_train, n_val, n_test, tokens_per_line, ...)` helper creates minimal dataset tree.
- Inject bad data (wrong tokens, missing labels) via kwargs to `_build_yolo_tree`.

## pyproject.toml Pattern for boto3 Stubs
```toml
[tool.poetry.group.dev.dependencies]
types-pyyaml = "^6.0"
boto3-stubs = {version = "^1.38.0", extras = ["s3"]}

[[tool.mypy.overrides]]
module = ["boto3", "boto3.*", "botocore", "botocore.*"]
ignore_missing_imports = true
```

## data.yaml Runtime Patching
- Always overwrite `data.yaml`'s `path` field to `str(Path(output_dir).resolve())`.
- If `data.yaml` was downloaded from S3, merge into it (preserve kpt_shape, names, etc.).
- Use `yaml.safe_load` / `yaml.dump` with `default_flow_style=False, sort_keys=False`.

## Sampling Determinism
- Use `random.Random(seed)` (not global `random.seed()`); module-local RNG.
- Sort image files before sampling for cross-platform reproducibility.
