# Datasets

The pipeline expects datasets in **YOLO Pose format**. This page explains what that means, where to put your data, and how to point the pipeline at it.

## Dataset Format

### YOLO Pose Format

Each image in your dataset needs a corresponding label file with keypoint annotations.

**Directory structure:**

```
dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── val/
│       ├── img_101.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img_001.txt
    │   ├── img_002.txt
    │   └── ...
    └── val/
        ├── img_101.txt
        └── ...
```

**Label file format** (one line per object):

```
<class_id> <x_center> <y_center> <width> <height> <kp1_x> <kp1_y> <kp1_v> ... <kpN_x> <kpN_y> <kpN_v>
```

All values are normalized to `[0, 1]` relative to the image dimensions.
Keypoint visibility `v` is `0` (not labelled), `1` (labelled but occluded), or `2` (labelled and visible).

### SPEED+ Dataset

The default dataset for this pipeline is **SPEED+** (Spacecraft Pose Estimation Dataset Plus), which is pre-formatted in YOLO Pose format. SPEED+ provides synthetic images of the Tango spacecraft with 11 keypoint annotations per image.

If you are using a different dataset, it must conform to the YOLO Pose format above.

## Data Storage

### Option 1: LakeFS (recommended)

LakeFS provides Git-like versioning for your dataset on top of S3. Use this option when you want to track exactly which version of the data was used for each experiment.

Set these fields in `pipeline_config.yaml`:

```yaml
dataset:
  source: "lakefs"
  lakefs_repo: "your-repo-name"
  lakefs_branch: "main"
  version: "v1"
```

The pipeline constructs the S3 path automatically:

```
s3://<bucket>/<lakefs_repo>/<lakefs_branch>/<version>/
```

### Option 2: Direct S3

Point directly to an S3 path using `path_override`:

```yaml
dataset:
  source: "s3"
  path_override: "s3://your-bucket/datasets/spacecraft-pose-v1/"
```

When `path_override` is set, the `lakefs_repo`, `lakefs_branch`, and `version` fields are ignored.

## Subsampling for Development

Training on the full dataset can take hours. While iterating on hyperparameters or debugging, use a random subset:

```yaml
dataset:
  sample_size: 200    # Use only 200 images
  seed: 42            # Same seed = same 200 images each time
```

Set `sample_size: null` to switch back to the full dataset for a real training run.
