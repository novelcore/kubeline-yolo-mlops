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

When submitting a workflow, set these parameters:

| Parameter | Value |
| --- | --- |
| `dataset-source` | `lakefs` |
| `dataset-version` | `v1` (or your version tag) |

The lakeFS repository and branch are pre-filled by the platform. The pipeline constructs the S3 path automatically from the version you provide.

### Option 2: Direct S3

Point directly to an S3 path using the `dataset-path-override` parameter:

| Parameter | Value |
| --- | --- |
| `dataset-source` | `s3` |
| `dataset-path-override` | `s3://your-bucket/datasets/spacecraft-pose-v1/` |

When `dataset-path-override` is set, the `dataset-version` parameter is ignored.

## Subsampling for Development

Training on the full dataset can take hours. While iterating on hyperparameters or debugging, use a random subset by setting these parameters at submission time:

| Parameter | Value |
| --- | --- |
| `dataset-sample-size` | `200` |
| `dataset-seed` | `42` |

This trains on 200 random images. Using the same seed gives you the same 200 images each time.

Leave `dataset-sample-size` empty (the default) to use the full dataset for a real training run.
