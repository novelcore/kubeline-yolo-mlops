# Uploading Data to LakeFS

Once you've prepared a dataset in [YOLO Pose format](datasets.md), the next step is getting it into LakeFS so the pipeline can read it. This page walks through the `kubecore-dataset` CLI that the app repo ships for that purpose.

The tool gets a YOLO-pose dataset from your laptop into your project's data store, so it shows up in the pipeline's `dataset-ref` dropdown. It signs you in through your browser (single sign-on) — **no LakeFS access keys are needed on your machine**. The app repo already knows its lakeFS URL and repo (from `.kubecore/dataset-config.yaml`, filled in per app by the platform), so you never type a URL.

!!! info "Data lands at the branch root"
    The tool uploads your dataset to the **root** of a lakeFS branch, i.e. `s3://<repo>/<branch>/`, with `data.yaml`, `images/{train,val}/`, and `labels/{train,val}/` directly under it. The branch name you sync to becomes the value shown in the `dataset-ref` dropdown. There is no `dataset/` prefix.

---

## Install

From the app repo root:

```bash
pip3 install ./dataset_tools
```

Python 3.9+. It only needs `requests` and `pyyaml`.

!!! tip "If `kubecore-dataset` is \"not found\""
    `pip install` sometimes puts the command in a directory that isn't on your `PATH`.
    The easiest fix is to run it as a module instead — the commands are identical:

    ```bash
    python -m dataset_cli login
    python -m dataset_cli validate ./my_dataset
    python -m dataset_cli sync ./my_dataset --branch my-dataset
    ```

    Everywhere below you can swap `kubecore-dataset <cmd>` for `python -m dataset_cli <cmd>`.

---

## The one command you'll use

The one-shot helper does login, validate, and sync in a single step:

```bash
./scripts/upload-dataset.py  datasets/my_dataset  my-first-dataset
```

It logs you in (browser), validates the dataset, uploads it to the branch root, and commits. The second argument (`my-first-dataset`) is the branch name — that becomes the `dataset-ref` value in the dropdown.

---

## The three subcommands

If you'd rather run the steps individually, the CLI exposes `login`, `validate`, and `sync`.

### `login` — browser sign-in

```bash
kubecore-dataset login
```

Opens your browser to sign in with single sign-on and caches the session locally (at `~/.config/kubecore-ml/`). No keys touch your machine. If the automatic browser flow can't finish, add `--paste` for a one-time guided sign-in.

### `validate` — check before you upload

```bash
kubecore-dataset validate datasets/my_dataset
```

Runs locally. Checks the dataset is a valid Ultralytics-pose set — `data.yaml` with `path/train/val/kpt_shape/names`, the `train` and `val` splits present, every image paired with a same-named label, and label rows in the right pose format. If anything is off it prints exactly what, and stops — so a bad dataset never reaches a run.

### `sync` — upload what changed

```bash
kubecore-dataset sync datasets/my_dataset --branch my-first-dataset
```

Compares your folder to the stored version, uploads new/changed files to the **root** of the branch, removes files you deleted, and creates one commit. Re-running is always safe.

---

## What a valid dataset looks like

```
my_dataset/
├── data.yaml
├── images/train/  *.jpg      # + images/val/
└── labels/train/  *.txt      # one .txt per image, same name; + labels/val/
```

```yaml
# data.yaml
path: .
train: images/train
val: images/val
kpt_shape: [11, 3]
names:
  0: spacecraft
```

---

## After uploading

Your dataset name (the branch you synced to) appears in the `dataset-ref` dropdown on the Argo Workflows submit form within ~30 minutes — the list refreshes on its own. Then run your pipeline from the Argo UI.

---

## Next steps

Once your data is committed to a LakeFS branch, head back to [Datasets](datasets.md) to point your pipeline at it via the `dataset-source` and `dataset-ref` parameters.
