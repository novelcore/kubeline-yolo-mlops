# Datasets & Pipelines — the simple guide

Everything you need to go from "a folder of images on my laptop" to "a trained,
registered model", in three steps:

1. **Upload** a dataset to lakeFS.
2. How it gets **discovered** (so it shows up as a dropdown).
3. **Run** a pipeline on it.

No ML knowledge needed. If you can run one command and click a dropdown, you're set.

---

## The mental model (30 seconds)

- **lakeFS is "git for data."** A dataset is a lakeFS **branch** whose top folder
  holds `data.yaml` + `images/` + `labels/`. Uploading = committing to a branch.
- **You never type a path.** You pick a dataset from a **dropdown** on the run
  form. The platform finds it, pins the exact commit, and trains on it — so a run
  is always reproducible.
- **The branch name is the dropdown value.** Sync to branch `speedplus-v2` and
  `speedplus-v2` is what you pick later.

---

## 1. Upload a dataset

Use the `kubecore-dataset` uploader (in `dataset_tools/`). It logs you in through
your browser, checks the dataset is valid, and syncs it — in one command.

```bash
# from the repo root
./scripts/upload-dataset.py  <local-dataset-dir>  <branch> \
    --url https://lakefs-<project>.<baseDns>  --repo <project>

# example:
./scripts/upload-dataset.py  ~/Downloads/speedplus_yolo  speedplus-v2 \
    --url https://lakefs-yolo.<baseDns>  --repo yolo
```

What happens:

1. **Login** — your browser opens; log in with your normal SSO. (No keys, no
   `kubectl port-forward`, no cookie copy-paste.)
2. **Validate** — the tool checks your folder is a proper Ultralytics-pose
   dataset *before* uploading. If something's wrong it tells you exactly what,
   and stops.
3. **Sync** — it uploads new/changed files, **deletes files you removed locally**,
   and makes one commit on the branch.

Don't know your lakeFS URL?

```bash
kubectl -n <org-namespace> get kubepool <pool> -o jsonpath='{.status.baseDns}'
# URL = https://lakefs-<project>.<that-baseDns>
```

### What a valid dataset looks like

```
speedplus_yolo/
├── data.yaml                 # path, train, val, kpt_shape, names
├── images/train/  *.jpg
├── images/val/    *.jpg
├── labels/train/  *.txt      # one .txt per image, same filename stem
└── labels/val/    *.txt
```

`data.yaml`:

```yaml
path: .
train: images/train
val: images/val
kpt_shape: [11, 3]
names:
  0: spacecraft
```

Want to check without uploading? `kubecore-dataset validate <dir>`.

### Updating a dataset later

Just re-run the same command. The sync is **incremental**: add images and only
those upload; delete images locally and they're **removed from lakeFS** too;
unchanged files are skipped. One clean commit per update.

---

## 2. How a dataset is discovered

- A **CronWorkflow `<app>-dataset-catalog-probe`** runs every 30 minutes. It asks
  lakeFS for every branch/tag, keeps the ones whose top folder has a `data.yaml`,
  and writes them into a ConfigMap (`kubecore-ml-dataset-catalog`).
- The run form reads that catalog and turns it into the **`dataset-ref` dropdown`.

So: **upload a dataset → within ~30 min it's in the dropdown.** Nothing else to do.

Want it *now* instead of waiting?

```bash
kubectl -n ml-<project> create job \
    --from=cronjob/<app>-dataset-catalog-probe  probe-now
```

(There's also a manual escape hatch — `dataset-path-override` on the run form
takes a full `s3://…` path — but the dropdown is the easy path.)

---

## 3. Run a pipeline

1. Open **Argo Workflows** → **Submit new workflow** → template
   `<app>-training-pipeline`.
2. Fill the form. The dropdowns that matter:
   - **`dataset-ref`** — pick your dataset (e.g. `speedplus-v2`).
   - **`<step>-class`** — the machine for each step. For training pick a GPU class
     (e.g. `model-training-class` → `gpu-t4`). The GPU pool scales up from zero
     for your run and back down after.
   - The rest (epochs, batch size, augmentation, quantization…) have sensible
     defaults — just hit **Submit** to run on defaults.
3. Watch it run: config-validation → dataset-loading → training → registration.
   The trained model lands in **MLflow**.

### The absolute-minimum "for kids" version

> Open Argo → Submit → pick a **dataset** from the dropdown → pick **`gpu-t4`** for
> training → **Submit**. Wait. Your model shows up in MLflow.

---

## FAQ / gotchas

- **"Dataset path not found or empty"** — your dataset is missing `data.yaml`,
  `images/`, or `labels/` at the branch root. Run `kubecore-dataset validate`
  locally; it catches this before you upload.
- **My dataset isn't in the dropdown** — the probe runs every 30 min; force it
  with the `create job --from=cronjob/...` command above.
- **A branch = a pinned dataset** — a run records the exact commit, so re-running
  the same `dataset-ref` later trains on the same data even if the branch moved on.
- **Login expired** — re-run `kubecore-dataset login`. Sessions last as long as
  your SSO session does.
