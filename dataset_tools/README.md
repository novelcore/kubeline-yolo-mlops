# `kubecore-dataset` — the dataset tool

The command-line tool that ships in this app repo. It gets a YOLO-pose dataset
from your laptop into your project's data store, so it shows up in the pipeline's
`dataset-ref` dropdown. No cluster access, no keys — just your browser.

> This app repo already knows its lakeFS URL and repo (`.kubecore/dataset-config.yaml`,
> filled in per app by the platform). You never type a URL. Run the tool from inside
> your app clone and it discovers everything.

For the full walkthrough see [`../docs/HOW-TO-UPLOAD.md`](../docs/HOW-TO-UPLOAD.md).

---

## Install

```bash
pip install ./dataset_tools           # from the app repo root
```

Python 3.9+. Only needs `requests` and `pyyaml`.

---

## The one command you'll use

```bash
./scripts/upload-dataset.py  datasets/my_dataset  my-first-dataset
```

Logs you in (browser), validates the dataset, uploads it, saves a version. Done.

---

## The three subcommands

### `login` — browser sign-in

```bash
kubecore-dataset login
```

Opens your browser to log in with single sign-on and remembers the session (cached
locally at `~/.config/kubecore-ml/`). No keys touch your machine. If the automatic
browser flow can't finish, add `--paste` for a one-time guided sign-in.

### `validate` — check before you upload

```bash
kubecore-dataset validate datasets/my_dataset
```

Runs locally. Checks the dataset is a valid Ultralytics-pose set — `data.yaml` with
`path/train/val/kpt_shape/names`, the `train` and `val` splits present, every image
paired with a same-named label, and label rows in the right pose format. If anything
is off it prints exactly what, and stops — so a bad dataset never reaches a run.

### `sync` — upload what changed

```bash
kubecore-dataset sync datasets/my_dataset --branch my-first-dataset
```

Compares your folder to the stored version, uploads new/changed files, removes files
you deleted, and saves one new version. Re-running is always safe.

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

Your dataset name appears in the `dataset-ref` dropdown on the Argo Workflows submit
form within ~30 minutes — the list refreshes on its own. Then run your pipeline from
the Argo UI. See [`../docs/HOW-TO-UPLOAD.md`](../docs/HOW-TO-UPLOAD.md) step 3.

---

## Options (rarely needed)

Everything is discovered from `.kubecore/dataset-config.yaml`. You only pass these if
you're running outside the app clone:

| Flag / env | What it is |
|---|---|
| `--url` / `LAKEFS_URL` | your lakeFS URL (auto-discovered otherwise) |
| `--repo` / `LAKEFS_REPO` | your lakeFS repo (auto-discovered otherwise) |
| `--branch` | the dataset name (the dropdown value) |
| `--paste` | guided sign-in instead of the automatic browser flow |
| `--dry-run` | show what would change without uploading |

`kubecore-dataset --help` for everything.
