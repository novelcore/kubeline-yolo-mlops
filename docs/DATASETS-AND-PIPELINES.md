# Datasets & Pipelines — how it works

The 30-second mental model behind [HOW-TO-UPLOAD.md](./HOW-TO-UPLOAD.md). You only
ever use your **browser** and this **app repo** — no cluster access, no `kubectl`.

---

## The idea

- **A dataset is a named version of your data.** You upload a folder of images +
  labels and give it a name; that name is what you pick when you run a pipeline.
  Uploading again under a new name keeps the old one — every run is reproducible.
- **You never type a URL or a path.** This app repo already knows where its data
  and pipelines live. The tools read that and do the right thing.
- **Two surfaces, two jobs:** the **terminal** (the `kubecore-dataset` tool in this
  repo) is how you *upload* data; the **Argo Workflows UI** is how you *run* training.

---

## The flow

```
   your laptop                         the platform
   ───────────                         ────────────
   dataset folder ──upload (terminal)──►  stored & versioned
                                              │
                                              ▼
   Argo UI  ──pick dataset, Submit──►  validate → load → train → register → MLflow
```

1. **Upload** — `./scripts/upload-dataset.py datasets/my_data my-name`. It logs you
   in through the browser, checks the dataset, and stores it. See
   [HOW-TO-UPLOAD.md](./HOW-TO-UPLOAD.md).
2. **Discover** — within ~30 minutes your dataset name shows up in the `dataset-ref`
   dropdown on the run form. Nothing for you to do; the list refreshes on its own.
3. **Run** — open the Argo Workflows UI, Submit the training pipeline, pick your
   `dataset-ref` and a GPU class, and go. The model lands in MLflow.

---

## What "a dataset" must contain

At the top level of your dataset folder:

```
my_dataset/
├── data.yaml                 # path, train, val, kpt_shape, names
├── images/train/  *.jpg      # + images/val/
└── labels/train/  *.txt      # one .txt per image, same filename; + labels/val/
```

If anything is missing or mismatched, the upload tool tells you before it uploads —
so a bad dataset never reaches a pipeline run. Check anytime with
`kubecore-dataset validate <folder>`.

---

## FAQ

- **"Dataset path not found or empty" during a run** — your `data.yaml` / `images/`
  / `labels/` aren't at the top of the folder you uploaded. Re-check with
  `kubecore-dataset validate`.
- **My dataset isn't in the dropdown** — the list refreshes ~every 30 min after an
  upload; reopen the submit form shortly.
- **A name = a frozen dataset** — a run records the exact version, so re-running the
  same `dataset-ref` later trains on the same data.
- **I need to change my data** — upload again with the same command; only the diff
  is sent and a new version is saved.

Full commands: [HOW-TO-UPLOAD.md](./HOW-TO-UPLOAD.md) · `kubecore-dataset --help`.
