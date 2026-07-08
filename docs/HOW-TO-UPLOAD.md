# Upload a dataset & run a pipeline

Everything you need, using only your **browser** and this **app repo**. No cluster
access, no `kubectl`, no keys.

You have three things:
- **this app repo** — it ships the `kubecore-dataset` tool and already knows your
  lakeFS (from `.kubecore/dataset-config.yaml`).
- your **browser** — to log in (single sign-on).
- the **Argo Workflows UI** — to run training.

---

## 1. Get set up (once)

```bash
git clone <this-app-repo-url>
cd <this-app-repo>
pip install ./dataset_tools          # installs the `kubecore-dataset` command
```

Requires **Python 3.9+**. Your lakeFS URL and repo are already filled in for this
app, so you never type a URL.

---

## 2. Upload your dataset

Put your dataset in the repo (a `datasets/` folder is ignored by git, so raw data
is never committed), then run one command:

```bash
./scripts/upload-dataset.py  datasets/my_dataset  my-first-dataset
```

- `datasets/my_dataset` — your dataset folder (layout below).
- `my-first-dataset` — the name you'll pick in the dropdown later.

**What happens:**
1. **Your browser opens** to log in. Sign in with your normal account.
2. The tool **checks your dataset** is valid — if not, it tells you exactly what to fix.
3. It **uploads** what changed (and removes what you deleted) and saves a version.

Your dataset is now stored and versioned.

### What a valid dataset looks like

```
my_dataset/
├── data.yaml
├── images/
│   ├── train/   img_0001.jpg  …
│   └── val/     img_5001.jpg  …
└── labels/
    ├── train/   img_0001.txt  …   (one .txt per image, same name)
    └── val/     img_5001.txt  …
```

`data.yaml`:

```yaml
path: .
train: images/train
val: images/val
kpt_shape: [11, 3]      # 11 keypoints, (x, y, visibility)
names:
  0: spacecraft
```

Want to check it first, without uploading?

```bash
kubecore-dataset validate datasets/my_dataset
```

---

## 3. Run a training pipeline

1. Open the **Argo Workflows UI** in your browser (the link is on your platform
   dashboard) and click **Submit new workflow**.
2. Choose your pipeline template (e.g. `*-training-pipeline`).
3. On the form:
   - **`dataset-ref`** — pick the dataset you just uploaded. *(It appears within
     ~5 minutes of uploading. If it's not there yet, wait a few minutes and
     reopen the form — the list refreshes automatically.)*
   - **`model-training-class`** — pick a GPU option (e.g. `gpu-t4`).
   - Everything else has good defaults.
4. Click **Submit**. Watch it run: validate → load data → train → register. The
   trained model appears in **MLflow**.

That's the whole loop: **upload in the terminal, run in the browser.**

---

## Updating a dataset later

Re-run the same upload command. Only what changed is sent, deletions are mirrored,
and a new version is saved — so re-running is always safe.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `kubecore-dataset: command not found` / `not recognized` | pip installed the command somewhere not on your `PATH`. Run it as a module instead: `python -m dataset_cli login` (same for `validate` / `sync`). Or add pip's scripts dir (it printed the path in a warning, e.g. `~/.local/bin`) to your `PATH`. |
| Browser didn't open | Click the link the tool printed on screen. |
| "Log in again" | Re-run `kubecore-dataset login`. |
| "Dataset path not found or empty" while running | Your `data.yaml` / `images/` / `labels/` aren't at the top of your dataset folder. Run `kubecore-dataset validate` to see what's missing. |
| My dataset isn't in the `dataset-ref` list yet | The list refreshes every ~5 min after an upload — reopen the submit form shortly. |
| Loopback login didn't complete | Add `--paste` to `login` and follow the one-time prompt. |

Full command reference: `kubecore-dataset --help`.
