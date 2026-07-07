# How to upload a dataset — step by step

Upload a YOLO-pose dataset to **your project's** lakeFS so it shows up in the
pipeline's `dataset-ref` dropdown. Three commands. You log in once through your
browser.

> **Nothing here is hard-coded to a specific instance.** The tool reads your
> project's lakeFS URL and repo from `.kubecore/dataset-config.yaml` in this
> repo (the platform renders it per app). So the same commands work for every
> app and every lakeFS instance — you don't pass a URL.

---

## 0. One-time setup

```bash
# from the root of THIS app repo (the one you cloned)
pip install ./dataset_tools          # installs the `kubecore-dataset` command
```

You need **Python 3.9+**. No lakeFS keys, no `kubectl`, no port-forward.

Put your dataset anywhere — a `datasets/` folder inside this repo is already
**gitignored**, so you can drop images there and they won't be committed.

---

## 1. Upload — the one-liner

Run it **from inside your app clone** so it auto-discovers your lakeFS URL + repo:

```bash
./scripts/upload-dataset.py  <your-dataset-dir>  <branch-name>
```

Example:

```bash
./scripts/upload-dataset.py  datasets/speedplus_yolo  speedplus-v2
```

**What happens, in order:**

1. **A browser tab opens** to *your project's* lakeFS login. Sign in with your
   normal SSO. The tool captures your session automatically — nothing to copy.
2. **It validates** your folder is a proper dataset (layout below) — fails fast
   with a clear message if not.
3. **It syncs** — uploads new/changed files, deletes files you removed locally,
   one commit on the branch. All through the ingress.

`<branch-name>` is what you'll pick in the dropdown. Make it human-friendly.

> Not inside the app clone? Point the tool at your lakeFS explicitly:
> `--url https://lakefs-<project>.<baseDns> --repo <project>`, or set
> `LAKEFS_URL` / `LAKEFS_REPO`. Your values are in `.kubecore/dataset-config.yaml`.

---

## 2. What a valid dataset looks like

Your `<your-dataset-dir>` must have this at its **root**:

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

Check it without uploading:

```bash
kubecore-dataset validate datasets/my_dataset
```

---

## 3. After uploading — run a pipeline on it

1. The catalog probe runs every 30 min and adds your branch to the dropdown.
   To see it **now** (the probe name is in your `.kubecore/dataset-config.yaml`):
   ```bash
   PROBE=$(python3 -c "import yaml;print(yaml.safe_load(open('.kubecore/dataset-config.yaml'))['probeCron'])")
   NS=$(python3 -c "import yaml;print(yaml.safe_load(open('.kubecore/dataset-config.yaml'))['namespace'])")
   kubectl -n "$NS" create job --from=cronjob/"$PROBE" probe-now
   ```
2. Open **Argo Workflows** → **Submit new workflow** → your `*-training-pipeline`
   template.
3. Pick your branch in **`dataset-ref`**, pick a GPU class for
   `model-training-class`, hit **Submit**.
4. It trains and registers the model in **MLflow**.

---

## The full CLI (steps separately)

```bash
kubecore-dataset login                    # opens the browser (URL auto-discovered)
kubecore-dataset validate datasets/my_dataset
kubecore-dataset sync     datasets/my_dataset --branch speedplus-v2
kubecore-dataset --help
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "could not determine the lakeFS URL" | Run from inside your app clone (it has `.kubecore/dataset-config.yaml`), or pass `--url`/`--repo`. |
| Browser didn't open | Click the link the tool printed — it's on screen. |
| `login` says the session expired | Run `kubecore-dataset login` again. |
| "Dataset path not found or empty" in a run | Your `data.yaml`/`images/`/`labels/` aren't at the branch root. Run `kubecore-dataset validate` first. |
| Branch not in the dropdown | The probe runs every 30 min; force it with the `create job` command above. |
| Loopback can't complete | Add `--paste`; the tool walks you through pasting the `_lakefs_oauth2` cookie once. |
