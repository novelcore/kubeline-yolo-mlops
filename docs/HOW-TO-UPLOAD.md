# How to upload a dataset — step by step

Upload a YOLO-pose dataset to lakeFS so it shows up in the pipeline's
`dataset-ref` dropdown. Three commands. You log in once through your browser.

---

## 0. One-time setup

```bash
git clone https://github.com/novelcore/kubeline-yolo-mlops
cd kubeline-yolo-mlops
pip install ./dataset_tools          # installs the `kubecore-dataset` command
```

You need **Python 3.9+**. That's it — no lakeFS keys, no `kubectl`, no port-forward.

---

## 1. Upload — the one-liner

```bash
./scripts/upload-dataset.py  <your-dataset-dir>  <branch-name> \
    --url https://lakefs-yolo.gke-dev.europe-central2.testing.kubecore.eu \
    --repo yolo
```

Real example:

```bash
./scripts/upload-dataset.py  ~/Downloads/speedplus_yolo  speedplus-v2 \
    --url https://lakefs-yolo.gke-dev.europe-central2.testing.kubecore.eu \
    --repo yolo
```

**What happens, in order:**

1. **A browser tab opens** to the lakeFS login:
   ```
   🔑  Opening your browser to log in…
       https://lakefs-yolo.gke-dev.europe-central2.testing.kubecore.eu/oauth2/start?rd=http://localhost:8765/callback
   ```
   Sign in with your normal SSO (the Zitadel login). The tool captures your
   session automatically — nothing to copy.

2. **It validates** your folder is a proper dataset (see the layout below). If
   something's wrong it tells you exactly what and stops before uploading.

3. **It syncs** — uploads new/changed files, deletes files you removed locally,
   and makes one commit on the branch. All through the ingress.

`<branch-name>` is what you'll pick in the dropdown later. Make it human-friendly.

---

## 2. What a valid dataset looks like

Your `<your-dataset-dir>` must have this at its **root**:

```
speedplus_yolo/
├── data.yaml
├── images/
│   ├── train/   img_0001.jpg  img_0002.jpg  …
│   └── val/     img_5001.jpg  …
└── labels/
    ├── train/   img_0001.txt  img_0002.txt  …   (one .txt per image, same name)
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
kubecore-dataset validate ~/Downloads/speedplus_yolo
```

---

## 3. After uploading — run a pipeline on it

1. The catalog probe runs every 30 min and adds your branch to the dropdown.
   To see it **now**:
   ```bash
   kubectl -n ml-yolo create job --from=cronjob/yolo-training-dataset-catalog-probe probe-now
   ```
2. Open **Argo Workflows** → **Submit new workflow** → template
   `yolo-training-pipeline`.
3. Pick your branch in **`dataset-ref`**, pick **`gpu-t4`** for
   `model-training-class`, hit **Submit**.
4. It trains and registers the model in **MLflow**.

---

## The full CLI (if you want the steps separately)

```bash
# log in (opens the browser)
kubecore-dataset login    --url https://lakefs-yolo.gke-dev.europe-central2.testing.kubecore.eu

# validate locally
kubecore-dataset validate ~/Downloads/speedplus_yolo

# upload (incremental: adds + deletes, one commit)
kubecore-dataset sync     ~/Downloads/speedplus_yolo \
    --url https://lakefs-yolo.gke-dev.europe-central2.testing.kubecore.eu \
    --repo yolo --branch speedplus-v2

kubecore-dataset --help   # everything else
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Browser didn't open | Click the link the tool printed — it's on screen. |
| `login` says the session expired | Run `kubecore-dataset login` again. |
| "Dataset path not found or empty" in a run | Your `data.yaml` / `images/` / `labels/` aren't at the branch root. Run `kubecore-dataset validate` locally first. |
| My branch isn't in the dropdown | The probe runs every 30 min; force it with the `create job` command above. |
| `--paste` mode | If the browser loopback can't complete, add `--paste` and the tool walks you through pasting the `_lakefs_oauth2` cookie once. |
