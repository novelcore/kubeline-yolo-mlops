# `kubecore-dataset` — the dataset uploader

A small Python CLI to get a YOLO-pose dataset from your laptop into lakeFS so it
shows up in the ML pipeline's **`dataset-ref` dropdown**. It does three things,
in one command:

1. **Logs you in** through your browser (no cookie copy-paste).
2. **Validates** the dataset against the exact structure the pipeline requires —
   so a bad dataset fails *here in one second*, not 20 minutes into a training run.
3. **Incrementally syncs** it to a lakeFS branch — uploading new/changed files
   **and deleting files you removed locally** — then makes one commit.

> **Why not the old `upload-dataset.sh`?** It needed you to `kubectl port-forward`
> and dig lakeFS admin keys out of a Secret, did no validation, and never synced
> deletions (its `aws s3 sync` had no `--delete`). This replaces it. The old
> script now just forwards here.

---

## Install

```bash
pip install ./dataset_tools           # from the repo root
# or, without installing, run it in place:
python3 -m dataset_cli --help
```

Only needs `requests` and `pyyaml`. Python 3.9+.

---

## Quick start (the one-liner)

```bash
./scripts/upload-dataset.py  ~/Downloads/speedplus_yolo  main \
    --url https://lakefs-<project>.<baseDns> --repo <project>
```

That logs you in, validates `~/Downloads/speedplus_yolo`, and syncs it to the
`main` branch of your lakeFS repo. After the catalog probe runs (≤30 min) the
branch appears in the `dataset-ref` dropdown on the Argo submit form.

Prefer the full CLI for anything else:

```bash
kubecore-dataset --help
```

---

## The three subcommands

### `login` — browser sign-in

```bash
kubecore-dataset login --url https://lakefs-<project>.<baseDns>
```

lakeFS OSS has no per-user login; it sits behind an **oauth2-proxy(Zitadel)**
that injects the shared lakeFS admin credential server-side. So the client only
needs a valid **`_lakefs_oauth2` session cookie** — no admin keys ever touch
your machine.

The CLI gets that cookie for you:

```
🔑  Opening your browser to log in…
    https://lakefs-yolo.<baseDns>/oauth2/start?rd=http://localhost:8765/callback
    (click the link above if the browser didn't open)

✓ Logged in — session cached.
```

- **Loopback (default):** a tiny localhost server catches the cookie after you
  log in. Fully automatic. Requires the `http://localhost:8765/callback` redirect
  to be allowed on the lakeFS OIDC app (the platform registers it; for testing it
  can be added in the Zitadel console by hand).
- **Guided paste (`--paste`, or automatic fallback):** if the loopback redirect
  isn't allowed yet, the CLI opens the lakeFS UI and asks you to paste the
  `_lakefs_oauth2` cookie once (DevTools → Application/Storage → Cookies).

The session is cached at `~/.config/kubecore-ml/lakefs-session.json` (mode 0600)
and reused by `validate`/`sync` until it expires. Re-run `login` when it does.

### `validate` — check the dataset before uploading

```bash
kubecore-dataset validate ~/Downloads/speedplus_yolo
```

Runs offline. Mirrors the pipeline's own contract, so if it passes here it won't
fail on structure in the pipeline. It checks:

- **`data.yaml`** at the dataset root, with keys `path`, `train`, `val`,
  `kpt_shape`, `names`. `kpt_shape` is `[N, 2]` or `[N, 3]`; `names` is a
  `{int: str}` map; `nc` (if present) equals `len(names)`.
- **`images/train/`, `images/val/`, `labels/train/`, `labels/val/`** present and
  non-empty (`test` split optional).
- **Every image has a matching-stem label** in the same split
  (`images/train/foo.jpg` ↔ `labels/train/foo.txt`).
- **Label rows** parse as Ultralytics pose:
  `class cx cy w h  kp1x kp1y [kp1v] … kpNx kpNy [kpNv]` — token count matches
  `kpt_shape`, bbox in `[0,1]`, class in range.

A failure prints an itemized report and exits non-zero:

```
✗ Dataset is INVALID (1 error(s)):
    ✗ 3 image(s) in 'train' have no matching label: train/img_041, train/img_207 ...
```

### `sync` — incremental upload + delete + commit

```bash
kubecore-dataset sync ~/Downloads/speedplus_yolo \
    --url https://lakefs-<project>.<baseDns> --repo <project> --branch main
```

`sync` validates first (skip with `--skip-validation`), logs you in if needed,
creates the branch if it's new, then **diffs your local tree against the branch
by content checksum** and applies exactly what changed:

```
Diffing local tree against lakefs://yolo/main (root) …
  add/changed: 12
  delete:      3
  unchanged:   1180
```

- **add/changed** — files that are new locally or whose content differs → uploaded.
- **delete** — files on the branch that you removed locally → **deleted from lakeFS**.
- **unchanged** — same checksum → skipped (no needless re-upload).

Then it makes **one commit** on the branch with the change counts in the commit
metadata. `--dry-run` shows the plan without touching anything.

The dataset lands at the **branch root** (`s3://<repo>/<branch>/`) — that's what
the pipeline reads (the ref *is* the version). A nested `--prefix` is refused
unless you pass `--allow-prefix`.

---

## What a valid dataset looks like

```
speedplus_yolo/
├── data.yaml                 # path, train, val, kpt_shape, names
├── images/
│   ├── train/  img_0001.jpg …
│   └── val/    img_5001.jpg …
└── labels/
    ├── train/  img_0001.txt …   # one .txt per image, same stem
    └── val/    img_5001.txt …
```

`data.yaml`:

```yaml
path: .
train: images/train
val: images/val
kpt_shape: [11, 3]     # 11 keypoints, (x, y, visibility)
names:
  0: spacecraft
```

---

## Configuration

Flags override environment variables, which override defaults.

| Flag / env                     | What it is |
|--------------------------------|------------|
| `--url` / `LAKEFS_URL`         | lakeFS ingress, e.g. `https://lakefs-<project>.<baseDns>` |
| `--repo` / `LAKEFS_REPO`       | lakeFS repository name |
| `--branch` / `LAKEFS_BRANCH`   | target branch (default `main`) — this is the dropdown value |
| `--concurrency`                | parallel uploads (default 16) |
| `--paste`                      | skip loopback; paste the cookie |
| `--skip-validation`            | force a sync without validating |
| `--dry-run`                    | show the plan; change nothing |

Find your lakeFS URL if you don't know it:

```bash
kubectl -n <org-namespace> get kubepool <pool> -o jsonpath='{.status.baseDns}'
# lakeFS URL = https://lakefs-<project>.<that-baseDns>
```

---

## After uploading

The `dataset-ref` dropdown is built by a catalog probe CronWorkflow that runs
every 30 minutes. To see your branch immediately:

```bash
kubectl -n ml-<project> create job \
    --from=cronjob/<app>-dataset-catalog-probe  probe-now
```

Then open the Argo submit form, pick your branch from `dataset-ref`, and run.
See `../docs/DATASETS-AND-PIPELINES.md` for the full upload → discover → run guide.
