# Uploading a Local Directory to LakeFS

A first-time-user guide to the LakeFS upload script — including the actual
Python code for each step, so you can read and understand the full flow inline
without opening a separate file.

The script walks a local directory tree and uploads every file in it to a
branch of a LakeFS repository, then creates a single commit containing all of
the uploaded files. It talks to LakeFS through the SSO-protected ingress using
only your browser session cookie — no LakeFS access keys are needed on your
machine, because an `nginx` sidecar behind `oauth2-proxy` injects Basic auth to
LakeFS server-side on every request.

---

## 1. Prerequisites

- **Python 3.9+** (the script uses modern type-hint syntax).
- **`requests`** library: `pip install requests`.
- **An active SSO login** to the LakeFS web UI in your browser, so you can
  copy the session cookie.
- **A LakeFS repository and branch** that already exist. The script does
  *not* create them; it will exit if the target branch isn't reachable.

The script depends on these standard library and third-party imports:

```python
from __future__ import annotations

import concurrent.futures as cf
import os
import pathlib
import sys
import threading
import time
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
```

---

## 2. Required environment variables

The script reads its entire configuration from environment variables. The
recommended workflow is to keep them in a `.env` file next to the script and
source that file into your shell before running.

| Variable        | Required | Default     | What it is                                                                                  |
|-----------------|----------|-------------|---------------------------------------------------------------------------------------------|
| `LAKEFS_URL`    | yes      | —           | Base URL of the LakeFS ingress, e.g. `https://lakefs-sso-test.ml-training.kaos.io.kubecore.eu`. Trailing slash is OK; the script strips it. |
| `LAKEFS_COOKIE` | yes      | —           | The raw value of the `_lakefs_oauth2` cookie from your browser. This is your SSO session — it expires when the SSO session does. |
| `LAKEFS_REPO`   | yes      | —           | The LakeFS repository name (e.g. `data-sampling-test`).                                     |
| `LAKEFS_BRANCH` | yes      | —           | Target branch within the repo. Must already exist.                                          |
| `LOCAL_DIR`     | yes      | —           | Absolute path to the directory you want to upload. Must be a directory, not a file.         |
| `UPLOAD_PREFIX` | no       | `dataset`   | Prefix prepended to every uploaded path inside the repo. Set to an empty string for none.   |
| `CONCURRENCY`   | no       | `16`        | Number of parallel HTTP uploads.                                                            |

### How to get `LAKEFS_COOKIE`

1. Log in to the LakeFS UI (`LAKEFS_URL`) in your browser.
2. Open DevTools → **Application** (Chrome) or **Storage** (Firefox) → **Cookies**.
3. Find the cookie named `_lakefs_oauth2`.
4. Copy its **Value** (a long base64-ish string ending in `=|<digits>|<base64>=`).
5. Paste it into the `.env` file. Wrap it in single quotes — the value contains
   `=` and `|` which can confuse some shells:
   ```
   LAKEFS_COOKIE='pzzUFbI7XB-...|1776931837|...='
   ```

The cookie expires when your SSO session does. If the script's auth probe fails
with a non-200 status, refresh the cookie.

### How `UPLOAD_PREFIX` works

A file at `LOCAL_DIR/foo/bar.jpg` becomes `UPLOAD_PREFIX/foo/bar.jpg` in the
repo. With the default prefix `dataset` it lands at
`lakefs://<repo>/<branch>/dataset/foo/bar.jpg`.

---

## 3. Setting up `.env`

Create a `.env` file alongside the script:

```bash
LAKEFS_URL=https://lakefs-sso-test.ml-training.kaos.io.kubecore.eu
LAKEFS_COOKIE='<paste cookie value here>'
LAKEFS_REPO=data-sampling-test
LAKEFS_BRANCH=upload-sample-1000
LOCAL_DIR=/absolute/path/to/your/data
UPLOAD_PREFIX=dataset
CONCURRENCY=16
```

The cookie value **must be on a single line**. If your editor soft-wraps it
visually that's fine; what matters is that there is no real newline inside the
value.

---

## 4. Running the script

```bash
set -a; source .env; set +a    # export every variable defined in .env
python3 upload_to_lakefs_api.py
```

The `set -a; source .env; set +a` pattern auto-exports every variable from the
file. The script reads from `os.environ` directly (no `python-dotenv`), so you
must export the variables into the shell before invoking Python.

---

## 5. What the script does, step by step

### Step 1 — Read configuration

A small helper reads each variable from `os.environ`, exits with a clear error
if a required one is missing, and falls back to a default otherwise:

```python
def env(name: str, default: str | None = None, required: bool = True) -> str:
    v = os.environ.get(name, default)
    if required and not v:
        sys.exit(f"ERROR: env var {name} is required")
    return v

LAKEFS_URL    = env("LAKEFS_URL").rstrip("/")
LAKEFS_COOKIE = env("LAKEFS_COOKIE")
LAKEFS_REPO   = env("LAKEFS_REPO")
LAKEFS_BRANCH = env("LAKEFS_BRANCH")
LOCAL_DIR     = pathlib.Path(env("LOCAL_DIR"))
UPLOAD_PREFIX = env("UPLOAD_PREFIX", "dataset", required=False).strip("/")
CONCURRENCY   = int(env("CONCURRENCY", "16", required=False))

if not LOCAL_DIR.is_dir():
    sys.exit(f"ERROR: LOCAL_DIR {LOCAL_DIR} is not a directory")
```

The trailing `is_dir()` check guarantees a single file or non-existent path is
caught up-front rather than failing mid-upload.

### Step 2 — Build the HTTP session

A single `requests.Session` is created and reused for every request, with
automatic retries on transient gateway errors, a connection pool sized for the
configured concurrency, and the SSO cookie attached:

```python
def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=CONCURRENCY * 2,
        pool_maxsize=CONCURRENCY * 2,
    )
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "lakefs-api-uploader/1.0"})
    s.cookies.set("_lakefs_oauth2", LAKEFS_COOKIE)
    return s
```

Only 502/503/504 are retried — 4xx errors (e.g. 401 from an expired cookie) are
returned to the caller immediately so the script can fail fast rather than
hammering the ingress.

### Step 3 — Auth probe

Before doing anything else, the script verifies the cookie works by listing
repositories:

```python
def check_auth(s: requests.Session) -> None:
    r = s.get(
        f"{LAKEFS_URL}/api/v1/repositories",
        timeout=30,
        allow_redirects=False,
        params={"amount": 1},
    )
    if r.status_code != 200:
        sys.exit(
            f"ERROR: auth probe returned HTTP {r.status_code}. "
            f"Cookie likely expired or invalid. Body: {r.text[:200]}"
        )
    print("Auth probe OK.")
```

`allow_redirects=False` is important here: if `oauth2-proxy` decides the cookie
is invalid, it issues a 302 to the SSO login page. Without this flag, the
client would follow the redirect and the script wouldn't notice the auth
failure.

### Step 4 — Branch check

Next it confirms the target branch actually exists on the repo:

```python
def check_branch(s: requests.Session) -> None:
    r = s.get(
        f"{LAKEFS_URL}/api/v1/repositories/{LAKEFS_REPO}/branches/{LAKEFS_BRANCH}",
        timeout=30,
        allow_redirects=False,
    )
    if r.status_code != 200:
        sys.exit(
            f"ERROR: branch {LAKEFS_BRANCH} of {LAKEFS_REPO} not reachable: "
            f"HTTP {r.status_code} {r.text[:200]}"
        )
    print(f"Branch {LAKEFS_BRANCH} ready.")
```

The branch must be created beforehand via the LakeFS UI or API; this script
never creates branches.

### Step 5 — Walk the local directory

The script recursively walks `LOCAL_DIR`, collecting only regular files, and
computes each file's destination path inside the repo:

```python
files: list[tuple[pathlib.Path, str]] = []
for p in LOCAL_DIR.rglob("*"):
    if p.is_file():
        rel = p.relative_to(LOCAL_DIR).as_posix()
        repo_path = f"{UPLOAD_PREFIX}/{rel}" if UPLOAD_PREFIX else rel
        files.append((p, repo_path))

total = len(files)
print(f"Found {total} files to upload to "
      f"lakefs://{LAKEFS_REPO}/{LAKEFS_BRANCH}/{UPLOAD_PREFIX}/")
if total == 0:
    sys.exit("Nothing to upload.")
```

Empty subdirectories are skipped — LakeFS, like S3, has no concept of empty
directories, so they would have nothing to upload anyway.

### Step 6 — Upload one file

Each file is sent as a multipart/form-data POST to the LakeFS objects endpoint.
The `repo_path` is URL-encoded so paths containing spaces or special characters
work correctly:

```python
def upload_one(s: requests.Session, local: pathlib.Path, repo_path: str) -> tuple[bool, str]:
    url = (
        f"{LAKEFS_URL}/api/v1/repositories/{LAKEFS_REPO}/branches/{LAKEFS_BRANCH}"
        f"/objects?path={quote(repo_path, safe='')}"
    )
    try:
        with local.open("rb") as fh:
            r = s.post(
                url,
                files={"content": (local.name, fh, "application/octet-stream")},
                timeout=300,
                allow_redirects=False,
            )
        if r.status_code in (200, 201):
            return True, repo_path
        return False, f"{repo_path}: HTTP {r.status_code} {r.text[:200]}"
    except Exception as exc:
        return False, f"{repo_path}: {exc!r}"
```

Returning a `(success, info)` tuple instead of raising means the worker thread
records the failure and moves on instead of taking down the whole pool.

### Step 7 — Run uploads in parallel with progress

A `ThreadPoolExecutor` runs `CONCURRENCY` workers in parallel. Each worker
calls `upload_one` and updates a shared counter under a lock; every 500 files
(and at the end), the worker prints a progress line with rate and ETA:

```python
lock = threading.Lock()
done = 0
failures: list[str] = []
start = time.time()

def worker(args: tuple[pathlib.Path, str]) -> None:
    nonlocal done
    local, repo_path = args
    ok, info = upload_one(session, local, repo_path)
    with lock:
        done += 1
        if not ok:
            failures.append(info)
        if done % 500 == 0 or done == total:
            elapsed = time.time() - start
            rate = done / elapsed if elapsed else 0
            eta = (total - done) / rate if rate else 0
            print(
                f"  {done}/{total} uploaded "
                f"({rate:.1f}/s, ETA {eta/60:.1f}m, failures {len(failures)})"
            )

with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
    list(ex.map(worker, files))
```

The `lock` protects both `done` and `failures` from concurrent updates, since
multiple worker threads may finish at the same instant. A `ThreadPoolExecutor`
is the right primitive here (rather than `ProcessPoolExecutor`) because
uploads are I/O-bound and the GIL is released during socket reads/writes.

### Step 8 — Decide whether to commit

After all uploads finish, the script summarizes the run. **If any upload
failed, it prints the first 20 failures and exits without committing** —
successfully uploaded objects remain on the branch as uncommitted changes:

```python
elapsed = time.time() - start
print(f"Upload pass complete in {elapsed/60:.1f}m. "
      f"Successes: {total - len(failures)}. Failures: {len(failures)}.")
if failures:
    print("First 20 failures:")
    for f in failures[:20]:
        print(f"  {f}")
    sys.exit(f"ERROR: {len(failures)} upload(s) failed; not committing.")
```

You can re-run the script to retry — LakeFS overwrites objects at the same
path, so this is safe.

### Step 9 — Commit

If every upload succeeded, the script POSTs a commit and prints the resulting
commit id. The commit message is **hardcoded** — edit the script if you need a
different one:

```python
print("Committing...")
r = session.post(
    f"{LAKEFS_URL}/api/v1/repositories/{LAKEFS_REPO}/branches/{LAKEFS_BRANCH}/commits",
    json={"message": f"Add sample_pose_yolo dataset ({total} files)"},
    timeout=600,
    allow_redirects=False,
)
if r.status_code not in (200, 201):
    sys.exit(f"ERROR: commit failed: HTTP {r.status_code} {r.text[:400]}")
print(f"Committed. id={r.json().get('id')}")
```

A 600-second timeout is used here because committing thousands of objects can
take noticeably longer server-side than uploading any individual one.

---

## 6. Expected output

A successful end-to-end run looks like:

```
Auth probe OK.
Branch upload-sample-1000 ready.
Found 4 files to upload to lakefs://data-sampling-test/upload-sample-1000/dataset/
  4/4 uploaded (7.7/s, ETA 0.0m, failures 0)
Upload pass complete in 0.0m. Successes: 4. Failures: 0.
Committing...
Committed. id=e9f56e7d6bcac9cbdcded22a503eeafcb5fe10ccfacefee76d9bd5500585b0d0
```

---

## 7. Troubleshooting

| Symptom                                              | Likely cause / fix                                                                                  |
|------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `ERROR: env var X is required`                        | You didn't `source .env` (or the variable line is malformed). Re-run with `set -a; source .env; set +a`. |
| `ERROR: LOCAL_DIR ... is not a directory`             | You pointed at a single file. Stage files in a folder and point `LOCAL_DIR` at the folder.          |
| `ERROR: auth probe returned HTTP 302/401/403`         | Cookie expired or wrong. Re-copy `_lakefs_oauth2` from the browser.                                 |
| `ERROR: branch X of Y not reachable: HTTP 404`        | The branch doesn't exist on that repo. Create it via the LakeFS UI or API first.                    |
| `Nothing to upload.`                                  | `LOCAL_DIR` exists but contains no files (only empty subdirs). The script doesn't create empty dirs in LakeFS. |
| Many failures with HTTP 5xx                           | Transient ingress / LakeFS issues. The session retries 502/503/504 up to 5 times automatically; if they still fail, lower `CONCURRENCY` and re-run. |

---

## 8. Notes & limitations

- **No resume.** A re-run uploads every file again (LakeFS overwrites at the
  same path, so this is safe but wasteful for large datasets).
- **All-or-nothing commit.** Any single upload failure aborts the commit step;
  partially-uploaded objects remain on the branch as uncommitted changes.
- **Hardcoded commit message.** Currently `"Add sample_pose_yolo dataset (N files)"`.
  Edit the script source if you need a different message.
- **Cookie-based auth.** Convenient for one-off interactive runs; not suitable
  for unattended automation. For scheduled jobs, use LakeFS access keys against
  an internal endpoint that bypasses `oauth2-proxy`.
