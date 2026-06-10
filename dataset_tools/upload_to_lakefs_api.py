#!/usr/bin/env python3
"""
Upload a local directory tree to a LakeFS repository's branch using only the
LakeFS REST API through the SSO-protected ingress.

Client auth: only a Zitadel session cookie (`_lakefs_oauth2`) passed via env.
LakeFS admin credentials are NOT required on the client side — the nginx
sidecar behind oauth2-proxy injects Basic auth to LakeFS on every request.

Env vars:
    LAKEFS_URL        e.g. https://lakefs-sso-test.ml-training.kaos.io.kubecore.eu
    LAKEFS_COOKIE     the raw value of the _lakefs_oauth2 cookie from the browser
    LAKEFS_REPO       e.g. kaos-yolo
    LAKEFS_BRANCH     e.g. main
    LOCAL_DIR         absolute path to the local tree to upload
    UPLOAD_PREFIX     optional prefix in the repo (default: dataset)
    CONCURRENCY       parallel uploads (default: 16)
"""
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


def env(name: str, default: str | None = None, required: bool = True) -> str:
    v = os.environ.get(name, default)
    if required and not v:
        sys.exit(f"ERROR: env var {name} is required")
    return v  # type: ignore[return-value]


LAKEFS_URL = env("LAKEFS_URL").rstrip("/")
LAKEFS_COOKIE = env("LAKEFS_COOKIE")
LAKEFS_REPO = env("LAKEFS_REPO")
LAKEFS_BRANCH = env("LAKEFS_BRANCH")
LOCAL_DIR = pathlib.Path(env("LOCAL_DIR"))
UPLOAD_PREFIX = env("UPLOAD_PREFIX", "dataset", required=False).strip("/")
CONCURRENCY = int(env("CONCURRENCY", "16", required=False))

if not LOCAL_DIR.is_dir():
    sys.exit(f"ERROR: LOCAL_DIR {LOCAL_DIR} is not a directory")


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=CONCURRENCY * 2,
                          pool_maxsize=CONCURRENCY * 2)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "lakefs-api-uploader/1.0"})
    s.cookies.set("_lakefs_oauth2", LAKEFS_COOKIE)
    return s


def check_auth(s: requests.Session) -> None:
    r = s.get(f"{LAKEFS_URL}/api/v1/repositories", timeout=30, allow_redirects=False,
              params={"amount": 1})
    if r.status_code != 200:
        sys.exit(
            f"ERROR: auth probe returned HTTP {r.status_code}. "
            f"Cookie likely expired or invalid. Body: {r.text[:200]}"
        )
    print("Auth probe OK.")


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
    except Exception as exc:  # noqa: BLE001
        return False, f"{repo_path}: {exc!r}"


def main() -> None:
    session = make_session()
    check_auth(session)
    check_branch(session)

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

    elapsed = time.time() - start
    print(f"Upload pass complete in {elapsed/60:.1f}m. "
          f"Successes: {total - len(failures)}. Failures: {len(failures)}.")
    if failures:
        print("First 20 failures:")
        for f in failures[:20]:
            print(f"  {f}")
        sys.exit(f"ERROR: {len(failures)} upload(s) failed; not committing.")

    print("Committing...")
    r = session.post(
        f"{LAKEFS_URL}/api/v1/repositories/{LAKEFS_REPO}/branches/{LAKEFS_BRANCH}/commits",
        json={"message": f"Add speedplus_yolo dataset ({total} files)"},
        timeout=600,
        allow_redirects=False,
    )
    if r.status_code not in (200, 201):
        sys.exit(f"ERROR: commit failed: HTTP {r.status_code} {r.text[:400]}")
    print(f"Committed. id={r.json().get('id')}")


if __name__ == "__main__":
    main()
