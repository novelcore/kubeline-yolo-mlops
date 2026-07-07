"""kubecore-dataset CLI entrypoint.

Subcommands:
  login     Log in to lakeFS via the browser (loopback OIDC, paste fallback).
  validate  Check a local dataset against the Ultralytics-pose contract.
  sync      Validate, then incrementally sync (adds+deletes) to a lakeFS branch.
  upload    Alias for `sync` with login + validate wired in (the old one-liner).

Config precedence for every value: CLI flag > environment variable > default.
  LAKEFS_URL     external ingress, e.g. https://lakefs-yolo.<baseDns>
  LAKEFS_REPO    lakeFS repository (default: derived / required)
  LAKEFS_BRANCH  target branch (default: main)
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

from . import __version__
from .lakefs_client import LakeFSClient
from .login import login as do_login
from .sync import sync as do_sync
from .validate import validate_dataset


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def _resolve_url(args) -> str:
    url = args.url or _env("LAKEFS_URL")
    if not url:
        sys.exit(
            "ERROR: lakeFS URL is required. Pass --url or set LAKEFS_URL "
            "(e.g. https://lakefs-<project>.<baseDns>)."
        )
    return url.rstrip("/")


def _resolve_repo(args) -> str:
    repo = args.repo or _env("LAKEFS_REPO")
    if not repo:
        sys.exit("ERROR: lakeFS repo is required. Pass --repo or set LAKEFS_REPO.")
    return repo


def cmd_login(args) -> int:
    url = _resolve_url(args)
    do_login(url, force=args.force, prefer_paste=args.paste)
    return 0


def cmd_validate(args) -> int:
    result = validate_dataset(args.dataset_dir)
    print(result.report())
    return 0 if result.ok else 2


def cmd_sync(args, *, do_auth: bool = True) -> int:
    url = _resolve_url(args)
    repo = _resolve_repo(args)
    branch = args.branch or _env("LAKEFS_BRANCH", "main")
    root = pathlib.Path(args.dataset_dir)

    # ref-native guard
    prefix = (args.prefix or "").strip("/")
    if prefix and not args.allow_prefix:
        sys.exit(
            f"ERROR: refusing to upload under nested prefix '{prefix}'. The "
            "pipeline expects the dataset at the branch ROOT (s3://repo/branch/). "
            "Pass --allow-prefix only if you really mean it."
        )

    # 1) validate (unless skipped)
    if not args.skip_validation:
        result = validate_dataset(root)
        print(result.report())
        if not result.ok:
            sys.exit("Aborting: dataset failed validation. Fix the errors above, "
                     "or re-run with --skip-validation to force.")

    # 2) auth
    cookie = do_login(url, prefer_paste=args.paste) if do_auth else do_login(url)
    client = LakeFSClient(url, cookie, concurrency=args.concurrency)

    # 3) ensure branch exists (create from default if new)
    if not client.branch_exists(repo, branch):
        source = client.default_branch(repo)
        print(f"Branch '{branch}' does not exist — creating from '{source}'.")
        client.ensure_branch(repo, branch, source)

    # 4) incremental sync + commit
    commit_id = do_sync(
        root, client, repo, branch,
        prefix=prefix, concurrency=args.concurrency, dry_run=args.dry_run,
        extra_metadata={"branch": branch},
    )
    if commit_id and not args.dry_run:
        print(
            f"\n✓ Done. Branch '{branch}' now matches your local dataset "
            f"(commit {commit_id[:12]}).\n"
            f"  It appears as a `dataset-ref` option after the catalog probe runs "
            f"(≤30 min).\n"
            f"  Force it now:  kubectl -n ml-<project> create job "
            f"--from=cronjob/<app>-dataset-catalog-probe probe-now"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kubecore-dataset",
        description="Log in, validate, and incrementally sync YOLO-pose datasets "
                    "to lakeFS for the ML pipeline dropdown.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--url", help="lakeFS ingress URL (or env LAKEFS_URL)")
        sp.add_argument("--repo", help="lakeFS repository (or env LAKEFS_REPO)")
        sp.add_argument("--branch", help="target branch (or env LAKEFS_BRANCH, default main)")
        sp.add_argument("--concurrency", type=int, default=16)
        sp.add_argument("--paste", action="store_true",
                        help="skip loopback; use guided cookie paste")
        sp.add_argument("--prefix", default="",
                        help="(discouraged) upload under a nested prefix")
        sp.add_argument("--allow-prefix", action="store_true",
                        help="permit --prefix (bypasses the ref-native guard)")
        sp.add_argument("--skip-validation", action="store_true")
        sp.add_argument("--dry-run", action="store_true")

    lp = sub.add_parser("login", help="log in to lakeFS via the browser")
    lp.add_argument("--url", help="lakeFS ingress URL (or env LAKEFS_URL)")
    lp.add_argument("--force", action="store_true", help="ignore any cached session")
    lp.add_argument("--paste", action="store_true", help="use guided cookie paste")
    lp.set_defaults(func=cmd_login)

    vp = sub.add_parser("validate", help="validate a local dataset directory")
    vp.add_argument("dataset_dir", help="path to the local dataset directory")
    vp.set_defaults(func=cmd_validate)

    spc = sub.add_parser("sync", help="incrementally sync a dataset to lakeFS")
    spc.add_argument("dataset_dir", help="path to the local dataset directory")
    add_common(spc)
    spc.set_defaults(func=cmd_sync)

    up = sub.add_parser("upload", help="login + validate + sync (the one-liner)")
    up.add_argument("dataset_dir", help="path to the local dataset directory")
    add_common(up)
    up.set_defaults(func=cmd_sync)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
