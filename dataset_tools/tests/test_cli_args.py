"""Argument-parsing tests — the branch may be given positionally OR via --branch.

Regression: `kubecore-dataset upload <dir> <branch>` (branch positional, matching
the docs + the scripts/upload-dataset.py one-liner) used to error with
"unrecognized arguments: <branch>" because only --branch was accepted.
"""
from dataset_cli.__main__ import build_parser


def _resolved_branch(args):
    # mirrors cmd_sync's precedence: positional > --branch > env/default
    return getattr(args, "branch_pos", None) or args.branch or "main"


def test_upload_branch_positional():
    args = build_parser().parse_args(["upload", "ds", "my-branch"])
    assert args.dataset_dir == "ds"
    assert _resolved_branch(args) == "my-branch"


def test_upload_branch_flag():
    args = build_parser().parse_args(["upload", "ds", "--branch", "my-branch"])
    assert _resolved_branch(args) == "my-branch"


def test_upload_branch_defaults_to_main():
    args = build_parser().parse_args(["upload", "ds"])
    assert _resolved_branch(args) == "main"


def test_sync_branch_positional():
    args = build_parser().parse_args(["sync", "ds", "b"])
    assert _resolved_branch(args) == "b"
