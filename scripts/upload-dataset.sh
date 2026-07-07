#!/usr/bin/env bash
# DEPRECATED. Replaced by the Python uploader, which adds a browser login (no
# cookie copy-paste), dataset validation, and true incremental sync (uploads
# AND deletions). Use:
#
#     ./scripts/upload-dataset.py <local-dataset-dir> <branch>
#     # or the full CLI:  kubecore-dataset --help
#
# This shim just forwards to the Python wrapper so old muscle-memory keeps working.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "note: upload-dataset.sh is deprecated — forwarding to upload-dataset.py" >&2
exec python3 "${HERE}/upload-dataset.py" "$@"
