#!/bin/bash
# Resumable raw-source transfer. Dry-run unless --execute is provided.

set -euo pipefail
execute=false
if [ "${1:-}" = "--execute" ]; then
  execute=true
  shift
fi
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 [--execute] USER@HOST /absolute/remote/BRATS" >&2
  exit 2
fi
remote_host="$1"
remote_root="${2%/}"
if [[ ! "$remote_host" =~ ^[A-Za-z0-9_.@-]+$ ]] || [[ ! "$remote_root" =~ ^/[A-Za-z0-9_./+-]+$ ]]; then
  echo "Invalid remote host or root" >&2
  exit 2
fi
workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
flags=(-a --partial --info=stats2)
if [ "$execute" != true ]; then
  flags+=(--dry-run)
  echo "DRY RUN: add --execute after reviewing the file list."
else
  ssh "$remote_host" mkdir -p "$remote_root/data/extracted"
fi
rsync "${flags[@]}" \
  "$workspace_root/data/extracted/BraTS-MEN-RT-Train-v2" \
  "$workspace_root/data/extracted/BraTS-MEN-RT-Val-v1" \
  "$workspace_root/data/extracted/BraTS-MEN-RT-0402-1" \
  "$remote_host:$remote_root/data/extracted/"

echo "Raw-source transfer finished. Build and validate raw_canonical on the destination before preprocessing."
