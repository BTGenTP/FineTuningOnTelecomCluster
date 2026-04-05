#!/usr/bin/env bash
# Shared rsync exclude list for benchmark ↔ cluster sync (sourced by other scripts).
# shellcheck shell=bash

cluster_sync_rsync_excludes() {
  # Usage: rsync "${excludes[@]}" ...  →  built from here
  local ex=(
    --exclude=.venv/
    --exclude=__pycache__/
    --exclude=.pytest_cache/
    --exclude=.mypy_cache/
    --exclude=.ruff_cache/
    --exclude=.pytype/
    --exclude=.ipynb_checkpoints/
    --exclude=.cursor/
    --exclude=.git/
    --exclude=*.pyc
    --exclude=*.pyo
    --exclude=*.egg-info/
    --exclude=.eggs/
    --exclude=dist/
    --exclude=build/
    --exclude=.tox/
    --exclude=htmlcov/
    --exclude=.coverage
    --exclude=artifacts/
    --exclude=.DS_Store
  )
  printf '%s\0' "${ex[@]}"
}

cluster_sync_read_excludes_to_array() {
  # Populates global _CLUSTER_RSYNC_EXCLUDES as array for rsync
  _CLUSTER_RSYNC_EXCLUDES=()
  while IFS= read -r -d '' x; do
    _CLUSTER_RSYNC_EXCLUDES+=("$x")
  done < <(cluster_sync_rsync_excludes)
}
