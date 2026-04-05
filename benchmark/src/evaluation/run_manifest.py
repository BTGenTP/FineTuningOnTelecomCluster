from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from ..contracts import ExperimentConfig, RunPaths


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _git_head() -> tuple[Optional[str], Optional[bool]]:
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if root.returncode != 0:
            return None, None
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            cwd=root.stdout.strip(),
        )
        if commit.returncode != 0:
            return None, None
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            cwd=root.stdout.strip(),
        )
        is_dirty = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
        return commit.stdout.strip(), is_dirty
    except (OSError, subprocess.TimeoutExpired):
        return None, None


def _torch_runtime() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import torch

        out["torch_version"] = torch.__version__
        out["cuda_version_torch"] = torch.version.cuda
        if torch.cuda.is_available():
            out["cuda_device_name"] = torch.cuda.get_device_name(0)
        else:
            out["cuda_device_name"] = None
    except Exception as exc:
        out["torch_import_error"] = str(exc)
    return out


def _slurm_env() -> dict[str, Optional[str]]:
    keys = (
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_ID",
        "SLURM_GPUS_ON_NODE",
        "SLURM_STEP_GPUS",
        "SLURM_SUBMIT_DIR",
        "SLURM_JOB_NAME",
    )
    return {k: os.environ.get(k) for k in keys}


def build_run_manifest(
    *,
    config_path: Optional[Path],
    cfg: ExperimentConfig,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    commit, dirty = _git_head()
    payload: dict[str, Any] = {
        "created_at_utc": _iso_now(),
        "config_path_resolved": str(config_path.resolve()) if config_path else None,
        "config_sha256": sha256_file(config_path) if config_path else None,
        "experiment_name": cfg.name,
        "task": cfg.task,
        "method": cfg.method,
        "model_name_or_path": cfg.model.model_name_or_path,
        "peft_adapter_path": cfg.peft.adapter_path,
        "metadata": dict(cfg.metadata),
        "git_commit": commit,
        "git_dirty": dirty,
        "slurm": _slurm_env(),
        "benchmark_env": {
            "BENCHMARK_GPU_PROFILE": os.environ.get("BENCHMARK_GPU_PROFILE"),
            "BENCHMARK_FP16": os.environ.get("BENCHMARK_FP16"),
            "BENCHMARK_DISABLE_4BIT": os.environ.get("BENCHMARK_DISABLE_4BIT"),
        },
        "runtime": _torch_runtime(),
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def write_run_manifest(paths: RunPaths, payload: Mapping[str, Any]) -> None:
    paths.run_manifest_json.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_manifest_for_run(
    paths: RunPaths,
    *,
    config_path: Optional[Path],
    cfg: ExperimentConfig,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    payload = build_run_manifest(config_path=config_path, cfg=cfg, extra=extra)
    write_run_manifest(paths, payload)
