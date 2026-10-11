"""Per-run result files: config hash, provenance, atomic JSON write."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import torch

RESULT_SCHEMA_VERSION = 1


def config_hash(factors: Mapping) -> str:
    """sha1 of the canonical JSON of the factors that determine a run's numbers."""
    payload = json.dumps(factors, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()


def run_path(results_dir: Path, digest: str) -> Path:
    return Path(results_dir) / f"{digest}.json"


def write_run(results_dir: Path, record: Mapping) -> Path:
    """Atomic write to <results_dir>/<config_hash>.json (temp file + rename)."""
    path = run_path(results_dir, record["config_hash"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True))
    os.replace(tmp, path)
    return path


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def provenance(repo: Path | None = None) -> dict:
    """Where and with what code a run happened; None for whatever cannot be determined."""
    repo = repo or Path(__file__).resolve().parents[2]
    # untracked files (a scratch note, a cluster-only constraints file) don't change the code
    status = _git(["status", "--porcelain", "--untracked-files=no"], repo)
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "git_sha": _git(["rev-parse", "HEAD"], repo),
        "git_dirty": None if status is None else bool(status),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "container": os.environ.get("SINGULARITY_CONTAINER")
        or os.environ.get("APPTAINER_CONTAINER"),
    }
