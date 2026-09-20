"""Immutable run metadata and artifact hashing for experiment provenance."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_run_metadata(output: str | Path, root: str | Path, config: dict,
                       inputs: Iterable[str | Path] = ()) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(root),
        "python": sys.version,
        "platform": platform.platform(),
        "config": config,
        "inputs": {str(path): sha256(path) for path in inputs if Path(path).is_file()},
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return output
