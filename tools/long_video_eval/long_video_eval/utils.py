"""Small utility helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_command(cmd: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> int:
    printable = " ".join(cmd)
    if dry_run:
        print(f"[dry-run] {printable}")
        return 0
    print(f"[run] {printable}")
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True).returncode


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
