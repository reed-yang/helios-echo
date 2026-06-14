"""Resource checks and setup actions for metric backends."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RuntimeConfig, deep_format, runtime_context


@dataclass
class CheckResult:
    id: str
    kind: str
    ok: bool
    message: str
    path: str = ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ResourceResolver:
    def __init__(self, runtime: RuntimeConfig):
        self.runtime = runtime
        self.context = runtime_context(runtime)
        self.resources: list[dict[str, Any]] = [
            deep_format(item, self.context) for item in runtime.raw.get("resources", [])
        ]
        self.python_packages: list[dict[str, Any]] = runtime.raw.get("python_packages", [])

    def check_python_packages(self) -> list[CheckResult]:
        results: list[CheckResult] = []
        for item in self.python_packages:
            import_name = item.get("import") or item.get("name")
            if not import_name:
                continue
            ok = importlib.util.find_spec(import_name) is not None
            message = "import ok" if ok else f"missing import: {import_name}"
            results.append(CheckResult(id=import_name, kind="python_package", ok=ok, message=message))
        return results

    def check_resources(self) -> list[CheckResult]:
        results: list[CheckResult] = []
        for item in self.resources:
            rid = item.get("id", "unnamed")
            kind = item.get("type", "local_path")
            path_value = item.get("path") or item.get("local_dir") or item.get("local_path") or ""
            path = Path(path_value).expanduser() if path_value else None

            if kind in {"local_path", "git", "hf_file", "hf_snapshot"}:
                if path is None:
                    results.append(CheckResult(rid, kind, False, "missing path in resource spec"))
                    continue
                if not path.exists():
                    results.append(CheckResult(rid, kind, False, f"missing path: {path}", str(path)))
                    continue
                expected_sha = item.get("sha256")
                if expected_sha and path.is_file():
                    actual = sha256_file(path)
                    if actual != expected_sha:
                        results.append(
                            CheckResult(rid, kind, False, f"sha256 mismatch: expected {expected_sha}, got {actual}", str(path))
                        )
                        continue
                results.append(CheckResult(rid, kind, True, "ok", str(path)))
            else:
                results.append(CheckResult(rid, kind, False, f"unknown resource type: {kind}"))
        return results

    def check_all(self) -> list[CheckResult]:
        return self.check_python_packages() + self.check_resources()

    def setup(self, *, yes: bool = False, dry_run: bool = False) -> None:
        for item in self.resources:
            kind = item.get("type", "local_path")
            rid = item.get("id", "unnamed")
            if kind == "local_path":
                path = Path(item["path"]).expanduser()
                if path.exists():
                    print(f"[setup] {rid}: exists {path}")
                else:
                    print(f"[setup] {rid}: missing local path {path}; no automatic action")
            elif kind == "git":
                self._setup_git(item, yes=yes, dry_run=dry_run)
            elif kind == "hf_file":
                self._setup_hf_file(item, yes=yes, dry_run=dry_run)
            elif kind == "hf_snapshot":
                self._setup_hf_snapshot(item, yes=yes, dry_run=dry_run)
            else:
                print(f"[setup] {rid}: unsupported resource type {kind}")

    def _confirm(self, action: str, *, yes: bool) -> bool:
        if yes:
            return True
        answer = input(f"{action} [y/N] ").strip().lower()
        return answer in {"y", "yes"}

    def _setup_git(self, item: dict[str, Any], *, yes: bool, dry_run: bool) -> None:
        rid = item.get("id", "git")
        path = Path(item["path"]).expanduser()
        url = item["url"]
        revision = item.get("revision")
        if path.exists():
            print(f"[setup] {rid}: exists {path}")
            return
        if not self._confirm(f"[setup] clone {url} -> {path}?", yes=yes):
            return
        cmd = ["git", "clone", url, str(path)]
        print("[setup]", " ".join(cmd))
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, check=True)
            if revision:
                subprocess.run(["git", "-C", str(path), "checkout", revision], check=True)

    def _setup_hf_file(self, item: dict[str, Any], *, yes: bool, dry_run: bool) -> None:
        rid = item.get("id", "hf_file")
        path = Path(item["path"]).expanduser()
        if path.exists():
            print(f"[setup] {rid}: exists {path}")
            return
        if not self._confirm(f"[setup] download HF file {item['repo_id']}:{item['filename']} -> {path}?", yes=yes):
            return
        print(f"[setup] hf_hub_download {item['repo_id']} {item['filename']} -> {path}")
        if dry_run:
            return
        from huggingface_hub import hf_hub_download

        path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=item["repo_id"],
            filename=item["filename"],
            repo_type=item.get("repo_type", "model"),
            revision=item.get("revision"),
            local_dir=str(path.parent),
        )

    def _setup_hf_snapshot(self, item: dict[str, Any], *, yes: bool, dry_run: bool) -> None:
        rid = item.get("id", "hf_snapshot")
        path = Path(item["path"]).expanduser()
        if path.exists() and any(path.iterdir()):
            print(f"[setup] {rid}: exists {path}")
            return
        if not self._confirm(f"[setup] download HF snapshot {item['repo_id']} -> {path}?", yes=yes):
            return
        print(f"[setup] snapshot_download {item['repo_id']} -> {path}")
        if dry_run:
            return
        from huggingface_hub import snapshot_download

        path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=item["repo_id"],
            repo_type=item.get("repo_type", "model"),
            revision=item.get("revision"),
            local_dir=str(path),
        )
