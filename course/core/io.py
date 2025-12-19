from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

JsonDict = Dict[str, Any]


def to_jsonable(obj: Any) -> Any:
    """Best-effort conversion to something JSON serializable.

    This is mainly used for manifests and logs where we want robustness.
    """

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, set):
        return [to_jsonable(x) for x in sorted(obj, key=lambda x: str(x))]
    # Fallback: make it explicit rather than crashing.
    return str(obj)


def utc_now_iso() -> str:
    """UTC timestamp in an ISO-ish format suitable for filenames/logs."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Best-effort atomic text write (write temp file then replace)."""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[JsonDict]:
    """Read a JSONL file into a list of dicts with friendly error messages."""
    records: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {e.msg}. Line={line!r}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object per line in {path}. Line {i} was {type(obj)}")
            records.append(obj)
    return records


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write iterable of dict-like records to JSONL."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            if is_dataclass(rec):
                rec = asdict(rec)  # type: ignore[assignment]
            f.write(json.dumps(to_jsonable(dict(rec)), ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, sort_keys=True, indent=indent)
        f.write("\n")


def make_run_dir(root: Path, prefix: str) -> Path:
    """Create a unique run directory under `root` with a timestamp prefix."""
    ensure_dir(root)
    ts = utc_now_iso().replace(":", "").replace("+", "").replace("-", "")
    base = f"{prefix}_{ts}"
    run_dir = root / base
    # Rare but deterministic behavior: if a directory exists (e.g., two runs in same second),
    # append an incrementing suffix.
    k = 1
    while run_dir.exists():
        k += 1
        run_dir = root / f"{base}_{k}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_fingerprint(path: Path) -> JsonDict:
    st = path.stat()
    # Use UTC timestamps in manifest to avoid timezone surprises.
    mtime_utc = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat()
    return {
        "path": str(path),
        "size_bytes": int(st.st_size),
        "sha256": sha256_file(path),
        "modified_utc": mtime_utc,
    }


def _run_cmd(cmd: list[str], *, cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    try:
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def try_get_git_info(repo_root: Optional[Path] = None) -> Optional[JsonDict]:
    """Best-effort git metadata.

    Returns None if:
    - git isn't installed
    - we're not in a git work tree
    - any command fails
    """

    root = repo_root or Path.cwd()
    inside = _run_cmd(["git", "rev-parse", "--is-inside-work-tree"], cwd=root)
    if inside != "true":
        return None

    commit = _run_cmd(["git", "rev-parse", "HEAD"], cwd=root)
    if not commit:
        return None

    status = _run_cmd(["git", "status", "--porcelain"], cwd=root)
    dirty = bool(status)

    return {
        "commit": commit,
        "dirty": dirty,
    }


def get_env_info() -> JsonDict:
    return {
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": str(Path.cwd()),
    }


def write_manifest(
    run_dir: Path,
    *,
    created_utc: str,
    script: str,
    argv: Optional[list[str]] = None,
    args: Optional[JsonDict] = None,
    inputs: Optional[list[Path]] = None,
    scorer: Optional[JsonDict] = None,
    extra: Optional[JsonDict] = None,
    repo_root: Optional[Path] = None,
) -> None:
    """Write a production-style run manifest.

    The manifest is meant to be the *minimum viable evidence* needed to reproduce
    or debug a number later.
    """

    input_fps: list[JsonDict] = []
    for p in (inputs or []):
        if p.exists() and p.is_file():
            input_fps.append(file_fingerprint(p))
        else:
            input_fps.append({"path": str(p), "missing": True})

    manifest: JsonDict = {
        "run_id": run_dir.name,
        "created_utc": created_utc,
        "command": {
            "script": script,
            "argv": argv or [],
            "args": args or {},
        },
        "inputs": input_fps,
        "scorer": scorer or {},
        "environment": get_env_info(),
        "git": try_get_git_info(repo_root),
        "extra": extra or {},
    }

    write_json(run_dir / "manifest.json", manifest)
