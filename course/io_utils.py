from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence


JsonDict = Dict[str, Any]


def utc_now_iso() -> str:
    """UTC timestamp in an ISO-ish format suitable for filenames/logs."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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
            f.write(json.dumps(dict(rec), ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=indent)
        f.write("\n")


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Best-effort atomic text write (write temp file then replace)."""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


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
