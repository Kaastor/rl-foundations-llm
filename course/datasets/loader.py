from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional

from course.io_utils import read_jsonl
from course.types import Example


def load_examples(path: Path) -> list[Example]:
    """Load and validate a dataset JSONL file."""
    records = read_jsonl(path)
    examples: list[Example] = []
    seen_ids: set[str] = set()
    for rec in records:
        ex = Example.from_record(rec)
        if ex.id in seen_ids:
            raise ValueError(f"Duplicate example id {ex.id!r} in dataset {path}")
        seen_ids.add(ex.id)
        examples.append(ex)
    return examples


def index_by_id(examples: Iterable[Example]) -> dict[str, Example]:
    """Index examples by id, raising if ids collide."""
    out: dict[str, Example] = {}
    for ex in examples:
        if ex.id in out:
            raise ValueError(f"Duplicate id {ex.id!r}")
        out[ex.id] = ex
    return out
