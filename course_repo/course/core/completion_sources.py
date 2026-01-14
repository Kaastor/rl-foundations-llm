from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from course.core.rollouts import load_frozen_rollouts
from course.core.types import RolloutSample


class CompletionSource(Protocol):
    """Minimal interface for "where completions come from".

    Production systems often have a generation step (model call) followed by
    scoring. This interface lets the harness keep that wiring explicit without
    importing any model libraries.
    """

    def describe(self) -> Dict[str, Any]:
        ...

    def get(self, example_id: str) -> Optional[RolloutSample]:
        ...


@dataclass
class DictCompletionSource:
    """In-memory completion source (useful for tests)."""

    mapping: Dict[str, RolloutSample]
    name: str = "dict"

    def describe(self) -> Dict[str, Any]:
        return {"type": self.name, "n": len(self.mapping)}

    def get(self, example_id: str) -> Optional[RolloutSample]:
        return self.mapping.get(example_id)


class JsonlCompletionSource:
    """Completion source backed by a JSONL file."""

    def __init__(self, path: Path):
        self.path = path
        self._map = load_frozen_rollouts(path)

    def describe(self) -> Dict[str, Any]:
        return {"type": "jsonl", "path": str(self.path), "n": len(self._map)}

    def get(self, example_id: str) -> Optional[RolloutSample]:
        return self._map.get(example_id)
