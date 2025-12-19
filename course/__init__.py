"""RL Foundations for LLMs — course harness.

This repository is intentionally scope-limited:
- single-turn task
- deterministic verifier (rule-based reward)
- no RL frameworks

The *core* reusable library lives in `course.core`.
Student-editable code lives in `course.assignments`.

CLI entrypoints are exposed via `python -m course.<script>`.
"""

__all__ = ["core", "assignments"]
