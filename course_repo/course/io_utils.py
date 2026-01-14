"""Compatibility wrapper.

The canonical implementation lives in `course.core.io`.
"""

from course.core.io import (  # noqa: F401
    atomic_write_text,
    ensure_dir,
    file_fingerprint,
    get_env_info,
    make_run_dir,
    read_jsonl,
    sha256_file,
    utc_now_iso,
    write_json,
    write_jsonl,
    write_manifest,
)

__all__ = [
    "utc_now_iso",
    "ensure_dir",
    "atomic_write_text",
    "read_jsonl",
    "write_jsonl",
    "write_json",
    "make_run_dir",
    "sha256_file",
    "file_fingerprint",
    "get_env_info",
    "write_manifest",
]
