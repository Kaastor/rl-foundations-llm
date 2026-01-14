"""Dataset loading utilities.

Datasets are JSONL files where each line is a record:
  {"id": "...", "prompt": "...", "expected_answer": 123}

The anchor task is intentionally narrow: expected_answer is always an integer.
"""
