from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from course.datasets.loader import load_examples, index_by_id
from course.io_utils import read_jsonl
from course.score import SCORER_NAME, SCORER_VERSION, score


def validate(
    dataset_path: Path,
    golden_path: Path,
) -> int:
    examples = load_examples(dataset_path)
    ex_by_id = index_by_id(examples)
    gold = read_jsonl(golden_path)

    failures = 0
    for rec in gold:
        ex_id = str(rec.get("id"))
        if ex_id not in ex_by_id:
            print(f"[FAIL] golden id not found in dataset: {ex_id}")
            failures += 1
            continue

        completion = rec.get("completion", "")
        expected_reward = float(rec.get("expected_reward", 0.0))

        out = score(ex_by_id[ex_id], completion)
        got = float(out["reward"])

        if got != expected_reward:
            failures += 1
            parse = (out.get("details") or {}).get("parse") or {}
            print(f"[FAIL] id={ex_id} expected_reward={expected_reward} got={got} error_code={parse.get('error_code')} completion={completion!r}")

    if failures == 0:
        print(f"OK: {golden_path} ({len(gold)} cases) under scorer {SCORER_NAME} v{SCORER_VERSION}")
    else:
        print(f"FAIL: {failures} / {len(gold)} golden cases failed under scorer {SCORER_NAME} v{SCORER_VERSION}")
    return failures


def main() -> None:
    p = argparse.ArgumentParser(description="Validate scorer against a golden JSONL file.")
    p.add_argument("--dataset", type=Path, required=True, help="Dataset JSONL (to look up expected answers)")
    p.add_argument("--golden", type=Path, required=True, help="Golden JSONL (id, completion, expected_reward)")
    args = p.parse_args()

    failures = validate(args.dataset, args.golden)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
