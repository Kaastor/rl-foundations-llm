from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq

from course.core.datasets import load_examples
from course.core.io import make_run_dir, utc_now_iso, write_json, write_jsonl, write_manifest


DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def _make_messages(prompt: str, system_prompt: Optional[str]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _sample_completion(
    client: Groq,
    *,
    model: str,
    prompt: str,
    system_prompt: Optional[str],
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> tuple[str, Dict[str, Any]]:
    resp = client.chat.completions.create(
        model=model,
        messages=_make_messages(prompt, system_prompt),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    choice = resp.choices[0]
    text = (choice.message.content or "").strip("\n")
    meta: Dict[str, Any] = {
        "model": model,
        "finish_reason": choice.finish_reason,
    }

    usage = getattr(resp, "usage", None)
    if usage:
        meta["prompt_tokens"] = usage.prompt_tokens
        meta["completion_tokens"] = usage.completion_tokens
        meta["total_tokens"] = usage.total_tokens

    return text, meta


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description="Sample real model rollouts via Groq (optional extension).")
    p.add_argument("--dataset", type=Path, required=True, help="Dataset JSONL (id, prompt, expected_answer)")
    p.add_argument("--outdir", type=Path, default=None, help="Output directory (defaults to runs/rollouts_<timestamp>)")
    p.add_argument("--out", type=str, default=None, help="Output filename (default depends on format)")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Groq model name")
    p.add_argument("--n", type=int, default=1, help="Samples per example")
    p.add_argument(
        "--format",
        type=str,
        choices=["completions", "selection"],
        default=None,
        help="Output format; defaults to completions for n=1, selection for n>1",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--system", type=str, default=None, help="Optional system prompt")
    p.add_argument("--max", type=int, default=None, dest="max_examples", help="Optional cap for quick runs")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests")
    p.add_argument("--api-key", type=str, default=None, help="Override GROQ_API_KEY")
    args = p.parse_args()

    api_key = args.api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Missing GROQ_API_KEY (or pass --api-key).")

    out_dir = args.outdir
    if out_dir is None:
        out_dir = make_run_dir(Path("runs"), prefix="rollouts")

    fmt = args.format
    if fmt is None:
        fmt = "selection" if args.n > 1 else "completions"

    if fmt == "completions" and args.n != 1:
        raise SystemExit("--format completions requires --n 1")

    out_name = args.out
    if out_name is None:
        out_name = "selection_pack.jsonl" if fmt == "selection" else "completions.jsonl"
    out_path = out_dir / out_name

    client = Groq(api_key=api_key)

    examples = load_examples(args.dataset)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    rows: list[Dict[str, Any]] = []
    created_utc = utc_now_iso()

    for ex in examples:
        if fmt == "completions":
            text, meta = _sample_completion(
                client,
                model=args.model,
                prompt=ex.prompt,
                system_prompt=args.system,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            row = {"id": ex.id, "completion": text, **meta}
            rows.append(row)
        else:
            samples: List[Dict[str, Any]] = []
            for _ in range(args.n):
                text, meta = _sample_completion(
                    client,
                    model=args.model,
                    prompt=ex.prompt,
                    system_prompt=args.system,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
                samples.append({"completion": text, **meta})
                if args.sleep:
                    time.sleep(args.sleep)
            rows.append({"id": ex.id, "samples": samples})

        if args.sleep and fmt == "completions":
            time.sleep(args.sleep)

    write_jsonl(out_path, rows)

    summary = {
        "run": {
            "created_utc": created_utc,
            "provider": "groq",
            "model": args.model,
            "dataset_path": str(args.dataset),
            "format": fmt,
            "n_examples": len(examples),
            "n_samples": args.n,
        },
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "system_prompt": bool(args.system),
        },
        "output": {"path": str(out_path)},
    }
    write_json(out_dir / "summary.json", summary)

    write_manifest(
        out_dir,
        created_utc=created_utc,
        script="rollout_sample",
        argv=sys.argv,
        args=vars(args),
        inputs=[args.dataset],
        extra={"provider": "groq", "format": fmt, "output": str(out_path)},
    )

    print(f"Wrote rollouts to: {out_path}")
    print(f"format={fmt} n_examples={len(examples)} n_samples={args.n}")


if __name__ == "__main__":
    main()
