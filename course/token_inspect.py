from __future__ import annotations

import argparse
from typing import Any, List, Tuple


def _try_tiktoken(text: str) -> tuple[str, list[tuple[int, str]]]:
    """Try to tokenize using tiktoken if installed.

    Returns (tokenizer_name, [(token_id, token_str), ...])
    """
    import tiktoken  # type: ignore

    enc = tiktoken.get_encoding("cl100k_base")
    ids = enc.encode(text)
    toks = [(i, enc.decode([i])) for i in ids]
    return "tiktoken/cl100k_base", toks


def _fallback_bytes(text: str) -> tuple[str, list[tuple[int, str]]]:
    b = text.encode("utf-8", errors="replace")
    toks = [(int(x), bytes([x]).decode("utf-8", errors="replace")) for x in b]
    return "fallback/utf8_bytes", toks


def tokenize(text: str) -> tuple[str, list[tuple[int, str]]]:
    """Tokenize text for inspection.

    Prefer a real LLM tokenizer (tiktoken). If unavailable, fall back to UTF-8 bytes.
    The fallback is still useful for seeing the token/text boundary and whitespace quirks,
    but it will not match real model tokenization.
    """
    try:
        return _try_tiktoken(text)
    except Exception:
        return _fallback_bytes(text)


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect tokenization for a string (real tokenizer if available)." )
    p.add_argument("text", type=str, nargs="?", default=None, help="Text to tokenize. If omitted, reads stdin.")
    args = p.parse_args()

    if args.text is None:
        import sys

        text = sys.stdin.read()
    else:
        text = args.text

    name, toks = tokenize(text)
    print(f"Tokenizer: {name}")
    print(f"Input repr: {text!r}")
    print(f"Token count: {len(toks)}\n")

    for idx, (tok_id, tok_str) in enumerate(toks):
        # Show whitespace clearly
        shown = tok_str.replace("\n", "\\n").replace("\t", "\\t")
        print(f"{idx:03d}  id={tok_id:<6}  token={shown!r}")


if __name__ == "__main__":
    main()
