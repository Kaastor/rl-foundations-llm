from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Union

from course.core.types import Example


# -----------------------------------------------------------------------------
# Scorer identity (version this like software)
# -----------------------------------------------------------------------------

SCORER_NAME = "math_final_line_verifier"
SCORER_VERSION = "1.0.0"

# Human-readable spec for auditing / diffs.
# Treat this as a contract: if you change reward behavior, bump SCORER_VERSION.
REWARD_SPEC = {
    "task": "single_turn_math_answer_verification",
    "format": "exactly one line: 'Final: <int>'",
    "int_format": {
        "base": 10,
        "sign": "optional leading '-'",
        "no_commas": True,
        "no_units": True,
        "no_words": True,
        "no_leading_zeros": True,  # except the literal '0'
        "no_negative_zero": True,
    },
    "normalization": ["strip surrounding whitespace"],
    "reward": "1.0 if parsed int == expected_answer else 0.0",
    "no_partial_credit": True,
}

# Stable result codes for debugging / grouping.
# These are deliberately small and stringly-typed for ease of use.
PARSE_ERROR_CODES = {
    "not_single_line",
    "missing_prefix",
    "extra_whitespace",
    "missing_integer",
    "plus_sign_disallowed",
    "missing_digits",
    "non_digit_characters",
    "negative_zero_disallowed",
    "leading_zeros",
    "int_parse_failed",
}

RESULT_CODES = {"ok", "wrong_answer", "invalid_example"} | PARSE_ERROR_CODES


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

_MAX_DETAIL_CHARS = 500  # keep logs inspectable


def _safe_preview(text: str, limit: int = _MAX_DETAIL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... <truncated {len(text) - limit} chars>"


def _extract_example_fields(example: Union[Example, Mapping[str, Any]]) -> tuple[str, str, Optional[int], Dict[str, Any]]:
    """Extract (id, prompt, expected_answer, extra_details) without crashing.

    This scorer is *total*: even if example is malformed, it should return a result.
    """

    extra: Dict[str, Any] = {}
    try:
        if isinstance(example, Example):
            return example.id, example.prompt, example.expected_answer, extra

        ex_id = str(example.get("id", "<missing-id>"))
        prompt = str(example.get("prompt", ""))
        expected_raw = example.get("expected_answer", None)
        expected_int: Optional[int]
        try:
            expected_int = int(expected_raw) if expected_raw is not None else None
        except Exception:
            expected_int = None
            extra["example_error"] = {
                "code": "expected_answer_not_int",
                "message": f"expected_answer must be int-coercible; got {expected_raw!r}",
            }
        return ex_id, prompt, expected_int, extra
    except Exception as e:  # defensive
        return (
            "<example-extraction-failed>",
            "",
            None,
            {"example_error": {"code": "example_extraction_failed", "message": str(e)}},
        )


def _parse_final_line(line: str) -> tuple[Optional[int], Dict[str, Any]]:
    """Parse a single line that should look like: 'Final: <int>'.

    Returns (value, parse_details). value is None if parsing fails.

    Important: we are intentionally strict. The format is part of the environment.
    """

    details: Dict[str, Any] = {
        "format_expected": "Final: <int>",
        "format_ok": False,
        "int_ok": False,
        "answer_str": None,
        "answer_int": None,
        "error_code": None,
        "error_message": None,
    }

    prefix = "Final: "
    if not line.startswith(prefix):
        details["error_code"] = "missing_prefix"
        details["error_message"] = "Completion must start with exactly 'Final: '"
        return None, details

    rest = line[len(prefix) :]

    # No leading/trailing spaces in the integer part.
    if rest != rest.strip():
        details["error_code"] = "extra_whitespace"
        details["error_message"] = "No extra spaces allowed after 'Final: ' or after the integer"
        return None, details

    if rest == "":
        details["error_code"] = "missing_integer"
        details["error_message"] = "Missing integer after 'Final: '"
        return None, details

    if rest[0] == "+":
        details["error_code"] = "plus_sign_disallowed"
        details["error_message"] = "Plus sign is not allowed. Use plain digits (or leading '-')"
        return None, details

    if rest.startswith("-"):
        digits = rest[1:]
        if digits == "":
            details["error_code"] = "missing_digits"
            details["error_message"] = "A '-' must be followed by digits"
            return None, details
        if not digits.isdigit():
            details["error_code"] = "non_digit_characters"
            details["error_message"] = "Integer must contain digits only"
            return None, details
        if digits == "0":
            details["error_code"] = "negative_zero_disallowed"
            details["error_message"] = "Negative zero (-0) is not allowed"
            return None, details
        if len(digits) > 1 and digits.startswith("0"):
            details["error_code"] = "leading_zeros"
            details["error_message"] = "Leading zeros are not allowed"
            return None, details
    else:
        if not rest.isdigit():
            details["error_code"] = "non_digit_characters"
            details["error_message"] = "Integer must contain digits only"
            return None, details
        if len(rest) > 1 and rest.startswith("0"):
            details["error_code"] = "leading_zeros"
            details["error_message"] = "Leading zeros are not allowed"
            return None, details

    try:
        value = int(rest)
    except Exception as e:  # defensive
        details["error_code"] = "int_parse_failed"
        details["error_message"] = f"int() failed unexpectedly: {e}"
        return None, details

    details["format_ok"] = True
    details["int_ok"] = True
    details["answer_str"] = rest
    details["answer_int"] = value
    return value, details


# -----------------------------------------------------------------------------
# Public scorer API (the course contract)
# -----------------------------------------------------------------------------


def score(example: Union[Example, Mapping[str, Any]], completion: Any) -> Dict[str, Any]:
    """Score a completion against an example.

    Contract (non-negotiable):
        score(example, completion) -> {"reward": float, "details": dict}

    Requirements:
    - deterministic
    - total (never raises)
    - explainable (details show why)
    """

    ex_id, _prompt, expected, ex_extra = _extract_example_fields(example)

    # Normalize completion to a string, but never crash.
    try:
        completion_str = "" if completion is None else str(completion)
    except Exception as e:  # defensive
        completion_str = ""
        ex_extra["completion_error"] = {"code": "completion_to_str_failed", "message": str(e)}

    completion_preview = _safe_preview(completion_str)
    normalized = completion_str.strip()

    # Enforce "exactly one line" after stripping surrounding whitespace.
    lines = normalized.splitlines() if normalized else []
    line_count = len(lines)

    details: Dict[str, Any] = {
        "scorer": {"name": SCORER_NAME, "version": SCORER_VERSION},
        "reward_spec": REWARD_SPEC,
        "example": {
            "id": ex_id,
            "expected_answer": expected,
        },
        "completion": {
            "raw_preview": completion_preview,
            "normalized_preview": _safe_preview(normalized),
            "line_count": line_count,
        },
        "parse": None,
        "match": False,
        "result": {
            "code": "invalid_example",  # overwritten below
            "message": None,
        },
        "notes": [],
    }
    if ex_extra:
        details["example_warnings"] = ex_extra

    # If example is malformed, fail safe.
    if expected is None:
        details["notes"].append("Example expected_answer is invalid; reward forced to 0.")
        details["result"] = {"code": "invalid_example", "message": "expected_answer missing or not int"}
        return {"reward": 0.0, "details": details}

    if line_count != 1:
        details["parse"] = {
            "format_expected": "exactly one line: 'Final: <int>'",
            "format_ok": False,
            "int_ok": False,
            "answer_str": None,
            "answer_int": None,
            "error_code": "not_single_line",
            "error_message": f"Expected 1 line after stripping; got {line_count}",
        }
        details["result"] = {"code": "not_single_line", "message": details["parse"]["error_message"]}
        return {"reward": 0.0, "details": details}

    value, parse_details = _parse_final_line(lines[0])
    details["parse"] = parse_details

    if value is None:
        code = str(parse_details.get("error_code") or "unknown")
        msg = parse_details.get("error_message")
        details["result"] = {"code": code, "message": msg}
        return {"reward": 0.0, "details": details}

    details["match"] = value == expected
    if details["match"]:
        details["result"] = {"code": "ok", "message": None}
        return {"reward": 1.0, "details": details}

    # Wrong answer: parsing succeeded, but semantics mismatch.
    details["notes"].append("Parsed successfully but did not match expected_answer.")
    details["result"] = {"code": "wrong_answer", "message": "parsed int != expected_answer"}
    return {"reward": 0.0, "details": details}
