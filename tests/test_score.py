import pytest

from course.score import score
from course.types import Example


def ex(expected: int = 323) -> Example:
    return Example(id="x", prompt="Compute 17*19.", expected_answer=expected)


def test_correct_exact_format():
    r = score(ex(323), "Final: 323")
    assert r["reward"] == 1.0
    assert r["details"]["parse"]["format_ok"] is True
    assert r["details"]["match"] is True


def test_wrong_answer_parses_but_mismatch():
    r = score(ex(323), "Final: 324")
    assert r["reward"] == 0.0
    assert r["details"]["parse"]["format_ok"] is True
    assert r["details"]["match"] is False


def test_missing_prefix_fails():
    r = score(ex(323), "323")
    assert r["reward"] == 0.0
    assert r["details"]["parse"]["error_code"] == "missing_prefix"


def test_extra_punctuation_fails():
    r = score(ex(323), "Final: 323.")
    assert r["reward"] == 0.0
    assert r["details"]["parse"]["error_code"] == "non_digit_characters"


def test_double_space_fails():
    r = score(ex(323), "Final:  323")
    assert r["reward"] == 0.0
    assert r["details"]["parse"]["error_code"] == "extra_whitespace"


def test_leading_zeros_fail():
    r = score(ex(323), "Final: 0323")
    assert r["reward"] == 0.0
    assert r["details"]["parse"]["error_code"] == "leading_zeros"


def test_not_single_line_fails():
    r = score(ex(323), "Final:\n323")
    assert r["reward"] == 0.0
    assert r["details"]["parse"]["error_code"] == "not_single_line"


def test_negative_zero_disallowed():
    r = score(ex(0), "Final: -0")
    assert r["reward"] == 0.0
    assert r["details"]["parse"]["error_code"] == "negative_zero_disallowed"


def test_deterministic_for_same_inputs():
    a = score(ex(323), "Final: 323")
    b = score(ex(323), "Final: 323")
    assert a == b


def test_total_even_if_completion_str_raises():
    class Bad:
        def __str__(self):
            raise RuntimeError("boom")

    r = score(ex(323), Bad())
    assert r["reward"] == 0.0
    assert "completion_error" in r["details"].get("example_warnings", {}) or r["details"]["completion"]["raw_preview"] == ""
