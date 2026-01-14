from course.core.scoring import score
from course.core.types import Example


def test_valid_exact_format_positive():
    ex = Example(id="x", prompt="Compute 11*2.", expected_answer=22)
    out = score(ex, "Final: 22")
    assert out["reward"] == 1.0
    assert out["details"]["result"]["code"] == "ok"


def test_valid_exact_format_negative():
    ex = Example(id="x", prompt="Compute -5.", expected_answer=-5)
    out = score(ex, "Final: -5")
    assert out["reward"] == 1.0
    assert out["details"]["result"]["code"] == "ok"


def test_leading_zeros_rejected():
    ex = Example(id="x", prompt="Compute 11*2.", expected_answer=22)
    out = score(ex, "Final: 022")
    assert out["reward"] == 0.0
    assert out["details"]["result"]["code"] == "leading_zeros"


def test_plus_sign_rejected():
    ex = Example(id="x", prompt="Compute 11*2.", expected_answer=22)
    out = score(ex, "Final: +22")
    assert out["reward"] == 0.0
    assert out["details"]["result"]["code"] == "plus_sign_disallowed"


def test_extra_whitespace_rejected():
    ex = Example(id="x", prompt="Compute 11*2.", expected_answer=22)
    out = score(ex, "Final:  22")
    assert out["reward"] == 0.0
    assert out["details"]["result"]["code"] == "extra_whitespace"


def test_multiline_rejected():
    ex = Example(id="x", prompt="Compute 11*2.", expected_answer=22)
    out = score(ex, "Final:\n22")
    assert out["reward"] == 0.0
    assert out["details"]["result"]["code"] == "not_single_line"
