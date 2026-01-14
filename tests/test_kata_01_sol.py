import pytest

from course.assignments.kata_01 import classify
from course.core.scoring import PARSE_ERROR_CODES


def test_wrong_answer_maps_to_model_math():
    assert classify("wrong_answer") == "model_math"


@pytest.mark.parametrize(
    "code",
    [
        "missing_prefix",
        "extra_whitespace",
        "not_single_line",
        "leading_zeros",
    ],
)
def test_parse_codes_map_to_model_format(code: str):
    assert classify(code) == "model_format"
    assert code in PARSE_ERROR_CODES


def test_invalid_example_maps_to_data_invalid():
    assert classify("invalid_example") == "data_invalid"


def test_unknown_code_maps_to_unknown():
    assert classify("something_else") == "unknown"
