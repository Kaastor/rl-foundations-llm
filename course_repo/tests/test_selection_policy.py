from course.assignments.selection_policy import pick_best
from course.core.types import Example, RolloutSample


def test_pick_best_is_deterministic():
    # TODO: call pick_best twice and assert the same index.
    raise NotImplementedError


def test_tie_break_prefers_higher_logprob_then_shorter_then_lex():
    # TODO: construct samples with equal reward and test tie-break ordering.
    raise NotImplementedError
