from course.assignments.selection_policy_sol import pick_best
from course.core.types import Example, RolloutSample


def test_pick_best_is_deterministic():
    ex = Example(id="x", prompt="Compute 1+1.", expected_answer=2)
    samples = [
        RolloutSample(completion="Final: 2", sum_logprob=-1.0),
        RolloutSample(completion="Final: 2", sum_logprob=-1.0),
    ]
    first = pick_best(ex, samples)
    second = pick_best(ex, samples)
    assert first.best_index == second.best_index


def test_tie_break_prefers_higher_logprob_then_shorter_then_lex():
    ex = Example(id="x", prompt="Compute 1+1.", expected_answer=2)
    samples = [
        RolloutSample(completion="Final: 3", sum_logprob=-1.0),
        RolloutSample(completion="Final: 3", sum_logprob=-0.5),
    ]
    result = pick_best(ex, samples)
    assert result.best_index == 1
