"""Pure implementation of the official BEHAVIOR task Q-score formula."""

from __future__ import annotations

from collections.abc import Sequence


PredicateOptions = Sequence[Sequence[bool]]


def q_score(initial: PredicateOptions, current: PredicateOptions, *, success: bool) -> float:
    """Return the task Q-score used by ``omnigibson.metrics.task_metric``.

    A predicate contributes partial credit only when it was false initially and
    is true now. A fully successful task always receives 1.0.
    """

    if success:
        return 1.0
    if len(initial) != len(current) or not initial:
        raise ValueError("initial and current predicate options must have the same non-zero length")

    option_scores: list[float] = []
    for initial_option, current_option in zip(initial, current):
        if len(initial_option) != len(current_option) or not initial_option:
            raise ValueError("predicate option shapes must match and each option must be non-empty")
        newly_satisfied = sum(
            int(not initially_true and currently_true)
            for initially_true, currently_true in zip(initial_option, current_option)
        )
        option_scores.append(newly_satisfied / len(current_option))
    return max(option_scores)

