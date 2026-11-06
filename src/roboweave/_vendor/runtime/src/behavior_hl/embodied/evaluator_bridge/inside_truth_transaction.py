"""Pure selected-option truth transactions for nested Inside execution.

The simulator adapter owns cache invalidation and official predicate reads.
This module accepts only attested, freshly captured truth mappings and keeps
the action-start, precommit, candidate, postcommit, and final projections
consistent.  It has no simulator imports and cannot write predicate state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


class InsideTruthTransactionError(ValueError):
    """Raised when a selected-option transaction is stale or inconsistent."""


ACTION_START = "action_start"
PRECOMMIT = "precommit"
CANDIDATE = "candidate"
POSTCOMMIT = "postcommit"
FINAL = "final"
_BOUNDARY_ORDER = {
    ACTION_START: 0,
    PRECOMMIT: 1,
    CANDIDATE: 2,
    POSTCOMMIT: 3,
    FINAL: 4,
}


def _leaf_ids(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    result = tuple(values)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise InsideTruthTransactionError(f"{label} must be non-empty leaf IDs")
    if len(set(result)) != len(result):
        raise InsideTruthTransactionError(f"{label} must be unique")
    return result


def project_truth(
    truth_by_leaf: Mapping[str, bool],
    leaf_ids: Sequence[str],
) -> tuple[bool, ...]:
    """Project one authoritative mapping in declared leaf order."""

    if not isinstance(truth_by_leaf, Mapping):
        raise InsideTruthTransactionError("truth must be a mapping")
    ordered = _leaf_ids(leaf_ids, label="projection leaf IDs")
    missing = tuple(leaf for leaf in ordered if leaf not in truth_by_leaf)
    if missing:
        raise InsideTruthTransactionError(f"truth is missing leaves: {missing}")
    values = tuple(truth_by_leaf[leaf] for leaf in ordered)
    if any(not isinstance(value, bool) for value in values):
        raise InsideTruthTransactionError("truth values must be booleans")
    return values


@dataclass(frozen=True, slots=True)
class FreshSelectedTruthCapture:
    """One adapter-attested official truth read at a transaction boundary."""

    boundary: str
    sequence: int
    selected_truth: Mapping[str, bool]
    cache_refresh_succeeded: bool
    physics_step_count: int
    predicate_setter_called: bool

    def __post_init__(self) -> None:
        if self.boundary not in _BOUNDARY_ORDER:
            raise InsideTruthTransactionError("unknown truth capture boundary")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise InsideTruthTransactionError(
                "truth capture sequence must be non-negative"
            )
        if self.cache_refresh_succeeded is not True:
            raise InsideTruthTransactionError("fresh cache refresh is required")
        if self.physics_step_count != 0:
            raise InsideTruthTransactionError("truth refresh must advance zero physics")
        if self.predicate_setter_called is not False:
            raise InsideTruthTransactionError("predicate setters are forbidden")
        if not isinstance(self.selected_truth, Mapping):
            raise InsideTruthTransactionError("selected truth must be a mapping")
        if any(
            not isinstance(leaf, str)
            or not leaf
            or not isinstance(value, bool)
            for leaf, value in self.selected_truth.items()
        ):
            raise InsideTruthTransactionError(
                "selected truth must contain concrete leaf IDs and booleans"
            )


@dataclass(frozen=True, slots=True)
class InsideTruthDecision:
    boundary: str
    attempted_leaf_true: bool
    regressed_leaf_ids: tuple[str, ...]
    accepted: bool


@dataclass(frozen=True, slots=True)
class FinalTruthProjection:
    inside_truth_vector: tuple[bool, ...]
    option_truth_vector: tuple[bool, ...]
    inside_complete: bool
    selected_option_complete: bool


@dataclass(slots=True)
class InsideSelectedTruthTransaction:
    """Fail-closed preservation gate for one selected Inside leaf action."""

    option_leaf_ids: Sequence[str]
    inside_leaf_ids: Sequence[str]
    attempted_leaf_id: str
    action_start: FreshSelectedTruthCapture
    _option_leaf_ids: tuple[str, ...] = field(init=False, repr=False)
    _inside_leaf_ids: tuple[str, ...] = field(init=False, repr=False)
    _last_boundary_order: int = field(init=False, repr=False)
    _last_sequence: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._option_leaf_ids = _leaf_ids(
            self.option_leaf_ids, label="selected option leaf IDs"
        )
        self._inside_leaf_ids = _leaf_ids(
            self.inside_leaf_ids, label="Inside leaf IDs"
        )
        if not set(self._inside_leaf_ids).issubset(self._option_leaf_ids):
            raise InsideTruthTransactionError(
                "Inside leaves must be a selected-option subset"
            )
        if self.attempted_leaf_id not in self._inside_leaf_ids:
            raise InsideTruthTransactionError(
                "attempted leaf must be a selected Inside leaf"
            )
        if self.action_start.boundary != ACTION_START:
            raise InsideTruthTransactionError(
                "transaction must begin at action_start"
            )
        project_truth(self.action_start.selected_truth, self._option_leaf_ids)
        self._last_boundary_order = _BOUNDARY_ORDER[ACTION_START]
        self._last_sequence = self.action_start.sequence

    def observe(self, capture: FreshSelectedTruthCapture) -> InsideTruthDecision:
        """Evaluate one later fresh read against the action-start baseline."""

        order = _BOUNDARY_ORDER[capture.boundary]
        if capture.sequence <= self._last_sequence:
            raise InsideTruthTransactionError(
                "truth capture sequence must be strictly monotonic"
            )
        if order < self._last_boundary_order or (
            order == self._last_boundary_order and capture.boundary != CANDIDATE
        ):
            raise InsideTruthTransactionError(
                "truth stages must be monotonic; only candidate may repeat"
            )
        project_truth(capture.selected_truth, self._option_leaf_ids)
        baseline = self.action_start.selected_truth
        regressed = tuple(
            leaf
            for leaf in self._option_leaf_ids
            if baseline[leaf] and not capture.selected_truth[leaf]
        )
        attempted_true = capture.selected_truth[self.attempted_leaf_id]
        accepted = attempted_true and not regressed
        self._last_boundary_order = order
        self._last_sequence = capture.sequence
        return InsideTruthDecision(
            boundary=capture.boundary,
            attempted_leaf_true=attempted_true,
            regressed_leaf_ids=regressed,
            accepted=accepted,
        )

    def final_projection(
        self,
        capture: FreshSelectedTruthCapture,
    ) -> FinalTruthProjection:
        """Split final tracker input from authoritative full-option truth."""

        if capture.boundary != FINAL:
            raise InsideTruthTransactionError("final projection requires final capture")
        self.observe(capture)
        inside = project_truth(capture.selected_truth, self._inside_leaf_ids)
        option = project_truth(capture.selected_truth, self._option_leaf_ids)
        return FinalTruthProjection(
            inside_truth_vector=inside,
            option_truth_vector=option,
            inside_complete=all(inside),
            selected_option_complete=all(option),
        )
