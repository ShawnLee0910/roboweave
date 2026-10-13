"""Selected-option goal-leaf progress and bounded recovery bookkeeping.

This module deliberately has no simulator imports.  A caller selects the
*actual* official grounded option used by its execution plan, supplies stable
leaf identities plus their dependency graph, and feeds authoritative truth
vectors after every action / settle boundary.

The tracker never treats an action primitive's local relation report as goal
truth.  Local success is retained only as audit evidence; completion requires
every required leaf in the latest authoritative vector to be true.  When a
previously true leaf regresses, the leaf and its downstream dependants are
queued in deterministic topological order.  Recovery claims are bounded so a
caller cannot accidentally create an infinite retry loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence


class GoalLeafProgressError(ValueError):
    """Raised when the selected-option progress contract is invalid."""


@dataclass(frozen=True, slots=True)
class GoalLeafRecoveryDirective:
    """One bounded request to re-establish an official goal leaf."""

    option_index: int
    leaf_id: str
    attempt_number: int
    maximum_attempts: int
    prerequisite_leaf_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GoalLeafProgressStep:
    """Immutable audit record for one authoritative truth observation."""

    option_index: int
    step_index: int
    action_id: str | None
    attempted_leaf_id: str | None
    action_relation_true: bool | None
    action_relation_true_without_q_gain: bool
    truth_vector: tuple[bool, ...]
    newly_true_leaf_ids: tuple[str, ...]
    regressed_leaf_ids: tuple[str, ...]
    stable_true_leaf_ids: tuple[str, ...]
    stable_false_leaf_ids: tuple[str, ...]
    required_true_count: int
    required_leaf_count: int
    required_fraction: float
    credited_true_count: int
    credited_q: float
    credited_q_delta: float
    requeued_leaf_ids: tuple[str, ...]
    pending_recovery_leaf_ids: tuple[str, ...]
    exhausted_recovery_leaf_ids: tuple[str, ...]
    complete: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SelectedOptionGoalLeafProgress:
    """Track one execution plan's selected official goal option.

    ``dependencies_by_leaf`` maps a leaf to prerequisite leaves.  For example,
    if moving an inner container into an outer container depends on first
    placing an object in the inner container, the outer relation depends on
    the inner relation.  Regressing the prerequisite therefore queues both in
    dependency order.  Ties are resolved by ``required_leaf_ids`` order.

    Recovery attempts are counted only when :meth:`claim_next_recovery` hands
    a directive to the caller.  A normal first-pass plan action can be audited
    with ``attempted_leaf_id`` without consuming recovery budget.  The claim
    remains in flight until the next :meth:`observe` call.
    """

    VERSION = "selected_option_goal_leaf_progress/v1"

    def __init__(
        self,
        *,
        option_index: int,
        required_leaf_ids: Sequence[str],
        initial_truth_vector: Sequence[bool],
        dependencies_by_leaf: Mapping[str, Iterable[str]] | None = None,
        maximum_recovery_attempts_per_leaf: int = 3,
        maximum_total_recovery_attempts: int | None = None,
    ) -> None:
        if (
            isinstance(option_index, bool)
            or not isinstance(option_index, int)
            or option_index < 0
        ):
            raise GoalLeafProgressError("option index must be non-negative")
        if not isinstance(required_leaf_ids, (list, tuple)) or not required_leaf_ids:
            raise GoalLeafProgressError("required leaf IDs must be non-empty")
        leaves = tuple(required_leaf_ids)
        if any(not isinstance(leaf_id, str) or not leaf_id for leaf_id in leaves):
            raise GoalLeafProgressError("required leaf IDs must be non-empty strings")
        if len(set(leaves)) != len(leaves):
            raise GoalLeafProgressError("required leaf IDs must be unique")
        if (
            isinstance(maximum_recovery_attempts_per_leaf, bool)
            or not isinstance(maximum_recovery_attempts_per_leaf, int)
            or maximum_recovery_attempts_per_leaf < 1
        ):
            raise GoalLeafProgressError(
                "maximum recovery attempts per leaf must be positive"
            )
        if maximum_total_recovery_attempts is not None and (
            isinstance(maximum_total_recovery_attempts, bool)
            or not isinstance(maximum_total_recovery_attempts, int)
            or maximum_total_recovery_attempts < 1
        ):
            raise GoalLeafProgressError(
                "maximum total recovery attempts must be positive"
            )

        self.option_index = option_index
        self.required_leaf_ids = leaves
        self.maximum_recovery_attempts_per_leaf = (
            maximum_recovery_attempts_per_leaf
        )
        self.maximum_total_recovery_attempts = maximum_total_recovery_attempts
        self._leaf_rank = {leaf_id: index for index, leaf_id in enumerate(leaves)}
        self._dependencies = self._normalize_dependencies(
            dependencies_by_leaf or {}
        )
        self._topological_leaf_ids = self._build_topological_order()
        self._dependants = self._build_dependants()

        initial = self._normalize_truth_vector(initial_truth_vector)
        self._initial_truth_vector = initial
        self._truth_vector = initial
        self._credited_q = self._compute_credited_q(initial)
        self._step_index = 0
        self._pending_recovery: set[str] = set()
        self._attempts_by_leaf = {leaf_id: 0 for leaf_id in leaves}
        self._total_recovery_attempts = 0
        self._exhausted_recovery: set[str] = set()
        self._inflight_recovery_leaf_id: str | None = None
        self._audit: list[GoalLeafProgressStep] = []

    @property
    def topological_leaf_ids(self) -> tuple[str, ...]:
        return self._topological_leaf_ids

    @property
    def truth_vector(self) -> tuple[bool, ...]:
        return self._truth_vector

    @property
    def complete(self) -> bool:
        return all(self._truth_vector)

    @property
    def audit_steps(self) -> tuple[GoalLeafProgressStep, ...]:
        return tuple(self._audit)

    @property
    def pending_recovery_leaf_ids(self) -> tuple[str, ...]:
        return self._ordered_pending()

    @property
    def exhausted_recovery_leaf_ids(self) -> tuple[str, ...]:
        return tuple(
            leaf_id
            for leaf_id in self._topological_leaf_ids
            if leaf_id in self._exhausted_recovery
        )

    @property
    def attempts_by_leaf(self) -> Mapping[str, int]:
        return dict(self._attempts_by_leaf)

    def initial_snapshot(self) -> dict[str, object]:
        """Return JSON-safe immutable construction evidence for artifacts."""

        return {
            "version": self.VERSION,
            "option_index": self.option_index,
            "required_leaf_ids": list(self.required_leaf_ids),
            "topological_leaf_ids": list(self._topological_leaf_ids),
            "initial_truth_vector": list(self._initial_truth_vector),
            "dependencies_by_leaf": {
                leaf_id: list(self._dependencies[leaf_id])
                for leaf_id in self._topological_leaf_ids
            },
            "maximum_recovery_attempts_per_leaf": (
                self.maximum_recovery_attempts_per_leaf
            ),
            "maximum_total_recovery_attempts": (
                self.maximum_total_recovery_attempts
            ),
        }

    def observe(
        self,
        truth_vector: Sequence[bool],
        *,
        action_id: str | None = None,
        attempted_leaf_id: str | None = None,
        action_relation_true: bool | None = None,
    ) -> GoalLeafProgressStep:
        """Record one authoritative truth vector and update recovery state.

        ``action_relation_true`` is deliberately not used as goal truth.  It
        only records the important case where an executor reports its local
        relation as true but the selected option's credited Q did not advance.
        Such an observation is recoverable evidence, not an exception.
        """

        if action_id is not None and (
            not isinstance(action_id, str) or not action_id
        ):
            raise GoalLeafProgressError("action ID must be a non-empty string")
        if attempted_leaf_id is not None and attempted_leaf_id not in self._leaf_rank:
            raise GoalLeafProgressError(
                "attempted leaf is outside the selected official option"
            )
        if action_relation_true is not None and not isinstance(
            action_relation_true, bool
        ):
            raise GoalLeafProgressError("action relation truth must be boolean")
        if self._inflight_recovery_leaf_id is not None:
            if attempted_leaf_id is None:
                attempted_leaf_id = self._inflight_recovery_leaf_id
            elif attempted_leaf_id != self._inflight_recovery_leaf_id:
                raise GoalLeafProgressError(
                    "observed attempted leaf does not match in-flight recovery"
                )

        current = self._normalize_truth_vector(truth_vector)
        previous = self._truth_vector
        newly_true = self._ordered_leaf_subset(
            leaf_id
            for leaf_id, before, after in zip(
                self.required_leaf_ids, previous, current, strict=True
            )
            if not before and after
        )
        regressed = self._ordered_leaf_subset(
            leaf_id
            for leaf_id, before, after in zip(
                self.required_leaf_ids, previous, current, strict=True
            )
            if before and not after
        )
        stable_true = self._ordered_leaf_subset(
            leaf_id
            for leaf_id, before, after in zip(
                self.required_leaf_ids, previous, current, strict=True
            )
            if before and after
        )
        stable_false = self._ordered_leaf_subset(
            leaf_id
            for leaf_id, before, after in zip(
                self.required_leaf_ids, previous, current, strict=True
            )
            if not before and not after
        )

        requeued: set[str] = set()
        for leaf_id in regressed:
            requeued.update(self._enqueue_recovery_closure(leaf_id))
        if attempted_leaf_id is not None and not current[self._leaf_rank[attempted_leaf_id]]:
            # A first-pass action that did not advance authoritative truth is
            # not itself evidence that already-false downstream leaves have
            # regressed.  Retry only its own leaf.  In contrast, a true->false
            # transition above queues the full dependent closure because
            # replaying the prerequisite may invalidate downstream work.
            if attempted_leaf_id not in self._exhausted_recovery:
                self._pending_recovery.add(attempted_leaf_id)
                requeued.add(attempted_leaf_id)

        # A queued dependant may remain true after the prerequisite recovery.
        # It does not need replay unless a later authoritative sample regresses
        # it, so prune currently true work at every action boundary.
        for leaf_id, truth in zip(self.required_leaf_ids, current, strict=True):
            if truth:
                self._pending_recovery.discard(leaf_id)
        self._inflight_recovery_leaf_id = None

        previous_q = self._credited_q
        credited_q = self._compute_credited_q(current)
        credited_q_delta = credited_q - previous_q
        true_count = sum(current)
        complete = true_count == len(current)
        if complete:
            self._pending_recovery.clear()

        self._step_index += 1
        self._truth_vector = current
        self._credited_q = credited_q
        step = GoalLeafProgressStep(
            option_index=self.option_index,
            step_index=self._step_index,
            action_id=action_id,
            attempted_leaf_id=attempted_leaf_id,
            action_relation_true=action_relation_true,
            action_relation_true_without_q_gain=bool(
                action_relation_true and credited_q_delta <= 0.0
            ),
            truth_vector=current,
            newly_true_leaf_ids=newly_true,
            regressed_leaf_ids=regressed,
            stable_true_leaf_ids=stable_true,
            stable_false_leaf_ids=stable_false,
            required_true_count=true_count,
            required_leaf_count=len(current),
            required_fraction=true_count / len(current),
            credited_true_count=sum(
                int(not initially_true and truth)
                for initially_true, truth in zip(
                    self._initial_truth_vector, current, strict=True
                )
            ),
            credited_q=credited_q,
            credited_q_delta=credited_q_delta,
            requeued_leaf_ids=self._ordered_leaf_subset(requeued),
            pending_recovery_leaf_ids=self._ordered_pending(),
            exhausted_recovery_leaf_ids=self.exhausted_recovery_leaf_ids,
            complete=complete,
        )
        self._audit.append(step)
        return step

    def claim_next_recovery(self) -> GoalLeafRecoveryDirective | None:
        """Claim one deterministic bounded recovery operation.

        The caller must feed a new authoritative observation before claiming
        another directive.  Once either bound is exhausted, the leaf is
        exposed in ``exhausted_recovery_leaf_ids`` and is never requeued.
        """

        if self._inflight_recovery_leaf_id is not None:
            raise GoalLeafProgressError(
                "an authoritative observation is required after recovery"
            )
        self._prune_true_pending()
        while self._pending_recovery:
            leaf_id = self._ordered_pending()[0]
            if self._recovery_budget_exhausted(leaf_id):
                self._pending_recovery.discard(leaf_id)
                self._exhausted_recovery.add(leaf_id)
                continue
            self._pending_recovery.discard(leaf_id)
            self._attempts_by_leaf[leaf_id] += 1
            self._total_recovery_attempts += 1
            self._inflight_recovery_leaf_id = leaf_id
            return GoalLeafRecoveryDirective(
                option_index=self.option_index,
                leaf_id=leaf_id,
                attempt_number=self._attempts_by_leaf[leaf_id],
                maximum_attempts=self.maximum_recovery_attempts_per_leaf,
                prerequisite_leaf_ids=self._dependencies[leaf_id],
            )
        return None

    def _normalize_dependencies(
        self,
        raw: Mapping[str, Iterable[str]],
    ) -> dict[str, tuple[str, ...]]:
        if not isinstance(raw, Mapping):
            raise GoalLeafProgressError("leaf dependencies must be a mapping")
        unknown_keys = set(raw) - set(self.required_leaf_ids)
        if unknown_keys:
            raise GoalLeafProgressError(
                "leaf dependency graph contains unknown leaves"
            )
        result: dict[str, tuple[str, ...]] = {}
        for leaf_id in self.required_leaf_ids:
            value = raw.get(leaf_id, ())
            if isinstance(value, (str, bytes)):
                raise GoalLeafProgressError(
                    "leaf prerequisites must be an iterable of leaf IDs"
                )
            try:
                prerequisites = tuple(value)
            except TypeError as exc:
                raise GoalLeafProgressError(
                    "leaf prerequisites must be iterable"
                ) from exc
            if len(set(prerequisites)) != len(prerequisites):
                raise GoalLeafProgressError("leaf prerequisites must be unique")
            if any(item not in self._leaf_rank for item in prerequisites):
                raise GoalLeafProgressError(
                    "leaf dependency graph contains unknown prerequisites"
                )
            if leaf_id in prerequisites:
                raise GoalLeafProgressError("a goal leaf cannot depend on itself")
            result[leaf_id] = tuple(
                sorted(prerequisites, key=self._leaf_rank.__getitem__)
            )
        return result

    def _build_topological_order(self) -> tuple[str, ...]:
        indegree = {
            leaf_id: len(self._dependencies[leaf_id])
            for leaf_id in self.required_leaf_ids
        }
        dependants = {leaf_id: [] for leaf_id in self.required_leaf_ids}
        for leaf_id, prerequisites in self._dependencies.items():
            for prerequisite in prerequisites:
                dependants[prerequisite].append(leaf_id)
        ready = sorted(
            (leaf_id for leaf_id, degree in indegree.items() if degree == 0),
            key=self._leaf_rank.__getitem__,
        )
        result: list[str] = []
        while ready:
            leaf_id = ready.pop(0)
            result.append(leaf_id)
            for dependant in sorted(
                dependants[leaf_id], key=self._leaf_rank.__getitem__
            ):
                indegree[dependant] -= 1
                if indegree[dependant] == 0:
                    ready.append(dependant)
                    ready.sort(key=self._leaf_rank.__getitem__)
        if len(result) != len(self.required_leaf_ids):
            raise GoalLeafProgressError("leaf dependency graph contains a cycle")
        return tuple(result)

    def _build_dependants(self) -> dict[str, tuple[str, ...]]:
        direct: dict[str, list[str]] = {
            leaf_id: [] for leaf_id in self.required_leaf_ids
        }
        for leaf_id, prerequisites in self._dependencies.items():
            for prerequisite in prerequisites:
                direct[prerequisite].append(leaf_id)
        return {
            leaf_id: tuple(
                sorted(values, key=self._topological_leaf_ids.index)
            )
            for leaf_id, values in direct.items()
        }

    def _enqueue_recovery_closure(self, root_leaf_id: str) -> tuple[str, ...]:
        closure = {root_leaf_id}
        frontier = [root_leaf_id]
        while frontier:
            current = frontier.pop()
            for dependant in self._dependants[current]:
                if dependant not in closure:
                    closure.add(dependant)
                    frontier.append(dependant)
        for leaf_id in closure:
            if leaf_id not in self._exhausted_recovery:
                self._pending_recovery.add(leaf_id)
        return self._ordered_leaf_subset(
            leaf_id
            for leaf_id in closure
            if leaf_id not in self._exhausted_recovery
        )

    def _normalize_truth_vector(
        self,
        value: Sequence[bool],
    ) -> tuple[bool, ...]:
        if not isinstance(value, (list, tuple)):
            raise GoalLeafProgressError("truth vector must be a list or tuple")
        if len(value) != len(self.required_leaf_ids):
            raise GoalLeafProgressError(
                "truth vector length does not match selected option"
            )
        if any(not isinstance(item, bool) for item in value):
            raise GoalLeafProgressError("truth vector entries must be boolean")
        return tuple(value)

    def _compute_credited_q(self, truth: tuple[bool, ...]) -> float:
        return (
            sum(
                int(not initially_true and current_true)
                for initially_true, current_true in zip(
                    self._initial_truth_vector, truth, strict=True
                )
            )
            / len(truth)
        )

    def _ordered_leaf_subset(self, leaf_ids: Iterable[str]) -> tuple[str, ...]:
        selected = set(leaf_ids)
        return tuple(
            leaf_id
            for leaf_id in self._topological_leaf_ids
            if leaf_id in selected
        )

    def _ordered_pending(self) -> tuple[str, ...]:
        return self._ordered_leaf_subset(self._pending_recovery)

    def _truth_for_leaf(self, leaf_id: str) -> bool:
        return self._truth_vector[self._leaf_rank[leaf_id]]

    def _prune_true_pending(self) -> None:
        self._pending_recovery = {
            leaf_id
            for leaf_id in self._pending_recovery
            if not self._truth_for_leaf(leaf_id)
        }

    def _recovery_budget_exhausted(self, leaf_id: str) -> bool:
        if (
            self._attempts_by_leaf[leaf_id]
            >= self.maximum_recovery_attempts_per_leaf
        ):
            return True
        return bool(
            self.maximum_total_recovery_attempts is not None
            and self._total_recovery_attempts
            >= self.maximum_total_recovery_attempts
        )
