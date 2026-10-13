"""Fail-closed recovery for an openable destination that remains closed.

This policy covers one deliberately narrow kinematic-research case.  A
nested destination has been staged onto a generic support, bounded official
OPEN attempts have finished, and official ``Open`` readback still says
``False``.  If the grounded goal requires that same destination to finish
closed, the remaining required ``Inside(child, destination)`` relations may
be attempted with the ordinary official PLACE_INSIDE executor.

The failed OPEN is never reported as successful and this module never writes
an ``Open`` or ``Inside`` state.  Every admitted placement is accepted only
from live official ``Inside`` readback while the destination continues to
read closed.  A session is complete only after all admitted relations and the
closed invariant are re-read.  Full task success remains the responsibility
of the official BDDL verifier and the runner's final stability gate.

Because placing into a closed destination bypasses physical access, every
report explicitly labels this path ``kinematic_research`` and not official-
challenge eligible.  Unreadable state, unexpected children, state drift, or
an exhausted official placement attempt latches the session failed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from behavior_hl.embodied.evaluator_bridge.nested_container_operations import (
    OpenCloseOutcome,
    OpenCloseRetryPolicy,
)


CLOSED_DESTINATION_INSIDE_RECOVERY_VERSION = (
    "behavior2026.closed-destination-inside-recovery/v1"
)
CLOSED_DESTINATION_INSIDE_RECOVERY_MODE = (
    "official_inside_closed_destination"
)


ReadOpen = Callable[[str], bool]
ReadInside = Callable[[str, str], bool]
ExecutePlaceInside = Callable[[str, str], Any]
BetweenInsideReadbacks = Callable[[str, str, int, int], None]


def _strict_bool(value: Any, *, source: str) -> bool:
    if isinstance(value, bool):
        return value
    # Avoid a NumPy dependency while accepting the scalar returned by some
    # official state implementations.
    if type(value).__name__ == "bool_":
        item = getattr(value, "item", None)
        if callable(item):
            scalar = item()
            if isinstance(scalar, bool):
                return scalar
    raise TypeError(f"{source} must return a boolean")


def _validate_scope_name(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")


def _normalize_scope_names(
    values: Iterable[str],
    *,
    field: str,
) -> tuple[str, ...]:
    names = tuple(values)
    for name in names:
        _validate_scope_name(name, field=field)
    return tuple(sorted(set(names)))


def _normalize_inside_pairs(
    values: Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    normalized: set[tuple[str, str]] = set()
    for value in values:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(
                "required_inside_pairs must contain (child, parent) pairs"
            )
        child, parent = value
        _validate_scope_name(child, field="Inside child scope name")
        _validate_scope_name(parent, field="Inside parent scope name")
        if child == parent:
            raise ValueError("an Inside goal cannot contain an object in itself")
        normalized.add((child, parent))
    return tuple(sorted(normalized))


def _error(stage: str, exc: Exception, **context: Any) -> dict[str, Any]:
    return {
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        **context,
    }


def _report_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _report_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_report_value(item) for item in value]
    return repr(value)


def _base_report() -> dict[str, Any]:
    return {
        "schema_version": CLOSED_DESTINATION_INSIDE_RECOVERY_VERSION,
        "recovery_mode": CLOSED_DESTINATION_INSIDE_RECOVERY_MODE,
        "execution_fidelity": "kinematic_research",
        "official_challenge_eligible": False,
        "official_open_succeeded": False,
        "official_state_write_by_policy": False,
        "official_verifier_success_claimed": False,
    }


@dataclass(frozen=True, slots=True)
class ClosedDestinationStagingEvidence:
    """Canonical readback evidence produced by generic staging."""

    destination_scope_name: str
    support_scope_name: str
    original_parent_scope_name: str
    official_on_top: bool
    official_inside_original_parent: bool
    operational: bool = True

    def __post_init__(self) -> None:
        _validate_scope_name(
            self.destination_scope_name,
            field="destination_scope_name",
        )
        _validate_scope_name(self.support_scope_name, field="support_scope_name")
        _validate_scope_name(
            self.original_parent_scope_name,
            field="original_parent_scope_name",
        )
        if self.support_scope_name == self.destination_scope_name:
            raise ValueError("staging support cannot be the destination")
        if self.original_parent_scope_name == self.destination_scope_name:
            raise ValueError("staging parent cannot be the destination")
        if not isinstance(self.official_on_top, bool):
            raise ValueError("official_on_top must be a boolean")
        if not isinstance(self.official_inside_original_parent, bool):
            raise ValueError(
                "official_inside_original_parent must be a boolean"
            )
        if not isinstance(self.operational, bool):
            raise ValueError("operational must be a boolean")

    @property
    def succeeded(self) -> bool:
        return (
            self.operational
            and self.official_on_top
            and not self.official_inside_original_parent
        )

    def to_report(self) -> dict[str, Any]:
        return {
            "destination_scope_name_private": self.destination_scope_name,
            "support_scope_name_private": self.support_scope_name,
            "original_parent_scope_name_private": self.original_parent_scope_name,
            "operational": self.operational,
            "official_on_top": self.official_on_top,
            "official_inside_original_parent": (
                self.official_inside_original_parent
            ),
            "succeeded": self.succeeded,
        }


@dataclass(frozen=True, slots=True)
class ClosedDestinationRecoveryDecision:
    destination_scope_name: str
    operational: bool
    authorized: bool
    status: str
    expected_child_scope_names: tuple[str, ...]
    staging_evidence: ClosedDestinationStagingEvidence
    open_outcome: OpenCloseOutcome
    errors: tuple[Mapping[str, Any], ...] = ()

    def to_report(self) -> dict[str, Any]:
        report = _base_report()
        report.update(
            {
                "destination_scope_name_private": self.destination_scope_name,
                "operational": self.operational,
                "authorization_granted": self.authorized,
                "status": self.status,
                "expected_child_scope_names_private": list(
                    self.expected_child_scope_names
                ),
                "staging_evidence_private": self.staging_evidence.to_report(),
                "failed_open_outcome_private": self.open_outcome.to_report(),
                "errors": [dict(item) for item in self.errors],
            }
        )
        return report


def authorize_closed_destination_inside_recovery(
    *,
    destination_scope_name: str,
    final_closed_scope_names: Iterable[str],
    required_inside_pairs: Iterable[tuple[str, str]],
    staging_evidence: ClosedDestinationStagingEvidence,
    open_outcome: OpenCloseOutcome,
    open_retry_policy: OpenCloseRetryPolicy = OpenCloseRetryPolicy(),
) -> ClosedDestinationRecoveryDecision:
    """Authorize only the narrow, graph-derived closed-destination fallback.

    ``required_inside_pairs`` and ``final_closed_scope_names`` must come from
    the selected grounded goal option, not from a task-name table.  A failed
    OPEN is eligible only after the supplied finite action budget is exhausted
    and its final official state is the desired final closed state.
    """

    _validate_scope_name(
        destination_scope_name,
        field="destination_scope_name",
    )
    final_closed = _normalize_scope_names(
        final_closed_scope_names,
        field="final Closed scope name",
    )
    inside_pairs = _normalize_inside_pairs(required_inside_pairs)
    expected_children = tuple(
        child for child, parent in inside_pairs if parent == destination_scope_name
    )

    def decision(
        *,
        operational: bool,
        authorized: bool,
        status: str,
    ) -> ClosedDestinationRecoveryDecision:
        return ClosedDestinationRecoveryDecision(
            destination_scope_name=destination_scope_name,
            operational=operational,
            authorized=authorized,
            status=status,
            expected_child_scope_names=expected_children,
            staging_evidence=staging_evidence,
            open_outcome=open_outcome,
        )

    if staging_evidence.destination_scope_name != destination_scope_name:
        return decision(
            operational=False,
            authorized=False,
            status="staging_target_mismatch",
        )
    if not staging_evidence.operational:
        return decision(
            operational=False,
            authorized=False,
            status="staging_evidence_unreadable",
        )
    if not staging_evidence.succeeded:
        return decision(
            operational=True,
            authorized=False,
            status="staging_postcondition_unsatisfied",
        )
    if destination_scope_name not in final_closed:
        return decision(
            operational=True,
            authorized=False,
            status="destination_not_required_closed",
        )
    if not expected_children:
        return decision(
            operational=True,
            authorized=False,
            status="no_required_inside_children",
        )
    if open_outcome.target_scope_name != destination_scope_name:
        return decision(
            operational=False,
            authorized=False,
            status="open_target_mismatch",
        )
    if open_outcome.operation != "open" or open_outcome.desired_open is not True:
        return decision(
            operational=False,
            authorized=False,
            status="not_a_failed_open_transition",
        )
    if open_outcome.success:
        return decision(
            operational=True,
            authorized=False,
            status="open_succeeded_recovery_not_needed",
        )
    if not open_outcome.operational or open_outcome.state_after is not False:
        return decision(
            operational=False,
            authorized=False,
            status="failed_open_state_not_confirmed_closed",
        )
    eligible_failure_statuses = {
        "executor_failed_and_postcondition_unsatisfied",
        "official_postcondition_unsatisfied",
    }
    if open_outcome.status not in eligible_failure_statuses:
        return decision(
            operational=False,
            authorized=False,
            status="failed_open_outcome_not_eligible",
        )
    if open_outcome.action_attempt_count != open_retry_policy.max_action_attempts:
        return decision(
            operational=False,
            authorized=False,
            status="open_retry_budget_not_exhausted",
        )
    if any(
        len(attempt.readbacks) != open_retry_policy.max_readbacks_per_attempt
        for attempt in open_outcome.attempts
    ):
        return decision(
            operational=False,
            authorized=False,
            status="open_readback_budget_not_exhausted",
        )
    return decision(
        operational=True,
        authorized=True,
        status="authorized_after_bounded_open_failure",
    )


@dataclass(frozen=True, slots=True)
class InsideRetryPolicy:
    max_action_attempts: int = 2
    max_readbacks_per_attempt: int = 2

    def __post_init__(self) -> None:
        if self.max_action_attempts < 1:
            raise ValueError("max_action_attempts must be positive")
        if self.max_readbacks_per_attempt < 1:
            raise ValueError("max_readbacks_per_attempt must be positive")


@dataclass(frozen=True, slots=True)
class InsidePlacementAttempt:
    action_attempt: int
    executor_status: str
    executor_report: Any
    inside_readbacks: tuple[bool | None, ...]
    destination_open_readbacks: tuple[bool | None, ...]
    errors: tuple[Mapping[str, Any], ...]

    def to_report(self) -> dict[str, Any]:
        return {
            "action_attempt": self.action_attempt,
            "executor_status": self.executor_status,
            "executor_report_private": _report_value(self.executor_report),
            "inside_readbacks": list(self.inside_readbacks),
            "destination_open_readbacks": list(
                self.destination_open_readbacks
            ),
            "errors": [dict(item) for item in self.errors],
        }


@dataclass(frozen=True, slots=True)
class InsidePlacementOutcome:
    child_scope_name: str
    destination_scope_name: str
    operational: bool
    success: bool
    status: str
    inside_before: bool | None
    inside_after: bool | None
    destination_open_before: bool | None
    destination_open_after: bool | None
    attempts: tuple[InsidePlacementAttempt, ...] = ()
    errors: tuple[Mapping[str, Any], ...] = ()

    @property
    def action_attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def readback_count(self) -> int:
        return sum(len(attempt.inside_readbacks) for attempt in self.attempts)

    def to_report(self) -> dict[str, Any]:
        report = _base_report()
        report.update(
            {
                "child_scope_name_private": self.child_scope_name,
                "destination_scope_name_private": self.destination_scope_name,
                "operational": self.operational,
                "success": self.success,
                "status": self.status,
                "inside_before_private": self.inside_before,
                "inside_after_private": self.inside_after,
                "destination_open_before_private": self.destination_open_before,
                "destination_open_after_private": self.destination_open_after,
                "action_attempt_count": self.action_attempt_count,
                "readback_count": self.readback_count,
                "attempts": [attempt.to_report() for attempt in self.attempts],
                "errors": [dict(item) for item in self.errors],
            }
        )
        return report


def execute_official_inside_placement(
    *,
    child_scope_name: str,
    destination_scope_name: str,
    read_open: ReadOpen,
    read_inside: ReadInside,
    execute_place_inside: ExecutePlaceInside,
    retry_policy: InsideRetryPolicy = InsideRetryPolicy(),
    between_readbacks: BetweenInsideReadbacks | None = None,
) -> InsidePlacementOutcome:
    """Execute bounded PLACE_INSIDE, accepting only official live readback."""

    _validate_scope_name(child_scope_name, field="child_scope_name")
    _validate_scope_name(destination_scope_name, field="destination_scope_name")
    if child_scope_name == destination_scope_name:
        raise ValueError("a destination cannot be placed inside itself")

    try:
        open_before = _strict_bool(
            read_open(destination_scope_name),
            source="official destination Open readback",
        )
    except Exception as exc:
        return InsidePlacementOutcome(
            child_scope_name=child_scope_name,
            destination_scope_name=destination_scope_name,
            operational=False,
            success=False,
            status="destination_open_state_unreadable",
            inside_before=None,
            inside_after=None,
            destination_open_before=None,
            destination_open_after=None,
            errors=(_error("initial_destination_open_readback", exc),),
        )
    if open_before:
        return InsidePlacementOutcome(
            child_scope_name=child_scope_name,
            destination_scope_name=destination_scope_name,
            operational=True,
            success=False,
            status="destination_not_closed",
            inside_before=None,
            inside_after=None,
            destination_open_before=True,
            destination_open_after=True,
        )
    try:
        inside_before = _strict_bool(
            read_inside(child_scope_name, destination_scope_name),
            source="official Inside readback",
        )
    except Exception as exc:
        return InsidePlacementOutcome(
            child_scope_name=child_scope_name,
            destination_scope_name=destination_scope_name,
            operational=False,
            success=False,
            status="official_inside_unreadable",
            inside_before=None,
            inside_after=None,
            destination_open_before=False,
            destination_open_after=False,
            errors=(_error("initial_official_inside_readback", exc),),
        )
    if inside_before:
        return InsidePlacementOutcome(
            child_scope_name=child_scope_name,
            destination_scope_name=destination_scope_name,
            operational=True,
            success=True,
            status="already_satisfied",
            inside_before=True,
            inside_after=True,
            destination_open_before=False,
            destination_open_after=False,
        )

    attempts: list[InsidePlacementAttempt] = []
    all_errors: list[Mapping[str, Any]] = []
    last_inside: bool | None = inside_before
    last_open: bool | None = open_before
    any_valid_inside_readback = False
    for action_attempt in range(1, retry_policy.max_action_attempts + 1):
        if action_attempt > 1:
            try:
                last_open = _strict_bool(
                    read_open(destination_scope_name),
                    source="official destination Open readback",
                )
            except Exception as exc:
                error = _error(
                    "pre_attempt_destination_open_readback",
                    exc,
                    action_attempt=action_attempt,
                )
                all_errors.append(error)
                return InsidePlacementOutcome(
                    child_scope_name=child_scope_name,
                    destination_scope_name=destination_scope_name,
                    operational=False,
                    success=False,
                    status="destination_open_state_unreadable",
                    inside_before=inside_before,
                    inside_after=last_inside,
                    destination_open_before=open_before,
                    destination_open_after=None,
                    attempts=tuple(attempts),
                    errors=tuple(all_errors),
                )
            if last_open:
                return InsidePlacementOutcome(
                    child_scope_name=child_scope_name,
                    destination_scope_name=destination_scope_name,
                    operational=True,
                    success=False,
                    status="destination_open_drift",
                    inside_before=inside_before,
                    inside_after=last_inside,
                    destination_open_before=open_before,
                    destination_open_after=True,
                    attempts=tuple(attempts),
                    errors=tuple(all_errors),
                )

        executor_report: Any = None
        attempt_errors: list[Mapping[str, Any]] = []
        try:
            executor_report = execute_place_inside(
                child_scope_name,
                destination_scope_name,
            )
            executor_status = "returned"
        except Exception as exc:
            executor_status = "raised"
            error = _error("executor", exc, action_attempt=action_attempt)
            attempt_errors.append(error)
            all_errors.append(error)

        inside_readbacks: list[bool | None] = []
        open_readbacks: list[bool | None] = []
        for readback_index in range(
            1, retry_policy.max_readbacks_per_attempt + 1
        ):
            try:
                last_open = _strict_bool(
                    read_open(destination_scope_name),
                    source="official destination Open readback",
                )
                open_readbacks.append(last_open)
            except Exception as exc:
                last_open = None
                open_readbacks.append(None)
                inside_readbacks.append(None)
                error = _error(
                    "destination_open_readback",
                    exc,
                    action_attempt=action_attempt,
                    readback_index=readback_index,
                )
                attempt_errors.append(error)
                all_errors.append(error)
            else:
                if last_open:
                    inside_readbacks.append(None)
                else:
                    try:
                        last_inside = _strict_bool(
                            read_inside(
                                child_scope_name,
                                destination_scope_name,
                            ),
                            source="official Inside readback",
                        )
                        any_valid_inside_readback = True
                        inside_readbacks.append(last_inside)
                    except Exception as exc:
                        last_inside = None
                        inside_readbacks.append(None)
                        error = _error(
                            "official_inside_readback",
                            exc,
                            action_attempt=action_attempt,
                            readback_index=readback_index,
                        )
                        attempt_errors.append(error)
                        all_errors.append(error)

            if last_open is True:
                attempts.append(
                    InsidePlacementAttempt(
                        action_attempt=action_attempt,
                        executor_status=executor_status,
                        executor_report=executor_report,
                        inside_readbacks=tuple(inside_readbacks),
                        destination_open_readbacks=tuple(open_readbacks),
                        errors=tuple(attempt_errors),
                    )
                )
                return InsidePlacementOutcome(
                    child_scope_name=child_scope_name,
                    destination_scope_name=destination_scope_name,
                    operational=True,
                    success=False,
                    status="destination_open_drift",
                    inside_before=inside_before,
                    inside_after=last_inside,
                    destination_open_before=open_before,
                    destination_open_after=True,
                    attempts=tuple(attempts),
                    errors=tuple(all_errors),
                )
            if last_open is False and last_inside is True:
                attempts.append(
                    InsidePlacementAttempt(
                        action_attempt=action_attempt,
                        executor_status=executor_status,
                        executor_report=executor_report,
                        inside_readbacks=tuple(inside_readbacks),
                        destination_open_readbacks=tuple(open_readbacks),
                        errors=tuple(attempt_errors),
                    )
                )
                return InsidePlacementOutcome(
                    child_scope_name=child_scope_name,
                    destination_scope_name=destination_scope_name,
                    operational=True,
                    success=True,
                    status="satisfied_after_execution",
                    inside_before=inside_before,
                    inside_after=True,
                    destination_open_before=open_before,
                    destination_open_after=False,
                    attempts=tuple(attempts),
                    errors=tuple(all_errors),
                )
            if (
                readback_index < retry_policy.max_readbacks_per_attempt
                and between_readbacks is not None
            ):
                try:
                    between_readbacks(
                        child_scope_name,
                        destination_scope_name,
                        action_attempt,
                        readback_index,
                    )
                except Exception as exc:
                    error = _error(
                        "between_readbacks",
                        exc,
                        action_attempt=action_attempt,
                        readback_index=readback_index,
                    )
                    attempt_errors.append(error)
                    all_errors.append(error)

        attempts.append(
            InsidePlacementAttempt(
                action_attempt=action_attempt,
                executor_status=executor_status,
                executor_report=executor_report,
                inside_readbacks=tuple(inside_readbacks),
                destination_open_readbacks=tuple(open_readbacks),
                errors=tuple(attempt_errors),
            )
        )

    executor_always_raised = all(
        attempt.executor_status == "raised" for attempt in attempts
    )
    return InsidePlacementOutcome(
        child_scope_name=child_scope_name,
        destination_scope_name=destination_scope_name,
        operational=any_valid_inside_readback,
        success=False,
        status=(
            "executor_failed_and_inside_unsatisfied"
            if executor_always_raised and any_valid_inside_readback
            else (
                "official_inside_unreadable"
                if not any_valid_inside_readback
                else "official_inside_unsatisfied"
            )
        ),
        inside_before=inside_before,
        inside_after=last_inside,
        destination_open_before=open_before,
        destination_open_after=last_open,
        attempts=tuple(attempts),
        errors=tuple(all_errors),
    )


@dataclass(frozen=True, slots=True)
class ClosedDestinationRecoveryFinalOutcome:
    destination_scope_name: str
    operational: bool
    success: bool
    status: str
    expected_child_scope_names: tuple[str, ...]
    placed_child_scope_names: tuple[str, ...]
    destination_open: bool | None
    inside_readbacks: tuple[tuple[str, bool | None], ...]
    errors: tuple[Mapping[str, Any], ...] = ()

    def to_report(self) -> dict[str, Any]:
        report = _base_report()
        report.update(
            {
                "destination_scope_name_private": self.destination_scope_name,
                "operational": self.operational,
                "success": self.success,
                "status": self.status,
                "expected_child_scope_names_private": list(
                    self.expected_child_scope_names
                ),
                "placed_child_scope_names_private": list(
                    self.placed_child_scope_names
                ),
                "destination_open_private": self.destination_open,
                "inside_readbacks_private": [
                    {"child_scope_name_private": child, "inside": inside}
                    for child, inside in self.inside_readbacks
                ],
                "errors": [dict(item) for item in self.errors],
            }
        )
        return report


class ClosedDestinationInsideRecovery:
    """A single fail-closed recovery session for one destination."""

    def __init__(
        self,
        *,
        decision: ClosedDestinationRecoveryDecision,
        read_open: ReadOpen,
        read_inside: ReadInside,
        execute_place_inside: ExecutePlaceInside,
        retry_policy: InsideRetryPolicy = InsideRetryPolicy(),
        between_readbacks: BetweenInsideReadbacks | None = None,
    ) -> None:
        if not decision.authorized:
            raise ValueError("recovery decision must be authorized")
        self._decision = decision
        self._read_open = read_open
        self._read_inside = read_inside
        self._execute_place_inside = execute_place_inside
        self._retry_policy = retry_policy
        self._between_readbacks = between_readbacks
        self._outcomes: dict[str, InsidePlacementOutcome] = {}
        self._failed_status: str | None = None

    @property
    def destination_scope_name(self) -> str:
        return self._decision.destination_scope_name

    @property
    def expected_child_scope_names(self) -> tuple[str, ...]:
        return self._decision.expected_child_scope_names

    @property
    def placed_child_scope_names(self) -> tuple[str, ...]:
        return tuple(
            child
            for child in self.expected_child_scope_names
            if child in self._outcomes and self._outcomes[child].success
        )

    @property
    def failed(self) -> bool:
        return self._failed_status is not None

    def _blocked_outcome(
        self,
        *,
        child_scope_name: str,
        status: str,
    ) -> InsidePlacementOutcome:
        return InsidePlacementOutcome(
            child_scope_name=child_scope_name,
            destination_scope_name=self.destination_scope_name,
            operational=False,
            success=False,
            status=status,
            inside_before=None,
            inside_after=None,
            destination_open_before=None,
            destination_open_after=None,
        )

    def place_child(self, child_scope_name: str) -> InsidePlacementOutcome:
        """Place one admitted child, latching every failure."""

        _validate_scope_name(child_scope_name, field="child_scope_name")
        if self.failed:
            return self._blocked_outcome(
                child_scope_name=child_scope_name,
                status="recovery_session_failed",
            )
        if child_scope_name not in self.expected_child_scope_names:
            self._failed_status = "unexpected_child"
            return self._blocked_outcome(
                child_scope_name=child_scope_name,
                status="unexpected_child",
            )
        outcome = execute_official_inside_placement(
            child_scope_name=child_scope_name,
            destination_scope_name=self.destination_scope_name,
            read_open=self._read_open,
            read_inside=self._read_inside,
            execute_place_inside=self._execute_place_inside,
            retry_policy=self._retry_policy,
            between_readbacks=self._between_readbacks,
        )
        self._outcomes[child_scope_name] = outcome
        if not outcome.success:
            self._failed_status = outcome.status
        return outcome

    def finalize(self) -> ClosedDestinationRecoveryFinalOutcome:
        """Re-read every required relation; this is not full task success."""

        destination = self.destination_scope_name
        expected = self.expected_child_scope_names
        placed = self.placed_child_scope_names
        if self.failed:
            return ClosedDestinationRecoveryFinalOutcome(
                destination_scope_name=destination,
                operational=False,
                success=False,
                status="recovery_session_failed",
                expected_child_scope_names=expected,
                placed_child_scope_names=placed,
                destination_open=None,
                inside_readbacks=(),
            )
        if set(placed) != set(expected):
            self._failed_status = "required_children_incomplete"
            return ClosedDestinationRecoveryFinalOutcome(
                destination_scope_name=destination,
                operational=True,
                success=False,
                status="required_children_incomplete",
                expected_child_scope_names=expected,
                placed_child_scope_names=placed,
                destination_open=None,
                inside_readbacks=(),
            )
        try:
            open_state = _strict_bool(
                self._read_open(destination),
                source="official destination Open readback",
            )
        except Exception as exc:
            self._failed_status = "destination_open_state_unreadable"
            return ClosedDestinationRecoveryFinalOutcome(
                destination_scope_name=destination,
                operational=False,
                success=False,
                status="destination_open_state_unreadable",
                expected_child_scope_names=expected,
                placed_child_scope_names=placed,
                destination_open=None,
                inside_readbacks=(),
                errors=(_error("final_destination_open_readback", exc),),
            )
        if open_state:
            self._failed_status = "destination_open_drift"
            return ClosedDestinationRecoveryFinalOutcome(
                destination_scope_name=destination,
                operational=True,
                success=False,
                status="destination_open_drift",
                expected_child_scope_names=expected,
                placed_child_scope_names=placed,
                destination_open=True,
                inside_readbacks=(),
            )

        readbacks: list[tuple[str, bool | None]] = []
        errors: list[Mapping[str, Any]] = []
        for child in expected:
            try:
                inside = _strict_bool(
                    self._read_inside(child, destination),
                    source="official Inside readback",
                )
                readbacks.append((child, inside))
            except Exception as exc:
                readbacks.append((child, None))
                errors.append(_error("final_official_inside_readback", exc, child=child))
        if errors:
            self._failed_status = "official_inside_unreadable"
            return ClosedDestinationRecoveryFinalOutcome(
                destination_scope_name=destination,
                operational=False,
                success=False,
                status="official_inside_unreadable",
                expected_child_scope_names=expected,
                placed_child_scope_names=placed,
                destination_open=False,
                inside_readbacks=tuple(readbacks),
                errors=tuple(errors),
            )
        if not all(inside is True for _, inside in readbacks):
            self._failed_status = "official_inside_regressed"
            return ClosedDestinationRecoveryFinalOutcome(
                destination_scope_name=destination,
                operational=True,
                success=False,
                status="official_inside_regressed",
                expected_child_scope_names=expected,
                placed_child_scope_names=placed,
                destination_open=False,
                inside_readbacks=tuple(readbacks),
            )
        return ClosedDestinationRecoveryFinalOutcome(
            destination_scope_name=destination,
            operational=True,
            success=True,
            status="official_inside_and_closed_confirmed",
            expected_child_scope_names=expected,
            placed_child_scope_names=placed,
            destination_open=False,
            inside_readbacks=tuple(readbacks),
        )

    def to_report(self) -> dict[str, Any]:
        report = _base_report()
        report.update(
            {
                "destination_scope_name_private": self.destination_scope_name,
                "expected_child_scope_names_private": list(
                    self.expected_child_scope_names
                ),
                "placed_child_scope_names_private": list(
                    self.placed_child_scope_names
                ),
                "failed": self.failed,
                "failed_status": self._failed_status,
                "authorization_private": self._decision.to_report(),
                "placement_outcomes_private": [
                    self._outcomes[child].to_report()
                    for child in self.expected_child_scope_names
                    if child in self._outcomes
                ],
            }
        )
        return report
