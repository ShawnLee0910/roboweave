"""Fail-closed operations for objects inside openable containers.

The policy in this module is deliberately simulator-independent.  Its
callbacks must read the live official ``Inside`` and ``Open`` states and must
execute the ordinary symbolic OPEN / CLOSE primitive.  The policy never calls
``set_value`` and never substitutes a cached or task-authored truth value for
an official readback.

There are two separate lifetimes:

* every target interaction first opens its closed containment ancestors in
  outer-to-inner order; and
* containers opened only to obtain access remain registered until
  :meth:`NestedContainerOperationPolicy.close_dependencies`, which closes them
  in inner-to-outer order.

OPEN and CLOSE are idempotent.  A matching initial official state is a success
without invoking the executor.  Otherwise the executor and official readback
are retried only within the explicit finite :class:`OpenCloseRetryPolicy`.
Expected runtime failures are returned as immutable outcomes rather than
escaping as exceptions, so the runner can persist them before failing closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping


NESTED_CONTAINER_OPERATION_POLICY_VERSION = (
    "behavior2026.nested-container-operations/v1"
)


ReadOpen = Callable[[str], bool]
ExecuteOpenClose = Callable[[str, str], Any]
IsOpenable = Callable[[str], bool]
IsInside = Callable[[str, str], bool]
BetweenReadbacks = Callable[[str, str, int, int], None]


def _strict_bool(value: Any, *, source: str) -> bool:
    """Accept Python / NumPy booleans, but reject truthy surrogate values."""

    if isinstance(value, bool):
        return value
    # Avoid importing NumPy into this pure policy module.  OmniGibson state
    # implementations can return np.bool_; ``item`` gives a Python bool.
    if type(value).__name__ == "bool_":
        item = getattr(value, "item", None)
        if callable(item):
            scalar = item()
            if isinstance(scalar, bool):
                return scalar
    raise TypeError(f"{source} must return a boolean")


def _error(stage: str, exc: Exception, **context: Any) -> dict[str, Any]:
    return {
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        **context,
    }


def _report_value(value: Any) -> Any:
    """Keep callback telemetry JSON-friendly without trusting its semantics."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _report_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_report_value(item) for item in value]
    return repr(value)


@dataclass(frozen=True, slots=True)
class OpenCloseRetryPolicy:
    """Finite action and readback budgets for one OPEN / CLOSE transition."""

    max_action_attempts: int = 2
    max_readbacks_per_attempt: int = 2

    def __post_init__(self) -> None:
        if self.max_action_attempts < 1:
            raise ValueError("max_action_attempts must be positive")
        if self.max_readbacks_per_attempt < 1:
            raise ValueError("max_readbacks_per_attempt must be positive")


@dataclass(frozen=True, slots=True)
class OpenCloseAttempt:
    action_attempt: int
    executor_status: str
    executor_report: Any
    readbacks: tuple[bool | None, ...]
    errors: tuple[Mapping[str, Any], ...]

    def to_report(self) -> dict[str, Any]:
        return {
            "action_attempt": self.action_attempt,
            "executor_status": self.executor_status,
            "executor_report": _report_value(self.executor_report),
            "readbacks": list(self.readbacks),
            "errors": [dict(item) for item in self.errors],
        }


@dataclass(frozen=True, slots=True)
class OpenCloseOutcome:
    target_scope_name: str
    operation: str
    desired_open: bool
    operational: bool
    success: bool
    status: str
    state_before: bool | None
    state_after: bool | None
    attempts: tuple[OpenCloseAttempt, ...] = ()
    errors: tuple[Mapping[str, Any], ...] = ()

    @property
    def action_attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def readback_count(self) -> int:
        return sum(len(attempt.readbacks) for attempt in self.attempts)

    def to_report(self) -> dict[str, Any]:
        return {
            "schema_version": NESTED_CONTAINER_OPERATION_POLICY_VERSION,
            "target_scope_name_private": self.target_scope_name,
            "operation": self.operation,
            "desired_open": self.desired_open,
            "operational": self.operational,
            "success": self.success,
            "status": self.status,
            "state_before_private": self.state_before,
            "state_after_private": self.state_after,
            "action_attempt_count": self.action_attempt_count,
            "readback_count": self.readback_count,
            "attempts": [attempt.to_report() for attempt in self.attempts],
            "errors": [dict(item) for item in self.errors],
            "official_state_write_by_policy": False,
        }


def execute_idempotent_open_close(
    *,
    target_scope_name: str,
    desired_open: bool,
    read_open: ReadOpen,
    execute_open_close: ExecuteOpenClose,
    retry_policy: OpenCloseRetryPolicy = OpenCloseRetryPolicy(),
    between_readbacks: BetweenReadbacks | None = None,
) -> OpenCloseOutcome:
    """Reach one official Open state with bounded execution and readback.

    An executor exception does not by itself decide the outcome: the official
    state is still read.  This handles a primitive that reports a stale
    postcondition while accepting success only when live official readback
    independently confirms it.
    """

    if not isinstance(target_scope_name, str) or not target_scope_name:
        raise ValueError("target_scope_name must be a non-empty string")
    if not isinstance(desired_open, bool):
        raise ValueError("desired_open must be a boolean")
    operation = "open" if desired_open else "close"
    try:
        before = _strict_bool(
            read_open(target_scope_name),
            source="official Open readback",
        )
    except Exception as exc:
        error = _error("initial_official_readback", exc)
        return OpenCloseOutcome(
            target_scope_name=target_scope_name,
            operation=operation,
            desired_open=desired_open,
            operational=False,
            success=False,
            status="official_state_unreadable",
            state_before=None,
            state_after=None,
            errors=(error,),
        )
    if before is desired_open:
        return OpenCloseOutcome(
            target_scope_name=target_scope_name,
            operation=operation,
            desired_open=desired_open,
            operational=True,
            success=True,
            status="already_satisfied",
            state_before=before,
            state_after=before,
        )

    attempts: list[OpenCloseAttempt] = []
    all_errors: list[Mapping[str, Any]] = []
    last_valid_state: bool | None = before
    any_valid_post_readback = False
    for action_attempt in range(1, retry_policy.max_action_attempts + 1):
        executor_report: Any = None
        attempt_errors: list[Mapping[str, Any]] = []
        try:
            executor_report = execute_open_close(target_scope_name, operation)
            executor_status = "returned"
        except Exception as exc:
            executor_status = "raised"
            error = _error(
                "executor",
                exc,
                action_attempt=action_attempt,
            )
            attempt_errors.append(error)
            all_errors.append(error)

        readbacks: list[bool | None] = []
        satisfied = False
        for readback_index in range(
            1, retry_policy.max_readbacks_per_attempt + 1
        ):
            try:
                observed = _strict_bool(
                    read_open(target_scope_name),
                    source="official Open readback",
                )
                any_valid_post_readback = True
                last_valid_state = observed
                readbacks.append(observed)
                if observed is desired_open:
                    satisfied = True
                    break
            except Exception as exc:
                readbacks.append(None)
                error = _error(
                    "official_readback",
                    exc,
                    action_attempt=action_attempt,
                    readback_index=readback_index,
                )
                attempt_errors.append(error)
                all_errors.append(error)
            if (
                readback_index < retry_policy.max_readbacks_per_attempt
                and between_readbacks is not None
            ):
                try:
                    between_readbacks(
                        target_scope_name,
                        operation,
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
            OpenCloseAttempt(
                action_attempt=action_attempt,
                executor_status=executor_status,
                executor_report=executor_report,
                readbacks=tuple(readbacks),
                errors=tuple(attempt_errors),
            )
        )
        if satisfied:
            return OpenCloseOutcome(
                target_scope_name=target_scope_name,
                operation=operation,
                desired_open=desired_open,
                operational=True,
                success=True,
                status="satisfied_after_execution",
                state_before=before,
                state_after=last_valid_state,
                attempts=tuple(attempts),
                errors=tuple(all_errors),
            )

    executor_always_raised = all(
        attempt.executor_status == "raised" for attempt in attempts
    )
    return OpenCloseOutcome(
        target_scope_name=target_scope_name,
        operation=operation,
        desired_open=desired_open,
        operational=any_valid_post_readback,
        success=False,
        status=(
            "executor_failed_and_postcondition_unsatisfied"
            if executor_always_raised and any_valid_post_readback
            else (
                "official_state_unreadable"
                if not any_valid_post_readback
                else "official_postcondition_unsatisfied"
            )
        ),
        state_before=before,
        state_after=last_valid_state,
        attempts=tuple(attempts),
        errors=tuple(all_errors),
    )


@dataclass(frozen=True, slots=True)
class ContainerAncestryOutcome:
    child_scope_name: str
    operational: bool
    success: bool
    status: str
    outer_to_inner: tuple[str, ...] = ()
    errors: tuple[Mapping[str, Any], ...] = ()

    def to_report(self) -> dict[str, Any]:
        return {
            "schema_version": NESTED_CONTAINER_OPERATION_POLICY_VERSION,
            "child_scope_name_private": self.child_scope_name,
            "operational": self.operational,
            "success": self.success,
            "status": self.status,
            "outer_to_inner_private": list(self.outer_to_inner),
            "errors": [dict(item) for item in self.errors],
        }


class _ContainmentCycle(RuntimeError):
    pass


def resolve_openable_container_ancestry(
    *,
    scope_names: Iterable[str],
    child_scope_name: str,
    is_openable: IsOpenable,
    is_inside: IsInside,
) -> ContainerAncestryOutcome:
    """Resolve all openable ancestors in deterministic outer-to-inner order."""

    names = tuple(scope_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("scope names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("scope names must be unique")
    ordered_names = tuple(sorted(names))
    if child_scope_name not in set(ordered_names):
        return ContainerAncestryOutcome(
            child_scope_name=child_scope_name,
            operational=False,
            success=False,
            status="target_unbound",
            errors=(
                {
                    "stage": "ancestry",
                    "error_type": "KeyError",
                    "error": f"scope target {child_scope_name!r} is unbound",
                },
            ),
        )
    try:
        openable = tuple(
            name
            for name in ordered_names
            if _strict_bool(
                is_openable(name),
                source="official openable probe",
            )
        )
        inside_cache: dict[tuple[str, str], bool] = {}

        def inside(child: str, parent: str) -> bool:
            key = (child, parent)
            if key not in inside_cache:
                inside_cache[key] = _strict_bool(
                    is_inside(child, parent),
                    source="official Inside readback",
                )
            return inside_cache[key]

        ordered: list[str] = []
        visited: set[str] = set()
        active: set[str] = set()

        def visit(current: str) -> None:
            if current in active:
                raise _ContainmentCycle(
                    "live openable containment ancestry contains a cycle"
                )
            if current in visited:
                return
            active.add(current)
            for parent in openable:
                if parent == current or not inside(current, parent):
                    continue
                visit(parent)
                if parent not in ordered:
                    ordered.append(parent)
            active.remove(current)
            visited.add(current)

        visit(child_scope_name)
    except _ContainmentCycle as exc:
        return ContainerAncestryOutcome(
            child_scope_name=child_scope_name,
            operational=False,
            success=False,
            status="containment_cycle",
            errors=(_error("ancestry", exc),),
        )
    except Exception as exc:
        return ContainerAncestryOutcome(
            child_scope_name=child_scope_name,
            operational=False,
            success=False,
            status="official_ancestry_unreadable",
            errors=(_error("ancestry", exc),),
        )
    return ContainerAncestryOutcome(
        child_scope_name=child_scope_name,
        operational=True,
        success=True,
        status="resolved",
        outer_to_inner=tuple(ordered),
    )


@dataclass(frozen=True, slots=True)
class ContainerAccessOutcome:
    target_scope_name: str
    operational: bool
    success: bool
    status: str
    ancestry: ContainerAncestryOutcome
    dependency_outcomes: tuple[OpenCloseOutcome, ...] = ()
    opened_now: tuple[str, ...] = ()

    def to_report(self) -> dict[str, Any]:
        return {
            "schema_version": NESTED_CONTAINER_OPERATION_POLICY_VERSION,
            "target_scope_name_private": self.target_scope_name,
            "operational": self.operational,
            "success": self.success,
            "status": self.status,
            "ancestry_private": self.ancestry.to_report(),
            "dependency_outcomes_private": [
                outcome.to_report() for outcome in self.dependency_outcomes
            ],
            "opened_now_private": list(self.opened_now),
        }


@dataclass(frozen=True, slots=True)
class NestedTransitionOutcome:
    target_scope_name: str
    operation: str
    operational: bool
    success: bool
    status: str
    access: ContainerAccessOutcome
    target_outcome: OpenCloseOutcome | None

    def to_report(self) -> dict[str, Any]:
        return {
            "schema_version": NESTED_CONTAINER_OPERATION_POLICY_VERSION,
            "target_scope_name_private": self.target_scope_name,
            "operation": self.operation,
            "operational": self.operational,
            "success": self.success,
            "status": self.status,
            "access_private": self.access.to_report(),
            "target_outcome_private": (
                None
                if self.target_outcome is None
                else self.target_outcome.to_report()
            ),
        }


@dataclass(frozen=True, slots=True)
class NestedInteractionOutcome:
    target_scope_name: str
    operation: str
    operational: bool
    success: bool
    status: str
    access: ContainerAccessOutcome
    executor_report: Any = None
    errors: tuple[Mapping[str, Any], ...] = ()

    def to_report(self) -> dict[str, Any]:
        return {
            "schema_version": NESTED_CONTAINER_OPERATION_POLICY_VERSION,
            "target_scope_name_private": self.target_scope_name,
            "operation": self.operation,
            "operational": self.operational,
            "success": self.success,
            "status": self.status,
            "access_private": self.access.to_report(),
            "executor_report_private": _report_value(self.executor_report),
            "errors": [dict(item) for item in self.errors],
        }


@dataclass(frozen=True, slots=True)
class DependencyCloseOutcome:
    operational: bool
    success: bool
    status: str
    close_order: tuple[str, ...]
    outcomes: tuple[OpenCloseOutcome, ...]
    remaining_dependencies: tuple[str, ...]

    def to_report(self) -> dict[str, Any]:
        return {
            "schema_version": NESTED_CONTAINER_OPERATION_POLICY_VERSION,
            "operational": self.operational,
            "success": self.success,
            "status": self.status,
            "close_order_private": list(self.close_order),
            "outcomes_private": [
                outcome.to_report() for outcome in self.outcomes
            ],
            "remaining_dependencies_private": list(
                self.remaining_dependencies
            ),
        }


class NestedContainerOperationPolicy:
    """One bounded, auditable access session over an official task scope."""

    def __init__(
        self,
        *,
        scope_names: Iterable[str],
        is_openable: IsOpenable,
        is_inside: IsInside,
        read_open: ReadOpen,
        execute_open_close: ExecuteOpenClose,
        retry_policy: OpenCloseRetryPolicy = OpenCloseRetryPolicy(),
        between_readbacks: BetweenReadbacks | None = None,
    ) -> None:
        names = tuple(scope_names)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("scope names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("scope names must be unique")
        self._scope_names = names
        self._is_openable = is_openable
        self._is_inside = is_inside
        self._read_open = read_open
        self._execute_open_close = execute_open_close
        self._retry_policy = retry_policy
        self._between_readbacks = between_readbacks
        self._opened_dependencies: list[str] = []

    @property
    def opened_dependencies(self) -> tuple[str, ...]:
        """Return policy-owned openings in acquisition order (outer first)."""

        return tuple(self._opened_dependencies)

    def _transition_direct(
        self,
        target_scope_name: str,
        *,
        desired_open: bool,
    ) -> OpenCloseOutcome:
        return execute_idempotent_open_close(
            target_scope_name=target_scope_name,
            desired_open=desired_open,
            read_open=self._read_open,
            execute_open_close=self._execute_open_close,
            retry_policy=self._retry_policy,
            between_readbacks=self._between_readbacks,
        )

    def _remember_opened_dependency(self, scope_name: str) -> None:
        if scope_name not in self._opened_dependencies:
            self._opened_dependencies.append(scope_name)

    def _forget_closed_dependency(self, scope_name: str) -> None:
        if scope_name in self._opened_dependencies:
            self._opened_dependencies.remove(scope_name)

    def ensure_access(self, target_scope_name: str) -> ContainerAccessOutcome:
        """Open every closed ancestor before an interaction with ``target``."""

        ancestry = resolve_openable_container_ancestry(
            scope_names=self._scope_names,
            child_scope_name=target_scope_name,
            is_openable=self._is_openable,
            is_inside=self._is_inside,
        )
        if not ancestry.success:
            return ContainerAccessOutcome(
                target_scope_name=target_scope_name,
                operational=False,
                success=False,
                status="ancestry_failed",
                ancestry=ancestry,
            )
        outcomes: list[OpenCloseOutcome] = []
        opened_now: list[str] = []
        for dependency in ancestry.outer_to_inner:
            outcome = self._transition_direct(
                dependency,
                desired_open=True,
            )
            outcomes.append(outcome)
            if outcome.success and outcome.status == "satisfied_after_execution":
                self._remember_opened_dependency(dependency)
                opened_now.append(dependency)
            if not outcome.success:
                return ContainerAccessOutcome(
                    target_scope_name=target_scope_name,
                    operational=outcome.operational,
                    success=False,
                    status="dependency_open_failed",
                    ancestry=ancestry,
                    dependency_outcomes=tuple(outcomes),
                    opened_now=tuple(opened_now),
                )
        return ContainerAccessOutcome(
            target_scope_name=target_scope_name,
            operational=True,
            success=True,
            status="accessible",
            ancestry=ancestry,
            dependency_outcomes=tuple(outcomes),
            opened_now=tuple(opened_now),
        )

    def transition(
        self,
        target_scope_name: str,
        *,
        desired_open: bool,
        restore_open_on_finalize: bool = False,
    ) -> NestedTransitionOutcome:
        """Prepare access, then idempotently OPEN or CLOSE the target."""

        access = self.ensure_access(target_scope_name)
        operation = "open" if desired_open else "close"
        if not access.success:
            return NestedTransitionOutcome(
                target_scope_name=target_scope_name,
                operation=operation,
                operational=access.operational,
                success=False,
                status="access_failed",
                access=access,
                target_outcome=None,
            )
        target_outcome = self._transition_direct(
            target_scope_name,
            desired_open=desired_open,
        )
        if target_outcome.success:
            if (
                desired_open
                and restore_open_on_finalize
                and target_outcome.status == "satisfied_after_execution"
            ):
                self._remember_opened_dependency(target_scope_name)
            elif not desired_open:
                self._forget_closed_dependency(target_scope_name)
        return NestedTransitionOutcome(
            target_scope_name=target_scope_name,
            operation=operation,
            operational=target_outcome.operational,
            success=target_outcome.success,
            status=("completed" if target_outcome.success else "target_failed"),
            access=access,
            target_outcome=target_outcome,
        )

    def perform_interaction(
        self,
        target_scope_name: str,
        *,
        operation: str,
        execute: Callable[[str, str], Any],
    ) -> NestedInteractionOutcome:
        """Prepare ancestor access, then perform one non-Open/Close action."""

        if not isinstance(operation, str) or not operation:
            raise ValueError("operation must be a non-empty string")
        if operation.lower() in {"open", "close"}:
            raise ValueError("Open/Close interactions must use transition()")
        access = self.ensure_access(target_scope_name)
        if not access.success:
            return NestedInteractionOutcome(
                target_scope_name=target_scope_name,
                operation=operation,
                operational=access.operational,
                success=False,
                status="access_failed",
                access=access,
            )
        try:
            report = execute(target_scope_name, operation)
        except Exception as exc:
            return NestedInteractionOutcome(
                target_scope_name=target_scope_name,
                operation=operation,
                operational=False,
                success=False,
                status="target_executor_failed",
                access=access,
                errors=(_error("target_executor", exc),),
            )
        return NestedInteractionOutcome(
            target_scope_name=target_scope_name,
            operation=operation,
            operational=True,
            success=True,
            status="completed",
            access=access,
            executor_report=report,
        )

    def close_dependencies(self) -> DependencyCloseOutcome:
        """Close policy-owned dependencies inner-to-outer, stopping on failure.

        Successful entries are removed immediately.  If a close fails, the
        failed container and every outer dependency remain registered, making
        a later bounded retry safe and preserving access for diagnosis.
        """

        requested_order = tuple(reversed(self._opened_dependencies))
        outcomes: list[OpenCloseOutcome] = []
        closed_order: list[str] = []
        while self._opened_dependencies:
            target = self._opened_dependencies[-1]
            outcome = self._transition_direct(target, desired_open=False)
            outcomes.append(outcome)
            if not outcome.success:
                return DependencyCloseOutcome(
                    operational=outcome.operational,
                    success=False,
                    status="dependency_close_failed",
                    close_order=tuple(closed_order),
                    outcomes=tuple(outcomes),
                    remaining_dependencies=tuple(self._opened_dependencies),
                )
            closed_order.append(target)
            self._opened_dependencies.pop()
        return DependencyCloseOutcome(
            operational=True,
            success=True,
            status=(
                "already_finalized" if not requested_order else "completed"
            ),
            close_order=tuple(closed_order),
            outcomes=tuple(outcomes),
            remaining_dependencies=(),
        )
