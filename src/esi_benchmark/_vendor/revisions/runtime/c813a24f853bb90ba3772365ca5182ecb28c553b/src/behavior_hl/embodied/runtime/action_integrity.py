"""Sticky world-integrity faults shared by execution and final scoring."""

from typing import Any, Mapping


class ActionIntegrityError(RuntimeError):
    """A mutation could not be restored; normal execution/scoring must stop."""

    def __init__(self, message: str, *, private_report: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.private_report = dict(private_report or {})


def _task(state: Any) -> Any:
    env = getattr(state, "env", state)
    return getattr(env, "task", env)


def execution_integrity_failure(state: Any) -> str | None:
    return getattr(_task(state), "_action20_integrity_failure", None)


def invalidate_execution(state: Any, error: BaseException) -> None:
    task = _task(state)
    if task is None:
        raise ActionIntegrityError("cannot record world-integrity failure without a task") from error
    if execution_integrity_failure(task) is None:
        task._action20_integrity_failure = f"{type(error).__name__}: {error}"


def require_execution_integrity(state: Any) -> None:
    failure = execution_integrity_failure(state)
    if failure is not None:
        raise ActionIntegrityError(f"world integrity is invalid: {failure}")
