"""Macro-step episode control independent of any simulator implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from behavior_hl.actions import HighLevelAction
from behavior_hl.tasks import TaskSpec


class ActionNotAllowed(ValueError):
    """Raised when an action is valid globally but outside the active task vocabulary."""


class EpisodeAlreadyDone(RuntimeError):
    """Raised when an action is submitted after termination."""


class EpisodeNotSuccessful(RuntimeError):
    """Raised when a gated episode is finalized before its BDDL goal is true."""


class HighLevelBackend(Protocol):
    def execute(self, action: HighLevelAction) -> tuple[int, dict[str, Any]]:
        """Execute one macro and return (simulator_steps, metadata)."""

    def success(self) -> bool:
        """Return current BDDL success."""

    def q_score(self) -> float:
        """Return current official-formula Q-score."""

    def goal_status(self) -> dict[str, Any]:
        """Return a serializable goal status."""


@dataclass(frozen=True, slots=True)
class EpisodeStep:
    action: dict[str, str]
    macro_step: int
    simulator_steps: int
    total_simulator_steps: int
    success: bool
    q_score: float
    terminated: bool
    reason: str | None = None
    goal_status: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class EpisodeController:
    """Runs a bounded high-level episode while leaving BDDL evaluation to the backend."""

    def __init__(
        self,
        task: TaskSpec,
        backend: HighLevelBackend,
        *,
        max_macro_steps: int = 200,
        max_simulator_steps: int = 100_000,
        terminate_on_success: bool = True,
    ) -> None:
        if max_macro_steps <= 0:
            raise ValueError("max_macro_steps must be positive")
        if max_simulator_steps <= 0:
            raise ValueError("max_simulator_steps must be positive")
        if not isinstance(terminate_on_success, bool):
            raise TypeError("terminate_on_success must be a bool")
        self.task = task
        self.backend = backend
        self.max_macro_steps = max_macro_steps
        self.max_simulator_steps = max_simulator_steps
        self.terminate_on_success = terminate_on_success
        self.macro_steps = 0
        self.simulator_steps = 0
        self.terminated = False

    def step(self, action: HighLevelAction) -> EpisodeStep:
        if self.terminated:
            raise EpisodeAlreadyDone("episode has already terminated")
        if action.name not in self.task.allowed_actions:
            raise ActionNotAllowed(f"{action.name.value!r} is not enabled for task {self.task.name!r}")

        step_simulator_steps = 0
        metadata: dict[str, Any] = {}
        reason: str | None = None
        self.macro_steps += 1

        if action.name.value == "stop":
            self.terminated = True
            reason = "model_stop"
        else:
            try:
                step_simulator_steps, metadata = self.backend.execute(action)
            except Exception as exc:
                consumed_steps = int(getattr(exc, "consumed_steps", 0))
                if consumed_steps < 0:
                    raise ValueError("backend exception reported negative consumed_steps") from exc
                self.simulator_steps += consumed_steps
                failure_reason = None
                if self.simulator_steps >= self.max_simulator_steps:
                    failure_reason = "simulator_timeout"
                elif self.macro_steps >= self.max_macro_steps:
                    failure_reason = "macro_timeout"
                if failure_reason is not None:
                    self.terminated = True
                    setattr(exc, "episode_terminated", True)
                    setattr(exc, "episode_reason", failure_reason)
                    details = getattr(exc, "details", None)
                    if isinstance(details, dict):
                        details["episode_terminated"] = True
                        details["episode_reason"] = failure_reason
                raise
            if step_simulator_steps < 0:
                raise ValueError("backend returned a negative simulator step count")
            self.simulator_steps += step_simulator_steps

        success = self.backend.success()
        if success:
            if self.terminate_on_success:
                self.terminated = True
                reason = "task_success"
            elif self.terminated:
                reason = "task_success_finalized"
            else:
                reason = "task_success_pending_finalization"
        elif self.simulator_steps >= self.max_simulator_steps:
            self.terminated = True
            reason = "simulator_timeout"
        elif self.macro_steps >= self.max_macro_steps:
            self.terminated = True
            reason = "macro_timeout"

        return EpisodeStep(
            action=action.to_dict(),
            macro_step=self.macro_steps,
            simulator_steps=step_simulator_steps,
            total_simulator_steps=self.simulator_steps,
            success=success,
            q_score=self.backend.q_score(),
            terminated=self.terminated,
            reason=reason,
            goal_status=self.backend.goal_status(),
            metadata=metadata,
        )

    def charge_simulator_steps(self, steps: int) -> str | None:
        """Charge evaluator-side intervention or settle steps to the episode budget."""

        if self.terminated:
            raise EpisodeAlreadyDone("episode has already terminated")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError("external simulator steps must be a non-negative integer")
        self.simulator_steps += steps
        if self.backend.success():
            return None
        if self.simulator_steps >= self.max_simulator_steps:
            self.terminated = True
            return "simulator_timeout"
        return None

    def finalize_success(self) -> None:
        """Terminate a gated episode without executing another action.

        This is used only after evaluator-side final checkpoints have been
        answered. It does not change task state, scores, or step counters.
        """

        if self.terminated:
            raise EpisodeAlreadyDone("episode has already terminated")
        if not self.backend.success():
            raise EpisodeNotSuccessful("cannot finalize before the BDDL goal is satisfied")
        self.terminated = True
