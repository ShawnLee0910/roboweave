"""Private canonical postcondition verifier for Action23 operations.

This module deliberately has no OmniGibson imports.  The resident worker owns
the only simulator-facing callbacks and keeps this report out of public
observations and prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from behavior_hl.embodied.policy.action_specs.wait import WAIT_SIMULATOR_STEPS


ACTION23_OUTCOME_VERIFIER_VERSION = "behavior2026.action23-outcome/v4"

_RELATION_BY_ACTION = {
    "place_inside": "Inside",
    "place_on_floor": "OnTop",
    "place_on_top": "OnTop",
    "place_next_to": "NextTo",
    "place_under": "Under",
    "place_touching": "Touching",
}


@dataclass(frozen=True, slots=True)
class Action23OutcomeVerifier:
    """Read one action-specific canonical postcondition, fail closed."""

    read_state: Callable[[Any, str, Any | None], bool]
    read_held: Callable[[], Any | None]

    def __post_init__(self) -> None:
        if not callable(self.read_state) or not callable(self.read_held):
            raise TypeError("Action23 verifier callbacks must be callable")

    def verify(
        self,
        *,
        operation: str,
        target: Any | None = None,
        acted_object: Any | None = None,
        simulator_steps: int | None = None,
    ) -> Mapping[str, Any]:
        if not isinstance(operation, str) or not operation:
            raise ValueError("operation must be a non-empty string")
        expected: bool
        observed: bool
        predicate: str | None = None
        if operation == "pick":
            expected = True
            observed = target is not None and self.read_held() is target
        elif operation == "release":
            expected = True
            observed = self.read_held() is None
        elif operation in {"open", "close"}:
            if target is None:
                raise ValueError("open/close verification requires a target")
            predicate = "Open"
            expected = operation == "open"
            observed = bool(self.read_state(target, predicate, None))
        elif operation in {"toggle_on", "toggle_off"}:
            if target is None:
                raise ValueError("toggle verification requires a target")
            predicate = "ToggledOn"
            expected = operation == "toggle_on"
            observed = bool(self.read_state(target, predicate, None))
        elif operation in _RELATION_BY_ACTION:
            if target is None or acted_object is None:
                raise ValueError("placement verification requires object and target")
            predicate = _RELATION_BY_ACTION[operation]
            expected = True
            observed = acted_object is not target and bool(
                self.read_state(acted_object, predicate, target)
            )
        elif operation == "wait":
            if isinstance(simulator_steps, bool) or not isinstance(
                simulator_steps, int
            ):
                raise ValueError("wait verification requires an integer step count")
            expected = True
            observed = simulator_steps == WAIT_SIMULATOR_STEPS
        else:
            raise ValueError(f"unsupported Action23 verifier operation {operation!r}")
        passed = observed is expected
        return {
            "verifier_version": ACTION23_OUTCOME_VERIFIER_VERSION,
            "operation": operation,
            "predicate": predicate,
            "expected": expected,
            "observed": observed,
            "semantic_state_pass": passed,
            "private": True,
        }
