"""Pure contract for private visual evidence at high-level action boundaries.

This module deliberately has no simulator, renderer, image, or artifact-writer
imports.  It describes *when* a caller must capture existing rendered state;
it never advances physics and never changes the public observation stream.
Predicate truth, when supplied, remains explicitly private metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence


KEY_ACTION_VISUAL_VERSION = "behavior2026.private-key-action-visuals/v1"
KEY_ACTION_PHASES = ("before", "state_changed", "settle_final")
NAVIGATION_PHASES = ("after",)

_GRASP_OPERATIONS = frozenset({"grasp", "pick_up"})
_OPEN_CLOSE_OPERATIONS = frozenset({"open", "close"})
_NAVIGATION_OPERATIONS = frozenset(
    {
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
        "turn_left",
        "turn_right",
        "navigate",
        "navigation",
    }
)


def _non_negative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _normalized_operation(operation: object) -> str:
    if not isinstance(operation, str) or not operation.strip():
        raise ValueError("operation must be a non-empty string")
    normalized = operation.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized != operation:
        raise ValueError("operation must already be normalized lowercase snake_case")
    return normalized


def visual_action_family(operation: str, *, recovery: bool = False) -> str:
    """Classify a generic action without task, object, or asset identities."""

    operation = _normalized_operation(operation)
    if not isinstance(recovery, bool):
        raise ValueError("recovery must be Boolean")
    if recovery or operation == "recovery" or operation.startswith("recovery_"):
        return "recovery"
    if operation in _GRASP_OPERATIONS:
        return "grasp"
    if operation == "place" or operation.startswith("place_"):
        return "place"
    if operation in _OPEN_CLOSE_OPERATIONS:
        return operation
    if operation in _NAVIGATION_OPERATIONS or operation.startswith("navigate_"):
        return "navigation"
    raise ValueError(f"unsupported visual action operation: {operation!r}")


def required_visual_phases(
    operation: str,
    *,
    recovery: bool = False,
) -> tuple[str, ...]:
    """Return the complete ordered phase requirement for one action."""

    family = visual_action_family(operation, recovery=recovery)
    return NAVIGATION_PHASES if family == "navigation" else KEY_ACTION_PHASES


def _private_predicate_truth(
    value: Mapping[str, bool] | None,
) -> Mapping[str, bool] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("predicate_truth_private must be a mapping")
    converted: dict[str, bool] = {}
    for leaf_id, truth in value.items():
        if not isinstance(leaf_id, str) or not leaf_id:
            raise ValueError("private predicate leaf IDs must be non-empty strings")
        if not isinstance(truth, bool):
            raise ValueError("private predicate truth values must be Boolean")
        converted[leaf_id] = truth
    return MappingProxyType(converted)


@dataclass(frozen=True, slots=True)
class PrivateKeyActionVisual:
    """One render-only capture request and its privacy/simulation contract."""

    action_index: int
    operation: str
    phase: str
    simulator_step: int
    action_family: str
    predicate_truth_private: Mapping[str, bool] | None = None
    schema_version: str = KEY_ACTION_VISUAL_VERSION
    render_only: bool = True
    additional_physics_steps: int = 0
    simulator_state_modified: bool = False
    public_observation_modified: bool = False

    def __post_init__(self) -> None:
        _non_negative_integer(self.action_index, field="action_index")
        _non_negative_integer(self.simulator_step, field="simulator_step")
        operation = _normalized_operation(self.operation)
        family = visual_action_family(
            operation,
            recovery=self.action_family == "recovery",
        )
        if family != self.action_family:
            raise ValueError("action_family does not match operation")
        phases = NAVIGATION_PHASES if family == "navigation" else KEY_ACTION_PHASES
        if self.phase not in phases:
            raise ValueError(
                f"phase {self.phase!r} is not allowed for {family} visual evidence"
            )
        if self.schema_version != KEY_ACTION_VISUAL_VERSION:
            raise ValueError("key-action visual schema version mismatch")
        if self.render_only is not True:
            raise ValueError("key-action visual capture must be render-only")
        if self.additional_physics_steps != 0:
            raise ValueError("key-action visual capture cannot add physics steps")
        if self.simulator_state_modified is not False:
            raise ValueError("key-action visual capture cannot modify simulator state")
        if self.public_observation_modified is not False:
            raise ValueError("private visual capture cannot modify public observation")
        object.__setattr__(
            self,
            "predicate_truth_private",
            _private_predicate_truth(self.predicate_truth_private),
        )

    def to_private_record(self) -> dict[str, object]:
        """Return strict-JSON metadata for the private visual-frame index."""

        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "action_index": self.action_index,
            "operation": self.operation,
            "action_family": self.action_family,
            "phase": self.phase,
            "simulator_step": self.simulator_step,
            "render_only": self.render_only,
            "additional_physics_steps": self.additional_physics_steps,
            "simulator_state_modified": self.simulator_state_modified,
            "public_observation_modified": self.public_observation_modified,
        }
        if self.predicate_truth_private is not None:
            record["predicate_truth_private"] = dict(
                self.predicate_truth_private
            )
        return record


def build_private_key_action_visual_plan(
    *,
    action_index: int,
    operation: str,
    phase_simulator_steps: Mapping[str, int],
    recovery: bool = False,
    predicate_truth_by_phase_private: (
        Mapping[str, Mapping[str, bool] | None] | None
    ) = None,
) -> tuple[PrivateKeyActionVisual, ...]:
    """Build one complete, ordered, render-only action evidence plan.

    The caller supplies already-observed simulator-step counters.  This
    function neither manufactures counters nor requests a physics step.
    """

    action_index = _non_negative_integer(action_index, field="action_index")
    operation = _normalized_operation(operation)
    family = visual_action_family(operation, recovery=recovery)
    phases = NAVIGATION_PHASES if family == "navigation" else KEY_ACTION_PHASES
    if not isinstance(phase_simulator_steps, Mapping):
        raise TypeError("phase_simulator_steps must be a mapping")
    if set(phase_simulator_steps) != set(phases):
        raise ValueError(
            "phase_simulator_steps must contain exactly " + repr(phases)
        )
    truths = predicate_truth_by_phase_private or {}
    if not isinstance(truths, Mapping):
        raise TypeError("predicate_truth_by_phase_private must be a mapping")
    if not set(truths).issubset(phases):
        raise ValueError("predicate truth is attached to an unsupported phase")

    simulator_steps = tuple(
        _non_negative_integer(phase_simulator_steps[phase], field="simulator_step")
        for phase in phases
    )
    if any(right < left for left, right in zip(simulator_steps, simulator_steps[1:])):
        raise ValueError("visual phase simulator steps cannot regress")
    return tuple(
        PrivateKeyActionVisual(
            action_index=action_index,
            operation=operation,
            action_family=family,
            phase=phase,
            simulator_step=simulator_step,
            predicate_truth_private=truths.get(phase),
        )
        for phase, simulator_step in zip(phases, simulator_steps, strict=True)
    )


def validate_private_key_action_visual_sequence(
    records: Sequence[PrivateKeyActionVisual],
) -> None:
    """Fail closed unless records contain one complete action contract."""

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("visual records must be a sequence")
    if not records:
        raise ValueError("visual evidence sequence cannot be empty")
    first = records[0]
    if not isinstance(first, PrivateKeyActionVisual) or any(
        not isinstance(record, PrivateKeyActionVisual) for record in records
    ):
        raise TypeError("visual evidence sequence contains an invalid record")
    if any(
        record.action_index != first.action_index
        or record.operation != first.operation
        or record.action_family != first.action_family
        for record in records
    ):
        raise ValueError("visual evidence records must describe one action")
    expected = (
        NAVIGATION_PHASES
        if first.action_family == "navigation"
        else KEY_ACTION_PHASES
    )
    if tuple(record.phase for record in records) != expected:
        raise ValueError("visual evidence phases are incomplete or out of order")
    steps = tuple(record.simulator_step for record in records)
    if any(right < left for left, right in zip(steps, steps[1:])):
        raise ValueError("visual evidence simulator steps regress")


__all__ = [
    "KEY_ACTION_PHASES",
    "KEY_ACTION_VISUAL_VERSION",
    "NAVIGATION_PHASES",
    "PrivateKeyActionVisual",
    "build_private_key_action_visual_plan",
    "required_visual_phases",
    "validate_private_key_action_visual_sequence",
    "visual_action_family",
]
