"""Frozen public/private profiles for B100 toggle-state tasks.

Only the natural-language instruction is public.  Target synsets, counts,
scene identity, and live object handles remain on the Oracle/verifier side.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


TOGGLE_TASK_PROFILE_VERSION = "behavior2026.toggle-task-profile/v1"


@dataclass(frozen=True, slots=True)
class ToggleTaskProfile:
    task_name: str
    task_id: int
    task_language: str
    scene_model: str
    split: str
    instance_id: int
    target_synsets: frozenset[str]
    expected_target_count: int
    expected_goal_leaf_count: int
    desired_toggled_on: bool
    schema_version: str = TOGGLE_TASK_PROFILE_VERSION

    def __post_init__(self) -> None:
        for name in ("task_name", "task_language", "scene_model", "split"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "task_id",
            "instance_id",
            "expected_target_count",
            "expected_goal_leaf_count",
        ):
            value = getattr(self, name)
            minimum = 0 if name == "instance_id" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        targets = frozenset(self.target_synsets)
        if not targets or any(not isinstance(value, str) or not value for value in targets):
            raise ValueError("target_synsets must contain non-empty strings")
        object.__setattr__(self, "target_synsets", targets)
        if self.expected_target_count != self.expected_goal_leaf_count:
            raise ValueError("first toggle profile requires one goal leaf per target")
        if not isinstance(self.desired_toggled_on, bool):
            raise TypeError("desired_toggled_on must be boolean")
        if self.schema_version != TOGGLE_TASK_PROFILE_VERSION:
            raise ValueError("unexpected toggle task profile version")

    def public_record(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {"task_name": self.task_name, "task_language": self.task_language}
        )

    def private_manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema_version": self.schema_version,
                "task_name": self.task_name,
                "task_id": self.task_id,
                "scene_model": self.scene_model,
                "split": self.split,
                "instance_id": self.instance_id,
                "target_synsets": sorted(self.target_synsets),
                "expected_target_count": self.expected_target_count,
                "expected_goal_leaf_count": self.expected_goal_leaf_count,
                "desired_toggled_on": self.desired_toggled_on,
            }
        )


TURNING_OUT_ALL_LIGHTS_BEFORE_SLEEP = ToggleTaskProfile(
    task_name="turning_out_all_lights_before_sleep",
    task_id=66,
    task_language=(
        "Before going to sleep, turn off every wall light switch and every "
        "table lamp in the home."
    ),
    scene_model="house_single_floor",
    split="train",
    instance_id=1,
    target_synsets=frozenset({"switch.n.01", "table_lamp.n.01"}),
    expected_target_count=5,
    expected_goal_leaf_count=5,
    desired_toggled_on=False,
)


def get_toggle_task_profile(task_name: str) -> ToggleTaskProfile:
    if task_name != TURNING_OUT_ALL_LIGHTS_BEFORE_SLEEP.task_name:
        raise KeyError(f"unregistered toggle task: {task_name!r}")
    return TURNING_OUT_ALL_LIGHTS_BEFORE_SLEEP


def _scope_synset(scope_name: str) -> str:
    head, separator, suffix = scope_name.rpartition("_")
    if not separator or not suffix.isdigit() or not head:
        raise RuntimeError(f"invalid official object_scope identifier: {scope_name!r}")
    return head


def private_toggle_task_targets(
    evaluator: Any,
    profile: ToggleTaskProfile,
) -> Mapping[str, Any]:
    """Return live target objects keyed by simulator object name."""

    task = getattr(getattr(evaluator, "env", None), "task", None)
    if task is None:
        raise RuntimeError("loaded task is unavailable")
    observed = (
        getattr(task, "activity_name", None),
        getattr(task, "scene_name", None),
    )
    expected = (profile.task_name, profile.scene_model)
    if observed != expected:
        raise RuntimeError(
            f"loaded task/scene differs from toggle profile: {observed} != {expected}"
        )
    scope = getattr(task, "object_scope", None)
    if not isinstance(scope, Mapping):
        raise RuntimeError("official object_scope is unavailable")
    targets: dict[str, Any] = {}
    for scope_name, entity in scope.items():
        if _scope_synset(str(scope_name)) not in profile.target_synsets:
            continue
        if entity is None:
            raise RuntimeError(f"toggle target {scope_name!r} is unbound")
        object_name = str(getattr(entity, "name", ""))
        if not object_name or object_name in targets:
            raise RuntimeError("toggle target has an invalid or duplicate live name")
        targets[object_name] = entity
    if len(targets) != profile.expected_target_count:
        raise RuntimeError(
            "toggle target count drifted: "
            f"expected {profile.expected_target_count}, got {len(targets)}"
        )
    return MappingProxyType(targets)


def private_toggle_task_roles(
    evaluator: Any,
    profile: ToggleTaskProfile,
) -> tuple[frozenset[str], frozenset[str]]:
    """Expose no grasp/container roles while retaining marker-runtime shape."""

    names = frozenset(private_toggle_task_targets(evaluator, profile))
    return names, frozenset()
