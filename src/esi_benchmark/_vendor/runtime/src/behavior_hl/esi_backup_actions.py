"""Restricted compatibility layer for a small, useful subset of ESI actions.

The original ESI active-exploration pipeline moves an OmniGibson viewer camera.
That is useful for visual QA, but it is not a valid robot navigation primitive:
the camera is not collision checked and two actions change its height directly.

This module intentionally has no OmniGibson imports.  It translates legacy ESI
action strings into the structured BEHAVIOR-HL protocol and predicts view/base
pose changes, but approves a base translation only after three caller-supplied
guards agree:

* the target belongs to the same floor;
* the target and local segment are traversable;
* the swept robot volume is collision-free.

The simulator adapter remains responsible for executing an approved structured
action and checking its physical postcondition.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class ESIBackupActionError(ValueError):
    """A legacy ESI action cannot be safely prepared for execution."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ESIBackupNavigationBlocked(ESIBackupActionError):
    """An otherwise valid base motion failed a navigation safety guard."""


class ActionDisposition(str, Enum):
    """How a prepared legacy action must be executed."""

    BASE_MOTION = "base_motion"
    VIEW_ONLY = "view_only"
    SEMANTIC_MACRO = "semantic_macro"
    TERMINATE = "terminate"


@dataclass(frozen=True, slots=True)
class BackupPose:
    """Robot base pose plus a view/head pitch, all in world coordinates.

    ``z`` belongs to the base.  ``pitch`` belongs only to the sensor head or
    view and must never be implemented by translating the base or viewer.
    """

    x: float
    y: float
    z: float
    yaw: float
    pitch: float = 0.0

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.z, self.yaw, self.pitch)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("pose values must all be finite")


@dataclass(frozen=True, slots=True)
class GuardDecision:
    """Optional richer result from a navigation guard."""

    allowed: bool
    reason: str = ""


PoseGuard = Callable[[BackupPose, BackupPose], bool | GuardDecision]
ObjectResolver = Callable[[str], str | None]


@dataclass(frozen=True, slots=True)
class NavigationGuards:
    """All three guards are mandatory; there is no permissive default."""

    same_floor: PoseGuard
    traversable: PoseGuard
    collision_free: PoseGuard

    def __post_init__(self) -> None:
        for name in ("same_floor", "traversable", "collision_free"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} guard must be callable")


@dataclass(frozen=True, slots=True)
class DoorCandidate:
    """Result of selecting the nearest segmented, openable door."""

    instance_id: str
    visible: bool
    openable: bool
    distance_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.instance_id, str) or not self.instance_id.strip():
            raise ValueError("door instance_id must be a non-empty string")
        if not math.isfinite(self.distance_m) or self.distance_m < 0:
            raise ValueError("door distance_m must be finite and non-negative")


DoorSelector = Callable[[bool], DoorCandidate | None]


@dataclass(frozen=True, slots=True)
class PreparedESIAction:
    """A checked ESI action ready for the appropriate downstream executor.

    ``core_action`` is a strict BEHAVIOR-HL action payload.  It is ``None`` for
    ``turn_up`` and ``turn_down`` because those are local sensor-head/view
    commands and deliberately are not robot-base actions.
    """

    original_action: str
    normalized_action: str
    disposition: ActionDisposition
    pose_before: BackupPose
    approved_pose: BackupPose
    core_action: dict[str, str] | None
    source_instance: str | None = None


_BASE_ACTIONS_TO_CORE = {
    "move_forward": "move_forward",
    "move_backward": "move_back",
    "move_back": "move_back",
    "move_left": "move_left",
    "move_right": "move_right",
    "turn_left": "turn_left",
    "turn_right": "turn_right",
}
_TRANSLATION_ACTIONS = {
    "move_forward",
    "move_backward",
    "move_back",
    "move_left",
    "move_right",
}
_VIEW_ACTIONS = {"turn_up", "turn_down"}
_FORBIDDEN_FREE_FLIGHT_ACTIONS = {"move_up", "move_down"}
_UNSAFE_OR_OUT_OF_SCOPE_PREFIXES = (
    "put back ",
    "place ",
    "fill ",
    "fill water in ",
    "pour ",
)

_PICK_UP_PATTERN = re.compile(r"^pick up (.+)$")
_PLACE_ON_TOP_PATTERN = re.compile(r"^place (.+) on top of (.+)$")
_PUT_INSIDE_PATTERN = re.compile(r"^put (.+) inside (.+)$")


def normalize_esi_action(value: str) -> str:
    """Normalize whitespace and case without accepting non-string actions."""

    if not isinstance(value, str):
        raise ESIBackupActionError("action_must_be_string", "ESI action must be a string")
    normalized = " ".join(value.strip().lower().split())
    if not normalized:
        raise ESIBackupActionError("empty_action", "ESI action must not be empty")
    return normalized


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _guard_result(name: str, guard: PoseGuard, start: BackupPose, target: BackupPose) -> None:
    try:
        result = guard(start, target)
    except ESIBackupActionError:
        raise
    except Exception as exc:
        raise ESIBackupNavigationBlocked(
            f"{name}_guard_error",
            f"{name} guard raised {type(exc).__name__}: {exc}",
        ) from exc

    if isinstance(result, GuardDecision):
        allowed = result.allowed
        reason = result.reason.strip()
    elif isinstance(result, bool):
        allowed = result
        reason = ""
    else:
        raise TypeError(f"{name} guard must return bool or GuardDecision")

    if not allowed:
        detail = f": {reason}" if reason else ""
        raise ESIBackupNavigationBlocked(
            f"{name}_blocked",
            f"ESI base motion rejected by {name} guard{detail}",
        )


class RestrictedMotionController:
    """Pure-Python predictor and gate for legacy ESI camera-like motions."""

    def __init__(
        self,
        guards: NavigationGuards,
        *,
        move_distance_m: float = 0.25,
        yaw_step_degrees: float = 15.0,
        pitch_step_degrees: float = 15.0,
        max_abs_pitch_degrees: float = 60.0,
    ) -> None:
        if move_distance_m <= 0:
            raise ValueError("move_distance_m must be positive")
        if not 0 < yaw_step_degrees <= 180:
            raise ValueError("yaw_step_degrees must be in (0, 180]")
        if not 0 < pitch_step_degrees <= 90:
            raise ValueError("pitch_step_degrees must be in (0, 90]")
        if not 0 < max_abs_pitch_degrees < 90:
            raise ValueError("max_abs_pitch_degrees must be in (0, 90)")

        self.guards = guards
        self.move_distance_m = float(move_distance_m)
        self.yaw_step = math.radians(yaw_step_degrees)
        self.pitch_step = math.radians(pitch_step_degrees)
        self.max_abs_pitch = math.radians(max_abs_pitch_degrees)

    def transition(self, pose: BackupPose, action: str) -> BackupPose:
        """Return an approved pose or raise before simulator state can change."""

        normalized = normalize_esi_action(action)
        if normalized in _FORBIDDEN_FREE_FLIGHT_ACTIONS:
            raise ESIBackupActionError(
                "free_flight_forbidden",
                f"{normalized!r} is disabled: robot/view height cannot be changed by an ESI movement action",
            )

        if normalized in _VIEW_ACTIONS:
            delta = self.pitch_step if normalized == "turn_up" else -self.pitch_step
            pitch = max(-self.max_abs_pitch, min(self.max_abs_pitch, pose.pitch + delta))
            return BackupPose(pose.x, pose.y, pose.z, pose.yaw, pitch)

        if normalized == "stop":
            return pose
        if normalized not in _BASE_ACTIONS_TO_CORE:
            raise ESIBackupActionError(
                "not_a_motion_action",
                f"{normalized!r} is not handled by RestrictedMotionController",
            )

        if normalized == "turn_left":
            target = BackupPose(pose.x, pose.y, pose.z, _normalize_angle(pose.yaw + self.yaw_step), pose.pitch)
            _guard_result("collision", self.guards.collision_free, pose, target)
            return target
        if normalized == "turn_right":
            target = BackupPose(pose.x, pose.y, pose.z, _normalize_angle(pose.yaw - self.yaw_step), pose.pitch)
            _guard_result("collision", self.guards.collision_free, pose, target)
            return target

        forward = (math.cos(pose.yaw), math.sin(pose.yaw))
        left = (-forward[1], forward[0])
        direction = {
            "move_forward": forward,
            "move_backward": (-forward[0], -forward[1]),
            "move_back": (-forward[0], -forward[1]),
            "move_left": left,
            "move_right": (-left[0], -left[1]),
        }[normalized]
        target = BackupPose(
            pose.x + self.move_distance_m * direction[0],
            pose.y + self.move_distance_m * direction[1],
            pose.z,
            pose.yaw,
            pose.pitch,
        )

        # Locking z is an invariant in addition to asking the floor-map guard.
        if target.z != pose.z:
            raise AssertionError("restricted ESI translation changed base z")
        _guard_result("same_floor", self.guards.same_floor, pose, target)
        _guard_result("traversability", self.guards.traversable, pose, target)
        _guard_result("collision", self.guards.collision_free, pose, target)
        return target


class RestrictedESIBackupActions:
    """Prepare guarded ESI-compatible actions for the BEHAVIOR-HL backend."""

    def __init__(
        self,
        *,
        motion: RestrictedMotionController,
        resolve_object: ObjectResolver,
        select_door: DoorSelector | None = None,
        max_door_reach_m: float = 1.75,
    ) -> None:
        if not callable(resolve_object):
            raise TypeError("resolve_object must be callable")
        if select_door is not None and not callable(select_door):
            raise TypeError("select_door must be callable")
        if max_door_reach_m <= 0:
            raise ValueError("max_door_reach_m must be positive")
        self.motion = motion
        self.resolve_object = resolve_object
        self.select_door = select_door
        self.max_door_reach_m = float(max_door_reach_m)

    def prepare(
        self,
        action: str,
        pose: BackupPose,
        *,
        held_instance: str | None = None,
    ) -> PreparedESIAction:
        """Validate and translate one legacy string without mutating the world."""

        original = action
        normalized = normalize_esi_action(action)

        if normalized in _FORBIDDEN_FREE_FLIGHT_ACTIONS:
            # Route through the motion controller so there is one stable error.
            self.motion.transition(pose, normalized)
            raise AssertionError("unreachable")

        if normalized in _BASE_ACTIONS_TO_CORE:
            approved = self.motion.transition(pose, normalized)
            return PreparedESIAction(
                original,
                normalized,
                ActionDisposition.BASE_MOTION,
                pose,
                approved,
                {"action": _BASE_ACTIONS_TO_CORE[normalized]},
            )

        if normalized in _VIEW_ACTIONS:
            approved = self.motion.transition(pose, normalized)
            return PreparedESIAction(
                original,
                normalized,
                ActionDisposition.VIEW_ONLY,
                pose,
                approved,
                None,
            )

        if normalized == "stop":
            return PreparedESIAction(
                original,
                normalized,
                ActionDisposition.TERMINATE,
                pose,
                pose,
                {"action": "stop"},
            )

        pick_up = _PICK_UP_PATTERN.fullmatch(normalized)
        if pick_up:
            target = self._resolve(pick_up.group(1))
            if held_instance not in (None, target):
                raise ESIBackupActionError(
                    "hand_not_empty",
                    f"cannot pick up {target!r} while holding {held_instance!r}",
                )
            return self._semantic(
                original,
                normalized,
                pose,
                {"action": "pick_up", "target": target},
            )

        place_on_top = _PLACE_ON_TOP_PATTERN.fullmatch(normalized)
        if place_on_top:
            source = self._resolve(place_on_top.group(1))
            target = self._resolve(place_on_top.group(2))
            self._require_held(source, held_instance)
            if source == target:
                raise ESIBackupActionError("same_object", "an object cannot be placed on top of itself")
            return self._semantic(
                original,
                normalized,
                pose,
                {"action": "place_on_top", "target": target},
                source=source,
            )

        put_inside = _PUT_INSIDE_PATTERN.fullmatch(normalized)
        if put_inside:
            source = self._resolve(put_inside.group(1))
            container = self._resolve(put_inside.group(2))
            self._require_held(source, held_instance)
            if source == container:
                raise ESIBackupActionError("same_object", "an object cannot be put inside itself")
            return self._semantic(
                original,
                normalized,
                pose,
                {"action": "place_inside", "target": container},
                source=source,
            )

        if normalized in {"open nearest visible door", "close nearest visible door"}:
            should_open = normalized.startswith("open ")
            if self.select_door is None:
                raise ESIBackupActionError(
                    "door_selector_required",
                    "nearest-door actions require a segmentation/reachability-aware selector",
                )
            candidate = self.select_door(should_open)
            if candidate is None:
                raise ESIBackupActionError(
                    "no_reachable_visible_door",
                    "no visible, openable door is within the configured reach limit",
                )
            if not candidate.visible:
                raise ESIBackupActionError("door_not_visible", f"door {candidate.instance_id!r} is not visible")
            if not candidate.openable:
                raise ESIBackupActionError("door_not_openable", f"door {candidate.instance_id!r} is not openable")
            if candidate.distance_m > self.max_door_reach_m:
                raise ESIBackupActionError(
                    "door_out_of_reach",
                    (
                        f"door {candidate.instance_id!r} is {candidate.distance_m:.2f} m away; "
                        f"limit is {self.max_door_reach_m:.2f} m"
                    ),
                )
            operation = "open" if should_open else "close"
            return self._semantic(
                original,
                normalized,
                pose,
                {"action": operation, "target": candidate.instance_id.strip()},
            )

        if normalized == "pickup":
            raise ESIBackupActionError(
                "ambiguous_object",
                "bare 'pickup' is disabled; name an object from the current BDDL scope",
            )
        if normalized.startswith(_UNSAFE_OR_OUT_OF_SCOPE_PREFIXES):
            raise ESIBackupActionError(
                "unsupported_legacy_physical_action",
                (
                    f"{normalized!r} is not safely reusable in the five-task pilot; "
                    "use a structured BEHAVIOR-HL semantic action"
                ),
            )
        raise ESIBackupActionError("unknown_action", f"unsupported ESI action {normalized!r}")

    def _resolve(self, label: str) -> str:
        normalized_label = " ".join(label.strip().split())
        resolved = self.resolve_object(normalized_label)
        if not isinstance(resolved, str) or not resolved.strip():
            raise ESIBackupActionError(
                "unknown_object",
                f"{normalized_label!r} does not resolve to exactly one current BDDL instance",
            )
        return resolved.strip()

    @staticmethod
    def _require_held(source: str, held_instance: str | None) -> None:
        if held_instance is None:
            raise ESIBackupActionError(
                "nothing_held",
                f"cannot place {source!r}: no canonical BDDL instance is currently held",
            )
        if source != held_instance:
            raise ESIBackupActionError(
                "held_object_mismatch",
                f"action names {source!r}, but the held instance is {held_instance!r}",
            )

    @staticmethod
    def _semantic(
        original: str,
        normalized: str,
        pose: BackupPose,
        core_action: dict[str, str],
        *,
        source: str | None = None,
    ) -> PreparedESIAction:
        return PreparedESIAction(
            original,
            normalized,
            ActionDisposition.SEMANTIC_MACRO,
            pose,
            pose,
            core_action,
            source,
        )


def supported_esi_backup_actions() -> tuple[str, ...]:
    """Return the fixed non-object vocabulary exposed in prompts."""

    return (
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
        "turn_left",
        "turn_right",
        "turn_up",
        "turn_down",
        "open nearest visible door",
        "close nearest visible door",
        "stop",
        "pick up <object>",
        "place <held object> on top of <target>",
        "put <held object> inside <container>",
    )
