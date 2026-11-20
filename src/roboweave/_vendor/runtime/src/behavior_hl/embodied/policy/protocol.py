"""Strict JSON envelope and reusable action-argument validators."""

from __future__ import annotations

HOUSEHOLD_TOOL_ACTIONS = ("cut", "soak_inside", "soak_under", "fill_with", "pour_into", "spread")
HOUSEHOLD_WAIT_ACTIONS = ("wait_for_cooked", "wait_for_washed", "wait_for_frozen")
HOUSEHOLD_ACTIONS = (*HOUSEHOLD_TOOL_ACTIONS, *HOUSEHOLD_WAIT_ACTIONS)

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol


MAX_ACTION_JSON_CHARS = 16_384
MAX_REASONING_CHARS = 2_000
CAMERA_NAMES = frozenset({"head", "left_wrist", "right_wrist"})
HAND_NAMES = frozenset({"left", "right"})
VISIBLE_OBJECT_CATEGORIES = frozenset(
    {"board_game", "jigsaw_puzzle", "tennis_ball"}
)
VISIBLE_OBJECT_SELECTORS = frozenset(
    {"bottommost", "center", "leftmost", "rightmost", "topmost"}
)


class ActionProtocolError(ValueError):
    """Raised when a model action is outside the frozen public protocol."""


class ActionArguments(Protocol):
    """Serializable arguments owned by one ActionSpec plugin."""

    def to_dict(self) -> dict[str, Any]:
        """Return the public JSON representation."""


@dataclass(frozen=True, slots=True)
class NoArgs:
    """Arguments for a parameter-free action."""

    def to_dict(self) -> dict[str, Any]:
        return {}


@dataclass(frozen=True, slots=True)
class PointActionArgs:
    """A visual target in the image actually shown to the model."""

    camera: str
    point: tuple[float, float]
    hand: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "point": [self.point[0], self.point[1]],
            "hand": self.hand,
        }


@dataclass(frozen=True, slots=True)
class VisibleObjectGraspArgs:
    """A task-object category selected only from the current head image."""

    object: str
    selector: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"object": self.object}
        if self.selector is not None:
            result["selector"] = self.selector
        return result


@dataclass(frozen=True, slots=True)
class MarkerGraspArgs:
    """One opaque, temporary label from an explicitly marked head frame."""

    marker: str

    def to_dict(self) -> dict[str, Any]:
        return {"marker": self.marker}


@dataclass(frozen=True, slots=True)
class MarkerPlaceInsideArgs:
    """A visible container marker from the latest explicitly marked frame."""

    marker: str

    def to_dict(self) -> dict[str, Any]:
        return {"marker": self.marker}


@dataclass(frozen=True, slots=True)
class MarkerTargetArgs:
    """One opaque temporary marker for the v34 symbolic action surface."""

    marker: str

    def to_dict(self) -> dict[str, Any]:
        return {"marker": self.marker}


@dataclass(frozen=True, slots=True)
class MarkerTargetOrRequestArgs:
    """Declare a marker operation, or select from its latest choice frame.

    ``marker=None`` is deliberately represented by an absent ``args`` object
    on the public wire.  It requests an operation-specific choice image but
    does not execute the object operation.  A later call to the same action
    with one opaque C-marker performs the next selection / execution stage.
    """

    marker: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {} if self.marker is None else {"marker": self.marker}


@dataclass(frozen=True, slots=True)
class ReleaseArgs:
    """Arguments for opening one physical gripper."""

    hand: str

    def to_dict(self) -> dict[str, Any]:
        return {"hand": self.hand}


@dataclass(frozen=True, slots=True)
class ActionEnvelope:
    """One parsed, registry-approved high-level action."""

    action: str
    args: ActionArguments
    reasoning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": self.action}
        args = self.args.to_dict()
        if args:
            payload["args"] = args
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        return payload


def decode_json_object(text: str) -> dict[str, Any]:
    """Decode one finite JSON object while rejecting duplicate keys."""

    if not isinstance(text, str):
        raise ActionProtocolError("model output must be a JSON string")
    if len(text) > MAX_ACTION_JSON_CHARS:
        raise ActionProtocolError(
            f"model output exceeds {MAX_ACTION_JSON_CHARS} characters"
        )

    def reject_constant(value: str) -> None:
        raise ActionProtocolError(f"non-finite JSON number is not allowed: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ActionProtocolError(f"duplicate JSON field: {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except ActionProtocolError:
        raise
    except json.JSONDecodeError as exc:
        raise ActionProtocolError(f"invalid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ActionProtocolError("action payload must be one JSON object")
    return payload


def parse_reasoning(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ActionProtocolError("'reasoning' must be a string")
    reasoning = value.strip()
    if not reasoning:
        raise ActionProtocolError("'reasoning' must not be empty")
    if len(reasoning) > MAX_REASONING_CHARS:
        raise ActionProtocolError(
            f"'reasoning' exceeds {MAX_REASONING_CHARS} characters"
        )
    return reasoning


def parse_no_args(payload: Mapping[str, Any]) -> NoArgs:
    if payload:
        raise ActionProtocolError(
            f"this action does not accept arguments: {sorted(payload)}"
        )
    return NoArgs()


def parse_point_action_args(payload: Mapping[str, Any]) -> PointActionArgs:
    _require_exact_fields(payload, {"camera", "point", "hand"})
    camera = _choice(payload["camera"], "camera", CAMERA_NAMES)
    hand = _choice(payload["hand"], "hand", HAND_NAMES)

    raw_point = payload["point"]
    if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
        raise ActionProtocolError("'point' must be a two-element array")
    point = (
        _normalized_number(raw_point[0], "point[0]"),
        _normalized_number(raw_point[1], "point[1]"),
    )
    return PointActionArgs(camera=camera, point=point, hand=hand)


def parse_release_args(payload: Mapping[str, Any]) -> ReleaseArgs:
    _require_exact_fields(payload, {"hand"})
    return ReleaseArgs(hand=_choice(payload["hand"], "hand", HAND_NAMES))


def parse_visible_object_grasp_args(
    payload: Mapping[str, Any],
) -> VisibleObjectGraspArgs:
    required = {"object"}
    optional = {"selector"}
    actual = set(payload)
    missing = required - actual
    unknown = actual - required - optional
    if missing:
        raise ActionProtocolError(f"missing argument fields: {sorted(missing)}")
    if unknown:
        raise ActionProtocolError(f"unknown argument fields: {sorted(unknown)}")
    category = _choice(
        payload["object"],
        "object",
        VISIBLE_OBJECT_CATEGORIES,
    )
    selector = None
    if "selector" in payload:
        selector = _choice(
            payload["selector"],
            "selector",
            VISIBLE_OBJECT_SELECTORS,
        )
    return VisibleObjectGraspArgs(object=category, selector=selector)


def parse_marker_grasp_args(payload: Mapping[str, Any]) -> MarkerGraspArgs:
    return MarkerGraspArgs(marker=_parse_marker(payload))


def parse_marker_place_inside_args(
    payload: Mapping[str, Any],
) -> MarkerPlaceInsideArgs:
    return MarkerPlaceInsideArgs(marker=_parse_marker(payload))


def parse_marker_target_args(payload: Mapping[str, Any]) -> MarkerTargetArgs:
    return MarkerTargetArgs(marker=_parse_marker(payload))


def parse_marker_target_or_request_args(
    payload: Mapping[str, Any],
) -> MarkerTargetOrRequestArgs:
    """Parse either ``{}`` (request choices) or ``{"marker": "C1"}``."""

    if not payload:
        return MarkerTargetOrRequestArgs()
    return MarkerTargetOrRequestArgs(marker=_parse_marker(payload))


def _parse_marker(payload: Mapping[str, Any]) -> str:
    _require_exact_fields(payload, {"marker"})
    marker = payload["marker"]
    if (
        not isinstance(marker, str)
        or len(marker) < 2
        or marker[0] != "C"
        or not marker[1:].isdigit()
        or marker[1] == "0"
        or len(marker) > 4
    ):
        raise ActionProtocolError(
            "'marker' must match C1..C999 from the latest marked frame"
        )
    return marker


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: set[str],
) -> None:
    actual = set(payload)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ActionProtocolError(f"missing argument fields: {sorted(missing)}")
    if unknown:
        raise ActionProtocolError(f"unknown argument fields: {sorted(unknown)}")


def _choice(value: Any, field: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ActionProtocolError(f"{field!r} must be one of: {choices}")
    return value


def _normalized_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionProtocolError(f"{field!r} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ActionProtocolError(f"{field!r} must be finite")
    if not 0.0 <= number <= 1.0:
        raise ActionProtocolError(f"{field!r} must be in [0, 1]")
    return number
