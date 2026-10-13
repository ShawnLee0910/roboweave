"""Strict JSON protocol for ESI-like navigation and semantic object actions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ActionProtocolError(ValueError):
    """Raised when a model action does not match the public protocol."""


class ActionName(str, Enum):
    # ESI-style ego navigation, executed by the robot base rather than the viewer camera.
    MOVE_FORWARD = "move_forward"
    MOVE_BACK = "move_back"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"

    # Object-centric navigation and manipulation.
    NAVIGATE_TO = "navigate_to"
    PICK_UP = "pick_up"
    PLACE_INSIDE = "place_inside"
    PLACE_ON_TOP = "place_on_top"
    PLACE_NEXT_TO = "place_next_to"
    OPEN = "open"
    CLOSE = "close"
    TOGGLE_ON = "toggle_on"
    TOGGLE_OFF = "toggle_off"

    STOP = "stop"


EGO_NAVIGATION_ACTIONS = frozenset(
    {
        ActionName.MOVE_FORWARD,
        ActionName.MOVE_BACK,
        ActionName.MOVE_LEFT,
        ActionName.MOVE_RIGHT,
        ActionName.TURN_LEFT,
        ActionName.TURN_RIGHT,
    }
)

TARGET_ACTIONS = frozenset(
    {
        ActionName.NAVIGATE_TO,
        ActionName.PICK_UP,
        ActionName.PLACE_INSIDE,
        ActionName.PLACE_ON_TOP,
        ActionName.PLACE_NEXT_TO,
        ActionName.OPEN,
        ActionName.CLOSE,
        ActionName.TOGGLE_ON,
        ActionName.TOGGLE_OFF,
    }
)


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ActionProtocolError(f"{field!r} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class HighLevelAction:
    """One model-selected macro action.

    ``target`` and ``support`` are BDDL instance identifiers from
    ``BehaviorTask.object_scope``, not unrestricted simulator object names.
    """

    name: ActionName
    target: str | None = None
    support: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HighLevelAction":
        if not isinstance(payload, dict):
            raise ActionProtocolError("action payload must be a JSON object")

        unknown = set(payload) - {"action", "target", "support"}
        if unknown:
            raise ActionProtocolError(f"unknown action fields: {sorted(unknown)}")
        if "action" not in payload:
            raise ActionProtocolError("missing required field 'action'")

        raw_name = _nonempty_string(payload["action"], "action")
        try:
            name = ActionName(raw_name)
        except ValueError as exc:
            allowed = ", ".join(action.value for action in ActionName)
            raise ActionProtocolError(f"unknown action {raw_name!r}; allowed: {allowed}") from exc

        target = payload.get("target")
        support = payload.get("support")

        if name in TARGET_ACTIONS:
            target = _nonempty_string(target, "target")
        elif target is not None:
            raise ActionProtocolError(f"{name.value!r} does not accept 'target'")

        if name is ActionName.PLACE_NEXT_TO:
            support = _nonempty_string(support, "support")
        elif support is not None:
            raise ActionProtocolError(f"{name.value!r} does not accept 'support'")

        return cls(name=name, target=target, support=support)

    @classmethod
    def from_json(cls, text: str) -> "HighLevelAction":
        if not isinstance(text, str):
            raise ActionProtocolError("model output must be a JSON string")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ActionProtocolError(f"invalid JSON: {exc.msg}") from exc
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, str]:
        payload = {"action": self.name.value}
        if self.target is not None:
            payload["target"] = self.target
        if self.support is not None:
            payload["support"] = self.support
        return payload

