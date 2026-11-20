"""Pure Action20 carry-state availability checks for the resident scene."""

from __future__ import annotations

from typing import Any, Mapping

from action20_protocol import HOUSEHOLD_TOOL_ACTIONS


PLACEMENT_ACTIONS = frozenset(
    {
        "place_inside",
        "place_on_floor",
        "place_on_top",
        "place_next_to",
        "place_under",
    }
)
EMPTY_HAND_ACTIONS = frozenset(
    {"pick", "open", "close", "toggle_on", "toggle_off"}
)


def public_precondition_block_reason(*, runtime: Any, action: str) -> str | None:
    """Return a private reason for normal public carry-state unavailability."""

    snapshot = runtime.carry_tracker.snapshot()
    if not isinstance(snapshot, Mapping):
        raise RuntimeError("public carry snapshot must be a mapping")
    likely = snapshot.get("likely")
    active_hand = snapshot.get("active_hand")
    if not isinstance(likely, bool):
        raise RuntimeError("public carry snapshot likely must be boolean")
    if active_hand not in {None, "left", "right"}:
        raise RuntimeError("public carry snapshot active_hand is invalid")
    if likely != (active_hand in {"left", "right"}):
        raise RuntimeError("public carry snapshot hand state is inconsistent")
    if action in PLACEMENT_ACTIONS or action in (*HOUSEHOLD_TOOL_ACTIONS, "wipe"):
        return (
            None
            if active_hand == "left"
            else "requires_object_in_left_hand"
        )
    if action in EMPTY_HAND_ACTIONS:
        if action in {
            "pick",
            "open",
            "close",
            "toggle_on",
            "toggle_off",
        } and active_hand == "left":
            return "requires_empty_left_hand"
        return "requires_empty_hands" if likely else None
    return None
