"""Provider-neutral public Action20 names and Marker protocol depth."""

from __future__ import annotations

import os


HOUSEHOLD_ENABLED = os.environ.get("ROBOWEAVE_EXPERIMENTAL_HOUSEHOLD_ACTIONS") == "1"
WIPE_ENABLED = HOUSEHOLD_ENABLED or os.environ.get("ROBOWEAVE_EXPERIMENTAL_WIPE") == "1"
HOUSEHOLD_TOOL_ACTIONS = ('cut', 'soak_inside', 'soak_under', 'fill_with', 'pour_into', 'spread')
HOUSEHOLD_WAIT_ACTIONS = ('wait_for_cooked', 'wait_for_washed', 'wait_for_frozen')
_EXPERIMENTAL_ACTIONS = (
    *(("wipe",) if WIPE_ENABLED else ()),
    *(HOUSEHOLD_TOOL_ACTIONS + HOUSEHOLD_WAIT_ACTIONS if HOUSEHOLD_ENABLED else ()),
)

BASE_ACTION20 = (
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
    "wait",
    "pick",
    "open",
    "close",
    "toggle_on",
    "toggle_off",
    "place_on_floor",
    "place_inside",
    "place_on_top",
    "place_under",
    "place_next_to",
)
TERMINATION_ACTION = "done"
STRICT_ACTION20 = (*BASE_ACTION20, *_EXPERIMENTAL_ACTIONS)
BASE_TOOL_ACTIONS = (*BASE_ACTION20, TERMINATION_ACTION)
PUBLIC_TOOL_ACTIONS = (*STRICT_ACTION20, TERMINATION_ACTION)
MODEL_VISIBLE_TOOL_ACTIONS = PUBLIC_TOOL_ACTIONS
ACTION_DECISION_LIMIT = 150
MARKER_ACTIONS = frozenset(
    {
        "pick",
        "open",
        "close",
        "toggle_on",
        "toggle_off",
        *_EXPERIMENTAL_ACTIONS,
        "place_inside",
        "place_on_top",
        "place_under",
        "place_next_to",
    }
)
TWO_CALL_MARKER_ACTIONS = frozenset(
    {
        "pick",
        "open",
        "close",
        "toggle_on",
        "toggle_off",
        *_EXPERIMENTAL_ACTIONS,
        "place_under",
    }
)
THREE_CALL_MARKER_ACTIONS = frozenset(
    {
        "place_inside",
        "place_on_top",
        "place_next_to",
    }
)
BASE_TRANSLATION_ACTIONS = frozenset(
    {"move_forward", "move_backward", "move_left", "move_right"}
)
BASE_TURN_ACTIONS = frozenset({"turn_left", "turn_right"})
BASE_ODOMETRY_ACTIONS = BASE_TRANSLATION_ACTIONS | BASE_TURN_ACTIONS
NAVIGATION_REPEAT_MAX = 10
NAVIGATION_MOVE_DISTANCES_M = (0.15, 0.30, 0.60)
NAVIGATION_TURN_ANGLES_DEG = (10.0, 30.0)
NAVIGATION_PHYSICAL_REPEAT_MAX = NAVIGATION_REPEAT_MAX * 4


def navigation_repeat_max(action: str) -> int:
    """Return the per-action repeat limit for base navigation."""

    if action not in BASE_ODOMETRY_ACTIONS:
        raise ValueError("action is not a navigation action")
    return NAVIGATION_REPEAT_MAX


def navigation_step_multiplier(
    action: str,
    *,
    move_distance_m: float | None = None,
    turn_angle_deg: float | None = None,
) -> int:
    """Expand one model-selected navigation step into safe physical steps."""

    if action in BASE_TRANSLATION_ACTIONS:
        value = 0.15 if move_distance_m is None else move_distance_m
        if turn_angle_deg is not None or value not in NAVIGATION_MOVE_DISTANCES_M:
            raise ValueError("move action selected an invalid distance")
        return round(value / 0.15)
    if action in BASE_TURN_ACTIONS:
        value = 10.0 if turn_angle_deg is None else turn_angle_deg
        if move_distance_m is not None or value not in NAVIGATION_TURN_ANGLES_DEG:
            raise ValueError("turn action selected an invalid angle")
        return round(value / 10.0)
    if move_distance_m is not None or turn_angle_deg is not None:
        raise ValueError("non-navigation action selected a navigation step size")
    return 1


def navigation_physical_repeat_max(action: str) -> int:
    """Return the largest safe-substep count accepted in one checkpoint."""

    if action in BASE_TRANSLATION_ACTIONS:
        return NAVIGATION_REPEAT_MAX * 4
    if action in BASE_TURN_ACTIONS:
        return NAVIGATION_REPEAT_MAX * 3
    raise ValueError("action is not a navigation action")

if (
    len(BASE_ACTION20) != 19
    or len(set(STRICT_ACTION20)) != len(STRICT_ACTION20)
    or len(PUBLIC_TOOL_ACTIONS) != 20 + len(_EXPERIMENTAL_ACTIONS)
    or MODEL_VISIBLE_TOOL_ACTIONS != PUBLIC_TOOL_ACTIONS
    or TWO_CALL_MARKER_ACTIONS & THREE_CALL_MARKER_ACTIONS
    or TWO_CALL_MARKER_ACTIONS | THREE_CALL_MARKER_ACTIONS != MARKER_ACTIONS
):
    raise RuntimeError("Action20 public protocol partition is invalid")
