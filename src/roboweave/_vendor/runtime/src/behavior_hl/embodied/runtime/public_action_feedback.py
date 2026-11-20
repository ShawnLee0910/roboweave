"""Allowlisted non-directive action feedback exposed to the model."""

from __future__ import annotations

from behavior_hl.embodied.policy.protocol import HOUSEHOLD_ACTIONS, HOUSEHOLD_TOOL_ACTIONS

import math
from typing import Any, Iterable, Mapping


TARGET_OUTSIDE_LOCAL_WORKSPACE = "TARGET_OUTSIDE_LOCAL_WORKSPACE"
ACTION20_BLOCKED_CATEGORY_METADATA_KEY = "action20_blocked_category"
ARTICULATED_PART_BLOCKED = "articulated_part_blocked"
EXECUTION_UNVERIFIED = "execution_unverified"
GEOMETRIC_FIT_BLOCKED = "geometric_fit_blocked"
NAVIGATION_BLOCKED = "navigation_blocked"
OUT_OF_REACH = "out_of_reach"
NO_SELECTABLE_CANDIDATES = "no_selectable_candidates"
PLACEMENT_COLLISION_BLOCKED = "placement_collision_blocked"
PLACEMENT_SUPPORT_UNAVAILABLE = "placement_support_unavailable"
RELATION_NOT_REALIZED = "relation_not_realized"
REQUIRES_EMPTY_LEFT_HAND = "requires_empty_left_hand"
REQUIRES_OBJECT_IN_LEFT_HAND = "requires_object_in_left_hand"
SELECTION_UNAVAILABLE = "selection_unavailable"
VIEW_LIMIT_REACHED = "view_limit_reached"
VIEW_MOTION_BLOCKED_WHILE_HOLDING = "view_motion_blocked_while_holding"
ACTION20_BLOCKED_CATEGORIES = frozenset(
    {
        ARTICULATED_PART_BLOCKED,
        EXECUTION_UNVERIFIED,
        GEOMETRIC_FIT_BLOCKED,
        NAVIGATION_BLOCKED,
        OUT_OF_REACH,
        NO_SELECTABLE_CANDIDATES,
        PLACEMENT_COLLISION_BLOCKED,
        PLACEMENT_SUPPORT_UNAVAILABLE,
        RELATION_NOT_REALIZED,
        REQUIRES_EMPTY_LEFT_HAND,
        REQUIRES_OBJECT_IN_LEFT_HAND,
        SELECTION_UNAVAILABLE,
        VIEW_LIMIT_REACHED,
        VIEW_MOTION_BLOCKED_WHILE_HOLDING,
    }
)
ACTION20_BLOCKED_DETAILS = {
    ARTICULATED_PART_BLOCKED: frozenset({
        "robot_obstruction", "external_obstruction", "contents_obstruction",
        "multiple_obstructions",
    }),
}


def action20_blocked_detail(category: Any, detail: Any) -> str | None:
    """Only finite, category-specific labels cross the public boundary."""
    if isinstance(category, str) and isinstance(detail, str):
        if detail in ACTION20_BLOCKED_DETAILS.get(category, ()):
            return detail
    return None
_EMPTY_HAND_MARKER_ACTIONS = frozenset(
    {"pick", "open", "close", "toggle_on", "toggle_off"}
)
_PLACEMENT_ACTIONS = frozenset(
    {
        "place_inside",
        "place_on_floor",
        "place_on_top",
        "place_next_to",
        "place_under",
    }
)
_MARKER_ACTIONS = _EMPTY_HAND_MARKER_ACTIONS | (
    _PLACEMENT_ACTIONS - {"place_on_floor"}
) | {"wipe", *HOUSEHOLD_ACTIONS}
_NAVIGATION_ACTIONS = frozenset(
    {
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
        "turn_left",
        "turn_right",
    }
)
_VIEW_ACTIONS = frozenset({"look_up", "look_down"})
_ACTION20_ACTIONS = (
    _MARKER_ACTIONS
    | _NAVIGATION_ACTIONS
    | _VIEW_ACTIONS
    | {"place_on_floor", "wait"}
)
# Preserve exact causes when normalizing private / legacy guard codes. A
# workspace rejection is not evidence that navigation was attempted or blocked.
_PRIVATE_CATEGORY_ALIASES = {
    "navigation_collision_blocked": NAVIGATION_BLOCKED,
    "navigation_path_blocked": NAVIGATION_BLOCKED,
    "target_outside_local_workspace": OUT_OF_REACH,
    TARGET_OUTSIDE_LOCAL_WORKSPACE: OUT_OF_REACH,
}
_CATEGORY_ACTIONS = {
    ARTICULATED_PART_BLOCKED: frozenset({"open", "close"}),
    EXECUTION_UNVERIFIED: _ACTION20_ACTIONS,
    GEOMETRIC_FIT_BLOCKED: _PLACEMENT_ACTIONS,
    NAVIGATION_BLOCKED: _NAVIGATION_ACTIONS,
    OUT_OF_REACH: _MARKER_ACTIONS | _PLACEMENT_ACTIONS,
    NO_SELECTABLE_CANDIDATES: _MARKER_ACTIONS,
    PLACEMENT_COLLISION_BLOCKED: _PLACEMENT_ACTIONS,
    PLACEMENT_SUPPORT_UNAVAILABLE: _PLACEMENT_ACTIONS,
    RELATION_NOT_REALIZED: _PLACEMENT_ACTIONS,
    REQUIRES_EMPTY_LEFT_HAND: _EMPTY_HAND_MARKER_ACTIONS,
    REQUIRES_OBJECT_IN_LEFT_HAND: _PLACEMENT_ACTIONS | {"wipe", *HOUSEHOLD_TOOL_ACTIONS},
    SELECTION_UNAVAILABLE: _MARKER_ACTIONS,
    VIEW_LIMIT_REACHED: _VIEW_ACTIONS,
    VIEW_MOTION_BLOCKED_WHILE_HOLDING: _VIEW_ACTIONS,
}
_RAW_CATEGORY_STATUSES = {
    ARTICULATED_PART_BLOCKED: frozenset({"blocked", "uncertain"}),
    EXECUTION_UNVERIFIED: frozenset({"uncertain", "blocked", "not_executed", "aborted"}),
    GEOMETRIC_FIT_BLOCKED: frozenset({"blocked", "uncertain"}),
    NAVIGATION_BLOCKED: frozenset({"blocked", "not_executed"}),
    OUT_OF_REACH: frozenset({"blocked", "not_executed"}),
    NO_SELECTABLE_CANDIDATES: frozenset({"blocked", "markers_unavailable"}),
    PLACEMENT_COLLISION_BLOCKED: frozenset({"blocked", "uncertain"}),
    PLACEMENT_SUPPORT_UNAVAILABLE: frozenset({"blocked"}),
    RELATION_NOT_REALIZED: frozenset({"blocked"}),
    REQUIRES_EMPTY_LEFT_HAND: frozenset({"blocked"}),
    REQUIRES_OBJECT_IN_LEFT_HAND: frozenset({"blocked"}),
    SELECTION_UNAVAILABLE: frozenset({"not_executed", "blocked"}),
    VIEW_LIMIT_REACHED: frozenset({"blocked", "limit_reached"}),
    VIEW_MOTION_BLOCKED_WHILE_HOLDING: frozenset({"blocked", "unsafe_while_carrying"}),
}
ACTION20_PUBLIC_TRANSLATION_LIMIT_M = 0.15
ACTION20_PUBLIC_YAW_LIMIT_DEG = 10.0
# Isaac/physics integration can finish a commanded boundary modestly beyond
# its exact decimal target after collision-prefix truncation and pose
# round-trips. Admit a 2 cm / 2 degree residual window, then clamp the public
# value back to the 0.15 m / 10 degree Action20 contract below.  Larger
# drift remains non-serializable so the resident recovery boundary can restore
# a checkpoint (or same-scene reset before the first checkpoint).
ACTION20_PUBLIC_TRANSLATION_TOLERANCE_M = 0.02
ACTION20_PUBLIC_YAW_TOLERANCE_DEG = 2.0
_ACTION20_PUBLIC_STATUSES = frozenset({"completed", "blocked"})
_BODY_ODOMETRY_FIELDS = ("forward_m", "left_m", "yaw_deg")


def placement_failure_category(categories: Iterable[str | None]) -> str | None:
    """Never turn a failed query into a claimed geometric or collision cause."""

    causes = set(categories)
    if EXECUTION_UNVERIFIED in causes:
        return EXECUTION_UNVERIFIED
    return next(iter(causes)) if len(causes) == 1 else None


def model_visible_action20_last_result(
    value: Any,
    *,
    action: str | None = None,
) -> dict[str, Any] | None:
    """Return the exact bounded Action20 public-action outcome allowlist.

    Every public action reports one two-valued outcome: ``completed`` passes
    through and any other executor status collapses to ``blocked``.  Only a
    blocked result may include one coarse category derived from an exact
    Action20 guard or terminal status.
    Base motion additionally reports finite body-relative odometry bounded by
    one public Action20 step; results without odometry carry only the status.
    The executor retains richer diagnostics privately.  Unknown fields are
    ignored, never forwarded.
    """

    if not isinstance(value, Mapping):
        return None
    status = value.get("status")
    if not isinstance(status, str) or not status.strip():
        return None
    public_status = "completed" if status == "completed" else "blocked"
    assert public_status in _ACTION20_PUBLIC_STATUSES

    result: dict[str, Any] = {"status": public_status}
    if public_status == "blocked":
        blocked_category = _action20_blocked_category(
            value,
            action=action,
            status=status,
        )
        if blocked_category is not None:
            result["blocked_category"] = blocked_category
            detail = action20_blocked_detail(blocked_category, value.get("blocked_detail"))
            if detail is not None:
                result["blocked_detail"] = detail

    odometry = value.get("body_odometry")
    if odometry is None:
        return result
    if not isinstance(odometry, Mapping):
        return None

    normalized: dict[str, float] = {}
    for field in _BODY_ODOMETRY_FIELDS:
        raw = odometry.get(field)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        number = float(raw)
        if not math.isfinite(number):
            return None
        normalized[field] = number

    translation_norm = math.hypot(
        normalized["forward_m"], normalized["left_m"]
    )
    if (
        translation_norm
        > ACTION20_PUBLIC_TRANSLATION_LIMIT_M
        + ACTION20_PUBLIC_TRANSLATION_TOLERANCE_M
    ):
        return None
    if (
        abs(normalized["yaw_deg"])
        > ACTION20_PUBLIC_YAW_LIMIT_DEG + ACTION20_PUBLIC_YAW_TOLERANCE_DEG
    ):
        return None
    if translation_norm > ACTION20_PUBLIC_TRANSLATION_LIMIT_M:
        scale = ACTION20_PUBLIC_TRANSLATION_LIMIT_M / translation_norm
        normalized["forward_m"] *= scale
        normalized["left_m"] *= scale
    normalized["yaw_deg"] = max(
        -ACTION20_PUBLIC_YAW_LIMIT_DEG,
        min(ACTION20_PUBLIC_YAW_LIMIT_DEG, normalized["yaw_deg"]),
    )
    # Physics keeps full precision privately.  The model-visible payload is
    # meters and degrees rounded to two decimal places so a successful
    # turn_left reads 0.00 m / 0.00 m / 10.00 deg instead of simulator
    # leftovers such as 3.5e-06 / 29.99999.
    result["body_odometry"] = {
        field: _round_public_odometry(normalized[field])
        for field in _BODY_ODOMETRY_FIELDS
    }
    return result


def _action20_blocked_category(
    value: Mapping[str, Any],
    *,
    action: str | None,
    status: str,
) -> str | None:
    if action not in _ACTION20_ACTIONS:
        return None
    raw_category = value.get("blocked_category")
    category = (
        _PRIVATE_CATEGORY_ALIASES.get(raw_category, raw_category)
        if isinstance(raw_category, str)
        else None
    )
    # Automatic approach is internal to the requested manipulation action.
    if category == NAVIGATION_BLOCKED and action in _MARKER_ACTIONS:
        category = OUT_OF_REACH
    if (
        isinstance(category, str)
        and action in _CATEGORY_ACTIONS.get(category, ())
        and status in _RAW_CATEGORY_STATUSES.get(category, ())
    ):
        return category
    if status == "markers_unavailable" and action in _MARKER_ACTIONS:
        return NO_SELECTABLE_CANDIDATES
    if (
        status == "not_executed"
        and value.get("reason_code") == TARGET_OUTSIDE_LOCAL_WORKSPACE
        and action in _MARKER_ACTIONS
    ):
        return OUT_OF_REACH
    if action in _VIEW_ACTIONS:
        if status == "limit_reached":
            return VIEW_LIMIT_REACHED
        if status == "unsafe_while_carrying":
            return VIEW_MOTION_BLOCKED_WHILE_HOLDING
        if status == "joint_stalled":
            return EXECUTION_UNVERIFIED
    return EXECUTION_UNVERIFIED if status in {"uncertain", "aborted"} else None


def _round_public_odometry(value: float) -> float:
    rounded = round(value, 2)
    return 0.0 if rounded == 0.0 else rounded


def model_visible_action_result(value: Any) -> dict[str, Any] | None:
    """Return only the exact public local-workspace failure contract.

    Distances, bearings, object identities, planner diagnostics, and suggested
    navigation actions are deliberately never serialized.
    """

    if not isinstance(value, Mapping):
        return None
    status = value.get("status")
    reason_code = value.get("reason_code")
    object_acquired = value.get("object_acquired")
    if (
        status != "not_executed"
        or reason_code != TARGET_OUTSIDE_LOCAL_WORKSPACE
        or not isinstance(object_acquired, bool)
    ):
        return None
    return {
        "status": status,
        "reason_code": reason_code,
        "object_acquired": object_acquired,
    }
