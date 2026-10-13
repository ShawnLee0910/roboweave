"""Resident model navigation batching; atoms remain ordinary Action20 actions."""

import json

from behavior_hl.embodied.policy.protocol import ActionProtocolError, decode_json_object


NAVIGATION_ACTIONS = (
    "move_forward", "move_backward", "move_left", "move_right",
    "turn_left", "turn_right",
)
NAVIGATION_REPEAT_MAX = 10
NAVIGATION_REPEAT_CONTRACT = {
    "field": "repeat",
    "minimum": 1,
    "maximum": NAVIGATION_REPEAT_MAX,
    "default": 1,
    "actions": list(NAVIGATION_ACTIONS),
    "semantics": (
        "Optional top-level repeat repeats the registered atomic navigation action. "
        "Each translation remains 0.15 m and each turn 10 degrees. "
        "One batch uses one model decision; every executed atom is recorded. "
        "Stops on any non-completed result, task completion, or V intervention. "
        "No move_distance_m or steps input is supported."
    ),
}


def parse_resident_navigation_action(registry, raw):
    payload = decode_json_object(raw)
    present = "repeat" in payload
    repeat = payload.pop("repeat", 1)
    if (type(repeat) is not int or not 1 <= repeat <= NAVIGATION_REPEAT_MAX
            or (present and payload.get("action") not in NAVIGATION_ACTIONS)):
        raise ActionProtocolError("repeat must be an integer 1..10 for navigation")
    return registry.parse_json(json.dumps(payload)), repeat
