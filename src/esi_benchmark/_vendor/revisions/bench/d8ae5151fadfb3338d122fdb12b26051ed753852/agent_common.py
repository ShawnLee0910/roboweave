"""Small shared constants and JSON helpers for the Action20 agent."""

from __future__ import annotations

import json
from typing import Any


PHYSICAL_HISTORY_MAX = 5
PHYSICAL_HISTORY_OPTIONS = frozenset({5, 20})
MODEL_OBSERVATION_MAX = max(PHYSICAL_HISTORY_OPTIONS) + 1
BLOCKED_OBSERVATION_REMINDER = (
    "The requested action did not complete. Reassess the latest observation "
    "before deciding whether to retry or change the situation."
)


def validate_physical_history_max(value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in PHYSICAL_HISTORY_OPTIONS
    ):
        raise ValueError("physical_history_max must be exactly 5 or 20")


def json_clone(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact value is not strict JSON") from exc
