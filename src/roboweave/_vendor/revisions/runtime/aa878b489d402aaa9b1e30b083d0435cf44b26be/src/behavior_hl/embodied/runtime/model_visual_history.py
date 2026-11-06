"""Advance and pack the Action20 model's completed-observation window."""

from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")

# Completed after-action observations retained across steps.
MODEL_COMPLETED_HISTORY_MAX = 5
# Those completed frames plus the current onboard view.
MODEL_SENT_FRAME_MAX = MODEL_COMPLETED_HISTORY_MAX + 1


def pack_model_visual_frames(
    completed_history: Sequence[T],
    current: T,
) -> tuple[T, ...]:
    """Return up to five completed observations plus the current view.

    Step 1 sends only the current onboard image.  Each later step prepends
    one completed after-action observation per finished action, oldest
    first, with the last entry the current view.  Same-action before /
    in-progress / after dumps never share this window.
    """

    if current is None:
        raise TypeError("current observation is required")
    history = tuple(completed_history[-MODEL_COMPLETED_HISTORY_MAX:])
    packed = (*history, current)
    if not 1 <= len(packed) <= MODEL_SENT_FRAME_MAX:
        raise ValueError("packed visual frames exceeded the public window")
    return packed


def advance_model_visual_window(
    completed_history: Sequence[T],
    current: T,
    completed_after: T,
) -> tuple[tuple[T, ...], T]:
    """Advance exactly once after an accepted public action.

    The observation used for the decision becomes one historical observation,
    and the action's final after-observation becomes the next current view.
    Reusing ``completed_after`` directly avoids capturing and sending the same
    post-action state twice as both ``action_after`` and a fresh ``current``.
    Protocol-invalid responses never call this transition, so they reuse the
    exact same public observation window.
    """

    if current is None:
        raise TypeError("current observation is required")
    if completed_after is None:
        raise TypeError("completed after-observation is required")
    history = (*tuple(completed_history), current)[
        -MODEL_COMPLETED_HISTORY_MAX:
    ]
    return history, completed_after
