"""Bounded in-process checkpoints for one resident Action20 episode.

The ring deliberately owns opaque payloads.  Simulator and Python runtime
state stay private to the resident scene process, while this module enforces
the action interval, safe-boundary deferral, and the two-checkpoint retention
limit without importing OmniGibson.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping


ROLLING_EPISODE_CHECKPOINT_VERSION = (
    "behavior-hl-resident-rolling-checkpoint/v1"
)
DEFAULT_ACTION_INTERVAL = 50
DEFAULT_MAXIMUM_RETAINED = 2


class RollingCheckpointError(RuntimeError):
    """A rolling checkpoint request violated the resident contract."""


@dataclass(frozen=True, slots=True)
class RollingCheckpointEntry:
    milestone_action_count: int
    captured_action_count: int
    captured_at: str
    payload: Any = field(repr=False)

    def to_private_record(self) -> dict[str, Any]:
        return {
            "milestone_action_count": self.milestone_action_count,
            "captured_action_count": self.captured_action_count,
            "captured_at": self.captured_at,
        }


class RollingEpisodeCheckpointRing:
    """Keep at most two safe action-boundary checkpoint payloads in memory."""

    def __init__(
        self,
        *,
        episode_id: str,
        action_interval: int = DEFAULT_ACTION_INTERVAL,
        maximum_retained: int = DEFAULT_MAXIMUM_RETAINED,
    ) -> None:
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("checkpoint episode_id must be non-empty")
        if (
            isinstance(action_interval, bool)
            or not isinstance(action_interval, int)
            or action_interval <= 0
        ):
            raise ValueError("checkpoint action interval must be positive")
        if (
            isinstance(maximum_retained, bool)
            or not isinstance(maximum_retained, int)
            or maximum_retained <= 0
        ):
            raise ValueError("maximum retained checkpoints must be positive")
        self.episode_id = episode_id
        self.action_interval = action_interval
        self.maximum_retained = maximum_retained
        self._next_milestone = action_interval
        self._entries: OrderedDict[int, RollingCheckpointEntry] = OrderedDict()
        self._deferred_boundaries = 0
        self._restore_count = 0

    @property
    def captured_action_counts(self) -> tuple[int, ...]:
        return tuple(self._entries)

    @property
    def next_milestone_action_count(self) -> int:
        return self._next_milestone

    def capture_if_due(
        self,
        *,
        completed_action_count: int,
        safe_boundary: bool,
        capture: Callable[[], Any],
    ) -> RollingCheckpointEntry | None:
        if (
            isinstance(completed_action_count, bool)
            or not isinstance(completed_action_count, int)
            or completed_action_count < 0
        ):
            raise ValueError("completed action count must be non-negative")
        if not isinstance(safe_boundary, bool):
            raise TypeError("safe_boundary must be boolean")
        if not callable(capture):
            raise TypeError("checkpoint capture must be callable")
        if completed_action_count < self._next_milestone:
            return None
        if not safe_boundary:
            self._deferred_boundaries += 1
            return None

        milestone = self._next_milestone
        # Build the new payload before evicting the oldest valid checkpoint.
        # If capture fails, the previous two recovery points remain intact.
        payload = capture()
        entry = RollingCheckpointEntry(
            milestone_action_count=milestone,
            captured_action_count=completed_action_count,
            captured_at=datetime.now(timezone.utc).isoformat(),
            payload=payload,
        )
        self._entries[completed_action_count] = entry
        while len(self._entries) > self.maximum_retained:
            self._entries.popitem(last=False)
        self._next_milestone = (
            completed_action_count // self.action_interval + 1
        ) * self.action_interval
        return entry

    def require(self, captured_action_count: int | None = None) -> RollingCheckpointEntry:
        if not self._entries:
            raise RollingCheckpointError("resident episode has no rolling checkpoint")
        if captured_action_count is None:
            return next(reversed(self._entries.values()))
        if (
            isinstance(captured_action_count, bool)
            or not isinstance(captured_action_count, int)
            or captured_action_count <= 0
        ):
            raise RollingCheckpointError("checkpoint action count is invalid")
        try:
            return self._entries[captured_action_count]
        except KeyError as exc:
            raise RollingCheckpointError(
                "requested rolling checkpoint is not retained"
            ) from exc

    def record_restore(self, *, captured_action_count: int) -> int:
        if captured_action_count not in self._entries:
            raise RollingCheckpointError(
                "restored checkpoint is not retained by this ring"
            )
        # Checkpoints from the abandoned future branch must never be offered
        # after rollback.  The next capture is due at the next 50-action
        # milestone following the restored sequence.
        for action_count in tuple(self._entries):
            if action_count > captured_action_count:
                del self._entries[action_count]
        self._next_milestone = (
            captured_action_count // self.action_interval + 1
        ) * self.action_interval
        self._restore_count += 1
        return self._restore_count

    def adopt_restored_payload(
        self,
        *,
        captured_action_count: int,
        payload: Any,
    ) -> RollingCheckpointEntry:
        """Seed a fresh resident ring from one verified durable checkpoint."""

        if self._entries:
            raise RollingCheckpointError("checkpoint ring is not empty")
        if (
            isinstance(captured_action_count, bool)
            or not isinstance(captured_action_count, int)
            or captured_action_count <= 0
        ):
            raise RollingCheckpointError("restored checkpoint action count is invalid")
        entry = RollingCheckpointEntry(
            milestone_action_count=captured_action_count,
            captured_action_count=captured_action_count,
            captured_at=datetime.now(timezone.utc).isoformat(),
            payload=payload,
        )
        self._entries[captured_action_count] = entry
        self._next_milestone = (
            captured_action_count // self.action_interval + 1
        ) * self.action_interval
        self._restore_count = 1
        return entry

    def release_payloads(self) -> None:
        """Release process-memory snapshots after a different episode begins."""

        self._entries.clear()

    def to_private_record(self) -> Mapping[str, Any]:
        return {
            "schema_version": ROLLING_EPISODE_CHECKPOINT_VERSION,
            "episode_id": self.episode_id,
            "action_interval": self.action_interval,
            "maximum_retained": self.maximum_retained,
            "next_milestone_action_count": self._next_milestone,
            "deferred_boundary_count": self._deferred_boundaries,
            "restore_count": self._restore_count,
            "retained": [
                entry.to_private_record() for entry in self._entries.values()
            ],
        }
