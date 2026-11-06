"""One carry fact with an explicit acquisition provenance.

The public runtime used to maintain a second inferred carry boolean beside the
Symbolic primitive's canonical held-object state.  Private Oracle setup could
therefore hold an object while the public action router still believed the
hand was empty.  ``CarryStateTracker`` keeps the public-safe controller view,
but, when a canonical provider is supplied, derives carry presence from that
single held-object fact and stores only how the fact was established.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Mapping

from behavior_hl.embodied.policy.protocol import (
    ActionEnvelope,
    MarkerGraspArgs,
    MarkerPlaceInsideArgs,
    MarkerTargetOrRequestArgs,
    MarkerTargetArgs,
    NoArgs,
    PointActionArgs,
    ReleaseArgs,
    VisibleObjectGraspArgs,
)
from behavior_hl.embodied.runtime.dispatch import ActionPluginResult


PUBLIC_CARRY_STATE_VERSION = "carry-state-with-provenance-v2"
_FORBIDDEN_FEEDBACK_KEY_FRAGMENTS = (
    "bddl",
    "contact",
    "goal_truth",
    "held",
    "inside",
    "object",
    "official",
    "predicate",
    "success_truth",
    "task_state",
    "world_pose",
)
_EXPLICIT_PUBLIC_SENSOR_FEEDBACK_KEYS = frozenset({"object_acquired"})
_GRASP_ACTIONS = frozenset({"grasp_at", "grasp", "pick"})
_PLACE_ACTIONS = frozenset(
    {
        "place_inside_at",
        "place_inside",
        "place_on_top",
        "place_next_to",
        "place_under",
        "place_touching",
        "place_on_floor",
        "put_inside",
        "put_back",
    }
)
_NO_STATE_ACTIONS = frozenset(
    {
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
        "turn_left",
        "turn_right",
        "look_up",
        "look_down",
        "look_center",
        "move_up",
        "move_down",
        "mark_visible_objects",
        "open",
        "close",
        "toggle_on",
        "toggle_off",
        "wait",
        "stop",
    }
)


class PublicCarryStateError(ValueError):
    """Raised before private or malformed feedback can affect carry state."""


class CarryConfidence(str, Enum):
    NOT_LIKELY = "not_likely"
    LIKELY = "likely"
    UNKNOWN = "unknown"


class CarryAcquisitionSource(str, Enum):
    """How the current canonical held-object fact was established."""

    NONE = "none"
    PUBLIC_PICK = "public_pick"
    PRIVATE_SETUP = "private_setup"
    PRIVATE_ORACLE = "private_oracle"
    UNKNOWN = "unknown"


def completed_action_carry_feedback(
    action: ActionEnvelope,
) -> dict[str, bool]:
    """Return the carry fact required when an interrupted action succeeded."""

    if action.action == "pick":
        return {"object_acquired": True}
    if action.action in _GRASP_ACTIONS:
        return {"grasp_proxy_likely": True}
    if action.action in _PLACE_ACTIONS:
        return {"gripper_open_commanded": True}
    return {}


class CarryStateTracker:
    """Track one carry fact and its provenance without exposing identity.

    ``held_object_provider`` is optional for the legacy physical-action path.
    When present, its return value is the sole carry-presence truth.  The
    tracker never serializes or publishes that object; snapshots expose only a
    boolean and the acquisition source needed for admission / audit.
    """

    def __init__(
        self,
        *,
        implicit_grasp_hand: str | None = None,
        held_object_provider: Callable[[], Any | None] | None = None,
    ) -> None:
        if implicit_grasp_hand not in {None, "left", "right"}:
            raise ValueError("implicit_grasp_hand must be left, right, or None")
        if held_object_provider is not None and not callable(
            held_object_provider
        ):
            raise TypeError("held_object_provider must be callable or None")
        self._implicit_grasp_hand = implicit_grasp_hand
        self._held_object_provider = held_object_provider
        self._confidence = CarryConfidence.NOT_LIKELY
        self._active_hand: str | None = None
        self._acquired_via = CarryAcquisitionSource.NONE
        self._transition_count = 0
        self._last_transition = "initial"

    def likely(self) -> bool:
        confidence, _hand, _source, _present = self._resolved_state()
        return confidence is CarryConfidence.LIKELY

    @property
    def implicit_grasp_hand(self) -> str | None:
        """Executor-selected hand for public grasp actions without a hand arg."""

        return self._implicit_grasp_hand

    def active_hand(self) -> str | None:
        confidence, hand, _source, _present = self._resolved_state()
        return hand if confidence is CarryConfidence.LIKELY else None

    def held_object(self) -> Any | None:
        """Return the canonical private object token, when configured.

        This method is executor-private.  It must never be copied into public
        feedback, prompts, or public observation artifacts.
        """

        if self._held_object_provider is None:
            return None
        try:
            return self._held_object_provider()
        except Exception as exc:
            raise PublicCarryStateError(
                "canonical held-object provider failed"
            ) from exc

    def record_private_acquisition(
        self,
        *,
        expected_object: Any,
        acquired_via: CarryAcquisitionSource | str,
    ) -> dict[str, Any]:
        """Attribute an already-established private canonical grasp.

        This is a private Oracle/setup adapter, not a public action.  Identity
        must match the canonical provider exactly before provenance changes.
        """

        source = _coerce_private_acquisition_source(acquired_via)
        if self._held_object_provider is None:
            raise PublicCarryStateError(
                "private acquisition requires a canonical held-object provider"
            )
        held = self.held_object()
        if held is None or held is not expected_object:
            raise PublicCarryStateError(
                "private acquisition does not match canonical held object"
            )
        if self._implicit_grasp_hand not in {"left", "right"}:
            raise PublicCarryStateError(
                "private acquisition requires one configured implicit hand"
            )
        self._transition(
            CarryConfidence.LIKELY,
            self._implicit_grasp_hand,
            f"canonical_{source.value}_acquired",
            acquired_via=source,
        )
        return self.snapshot()

    def record_private_release(self, *, reason: str) -> dict[str, Any]:
        """Clear provenance after a private action clears canonical carry."""

        if not isinstance(reason, str) or not reason:
            raise ValueError("private release reason must be non-empty")
        if self._held_object_provider is None:
            raise PublicCarryStateError(
                "private release requires a canonical held-object provider"
            )
        if self.held_object() is not None:
            raise PublicCarryStateError(
                "private release cannot clear a non-empty canonical hand"
            )
        self._transition(
            CarryConfidence.NOT_LIKELY,
            None,
            reason,
            acquired_via=CarryAcquisitionSource.NONE,
        )
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        confidence, hand, source, held_present = self._resolved_state()
        return {
            "schema_version": PUBLIC_CARRY_STATE_VERSION,
            "confidence": confidence.value,
            "likely": confidence is CarryConfidence.LIKELY,
            "active_hand": (
                hand if confidence is CarryConfidence.LIKELY else None
            ),
            "acquired_via": source.value,
            "canonical_held_object_present": held_present,
            "canonical_provider_enabled": (
                self._held_object_provider is not None
            ),
            "transition_count": self._transition_count,
            "last_transition": self._last_transition,
        }

    def _resolved_state(
        self,
    ) -> tuple[
        CarryConfidence,
        str | None,
        CarryAcquisitionSource,
        bool | None,
    ]:
        """Resolve carry presence from exactly one configured fact source."""

        if self._held_object_provider is None:
            return (
                self._confidence,
                self._active_hand,
                self._acquired_via,
                None,
            )
        held_present = self.held_object() is not None
        attributed = self._acquired_via in {
            CarryAcquisitionSource.PUBLIC_PICK,
            CarryAcquisitionSource.PRIVATE_SETUP,
            CarryAcquisitionSource.PRIVATE_ORACLE,
        }
        if held_present and attributed:
            if self._active_hand not in {"left", "right"}:
                return (
                    CarryConfidence.UNKNOWN,
                    None,
                    CarryAcquisitionSource.UNKNOWN,
                    True,
                )
            return (
                CarryConfidence.LIKELY,
                self._active_hand,
                self._acquired_via,
                True,
            )
        if not held_present and self._acquired_via is CarryAcquisitionSource.NONE:
            return (
                CarryConfidence.NOT_LIKELY,
                None,
                CarryAcquisitionSource.NONE,
                False,
            )
        # A canonical object without provenance, or stale provenance without
        # an object, is a real consistency fault and must never be guessed.
        return (
            CarryConfidence.UNKNOWN,
            None,
            CarryAcquisitionSource.UNKNOWN,
            held_present,
        )

    def update(
        self,
        action: ActionEnvelope,
        result: ActionPluginResult,
    ) -> dict[str, Any]:
        if not isinstance(action, ActionEnvelope):
            raise TypeError("action must be a parsed ActionEnvelope")
        if not isinstance(result, ActionPluginResult):
            raise TypeError("result must be ActionPluginResult")
        _reject_private_feedback(result.public_feedback)

        # Action-first selection frames are observation-only public actions.
        # They must never create, clear, or downgrade the existing carry
        # proxy, regardless of which placement operation requested them.
        if result.status in {"selection_required", "markers_unavailable"}:
            return self.snapshot()

        name = action.action
        if name in _NO_STATE_ACTIONS:
            return self.snapshot()
        if name in _GRASP_ACTIONS:
            self._update_grasp(action, result)
            return self.snapshot()
        if name == "release":
            self._update_release(action, result)
            return self.snapshot()
        if name in _PLACE_ACTIONS:
            self._update_place(action, result)
            return self.snapshot()
        # Future actions are inert until their public carry semantics are
        # explicitly versioned here.
        return self.snapshot()

    def _update_grasp(
        self,
        action: ActionEnvelope,
        result: ActionPluginResult,
    ) -> None:
        if action.action == "grasp_at":
            if not isinstance(action.args, PointActionArgs):
                raise PublicCarryStateError(
                    "grasp_at requires parsed PointActionArgs"
                )
            hand = action.args.hand
        elif action.action == "grasp":
            if not isinstance(
                action.args,
                (VisibleObjectGraspArgs, MarkerGraspArgs),
            ):
                raise PublicCarryStateError(
                    "grasp requires parsed visible-object or marker arguments"
                )
            hand = self._implicit_grasp_hand
            if hand not in {"left", "right"}:
                raise PublicCarryStateError(
                    "grasp requires one configured implicit hand"
                )
        else:
            if not isinstance(
                action.args,
                (MarkerTargetArgs, MarkerTargetOrRequestArgs),
            ):
                raise PublicCarryStateError(
                    "pick requires parsed marker arguments"
                )
            hand = self._implicit_grasp_hand
            if hand not in {"left", "right"}:
                raise PublicCarryStateError(
                    "pick requires one configured implicit hand"
                )
        proxy = result.public_feedback.get("grasp_proxy_likely")
        if proxy is not None and not isinstance(proxy, bool):
            raise PublicCarryStateError(
                "grasp_proxy_likely must be boolean"
            )
        object_acquired = result.public_feedback.get("object_acquired")
        if object_acquired is not None and not isinstance(
            object_acquired, bool
        ):
            raise PublicCarryStateError("object_acquired must be boolean")
        if result.status == "not_executed" and object_acquired is not False:
            raise PublicCarryStateError(
                "not_executed grasp must report object_acquired=false"
            )
        status_likely = result.status == "grasp_likely" or (
            action.action == "pick"
            and result.status == "completed"
            and object_acquired is True
        )
        if status_likely and proxy is False:
            raise PublicCarryStateError(
                "grasp status conflicts with public proxy"
            )
        if not status_likely and proxy is not True:
            # Failed or uncertain grasps cannot create carry state.
            return
        self._transition(
            CarryConfidence.LIKELY,
            hand,
            "public_grasp_proxy_likely",
            acquired_via=CarryAcquisitionSource.PUBLIC_PICK,
        )

    def _update_release(
        self,
        action: ActionEnvelope,
        result: ActionPluginResult,
    ) -> None:
        if isinstance(action.args, ReleaseArgs):
            hand = action.args.hand
        elif isinstance(action.args, NoArgs):
            hand = self._implicit_grasp_hand
            if hand not in {"left", "right"}:
                raise PublicCarryStateError(
                    "marker release requires one configured implicit hand"
                )
        else:
            raise PublicCarryStateError(
                "release requires parsed ReleaseArgs or marker-profile NoArgs"
            )
        # The canonical executor may have cleared the held object immediately
        # before its public result is recorded.  Match the provenance-bound
        # pre-action hand, not the already-updated canonical presence.
        if self._active_hand != hand:
            return
        if result.status == "completed":
            self._transition(
                CarryConfidence.NOT_LIKELY,
                None,
                "public_release_completed",
                acquired_via=CarryAcquisitionSource.NONE,
            )
        else:
            self._resolve_missing_release_feedback(
                empty_reason="canonical_release_result_empty",
                unknown_reason="public_release_failed_closed",
            )

    def _update_place(
        self,
        action: ActionEnvelope,
        result: ActionPluginResult,
    ) -> None:
        if not isinstance(
            action.args,
            (
                PointActionArgs,
                MarkerPlaceInsideArgs,
                MarkerTargetArgs,
                MarkerTargetOrRequestArgs,
                NoArgs,
            ),
        ):
            raise PublicCarryStateError(
                "place action requires parsed point or marker arguments"
            )
        gripper_open_commanded = result.public_feedback.get(
            "gripper_open_commanded"
        )
        if (
            gripper_open_commanded is not None
            and not isinstance(gripper_open_commanded, bool)
        ):
            raise PublicCarryStateError(
                "gripper_open_commanded must be boolean"
            )
        if isinstance(action.args, PointActionArgs):
            if self._active_hand != action.args.hand:
                return
        elif self._active_hand != self._implicit_grasp_hand:
            return
        if gripper_open_commanded is True:
            # This public execution fact is emitted immediately after the
            # gripper-open command.  Later retract / settle failures cannot
            # restore a carry proxy, and placement truth remains private.
            self._transition(
                CarryConfidence.NOT_LIKELY,
                None,
                "public_place_gripper_open_commanded",
                acquired_via=CarryAcquisitionSource.NONE,
            )
        elif gripper_open_commanded is False:
            # A stale / unknown visual marker is rejected before the official
            # placement primitive commands release.  The previous public
            # carry proxy therefore remains valid and the policy may re-mark
            # the current frame and try again.
            return
        else:
            self._resolve_missing_release_feedback(
                empty_reason="canonical_place_result_empty",
                unknown_reason="public_place_failed_closed",
            )

    def _resolve_missing_release_feedback(
        self,
        *,
        empty_reason: str,
        unknown_reason: str,
    ) -> None:
        """Use canonical presence only when public release feedback is absent."""

        if self._held_object_provider is None:
            self._transition(
                CarryConfidence.UNKNOWN,
                None,
                unknown_reason,
                acquired_via=CarryAcquisitionSource.UNKNOWN,
            )
        elif self.held_object() is None:
            self._transition(
                CarryConfidence.NOT_LIKELY,
                None,
                empty_reason,
                acquired_via=CarryAcquisitionSource.NONE,
            )

    def _transition(
        self,
        confidence: CarryConfidence,
        hand: str | None,
        reason: str,
        *,
        acquired_via: CarryAcquisitionSource,
    ) -> None:
        if confidence is CarryConfidence.LIKELY:
            if hand not in {"left", "right"}:
                raise PublicCarryStateError(
                    "likely carry state requires an active hand"
                )
        elif hand is not None:
            raise PublicCarryStateError(
                "non-likely carry state cannot retain an active hand"
            )
        self._confidence = confidence
        self._active_hand = hand
        self._acquired_via = acquired_via
        self._transition_count += 1
        self._last_transition = reason


# Import compatibility for the existing public runtime.  New code should use
# the less misleading name; both names denote the same single state class.
PublicCarryStateTracker = CarryStateTracker


def _coerce_private_acquisition_source(
    value: CarryAcquisitionSource | str,
) -> CarryAcquisitionSource:
    try:
        source = CarryAcquisitionSource(value)
    except (TypeError, ValueError) as exc:
        raise PublicCarryStateError(
            "private acquisition source is invalid"
        ) from exc
    if source not in {
        CarryAcquisitionSource.PRIVATE_SETUP,
        CarryAcquisitionSource.PRIVATE_ORACLE,
    }:
        raise PublicCarryStateError(
            "private acquisition source must be private_setup or private_oracle"
        )
    return source


def _reject_private_feedback(
    value: Any,
    *,
    path: str = "public_feedback",
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise PublicCarryStateError(
                    f"{path} keys must be non-empty strings"
                )
            folded = key.casefold()
            if key not in _EXPLICIT_PUBLIC_SENSOR_FEEDBACK_KEYS and any(
                fragment in folded
                for fragment in _FORBIDDEN_FEEDBACK_KEY_FRAGMENTS
            ):
                raise PublicCarryStateError(
                    f"{path}.{key} is private evaluator information"
                )
            _reject_private_feedback(item, path=f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_private_feedback(item, path=f"{path}[{index}]")
