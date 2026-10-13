"""Visibility-gated adapter for BEHAVIOR's official Starter GRASP.

The policy supplies only an onboard image point.  Private evaluator services
may bind that point to a task object visible in the synchronized instance
mask.  The object handle and all failure reasons remain private.  A visible
target is handed to the official Starter primitive under a tight base-motion
budget; an invisible target is a zero-step no-op.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from behavior_hl.embodied.evaluator_bridge.bounded_official_grasp import (
    OfficialGraspStatus,
    OfficialGraspTick,
)
from behavior_hl.embodied.evaluator_bridge.private_instance_visibility import (
    PrivateVisibilityResolution,
    VisibilityStatus,
)
from behavior_hl.embodied.policy.protocol import (
    ActionEnvelope,
    PointActionArgs,
    VisibleObjectGraspArgs,
)
from behavior_hl.embodied.runtime.dispatch import (
    ActionExecutionContext,
    ActionPluginResult,
    ActionPluginTick,
)
from behavior_hl.embodied.runtime.plugins.grasp import (
    PRIVATE_GRASP_AUDIT_SERVICE,
    PrivateGraspAuditEvent,
    PrivateGraspAuditSink,
)


VISIBLE_OFFICIAL_GRASP_BACKEND_SERVICE = "visible_official_grasp_backend"
VISIBLE_OFFICIAL_GRASP_CONTROLLER_KEY = (
    "manipulation.visible_official_grasp_at"
)
VISIBLE_OBJECT_OFFICIAL_GRASP_CONTROLLER_KEY = (
    "manipulation.visible_object_official_grasp"
)


class VisibleOfficialGraspContractError(RuntimeError):
    """Raised when a private backend violates the macro contract."""


class VisibleOfficialGraspBackend(Protocol):
    """Private evaluator-side service consumed by the plugin."""

    def resolve_visible_target(
        self,
        *,
        camera: str,
        point: tuple[float, float],
        hand: str,
    ) -> PrivateVisibilityResolution:
        """Resolve a point only in the synchronized current segmentation."""

    def begin_official_grasp(self, *, target_token: Any, hand: str) -> Any:
        """Begin a bounded official GRASP and return an opaque state token."""

    def tick_official_grasp(self, state_token: Any) -> OfficialGraspTick:
        """Advance the official primitive by at most one simulator step."""

    def abort_official_grasp(self, state_token: Any) -> OfficialGraspTick:
        """Stop commands without restoring already-observed physical motion."""

    def resolve_visible_object(
        self,
        *,
        object_category: str,
        selector: str | None,
    ) -> PrivateVisibilityResolution:
        """Resolve a category only from synchronized current segmentation."""


@dataclass(slots=True)
class _VisibleOfficialGraspLifecycle:
    backend: VisibleOfficialGraspBackend = field(repr=False)
    args: PointActionArgs
    audit: PrivateGraspAuditSink | None = field(default=None, repr=False)
    stage: str = "visibility"
    driver_token: Any = field(default=None, repr=False)
    terminal_status: str | None = None
    terminal_private_status: str | None = field(default=None, repr=False)


class VisibleOfficialGraspPlugin:
    """Opt-in ``grasp_at`` implementation with silent visual feedback only."""

    controller_key = VISIBLE_OFFICIAL_GRASP_CONTROLLER_KEY

    def begin(
        self,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> _VisibleOfficialGraspLifecycle:
        if action.action != "grasp_at" or not isinstance(
            action.args,
            PointActionArgs,
        ):
            raise VisibleOfficialGraspContractError(
                "visible official grasp accepts parsed grasp_at only"
            )
        try:
            backend = context.services[
                VISIBLE_OFFICIAL_GRASP_BACKEND_SERVICE
            ]
        except KeyError as exc:
            raise VisibleOfficialGraspContractError(
                "missing visible official grasp backend"
            ) from exc
        _validate_backend(backend)
        audit = context.services.get(PRIVATE_GRASP_AUDIT_SERVICE)
        if audit is not None and not callable(getattr(audit, "record", None)):
            raise VisibleOfficialGraspContractError(
                "private grasp audit sink must provide record(event)"
            )
        return _VisibleOfficialGraspLifecycle(
            backend=backend,
            args=action.args,
            audit=audit,
        )

    def tick(
        self,
        state: _VisibleOfficialGraspLifecycle,
        context: ActionExecutionContext,
    ) -> ActionPluginTick:
        del context
        if state.terminal_status is not None:
            raise VisibleOfficialGraspContractError(
                "cannot tick a terminal visible grasp"
            )
        if state.stage == "visibility":
            resolution = state.backend.resolve_visible_target(
                camera=state.args.camera,
                point=state.args.point,
                hand=state.args.hand,
            )
            if not isinstance(resolution, PrivateVisibilityResolution):
                raise VisibleOfficialGraspContractError(
                    "visibility backend returned an invalid result"
                )
            if resolution.status is not VisibilityStatus.VISIBLE:
                self._record(
                    state,
                    "visibility_gate_rejected",
                    {"private_status": resolution.status.value},
                )
                # No simulation command and no textual reason.  The next
                # policy turn sees only a newly captured onboard RGB frame.
                state.terminal_status = "grasp_uncertain"
                state.terminal_private_status = resolution.status.value
                return ActionPluginTick(
                    done=True,
                    simulator_steps=0,
                    capture_observation=True,
                )
            state.driver_token = state.backend.begin_official_grasp(
                target_token=resolution.target.target_token,
                hand=state.args.hand,
            )
            state.stage = "official_grasp"
            self._record(
                state,
                "visibility_gate_admitted",
                {
                    "object_name": resolution.target.object_name,
                    "visible_pixels": resolution.target.visible_pixels,
                },
            )
            return ActionPluginTick(done=False, simulator_steps=0)

        if state.stage != "official_grasp":
            raise VisibleOfficialGraspContractError(
                f"unknown visible grasp stage {state.stage!r}"
            )
        tick = state.backend.tick_official_grasp(state.driver_token)
        if not isinstance(tick, OfficialGraspTick):
            raise VisibleOfficialGraspContractError(
                "official grasp backend returned an invalid tick"
            )
        if not tick.done:
            return ActionPluginTick(
                done=False,
                simulator_steps=tick.simulator_steps,
            )
        state.terminal_private_status = tick.status.value
        state.terminal_status = (
            "grasp_likely"
            if tick.status is OfficialGraspStatus.SUCCEEDED
            else "grasp_uncertain"
        )
        self._record(
            state,
            "official_grasp_terminal",
            {
                "private_status": tick.status.value,
                "base_translation_m": tick.base_translation_m,
                "base_yaw_rad": tick.base_yaw_rad,
            },
        )
        return ActionPluginTick(
            done=True,
            simulator_steps=tick.simulator_steps,
            capture_observation=True,
        )

    def finish(
        self,
        state: _VisibleOfficialGraspLifecycle,
        context: ActionExecutionContext,
    ) -> ActionPluginResult:
        del context
        if state.terminal_status is None:
            raise VisibleOfficialGraspContractError(
                "cannot finish a non-terminal visible grasp"
            )
        # Deliberately empty: no visibility, reach, IK, contact, or held-state
        # truth is expressed in language to the policy.
        return ActionPluginResult(
            status=state.terminal_status,
            public_feedback={},
        )

    def abort(
        self,
        state: _VisibleOfficialGraspLifecycle,
        context: ActionExecutionContext,
        reason: str,
    ) -> ActionPluginResult:
        del context
        if state.driver_token is not None and state.terminal_status is None:
            state.backend.abort_official_grasp(state.driver_token)
        self._record(state, "official_grasp_aborted", {"reason": reason})
        state.terminal_status = "grasp_uncertain"
        state.terminal_private_status = "aborted"
        return ActionPluginResult(status="grasp_uncertain", public_feedback={})

    @staticmethod
    def _record(
        state: _VisibleOfficialGraspLifecycle,
        event: str,
        details: dict[str, Any],
    ) -> None:
        if state.audit is not None:
            state.audit.record(
                PrivateGraspAuditEvent(event=event, details=details)
            )


@dataclass(slots=True)
class _VisibleObjectOfficialGraspLifecycle:
    backend: VisibleOfficialGraspBackend = field(repr=False)
    args: VisibleObjectGraspArgs
    hand: str
    audit: PrivateGraspAuditSink | None = field(default=None, repr=False)
    stage: str = "visibility"
    driver_token: Any = field(default=None, repr=False)
    terminal_status: str | None = None
    official_simulator_steps: int = 0


class VisibleObjectOfficialGraspPlugin:
    """Category-level ``grasp`` with private masks and visual-only feedback."""

    controller_key = VISIBLE_OBJECT_OFFICIAL_GRASP_CONTROLLER_KEY

    def __init__(self, *, hand: str) -> None:
        if hand not in {"left", "right"}:
            raise ValueError("hand must be left or right")
        self.hand = hand

    def begin(
        self,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> _VisibleObjectOfficialGraspLifecycle:
        if action.action != "grasp" or not isinstance(
            action.args,
            VisibleObjectGraspArgs,
        ):
            raise VisibleOfficialGraspContractError(
                "visible-object official grasp accepts parsed grasp only"
            )
        try:
            backend = context.services[
                VISIBLE_OFFICIAL_GRASP_BACKEND_SERVICE
            ]
        except KeyError as exc:
            raise VisibleOfficialGraspContractError(
                "missing visible official grasp backend"
            ) from exc
        _validate_object_backend(backend)
        audit = context.services.get(PRIVATE_GRASP_AUDIT_SERVICE)
        if audit is not None and not callable(getattr(audit, "record", None)):
            raise VisibleOfficialGraspContractError(
                "private grasp audit sink must provide record(event)"
            )
        return _VisibleObjectOfficialGraspLifecycle(
            backend=backend,
            args=action.args,
            hand=self.hand,
            audit=audit,
        )

    def tick(
        self,
        state: _VisibleObjectOfficialGraspLifecycle,
        context: ActionExecutionContext,
    ) -> ActionPluginTick:
        del context
        if state.terminal_status is not None:
            raise VisibleOfficialGraspContractError(
                "cannot tick a terminal visible-object grasp"
            )
        if state.stage == "visibility":
            resolution = state.backend.resolve_visible_object(
                object_category=state.args.object,
                selector=state.args.selector,
            )
            if not isinstance(resolution, PrivateVisibilityResolution):
                raise VisibleOfficialGraspContractError(
                    "visibility backend returned an invalid result"
                )
            if resolution.status is not VisibilityStatus.VISIBLE:
                self._record(
                    state,
                    "visibility_category_gate_rejected",
                    {"private_status": resolution.status.value},
                )
                state.terminal_status = "grasp_uncertain"
                return ActionPluginTick(
                    done=True,
                    simulator_steps=0,
                    capture_observation=True,
                )
            state.driver_token = state.backend.begin_official_grasp(
                target_token=resolution.target.target_token,
                hand=state.hand,
            )
            state.stage = "official_grasp"
            self._record(
                state,
                "visibility_category_gate_admitted",
                {
                    "requested_category": state.args.object,
                    "selector": state.args.selector,
                    "object_name": resolution.target.object_name,
                    "visible_pixels": resolution.target.visible_pixels,
                },
            )
            return ActionPluginTick(done=False, simulator_steps=0)

        if state.stage != "official_grasp":
            raise VisibleOfficialGraspContractError(
                f"unknown visible-object grasp stage {state.stage!r}"
            )
        tick = state.backend.tick_official_grasp(state.driver_token)
        if not isinstance(tick, OfficialGraspTick):
            raise VisibleOfficialGraspContractError(
                "official grasp backend returned an invalid tick"
            )
        state.official_simulator_steps += tick.simulator_steps
        if not tick.done:
            return ActionPluginTick(
                done=False,
                simulator_steps=tick.simulator_steps,
            )
        state.terminal_status = (
            "grasp_likely"
            if tick.status is OfficialGraspStatus.SUCCEEDED
            else "grasp_uncertain"
        )
        self._record(
            state,
            "official_category_grasp_terminal",
            {
                "private_status": tick.status.value,
                "base_translation_m": tick.base_translation_m,
                "base_yaw_rad": tick.base_yaw_rad,
                "simulator_steps": state.official_simulator_steps,
            },
        )
        return ActionPluginTick(
            done=True,
            simulator_steps=tick.simulator_steps,
            capture_observation=True,
        )

    def finish(
        self,
        state: _VisibleObjectOfficialGraspLifecycle,
        context: ActionExecutionContext,
    ) -> ActionPluginResult:
        del context
        if state.terminal_status is None:
            raise VisibleOfficialGraspContractError(
                "cannot finish a non-terminal visible-object grasp"
            )
        return ActionPluginResult(
            status=state.terminal_status,
            public_feedback={},
        )

    def abort(
        self,
        state: _VisibleObjectOfficialGraspLifecycle,
        context: ActionExecutionContext,
        reason: str,
    ) -> ActionPluginResult:
        del context
        if state.driver_token is not None and state.terminal_status is None:
            state.backend.abort_official_grasp(state.driver_token)
        self._record(state, "official_category_grasp_aborted", {"reason": reason})
        state.terminal_status = "grasp_uncertain"
        return ActionPluginResult(status="grasp_uncertain", public_feedback={})

    @staticmethod
    def _record(
        state: _VisibleObjectOfficialGraspLifecycle,
        event: str,
        details: dict[str, Any],
    ) -> None:
        if state.audit is not None:
            state.audit.record(
                PrivateGraspAuditEvent(event=event, details=details)
            )


def _validate_backend(value: Any) -> None:
    for name in (
        "resolve_visible_target",
        "begin_official_grasp",
        "tick_official_grasp",
        "abort_official_grasp",
    ):
        if not callable(getattr(value, name, None)):
            raise VisibleOfficialGraspContractError(
                f"visible official grasp backend lacks {name}()"
            )


def _validate_object_backend(value: Any) -> None:
    for name in (
        "resolve_visible_object",
        "begin_official_grasp",
        "tick_official_grasp",
        "abort_official_grasp",
    ):
        if not callable(getattr(value, name, None)):
            raise VisibleOfficialGraspContractError(
                f"visible-object official grasp backend lacks {name}()"
            )
