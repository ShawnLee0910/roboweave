"""Visibility-gated adapter for BEHAVIOR's official PLACE_INSIDE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from behavior_hl.embodied.evaluator_bridge.bounded_official_place import (
    OfficialPlaceStatus,
    OfficialPlaceTick,
)
from behavior_hl.embodied.evaluator_bridge.private_instance_visibility import (
    PrivateVisibilityResolution,
    VisibilityStatus,
)
from behavior_hl.embodied.policy.protocol import ActionEnvelope, PointActionArgs
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


VISIBLE_OFFICIAL_PLACE_BACKEND_SERVICE = "visible_official_place_backend"
VISIBLE_OFFICIAL_PLACE_CONTROLLER_KEY = (
    "manipulation.visible_official_place_inside_at"
)


class VisibleOfficialPlaceContractError(RuntimeError):
    """Raised when a private placement backend violates the contract."""


class VisibleOfficialPlaceBackend(Protocol):
    def resolve_visible_container(
        self, *, camera: str, point: tuple[float, float], hand: str
    ) -> PrivateVisibilityResolution: ...

    def begin_official_place(self, *, container_token: Any, hand: str) -> Any: ...

    def tick_official_place(self, state_token: Any) -> OfficialPlaceTick: ...

    def abort_official_place(self, state_token: Any) -> OfficialPlaceTick: ...


@dataclass(slots=True)
class _VisibleOfficialPlaceLifecycle:
    backend: VisibleOfficialPlaceBackend = field(repr=False)
    args: PointActionArgs
    audit: PrivateGraspAuditSink | None = field(default=None, repr=False)
    stage: str = "visibility"
    driver_token: Any = field(default=None, repr=False)
    terminal_status: str | None = None
    terminal_private_status: str | None = field(default=None, repr=False)
    release_observed: bool = False


class VisibleOfficialPlacePlugin:
    """Opt-in place implementation with visual-only policy feedback."""

    controller_key = VISIBLE_OFFICIAL_PLACE_CONTROLLER_KEY

    def begin(
        self,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> _VisibleOfficialPlaceLifecycle:
        if action.action != "place_inside_at" or not isinstance(
            action.args, PointActionArgs
        ):
            raise VisibleOfficialPlaceContractError(
                "visible official place accepts parsed place_inside_at only"
            )
        try:
            backend = context.services[VISIBLE_OFFICIAL_PLACE_BACKEND_SERVICE]
        except KeyError as exc:
            raise VisibleOfficialPlaceContractError(
                "missing visible official place backend"
            ) from exc
        _validate_backend(backend)
        audit = context.services.get(PRIVATE_GRASP_AUDIT_SERVICE)
        if audit is not None and not callable(getattr(audit, "record", None)):
            raise VisibleOfficialPlaceContractError(
                "private place audit sink must provide record(event)"
            )
        return _VisibleOfficialPlaceLifecycle(
            backend=backend,
            args=action.args,
            audit=audit,
        )

    def tick(
        self,
        state: _VisibleOfficialPlaceLifecycle,
        context: ActionExecutionContext,
    ) -> ActionPluginTick:
        del context
        if state.terminal_status is not None:
            raise VisibleOfficialPlaceContractError(
                "cannot tick a terminal visible place"
            )
        if state.stage == "visibility":
            resolution = state.backend.resolve_visible_container(
                camera=state.args.camera,
                point=state.args.point,
                hand=state.args.hand,
            )
            if not isinstance(resolution, PrivateVisibilityResolution):
                raise VisibleOfficialPlaceContractError(
                    "visibility backend returned an invalid result"
                )
            if resolution.status is not VisibilityStatus.VISIBLE:
                self._record(
                    state,
                    "place_visibility_gate_rejected",
                    {"private_status": resolution.status.value},
                )
                state.terminal_status = "place_noop"
                state.terminal_private_status = resolution.status.value
                return ActionPluginTick(
                    done=True,
                    simulator_steps=0,
                    capture_observation=True,
                )
            state.driver_token = state.backend.begin_official_place(
                container_token=resolution.target.target_token,
                hand=state.args.hand,
            )
            state.stage = "official_place"
            self._record(
                state,
                "place_visibility_gate_admitted",
                {
                    "container_name": resolution.target.object_name,
                    "visible_pixels": resolution.target.visible_pixels,
                },
            )
            return ActionPluginTick(done=False, simulator_steps=0)

        if state.stage != "official_place":
            raise VisibleOfficialPlaceContractError(
                f"unknown visible place stage {state.stage!r}"
            )
        tick = state.backend.tick_official_place(state.driver_token)
        if not isinstance(tick, OfficialPlaceTick):
            raise VisibleOfficialPlaceContractError(
                "official place backend returned an invalid tick"
            )
        state.release_observed = state.release_observed or tick.release_observed
        if not tick.done:
            return ActionPluginTick(
                done=False,
                simulator_steps=tick.simulator_steps,
            )
        state.terminal_private_status = tick.status.value
        state.terminal_status = (
            "completed"
            if tick.status is OfficialPlaceStatus.SUCCEEDED
            else "place_uncertain"
        )
        self._record(
            state,
            "official_place_terminal",
            {
                "private_status": tick.status.value,
                "release_observed": state.release_observed,
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
        state: _VisibleOfficialPlaceLifecycle,
        context: ActionExecutionContext,
    ) -> ActionPluginResult:
        del context
        if state.terminal_status is None:
            raise VisibleOfficialPlaceContractError(
                "cannot finish a non-terminal visible place"
            )
        feedback = (
            {"gripper_open_commanded": True}
            if state.terminal_status == "completed"
            else {}
        )
        return ActionPluginResult(
            status=state.terminal_status,
            public_feedback=feedback,
        )

    def abort(
        self,
        state: _VisibleOfficialPlaceLifecycle,
        context: ActionExecutionContext,
        reason: str,
    ) -> ActionPluginResult:
        del context
        if state.driver_token is not None and state.terminal_status is None:
            tick = state.backend.abort_official_place(state.driver_token)
            state.release_observed = (
                state.release_observed or tick.release_observed
            )
        self._record(state, "official_place_aborted", {"reason": reason})
        state.terminal_status = "place_uncertain"
        state.terminal_private_status = "aborted"
        return ActionPluginResult(status="place_uncertain", public_feedback={})

    @staticmethod
    def _record(
        state: _VisibleOfficialPlaceLifecycle,
        event: str,
        details: dict[str, Any],
    ) -> None:
        if state.audit is not None:
            state.audit.record(
                PrivateGraspAuditEvent(event=event, details=details)
            )


def _validate_backend(value: Any) -> None:
    for name in (
        "resolve_visible_container",
        "begin_official_place",
        "tick_official_place",
        "abort_official_place",
    ):
        if not callable(getattr(value, name, None)):
            raise VisibleOfficialPlaceContractError(
                f"visible official place backend lacks {name}()"
            )
