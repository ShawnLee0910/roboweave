"""Explicit marker request and one-shot official Symbolic GRASP plugins."""

from __future__ import annotations

from behavior_hl.embodied.policy.protocol import HOUSEHOLD_ACTIONS

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from behavior_hl.embodied.evaluator_bridge.private_visual_markers import (
    MarkerPublicFailureReason,
    MarkerStatus,
    PrivateMarkerResolution,
)
from behavior_hl.embodied.policy.protocol import (
    ActionEnvelope,
    MarkerGraspArgs,
    MarkerPlaceInsideArgs,
    MarkerTargetOrRequestArgs,
    MarkerTargetArgs,
    NoArgs,
)
from behavior_hl.embodied.runtime.dispatch import (
    ActionExecutionContext,
    ActionPluginResult,
    ActionPluginTick,
)
from behavior_hl.embodied.runtime.public_action_feedback import (
    action20_blocked_detail,
    ARTICULATED_PART_BLOCKED,
    EXECUTION_UNVERIFIED,
    OUT_OF_REACH,
    SELECTION_UNAVAILABLE,
    TARGET_OUTSIDE_LOCAL_WORKSPACE,
)


MARKER_SYMBOLIC_BACKEND_SERVICE = "marker_symbolic_grasp_backend"
MARK_VISIBLE_CONTROLLER_KEY = "perception.mark_visible_objects"
MARKER_SYMBOLIC_GRASP_CONTROLLER_KEY = "manipulation.marker_symbolic_grasp"
MARKER_SYMBOLIC_PLACE_CONTROLLER_KEY = "manipulation.marker_symbolic_place_inside"
MARKER_SYMBOLIC_V34_CONTROLLER_KEYS = {
    "pick": "manipulation.marker_symbolic_pick",
    "place_inside": "manipulation.marker_symbolic_place_inside",
    "put_back": "manipulation.marker_symbolic_put_back",
    "open": "manipulation.marker_symbolic_open",
    "close": "manipulation.marker_symbolic_close",
    "place_on_top": "manipulation.marker_symbolic_place_on_top",
    "place_next_to": "manipulation.marker_symbolic_place_next_to",
    "place_under": "manipulation.marker_symbolic_place_under",
    "place_touching": "manipulation.marker_symbolic_place_touching",
    "toggle_on": "manipulation.marker_symbolic_toggle_on",
    "toggle_off": "manipulation.marker_symbolic_toggle_off",
    "wipe": "manipulation.marker_symbolic_wipe",
    **{name: f"manipulation.marker_symbolic_{name}" for name in HOUSEHOLD_ACTIONS},
    "release": "manipulation.marker_symbolic_release",
    "place_on_floor": "manipulation.marker_symbolic_place_on_floor",
}

_MARKERLESS_SYMBOLIC_OPERATIONS = frozenset(
    {"put_back", "release", "place_on_floor"}
)
_PLACEMENT_SYMBOLIC_OPERATIONS = frozenset(
    {
        "place_inside",
        "place_on_top",
        "place_next_to",
        "place_under",
        "place_touching",
        "put_back",
        "release",
        "place_on_floor",
    }
)
_ACTION_FIRST_MARKER_OPERATIONS = frozenset(
    {
        "pick",
        "place_inside",
        "place_on_top",
        "place_next_to",
        "place_under",
        "place_touching",
        "open",
        "close",
        "toggle_on",
        "toggle_off",
        "wipe",
        *HOUSEHOLD_ACTIONS,
    }
)
_SECOND_STAGE_SELECTION_KINDS = {
    "place_inside": "inside_cell",
    "place_next_to": "position",
    "place_touching": "direction",
}
_NAVIGATION_BLOCKED_CATEGORIES = frozenset(
    {"navigation_path_blocked", "navigation_collision_blocked"}
)


def _selection_public_feedback(operation: str) -> dict[str, Any]:
    """Describe unchanged public carry state during a zero-step selection."""

    if operation == "pick":
        return {"object_acquired": False}
    if operation in _PLACEMENT_SYMBOLIC_OPERATIONS:
        return {
            "object_acquired": True,
            "gripper_open_commanded": False,
        }
    return {}


class SymbolicGraspStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class SymbolicGraspTick:
    status: SymbolicGraspStatus
    simulator_steps: int
    blocked_category: str | None = None
    blocked_detail: str | None = None


class MarkerSymbolicBackend(Protocol):
    def publish_visible_markers(self, *, action_index: int) -> int:
        """Publish the exact current marked frame and return marker count."""

    def publish_operation_markers(
        self,
        *,
        operation: str,
        action_index: int,
        reference_binding: Any | None = None,
    ) -> int:
        """Publish one operation-specific object / region choice frame."""

    def consume_marker(
        self, *, marker: str, action_index: int, operation: str
    ) -> PrivateMarkerResolution:
        """Resolve one one-shot marker without category fallback."""

    def begin_symbolic_grasp(self, *, target_token: Any) -> Any:
        """Start official SymbolicSemanticActionPrimitives.GRASP."""

    def tick_symbolic_grasp(self, state_token: Any) -> SymbolicGraspTick:
        """Advance the official generator by at most one simulator step."""

    def abort_symbolic_grasp(self, state_token: Any) -> SymbolicGraspTick:
        """Stop the active symbolic grasp safely."""

    def begin_symbolic_place_inside(self, *, target_token: Any) -> Any:
        """Start official SymbolicSemanticActionPrimitives.PLACE_INSIDE."""

    def tick_symbolic_place_inside(self, state_token: Any) -> SymbolicGraspTick:
        """Advance the official placement generator by at most one step."""

    def abort_symbolic_place_inside(self, state_token: Any) -> SymbolicGraspTick:
        """Stop the active symbolic placement safely."""

    def begin_symbolic_operation(
        self, *, operation: str, target_token: Any | None
    ) -> Any:
        """Start one v34 restricted symbolic operation."""

    def tick_symbolic_operation(self, state_token: Any) -> SymbolicGraspTick:
        """Advance one v34 operation by at most one simulator step."""

    def abort_symbolic_operation(self, state_token: Any) -> SymbolicGraspTick:
        """Stop one v34 operation safely."""


class MarkerSymbolicContractError(RuntimeError):
    pass


@dataclass(slots=True)
class _MarkState:
    backend: MarkerSymbolicBackend = field(repr=False)
    action_index: int
    status: str | None = None


class MarkVisibleObjectsPlugin:
    controller_key = MARK_VISIBLE_CONTROLLER_KEY

    def begin(self, action: ActionEnvelope, context: ActionExecutionContext) -> _MarkState:
        if action.action != "mark_visible_objects" or not isinstance(action.args, NoArgs):
            raise MarkerSymbolicContractError("mark action has invalid parsed arguments")
        backend = _backend(context)
        return _MarkState(backend=backend, action_index=context.action_index)

    def tick(self, state: _MarkState, context: ActionExecutionContext) -> ActionPluginTick:
        del context
        if state.status is not None:
            raise MarkerSymbolicContractError("cannot tick a terminal mark action")
        count = state.backend.publish_visible_markers(action_index=state.action_index)
        state.status = "markers_ready" if count > 0 else "markers_unavailable"
        return ActionPluginTick(done=True, simulator_steps=0, capture_observation=True)

    def finish(self, state: _MarkState, context: ActionExecutionContext) -> ActionPluginResult:
        del context
        if state.status is None:
            raise MarkerSymbolicContractError("cannot finish a non-terminal mark action")
        return ActionPluginResult(status=state.status, public_feedback={})

    def abort(self, state: _MarkState, context: ActionExecutionContext, reason: str) -> ActionPluginResult:
        del context, reason
        state.status = "markers_unavailable"
        return ActionPluginResult(status=state.status, public_feedback={})


@dataclass(slots=True)
class _GraspState:
    backend: MarkerSymbolicBackend = field(repr=False)
    args: MarkerGraspArgs
    action_index: int
    stage: str = "binding"
    driver_token: Any = field(default=None, repr=False)
    status: str | None = None
    public_feedback: dict[str, Any] = field(default_factory=dict)


class MarkerSymbolicGraspPlugin:
    controller_key = MARKER_SYMBOLIC_GRASP_CONTROLLER_KEY

    def begin(self, action: ActionEnvelope, context: ActionExecutionContext) -> _GraspState:
        if action.action != "grasp" or not isinstance(action.args, MarkerGraspArgs):
            raise MarkerSymbolicContractError("marker grasp has invalid parsed arguments")
        return _GraspState(
            backend=_backend(context),
            args=action.args,
            action_index=context.action_index,
        )

    def tick(self, state: _GraspState, context: ActionExecutionContext) -> ActionPluginTick:
        del context
        if state.status is not None:
            raise MarkerSymbolicContractError("cannot tick a terminal marker grasp")
        if state.stage == "binding":
            resolution = state.backend.consume_marker(
                marker=state.args.marker,
                action_index=state.action_index,
                operation="grasp",
            )
            if not isinstance(resolution, PrivateMarkerResolution):
                raise MarkerSymbolicContractError("backend returned invalid marker resolution")
            if resolution.status is not MarkerStatus.READY:
                if (
                    resolution.public_failure_reason
                    is MarkerPublicFailureReason.TARGET_OUTSIDE_LOCAL_WORKSPACE
                ):
                    state.status = "not_executed"
                    state.public_feedback = {
                        "reason_code": TARGET_OUTSIDE_LOCAL_WORKSPACE,
                        "object_acquired": False,
                    }
                else:
                    state.status = "grasp_uncertain"
                return ActionPluginTick(done=True, simulator_steps=0, capture_observation=True)
            state.driver_token = state.backend.begin_symbolic_grasp(
                target_token=resolution.binding.target_token
            )
            state.stage = "symbolic_grasp"
            return ActionPluginTick(done=False, simulator_steps=0)
        tick = state.backend.tick_symbolic_grasp(state.driver_token)
        if not isinstance(tick, SymbolicGraspTick):
            raise MarkerSymbolicContractError("backend returned invalid symbolic tick")
        if tick.status is SymbolicGraspStatus.RUNNING:
            return ActionPluginTick(
                done=False,
                simulator_steps=tick.simulator_steps,
                capture_observation=tick.simulator_steps > 0,
            )
        state.status = (
            "grasp_likely"
            if tick.status is SymbolicGraspStatus.SUCCEEDED
            else "grasp_uncertain"
        )
        return ActionPluginTick(
            done=True,
            simulator_steps=tick.simulator_steps,
            capture_observation=True,
        )

    def finish(self, state: _GraspState, context: ActionExecutionContext) -> ActionPluginResult:
        del context
        if state.status is None:
            raise MarkerSymbolicContractError("cannot finish a non-terminal marker grasp")
        return ActionPluginResult(
            status=state.status,
            public_feedback=state.public_feedback,
        )

    def abort(self, state: _GraspState, context: ActionExecutionContext, reason: str) -> ActionPluginResult:
        del context, reason
        if state.driver_token is not None and state.status is None:
            state.backend.abort_symbolic_grasp(state.driver_token)
        state.status = "grasp_uncertain"
        return ActionPluginResult(status=state.status, public_feedback={})


@dataclass(slots=True)
class _PlaceState:
    backend: MarkerSymbolicBackend = field(repr=False)
    args: MarkerPlaceInsideArgs
    action_index: int
    stage: str = "binding"
    driver_token: Any = field(default=None, repr=False)
    status: str | None = None
    public_feedback: dict[str, Any] = field(default_factory=dict)


class MarkerSymbolicPlaceInsidePlugin:
    controller_key = MARKER_SYMBOLIC_PLACE_CONTROLLER_KEY

    def begin(self, action: ActionEnvelope, context: ActionExecutionContext) -> _PlaceState:
        if action.action != "place_inside" or not isinstance(
            action.args, MarkerPlaceInsideArgs
        ):
            raise MarkerSymbolicContractError("marker place has invalid parsed arguments")
        return _PlaceState(
            backend=_backend(context),
            args=action.args,
            action_index=context.action_index,
        )

    def tick(self, state: _PlaceState, context: ActionExecutionContext) -> ActionPluginTick:
        del context
        if state.status is not None:
            raise MarkerSymbolicContractError("cannot tick a terminal marker place")
        if state.stage == "binding":
            resolution = state.backend.consume_marker(
                marker=state.args.marker,
                action_index=state.action_index,
                operation="place_inside",
            )
            if resolution.status is not MarkerStatus.READY:
                if (
                    resolution.public_failure_reason
                    is MarkerPublicFailureReason.TARGET_OUTSIDE_LOCAL_WORKSPACE
                ):
                    state.status = "not_executed"
                    state.public_feedback = {
                        "reason_code": TARGET_OUTSIDE_LOCAL_WORKSPACE,
                        # The placement primitive was not entered, so its
                        # pre-existing public carry proxy remains true.
                        "object_acquired": True,
                        "gripper_open_commanded": False,
                    }
                else:
                    state.status = "placement_uncertain"
                return ActionPluginTick(done=True, simulator_steps=0, capture_observation=True)
            state.driver_token = state.backend.begin_symbolic_place_inside(
                target_token=resolution.binding.target_token
            )
            state.stage = "symbolic_place_inside"
            return ActionPluginTick(done=False, simulator_steps=0)
        tick = state.backend.tick_symbolic_place_inside(state.driver_token)
        if not isinstance(tick, SymbolicGraspTick):
            raise MarkerSymbolicContractError("backend returned invalid symbolic place tick")
        if tick.status is SymbolicGraspStatus.RUNNING:
            return ActionPluginTick(
                done=False,
                simulator_steps=tick.simulator_steps,
                capture_observation=tick.simulator_steps > 0,
            )
        state.status = (
            "completed"
            if tick.status is SymbolicGraspStatus.SUCCEEDED
            else "blocked"
            if tick.status is SymbolicGraspStatus.BLOCKED
            else "placement_uncertain"
        )
        return ActionPluginTick(
            done=True,
            simulator_steps=tick.simulator_steps,
            capture_observation=True,
        )

    def finish(self, state: _PlaceState, context: ActionExecutionContext) -> ActionPluginResult:
        del context
        if state.status is None:
            raise MarkerSymbolicContractError("cannot finish a non-terminal marker place")
        feedback = (
            state.public_feedback
            if state.public_feedback
            else {"gripper_open_commanded": state.status == "completed"}
        )
        return ActionPluginResult(
            status=state.status,
            public_feedback=feedback,
        )

    def abort(self, state: _PlaceState, context: ActionExecutionContext, reason: str) -> ActionPluginResult:
        del context, reason
        if state.driver_token is not None and state.status is None:
            state.backend.abort_symbolic_place_inside(state.driver_token)
        state.status = "placement_uncertain"
        return ActionPluginResult(status=state.status, public_feedback={})


@dataclass(slots=True)
class _V34OperationState:
    backend: MarkerSymbolicBackend = field(repr=False)
    operation: str
    action_index: int
    marker: str | None = None
    stage: str = "binding"
    driver_token: Any = field(default=None, repr=False)
    status: str | None = None
    public_feedback: dict[str, Any] = field(default_factory=dict)
    action_first_selection: bool = False


@dataclass(slots=True)
class MarkerSymbolicV34OperationPlugin:
    """One implementation for pick/put/open/close/put-back.

    The plugin deliberately knows no simulator object identity. Marker
    resolution and the pre-pick pose live behind the private backend.
    """

    action_name: str
    two_stage_on_top: bool = False
    controller_key: str = field(init=False)

    def __post_init__(self) -> None:
        if self.action_name not in MARKER_SYMBOLIC_V34_CONTROLLER_KEYS:
            raise ValueError("unsupported v34 symbolic operation")
        if self.two_stage_on_top and self.action_name != "place_on_top":
            raise ValueError("two-stage OnTop is only valid for place_on_top")
        self.controller_key = MARKER_SYMBOLIC_V34_CONTROLLER_KEYS[
            self.action_name
        ]

    def begin(
        self,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> _V34OperationState:
        if action.action != self.action_name:
            raise MarkerSymbolicContractError(
                "v34 symbolic action name differs from its plugin"
            )
        if self.action_name in _MARKERLESS_SYMBOLIC_OPERATIONS:
            if not isinstance(action.args, NoArgs):
                raise MarkerSymbolicContractError(
                    f"{self.action_name} does not accept marker arguments"
                )
            marker = None
        else:
            if isinstance(action.args, MarkerTargetOrRequestArgs):
                if self.action_name not in _ACTION_FIRST_MARKER_OPERATIONS:
                    raise MarkerSymbolicContractError(
                        "action-first marker request is unsupported for this action"
                    )
                marker = action.args.marker
                action_first_selection = True
            elif isinstance(action.args, MarkerTargetArgs):
                marker = action.args.marker
                action_first_selection = False
            else:
                raise MarkerSymbolicContractError(
                    "v34 marker action has invalid parsed arguments"
                )
        return _V34OperationState(
            backend=_backend(
                context,
                require_v34=True,
                require_marker=(
                    self.action_name not in _MARKERLESS_SYMBOLIC_OPERATIONS
                ),
            ),
            operation=self.action_name,
            marker=marker,
            action_index=context.action_index,
            action_first_selection=(
                action_first_selection
                if self.action_name not in _MARKERLESS_SYMBOLIC_OPERATIONS
                else False
            ),
        )

    def tick(
        self,
        state: _V34OperationState,
        context: ActionExecutionContext,
    ) -> ActionPluginTick:
        del context
        if state.status is not None:
            raise MarkerSymbolicContractError(
                "cannot tick a terminal v34 symbolic action"
            )
        if state.stage == "binding":
            if state.action_first_selection and state.marker is None:
                publish = getattr(
                    state.backend,
                    "publish_operation_markers",
                    None,
                )
                if not callable(publish):
                    raise MarkerSymbolicContractError(
                        "backend lacks action-first spatial marker publication"
                    )
                count = publish(
                    operation=state.operation,
                    action_index=state.action_index,
                    reference_binding=None,
                )
                state.status = (
                    "selection_required" if count > 0 else "markers_unavailable"
                )
                state.public_feedback = _selection_public_feedback(
                    state.operation
                )
                return ActionPluginTick(
                    done=True,
                    simulator_steps=0,
                    capture_observation=True,
                )
            target_token = None
            if state.marker is not None:
                resolution = state.backend.consume_marker(
                    marker=state.marker,
                    action_index=state.action_index,
                    operation=state.operation,
                )
                if not isinstance(resolution, PrivateMarkerResolution):
                    raise MarkerSymbolicContractError(
                        "backend returned invalid marker resolution"
                    )
                if resolution.status is not MarkerStatus.READY:
                    state.status = "not_executed"
                    if state.operation == "pick":
                        state.public_feedback = {
                            "object_acquired": False,
                        }
                    elif state.operation in _PLACEMENT_SYMBOLIC_OPERATIONS:
                        state.public_feedback = _selection_public_feedback(
                            state.operation
                        )
                    if (
                        resolution.public_failure_reason
                        is MarkerPublicFailureReason.TARGET_OUTSIDE_LOCAL_WORKSPACE
                    ):
                        state.public_feedback.update({
                            "blocked_category": OUT_OF_REACH,
                        })
                    elif (
                        resolution.public_failure_reason
                        is MarkerPublicFailureReason.SELECTION_UNAVAILABLE
                    ):
                        state.public_feedback.update({
                            "blocked_category": SELECTION_UNAVAILABLE,
                        })
                    elif (
                        resolution.public_failure_reason
                        is MarkerPublicFailureReason.EXECUTION_UNVERIFIED
                    ):
                        state.public_feedback["blocked_category"] = EXECUTION_UNVERIFIED
                    return ActionPluginTick(
                        done=True,
                        simulator_steps=0,
                        capture_observation=True,
                    )
                if state.action_first_selection:
                    kind = getattr(resolution.binding, "binding_kind", "object")
                    second_stage_kind = (
                        "surface_cell"
                        if self.two_stage_on_top
                        and state.operation == "place_on_top"
                        else _SECOND_STAGE_SELECTION_KINDS.get(state.operation)
                    )
                    if (
                        second_stage_kind is not None
                        and kind == "object"
                    ):
                        publish = getattr(
                            state.backend,
                            "publish_operation_markers",
                            None,
                        )
                        if not callable(publish):
                            raise MarkerSymbolicContractError(
                                "backend lacks second-stage marker publication"
                            )
                        count = publish(
                            operation=state.operation,
                            action_index=state.action_index,
                            reference_binding=resolution.binding,
                        )
                        state.status = (
                            "selection_required"
                            if count > 0
                            else "markers_unavailable"
                        )
                        state.public_feedback = _selection_public_feedback(
                            state.operation
                        )
                        return ActionPluginTick(
                            done=True,
                            simulator_steps=0,
                            capture_observation=True,
                        )
                    expected_kinds = (
                        {"object", "part"}
                        if state.operation in {
                            "open",
                            "close",
                            "toggle_on",
                            "toggle_off",
                        }
                        else {"object", "surface_cell"}
                        if state.operation == "place_on_top"
                        else {"object", "bundle"}
                        if state.operation == "pick"
                        else {
                            second_stage_kind or "object"
                        }
                    )
                    if kind not in expected_kinds:
                        state.status = "not_executed"
                        state.public_feedback = _selection_public_feedback(
                            state.operation
                        )
                        state.public_feedback["blocked_category"] = SELECTION_UNAVAILABLE
                        return ActionPluginTick(
                            done=True,
                            simulator_steps=0,
                            capture_observation=True,
                        )
                target_token = resolution.binding.target_token
            state.driver_token = state.backend.begin_symbolic_operation(
                operation=state.operation,
                target_token=target_token,
            )
            state.stage = "symbolic"
            return ActionPluginTick(done=False, simulator_steps=0)
        tick = state.backend.tick_symbolic_operation(state.driver_token)
        if not isinstance(tick, SymbolicGraspTick):
            raise MarkerSymbolicContractError(
                "backend returned invalid v34 symbolic tick"
            )
        if tick.status is SymbolicGraspStatus.RUNNING:
            return ActionPluginTick(
                done=False,
                simulator_steps=tick.simulator_steps,
                capture_observation=(
                    tick.simulator_steps > 0
                    and state.operation not in {"open", "close"}
                ),
            )
        blocked = bool(
            tick.status is SymbolicGraspStatus.BLOCKED
            or (
                tick.status is SymbolicGraspStatus.FAILED
                and tick.blocked_category in _NAVIGATION_BLOCKED_CATEGORIES
            )
        )
        state.status = (
            "completed"
            if tick.status is SymbolicGraspStatus.SUCCEEDED
            else "blocked"
            if blocked
            else "uncertain"
        )
        feedback: dict[str, Any] = {}
        if blocked:
            feedback = _selection_public_feedback(state.operation)
            if tick.blocked_category is not None:
                feedback["blocked_category"] = tick.blocked_category
        elif tick.blocked_category == EXECUTION_UNVERIFIED:
            feedback["blocked_category"] = EXECUTION_UNVERIFIED
        elif state.operation == "pick":
            feedback["object_acquired"] = state.status == "completed"
        elif state.operation in _PLACEMENT_SYMBOLIC_OPERATIONS:
            feedback["gripper_open_commanded"] = (
                state.status == "completed"
            )
        elif tick.blocked_category == ARTICULATED_PART_BLOCKED:
            feedback["blocked_category"] = ARTICULATED_PART_BLOCKED
        detail = action20_blocked_detail(feedback.get("blocked_category"), tick.blocked_detail)
        if state.status != "completed" and detail is not None:
            feedback["blocked_detail"] = detail
        state.public_feedback = feedback
        return ActionPluginTick(
            done=True,
            simulator_steps=tick.simulator_steps,
            capture_observation=True,
        )

    def finish(
        self,
        state: _V34OperationState,
        context: ActionExecutionContext,
    ) -> ActionPluginResult:
        del context
        if state.status is None:
            raise MarkerSymbolicContractError(
                "cannot finish a non-terminal v34 symbolic action"
            )
        return ActionPluginResult(
            status=state.status,
            public_feedback=state.public_feedback,
        )

    def abort(
        self,
        state: _V34OperationState,
        context: ActionExecutionContext,
        reason: str,
    ) -> ActionPluginResult:
        del context, reason
        if state.driver_token is not None and state.status is None:
            state.backend.abort_symbolic_operation(state.driver_token)
        state.status = "uncertain"
        return ActionPluginResult(status=state.status, public_feedback={})


def _backend(
    context: ActionExecutionContext,
    *,
    require_v34: bool = False,
    require_marker: bool = True,
) -> MarkerSymbolicBackend:
    try:
        backend = context.services[MARKER_SYMBOLIC_BACKEND_SERVICE]
    except KeyError as exc:
        raise MarkerSymbolicContractError("missing marker symbolic backend") from exc
    required = (
        ["publish_visible_markers", "consume_marker"]
        if require_marker
        else []
    )
    if not require_v34:
        required.extend(
            (
                "begin_symbolic_grasp",
                "tick_symbolic_grasp",
                "abort_symbolic_grasp",
                "begin_symbolic_place_inside",
                "tick_symbolic_place_inside",
                "abort_symbolic_place_inside",
            )
        )
    for method in required:
        if not callable(getattr(backend, method, None)):
            raise MarkerSymbolicContractError(f"marker symbolic backend lacks {method}()")
    if require_v34:
        for method in (
            "begin_symbolic_operation",
            "tick_symbolic_operation",
            "abort_symbolic_operation",
        ):
            if not callable(getattr(backend, method, None)):
                raise MarkerSymbolicContractError(
                    f"marker symbolic backend lacks {method}()"
                )
    return backend
