"""Live-evaluator adapter for the simulator-independent resident runtime.

This module intentionally imports no OmniGibson package.  It receives an
already-constructed official Evaluator and SharedCanaryControl as opaque
objects inside the evaluator process.
"""

from __future__ import annotations

import binascii
import hashlib
import struct
import zlib
from contextlib import nullcontext
from dataclasses import dataclass, field
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

import numpy as np
from PIL import Image

from behavior_hl.embodied.runtime.action_integrity import require_execution_integrity

from behavior_hl.embodied.evaluator_bridge.private_observation_ingress import (
    declassify_official_observation,
)
from behavior_hl.embodied.evaluator_bridge.private_visual_markers import (
    PrivateVisualMarkerStore,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_overhead_view import (
    OVERHEAD_CAMERA_IMAGE_SIZE_PX,
    capture_r1pro_overhead_rgb,
)

from behavior_hl.embodied.policy.observations import (
    HEAD_RGB_MINIMAL_CAMERA_PROFILE,
    HEAD_RGB_MINIMAL_OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
    THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
    THREE_RGBD_HEAD_EVAL_OBSERVATION_SCHEMA_VERSION,
    THREE_RGB_HEAD_EVAL_OBSERVATION_SCHEMA_VERSION,
    TASK_ID,
    sanitize_head_rgb_minimal_observation,
    sanitize_official_observation,
    sanitize_three_rgb_head_eval_observation,
)
from behavior_hl.embodied.policy.protocol import ActionEnvelope
from behavior_hl.embodied.runtime.dispatch import ActionPluginResult
from behavior_hl.embodied.runtime.observation import PublicRobotObservation
from behavior_hl.embodied.runtime.public_action_feedback import (
    model_visible_action20_last_result,
)


R1PRO_RESIDENT_SERVICE_FACTORY_VERSION = (
    "behavior2026.r1pro-resident-services/v2"
)
COMPOSITE_PADDING_RGB = (96, 96, 96)
NATIVE_CAMERA_LAYOUT_PROFILE = "native_horizontal_v2"
HEAD_DOMINANT_CAMERA_LAYOUT_PROFILE = "left_wrist_stack_head_square_2x_v1"
ACTION20_OVERHEAD_RGB_OBSERVATION_SCHEMA_VERSION = (
    "behavior2026.action20-three-rgb-overhead-observation/v2-mode"
)
OVERHEAD_VIEW_MODES = frozenset({"off", "always", "on_blocked"})
OVERHEAD_VIEW_TRIGGER_CATEGORIES = frozenset(
    {
        "articulated_part_blocked",
        "navigation_blocked",
        "out_of_reach",
    }
)


class ResidentEnvironmentBindingError(RuntimeError):
    """Raised when the already-live evaluator cannot satisfy the binding."""


class ResidentEvaluatorTerminalError(RuntimeError):
    """Typed control signal for one already-completed terminal evaluator step."""

    def __init__(
        self,
        *,
        terminated: bool,
        truncated: bool,
        simulator_steps: int = 1,
    ) -> None:
        if not isinstance(terminated, bool) or not isinstance(truncated, bool):
            raise TypeError("terminal evaluator flags must be boolean")
        if not terminated and not truncated:
            raise ValueError("terminal evaluator error requires a terminal flag")
        if (
            isinstance(simulator_steps, bool)
            or not isinstance(simulator_steps, int)
            or simulator_steps != 1
        ):
            raise ValueError(
                "terminal evaluator error must represent exactly one completed step"
            )
        self.terminated = terminated
        self.truncated = truncated
        self.simulator_steps = simulator_steps
        reason = (
            "terminated and truncated"
            if terminated and truncated
            else "terminated"
            if terminated
            else "truncated"
        )
        super().__init__(f"resident evaluator {reason} during bounded step")


class VersionedControllerServiceFactory(Protocol):
    """Build private plugin services for one action in the resident process."""

    api_contract_version: str

    def build(
        self,
        *,
        action_index: int,
        action: ActionEnvelope,
        evaluator: Any,
        control: Any,
        public_snapshot_provider: Callable[[], Mapping[str, Any]],
        step_once: Callable[[], int],
        physics_step_once: Callable[[], int] | None = None,
        mark_boundary_already_checkpointed: Callable[[], None] | None = None,
    ) -> Mapping[str, Any]:
        """Return the services inserted into ActionExecutionContext."""

    def public_carry_snapshot(self) -> Mapping[str, Any]:
        """Return only the public grasp proxy and its selected hand."""

    def record_public_action_result(
        self,
        action: ActionEnvelope,
        result: ActionPluginResult,
    ) -> None:
        """Update carry state from public action feedback only."""


@dataclass(frozen=True, slots=True)
class R1ProResidentCapabilityReport:
    """Read-only construction preflight report."""

    available: bool
    missing_capabilities: tuple[str, ...]
    evidence: Mapping[str, bool]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "missing_capabilities",
            tuple(self.missing_capabilities),
        )
        object.__setattr__(
            self,
            "evidence",
            MappingProxyType(dict(self.evidence)),
        )

    def require_available(self) -> None:
        if not self.available:
            raise ResidentEnvironmentBindingError(
                "R1Pro resident environment is missing capabilities: "
                f"{list(self.missing_capabilities)}"
            )


@dataclass(frozen=True, slots=True)
class PNGFrameArtifact:
    """A real PNG encoded from one exact frozen public RGB array."""

    camera: str
    simulator_step: int
    width: int
    height: int
    rgb_sha256: str
    png_sha256: str
    png_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not self.png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("png_bytes is not a PNG stream")


@dataclass(frozen=True, slots=True)
class _FrozenObservation:
    observation: PublicRobotObservation
    frame_hashes: Mapping[str, str]
    png_frames: Mapping[str, PNGFrameArtifact]
    frame_artifact: Mapping[str, Any]


_FULL_OBSERVATION_PROFILE = "r1pro_rgbd_fullres"


def _sanitize_observation_for_profile(
    observation: Mapping[str, Any],
    *,
    observation_profile: str,
    expected_task_id: int = TASK_ID,
) -> dict[str, Any]:
    """Apply the exact schema selected before Environment construction."""

    copied = _numpy_observation_copy(observation)
    if observation_profile == HEAD_RGB_MINIMAL_CAMERA_PROFILE:
        return sanitize_head_rgb_minimal_observation(
            copied,
            expected_task_id=expected_task_id,
        )
    if observation_profile == THREE_RGB_HEAD_EVAL_CAMERA_PROFILE:
        return sanitize_three_rgb_head_eval_observation(
            copied,
            expected_task_id=expected_task_id,
        )
    if observation_profile in {
        _FULL_OBSERVATION_PROFILE,
        THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
    }:
        return sanitize_official_observation(
            declassify_official_observation(copied),
            expected_task_id=expected_task_id,
            head_eval=(
                observation_profile == THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE
            ),
        )
    raise ResidentEnvironmentBindingError(
        f"unknown resident observation profile: {observation_profile!r}"
    )


def preflight_r1pro_resident_environment(
    *,
    evaluator: Any,
    control: Any,
    service_factory: Any,
    observation_profile: str = _FULL_OBSERVATION_PROFILE,
    expected_task_id: int = TASK_ID,
) -> R1ProResidentCapabilityReport:
    """Inspect and validate capabilities without stepping or commanding."""

    evaluator_obs = getattr(evaluator, "obs", None)
    observation_contract = False
    if isinstance(evaluator_obs, Mapping):
        try:
            _sanitize_observation_for_profile(
                evaluator_obs,
                observation_profile=observation_profile,
                expected_task_id=expected_task_id,
            )
            observation_contract = True
        except Exception:
            observation_contract = False

    public_state_contract = False
    public_state = None
    if callable(getattr(control, "public_state", None)):
        try:
            public_state = control.public_state()
            count = public_state.get("observation_count")
            public_state_contract = (
                isinstance(public_state, Mapping)
                and isinstance(count, int)
                and not isinstance(count, bool)
                and count >= 0
            )
        except Exception:
            public_state_contract = False

    evidence = {
        "live_evaluator_step": callable(getattr(evaluator, "step", None)),
        "ready_official_rgbd_proprio_observation": observation_contract,
        "private_evaluator_env": getattr(evaluator, "env", None) is not None,
        "shared_control_hold": callable(getattr(control, "hold", None)),
        "shared_control_public_state": public_state_contract,
        "versioned_service_factory": (
            getattr(service_factory, "api_contract_version", None)
            == R1PRO_RESIDENT_SERVICE_FACTORY_VERSION
        ),
        "service_factory_build": callable(
            getattr(service_factory, "build", None)
        ),
        "service_factory_public_carry_snapshot": callable(
            getattr(service_factory, "public_carry_snapshot", None)
        ),
        "service_factory_public_result_update": callable(
            getattr(service_factory, "record_public_action_result", None)
        ),
    }
    missing = tuple(
        name for name, present in evidence.items() if not present
    )
    return R1ProResidentCapabilityReport(
        available=not missing,
        missing_capabilities=missing,
        evidence=evidence,
    )


def _controller_snapshot(
    values: Mapping[str, Any],
    *,
    simulator_step: int,
) -> Mapping[str, Any]:
    """Freeze only the already-public arrays needed by live controllers."""

    if (
        isinstance(simulator_step, bool)
        or not isinstance(simulator_step, int)
        or simulator_step < 0
    ):
        raise ResidentEnvironmentBindingError(
            "controller simulator_step must be non-negative"
        )
    required = {"proprio", "cam_rel_poses", "cameras"}
    missing = required.difference(values)
    if missing:
        raise ResidentEnvironmentBindingError(
            f"controller snapshot is missing public keys: {sorted(missing)}"
        )
    return MappingProxyType(
        {
            "simulator_step": simulator_step,
            "proprio": values["proprio"],
            "cam_rel_poses": values["cam_rel_poses"],
            "cameras": values["cameras"],
        }
    )


def _checkpoint_action_tail(
    environment: Any,
    *,
    capture_action_boundary: bool,
    preserve_marker_overlay: bool,
    accepted_official_success_termination: bool,
) -> int:
    consume = getattr(environment, "consume_boundary_already_checkpointed", None)
    boundary_already_checkpointed = bool(consume()) if callable(consume) else False
    checkpoint = getattr(environment, "checkpoint_once", None)
    if (
        not capture_action_boundary
        or preserve_marker_overlay
        or accepted_official_success_termination
        or boundary_already_checkpointed
        or not callable(checkpoint)
    ):
        return 0
    return checkpoint()


class R1ProResidentEnvironment:
    """Bind one already-live R1Pro evaluator to ResidentEnvironment."""

    def __init__(
        self,
        *,
        evaluator: Any,
        control: Any,
        service_factory: VersionedControllerServiceFactory,
        settle_steps: int = 30,
        settle_step_policy: (
            Callable[[ActionEnvelope, ActionPluginResult], int] | None
        ) = None,
        visual_marker_store: PrivateVisualMarkerStore | None = None,
        internal_step_context_factory: Callable[[], Any] | None = None,
        physics_step_once: Callable[[], Any] | None = None,
        suppress_internal_video_write: bool = False,
        observation_profile: str = _FULL_OBSERVATION_PROFILE,
        expected_task_id: int = TASK_ID,
        camera_layout_profile: str | None = None,
        overhead_view_mode: str = "off",
    ) -> None:
        if (
            isinstance(settle_steps, bool)
            or not isinstance(settle_steps, int)
            or settle_steps < 0
        ):
            raise ValueError("settle_steps must be a non-negative integer")
        if settle_step_policy is not None and not callable(settle_step_policy):
            raise TypeError("settle_step_policy must be callable")
        self._evaluator = evaluator
        self._control = control
        self._service_factory = service_factory
        self._settle_steps = settle_steps
        self._settle_step_policy = settle_step_policy
        if visual_marker_store is not None and not isinstance(
            visual_marker_store, PrivateVisualMarkerStore
        ):
            raise TypeError("visual_marker_store must be PrivateVisualMarkerStore")
        self._visual_marker_store = visual_marker_store
        if (
            internal_step_context_factory is not None
            and not callable(internal_step_context_factory)
        ):
            raise TypeError("internal_step_context_factory must be callable")
        if physics_step_once is not None and not callable(physics_step_once):
            raise TypeError("physics_step_once must be callable")
        if not isinstance(suppress_internal_video_write, bool):
            raise TypeError("suppress_internal_video_write must be boolean")
        if observation_profile not in {
            _FULL_OBSERVATION_PROFILE,
            HEAD_RGB_MINIMAL_CAMERA_PROFILE,
            THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
            THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
        }:
            raise ValueError("unknown resident observation profile")
        if overhead_view_mode not in OVERHEAD_VIEW_MODES:
            raise ValueError("overhead_view_mode is invalid")
        if (
            overhead_view_mode != "off"
            and observation_profile != THREE_RGB_HEAD_EVAL_CAMERA_PROFILE
        ):
            raise ValueError(
                "overhead RGB requires the three-RGB head-eval profile"
            )
        if camera_layout_profile is None:
            camera_layout_profile = str(
                getattr(
                    evaluator,
                    "_behavior_hl_camera_layout_profile",
                    NATIVE_CAMERA_LAYOUT_PROFILE,
                )
            )
        if camera_layout_profile not in {
            NATIVE_CAMERA_LAYOUT_PROFILE,
            HEAD_DOMINANT_CAMERA_LAYOUT_PROFILE,
        }:
            raise ValueError("unknown resident camera layout profile")
        if (
            camera_layout_profile == HEAD_DOMINANT_CAMERA_LAYOUT_PROFILE
            and observation_profile
            not in {
                THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
                THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
            }
        ):
            raise ValueError(
                "head-dominant layout requires the three-RGB camera profile"
            )
        self._internal_step_context_factory = internal_step_context_factory
        self._physics_step_once = physics_step_once
        self._suppress_internal_video_write = suppress_internal_video_write
        self._observation_profile = observation_profile
        self._overhead_view_mode = overhead_view_mode
        self._overhead_trigger_reason: str | None = None
        self._camera_layout_profile = camera_layout_profile
        self._expected_task_id = expected_task_id
        self.capability_report = preflight_r1pro_resident_environment(
            evaluator=evaluator,
            control=control,
            service_factory=service_factory,
            observation_profile=observation_profile,
            expected_task_id=expected_task_id,
        )
        self.capability_report.require_available()
        self._baseline_observation_count = self._control_observation_count()
        self._steps_advanced = 0
        self._boundary_already_checkpointed = False
        self._last_simulator_step = 0
        self._latest: _FrozenObservation | None = None
        self._controller_snapshot: Mapping[str, Any] | None = None
        self._frozen: dict[int, _FrozenObservation] = {}

    def observe(self) -> PublicRobotObservation:
        """Freeze approved RGB-D/proprio arrays without advancing physics."""

        normalized = _sanitize_observation_for_profile(
            self._evaluator.obs,
            observation_profile=self._observation_profile,
            expected_task_id=self._expected_task_id,
        )
        control_delta = (
            self._control_observation_count()
            - self._baseline_observation_count
        )
        if control_delta < 0:
            raise ResidentEnvironmentBindingError(
                "shared observation counter regressed"
            )
        simulator_step = max(control_delta, self._steps_advanced)
        if simulator_step < self._last_simulator_step:
            raise ResidentEnvironmentBindingError(
                "resident simulator step regressed"
            )
        self._last_simulator_step = simulator_step
        cameras = normalized["cameras"]
        overhead_trigger_reason = (
            "always"
            if self._overhead_view_mode == "always"
            else self._overhead_trigger_reason
            if self._overhead_view_mode == "on_blocked"
            else None
        )
        overhead_capture_seconds: float | None = None
        if overhead_trigger_reason is not None:
            capture_started = perf_counter()
            overhead_rgb = capture_r1pro_overhead_rgb(self._evaluator.robot)
            overhead_capture_seconds = perf_counter() - capture_started
            cameras = {
                role: dict(modalities)
                for role, modalities in cameras.items()
            }
            cameras["overhead"] = {
                "rgb": _normalized_overhead_rgb(overhead_rgb)
            }
        self._overhead_trigger_reason = None
        if self._visual_marker_store is not None:
            marked_head = self._visual_marker_store.overlay_for(
                raw_rgb=cameras["head"]["rgb"],
                simulator_step=simulator_step,
            )
            if marked_head is not None:
                cameras = {
                    role: dict(modalities)
                    for role, modalities in cameras.items()
                }
                cameras["head"]["rgb"] = marked_head
        carry_state = self._public_carry_snapshot()
        public = PublicRobotObservation(
            schema_version=(
                ACTION20_OVERHEAD_RGB_OBSERVATION_SCHEMA_VERSION
                if overhead_trigger_reason is not None
                else HEAD_RGB_MINIMAL_OBSERVATION_SCHEMA_VERSION
                if self._observation_profile
                == HEAD_RGB_MINIMAL_CAMERA_PROFILE
                else THREE_RGB_HEAD_EVAL_OBSERVATION_SCHEMA_VERSION
                if self._observation_profile
                == THREE_RGB_HEAD_EVAL_CAMERA_PROFILE
                else THREE_RGBD_HEAD_EVAL_OBSERVATION_SCHEMA_VERSION
                if self._observation_profile
                == THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE
                else OBSERVATION_SCHEMA_VERSION
            ),
            simulator_step=simulator_step,
            values={
                "task_id": normalized["task_id"],
                "proprio": normalized["proprio"],
                "cam_rel_poses": normalized["cam_rel_poses"],
                "cameras": cameras,
                "grasp_proxy_likely": carry_state[
                    "grasp_proxy_likely"
                ],
            },
        )
        controller_values = public.values
        if overhead_trigger_reason is not None:
            controller_values = {
                **public.values,
                "cameras": {
                    role: modalities
                    for role, modalities in public.values["cameras"].items()
                    if role != "overhead"
                },
            }
        self._controller_snapshot = _controller_snapshot(
            controller_values,
            simulator_step=simulator_step,
        )
        frame_hashes: dict[str, str] = {}
        png_frames: dict[str, PNGFrameArtifact] = {}
        cameras = public.values["cameras"]
        for camera, modalities in cameras.items():
            rgb = np.asarray(modalities["rgb"])
            rgb_bytes = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
            rgb_sha256 = hashlib.sha256(rgb_bytes).hexdigest()
            png_bytes = _encode_rgb_png(rgb)
            frame_hashes[camera] = rgb_sha256
            png_frames[camera] = PNGFrameArtifact(
                camera=camera,
                simulator_step=simulator_step,
                width=int(rgb.shape[1]),
                height=int(rgb.shape[0]),
                rgb_sha256=rgb_sha256,
                png_sha256=hashlib.sha256(png_bytes).hexdigest(),
                png_bytes=png_bytes,
            )
        head_dominant_composite = (
            self._camera_layout_profile
            == HEAD_DOMINANT_CAMERA_LAYOUT_PROFILE
        )
        composite, camera_layout = _compose_public_rgb(
            cameras,
            head_dominant=head_dominant_composite,
        )
        composite_png = _encode_rgb_png(composite)
        frame_artifact = MappingProxyType(
            {
                "media_type": "image/png",
                "png_bytes": composite_png,
                "png_sha256": hashlib.sha256(composite_png).hexdigest(),
                "width": int(composite.shape[1]),
                "height": int(composite.shape[0]),
                "camera_order": tuple(cameras),
                "camera_layout": camera_layout,
                "source_rgb_sha256": MappingProxyType(
                    dict(frame_hashes)
                ),
                "composition": (
                    "native_resolution_head_rgb_v1"
                    if self._observation_profile
                    == HEAD_RGB_MINIMAL_CAMERA_PROFILE
                    else HEAD_DOMINANT_CAMERA_LAYOUT_PROFILE
                    if head_dominant_composite
                    else "native_resolution_horizontal_neutral_padding_v2"
                ),
                "padding_rgb": (
                    None
                    if (
                        self._observation_profile
                        == HEAD_RGB_MINIMAL_CAMERA_PROFILE
                        or head_dominant_composite
                    )
                    else COMPOSITE_PADDING_RGB
                ),
                **(
                    {
                        "overhead_view_mode": self._overhead_view_mode,
                        "overhead_trigger_reason": overhead_trigger_reason,
                        "overhead_capture_seconds": overhead_capture_seconds,
                    }
                    if overhead_trigger_reason is not None
                    else {}
                ),
            }
        )
        frozen = _FrozenObservation(
            observation=public,
            frame_hashes=MappingProxyType(frame_hashes),
            png_frames=MappingProxyType(png_frames),
            frame_artifact=frame_artifact,
        )
        self._latest = frozen
        self._frozen[id(public)] = frozen
        return public

    def frame_sha256(
        self,
        observation: PublicRobotObservation,
    ) -> Mapping[str, str]:
        """Return hashes calculated from that exact frozen observation."""

        return self._record_for(observation).frame_hashes

    def png_artifacts(
        self,
        observation: PublicRobotObservation,
    ) -> Mapping[str, PNGFrameArtifact]:
        """Return real PNG bytes for the exact frames used by the model."""

        return self._record_for(observation).png_frames

    def frame_artifact(
        self,
        observation: PublicRobotObservation,
    ) -> Mapping[str, Any]:
        """Return the exact configured public composite and its layout."""

        return self._record_for(observation).frame_artifact

    def public_snapshot(self) -> Mapping[str, Any]:
        """Expose the latest sanitized RGB-D/proprio arrays to plugins.

        ``observe`` freezes model/artifact frames, while ``step_once`` refreshes
        this controller-only snapshot after every real physics step.  The
        refresh uses the same public schema but does not append a model-history
        frame or encode PNG artifacts.
        """

        if self._controller_snapshot is None:
            raise ResidentEnvironmentBindingError(
                "observe must run before requesting a public snapshot"
            )
        return self._controller_snapshot

    def private_four_view_artifact(self, observation: PublicRobotObservation) -> Mapping[str, Any]:
        """Diagnostic only: never install overhead in public/controller history."""
        self._record_for(observation)
        if id(observation) != next(reversed(self._frozen)):
            raise RuntimeError("cannot capture overhead for a historical observation")
        cameras = {role: dict(value) for role, value in observation.values["cameras"].items()}
        if not {"head", "left_wrist", "right_wrist"} <= cameras.keys():
            raise RuntimeError("private four-view requires three onboard RGB cameras")
        reused = "overhead" in cameras
        if not reused:
            cameras["overhead"] = {"rgb": _normalized_overhead_rgb(
                capture_r1pro_overhead_rgb(self._evaluator.robot))}
        composite, layout = _compose_public_rgb(cameras, head_dominant=True)
        png = _encode_rgb_png(composite)
        return {"png_bytes": png, "png_sha256": hashlib.sha256(png).hexdigest(),
                "camera_layout": {role: dict(tile) for role, tile in layout.items()},
                "simulator_step": observation.simulator_step,
                "physics_steps": 0, "overhead_reused": reused,
                "capture_alignment": "same_action_boundary_render_only",
                "model_visible": False}

    def evaluator_state(self) -> Any:
        """Return the opaque evaluator only across the private verifier edge."""

        return self._evaluator

    def controller_services(
        self,
        action_index: int,
        action: ActionEnvelope,
    ) -> Mapping[str, Any]:
        require_execution_integrity(self._evaluator)
        if (
            isinstance(action_index, bool)
            or not isinstance(action_index, int)
            or action_index < 0
        ):
            raise ValueError("action_index must be non-negative")
        if not isinstance(action, ActionEnvelope):
            raise TypeError("action must be an ActionEnvelope")
        self._boundary_already_checkpointed = False
        if self._latest is None:
            raise ResidentEnvironmentBindingError(
                "observe must run before constructing controller services"
            )
        if (
            self._visual_marker_store is not None
            and action.action
            not in {
                "mark_visible_objects",
                "grasp",
                "place_inside",
                "pick",
                "put_inside",
                "place_on_top",
                "place_next_to",
                "place_under",
                "place_touching",
                "open",
                "close",
                "toggle_on",
                "toggle_off",
            }
        ):
            # Movement, stop, markerless placement, and put_back invalidate a prior marker
            # before any component build can fail. Marker-consuming actions
            # instead invalidate atomically through store.consume().
            self._visual_marker_store.clear()
        try:
            build_kwargs = {
                "action_index": action_index,
                "action": action,
                "evaluator": self._evaluator,
                "control": self._control,
                "public_snapshot_provider": self.public_snapshot,
                "step_once": (
                    self.step_internal_once
                    if self._internal_step_context_factory is not None
                    else self.step_once
                ),
            }
            if self._physics_step_once is not None:
                build_kwargs["physics_step_once"] = self.step_physics_once
                build_kwargs["mark_boundary_already_checkpointed"] = (
                    self.mark_boundary_already_checkpointed
                )
            services = self._service_factory.build(
                **build_kwargs,
            )
        except Exception:
            if self._visual_marker_store is not None:
                self._visual_marker_store.clear()
            raise
        if not isinstance(services, Mapping):
            raise ResidentEnvironmentBindingError(
                "controller service factory must return a mapping"
            )
        if any(not isinstance(key, str) or not key for key in services):
            raise ResidentEnvironmentBindingError(
                "controller service keys must be non-empty strings"
            )
        return MappingProxyType(dict(services))

    def record_public_action_result(
        self,
        action: ActionEnvelope,
        result: ActionPluginResult,
    ) -> None:
        if not isinstance(action, ActionEnvelope):
            raise TypeError("action must be an ActionEnvelope")
        if not isinstance(result, ActionPluginResult):
            raise TypeError("result must be an ActionPluginResult")
        try:
            self._service_factory.record_public_action_result(action, result)
        finally:
            if (
                self._visual_marker_store is not None
                and action.action != "mark_visible_objects"
                and result.status not in {
                    "selection_required",
                    "markers_unavailable",
                }
            ):
                self._visual_marker_store.clear()
        self._public_carry_snapshot()
        public_result = model_visible_action20_last_result(
            {"status": result.status, **dict(result.public_feedback)},
            action=action.action,
        )
        blocked_category = (
            public_result.get("blocked_category")
            if isinstance(public_result, Mapping)
            else None
        )
        self._overhead_trigger_reason = (
            str(blocked_category)
            if self._overhead_view_mode == "on_blocked"
            and blocked_category in OVERHEAD_VIEW_TRIGGER_CATEGORIES
            else None
        )

    def step_once(self) -> int:
        """Advance exactly once and fail closed on terminal evaluator state."""

        return self._advance_once(internal=False)

    def step_internal_once(self) -> int:
        """Advance physics/task once without rendering an intermediate frame."""

        if self._internal_step_context_factory is None:
            return self._advance_once(internal=False)
        return self._advance_once(internal=True)

    def step_physics_once(self) -> int:
        """Advance raw physics once without running the task verifier."""

        require_execution_integrity(self._evaluator)
        if self._physics_step_once is None:
            raise ResidentEnvironmentBindingError(
                "physics-only stepping is unavailable"
            )
        self._physics_step_once()
        self._steps_advanced += 1
        return 1

    def mark_boundary_already_checkpointed(self) -> None:
        """Record that the active private transaction crossed its boundary."""

        self._boundary_already_checkpointed = True

    def consume_boundary_already_checkpointed(self) -> bool:
        """Consume the active action's one-shot private boundary marker."""

        checkpointed = self._boundary_already_checkpointed
        self._boundary_already_checkpointed = False
        return checkpointed

    def checkpoint_once(self) -> int:
        """Advance once with rendering/video enabled for an action boundary."""

        return self._advance_once(internal=False)

    def _advance_once(self, *, internal: bool) -> int:
        require_execution_integrity(self._evaluator)
        context = (
            self._internal_step_context_factory()
            if internal and self._internal_step_context_factory is not None
            else nullcontext()
        )
        suppress_video = internal and self._suppress_internal_video_write
        original_video_path = None
        if suppress_video:
            if not hasattr(self._evaluator, "_video_path"):
                raise ResidentEnvironmentBindingError(
                    "Evaluator video-path contract changed"
                )
            original_video_path = self._evaluator._video_path
            self._evaluator._video_path = None
        try:
            with context:
                outcome = self._evaluator.step()
        finally:
            if suppress_video:
                self._evaluator._video_path = original_video_path

        if (
            not isinstance(outcome, tuple)
            or len(outcome) != 2
            or not all(isinstance(value, (bool, np.bool_)) for value in outcome)
        ):
            raise ResidentEnvironmentBindingError(
                "Evaluator.step must return (terminated, truncated)"
            )
        self._steps_advanced += 1
        terminated, truncated = map(bool, outcome)
        # Evaluator.step has now produced obs_(n+1).  Refresh the approved
        # controller snapshot immediately so the actuator's post-step sample
        # and its next command cannot reuse action-start proprioception.  This
        # is deliberately not a model observation and performs no PNG work.
        normalized = _sanitize_observation_for_profile(
            self._evaluator.obs,
            observation_profile=self._observation_profile,
            expected_task_id=self._expected_task_id,
        )
        control_delta = (
            self._control_observation_count()
            - self._baseline_observation_count
        )
        simulator_step = max(control_delta, self._steps_advanced)
        self._controller_snapshot = _controller_snapshot(
            normalized,
            simulator_step=simulator_step,
        )
        if terminated or truncated:
            # The step above is real and its obs_(n+1) is authoritative.  Keep
            # the refreshed controller snapshot available for zero-step hold /
            # cleanup, but never permit a caller to advance the terminal
            # evaluator again.  The episode owner must re-read its private
            # verifier before deciding whether this is success or failure.
            raise ResidentEvaluatorTerminalError(
                terminated=terminated,
                truncated=truncated,
            )
        return 1

    def settle(
        self,
        action: ActionEnvelope,
        result: ActionPluginResult,
        *,
        max_steps: int,
    ) -> int:
        """Hold the physical robot and perform an action-aware bounded settle."""

        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps < 0
        ):
            raise ValueError("max_steps must be a non-negative integer")
        requested = 0 if (
            action.action == "mark_visible_objects"
            or result.status in {"selection_required", "markers_unavailable"}
        ) else (
            self._settle_steps
            if self._settle_step_policy is None
            else self._settle_step_policy(action, result)
        )
        if (
            isinstance(requested, bool)
            or not isinstance(requested, int)
            or requested < 0
        ):
            raise ResidentEnvironmentBindingError(
                "settle policy must return a non-negative integer"
            )
        if requested > max_steps:
            raise ResidentEnvironmentBindingError(
                "remaining simulator budget cannot satisfy settle window"
            )
        if requested == 0:
            return 0
        self._control.hold()
        completed = 0
        for _ in range(requested):
            self.step_once()
            completed += 1
        return completed

    def _record_for(
        self,
        observation: PublicRobotObservation,
    ) -> _FrozenObservation:
        record = self._frozen.get(id(observation))
        if record is None or record.observation is not observation:
            raise ResidentEnvironmentBindingError(
                "observation was not frozen by this resident environment"
            )
        return record

    def _control_observation_count(self) -> int:
        state = self._control.public_state()
        if not isinstance(state, Mapping):
            raise ResidentEnvironmentBindingError(
                "SharedCanaryControl.public_state returned a non-mapping"
            )
        value = state.get("observation_count")
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ResidentEnvironmentBindingError(
                "shared observation_count must be non-negative"
            )
        return value

    def _public_carry_snapshot(self) -> Mapping[str, Any]:
        snapshot = self._service_factory.public_carry_snapshot()
        if not isinstance(snapshot, Mapping) or set(snapshot) != {
            "grasp_proxy_likely",
            "active_hand",
        }:
            raise ResidentEnvironmentBindingError(
                "public carry snapshot fields do not match"
            )
        likely = snapshot["grasp_proxy_likely"]
        hand = snapshot["active_hand"]
        if not isinstance(likely, bool):
            raise ResidentEnvironmentBindingError(
                "public grasp proxy must be boolean"
            )
        if hand not in {None, "left", "right"}:
            raise ResidentEnvironmentBindingError(
                "public active hand is invalid"
            )
        if likely != (hand is not None):
            raise ResidentEnvironmentBindingError(
                "public grasp proxy and active hand disagree"
            )
        return MappingProxyType(
            {
                "grasp_proxy_likely": likely,
                "active_hand": hand,
            }
        )


def _numpy_observation_copy(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    copied: dict[str, np.ndarray] = {}
    for key, value in observation.items():
        if not isinstance(key, str) or not key:
            raise ValueError("official observation keys must be strings")
        candidate = value
        detach = getattr(candidate, "detach", None)
        if callable(detach):
            candidate = detach()
        cpu = getattr(candidate, "cpu", None)
        if callable(cpu):
            candidate = cpu()
        to_numpy = getattr(candidate, "numpy", None)
        if callable(to_numpy):
            candidate = to_numpy()
        array = np.asarray(candidate)
        if array.dtype.kind in {"O", "V", "c"}:
            raise ValueError(f"unsupported official observation dtype: {key}")
        copied[key] = np.array(array, copy=True)
    return copied


def _encode_rgb_png(rgb: np.ndarray) -> bytes:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("PNG source must be HxWx3 uint8 RGB")
    height, width = map(int, array.shape[:2])
    scanlines = b"".join(
        b"\x00" + np.ascontiguousarray(row).tobytes()
        for row in array
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + _png_chunk(b"IEND", b"")
    )


def _normalized_overhead_rgb(value: Any) -> np.ndarray:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    to_numpy = getattr(candidate, "numpy", None)
    if callable(to_numpy):
        candidate = to_numpy()
    array = np.asarray(candidate)
    if array.shape not in {
        (OVERHEAD_CAMERA_IMAGE_SIZE_PX, OVERHEAD_CAMERA_IMAGE_SIZE_PX, 3),
        (OVERHEAD_CAMERA_IMAGE_SIZE_PX, OVERHEAD_CAMERA_IMAGE_SIZE_PX, 4),
    }:
        raise ResidentEnvironmentBindingError(
            "overhead VisionSensor RGB resolution changed"
        )
    rgb = np.asarray(array[..., :3])
    if not np.all(np.isfinite(rgb)):
        raise ResidentEnvironmentBindingError(
            "overhead VisionSensor RGB contains non-finite values"
        )
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = rgb * (255.0 if float(rgb.max(initial=0.0)) <= 1.0 else 1.0)
    return np.ascontiguousarray(np.clip(rgb, 0, 255), dtype=np.uint8)


def _compose_public_rgb(
    cameras: Mapping[str, Mapping[str, Any]],
    *,
    head_dominant: bool = False,
) -> tuple[np.ndarray, Mapping[str, Mapping[str, int]]]:
    canonical_order = ("left_wrist", "right_wrist", "head", "overhead")
    order = tuple(role for role in canonical_order if role in cameras)
    if not order or set(order) != set(cameras):
        raise ValueError("public camera roles do not match a supported R1Pro profile")
    arrays = {
        role: np.asarray(cameras[role]["rgb"], dtype=np.uint8)
        for role in order
    }
    if head_dominant:
        if order not in {canonical_order[:3], canonical_order}:
            raise ValueError(
                "head-dominant composition requires both wrists and head"
            )
        left = arrays["left_wrist"]
        right = arrays["right_wrist"]
        head = arrays["head"]
        for role, array in arrays.items():
            if (
                array.ndim != 3
                or array.shape[2] != 3
                or array.shape[0] != array.shape[1]
            ):
                raise ValueError(
                    f"head-dominant camera {role!r} must be square RGB"
                )
        if left.shape != right.shape:
            raise ValueError(
                "head-dominant wrist camera resolutions must match"
            )
        wrist_edge = int(left.shape[0])
        head_edge = wrist_edge * 2
        if tuple(head.shape[:2]) != (head_edge, head_edge):
            head = np.asarray(
                Image.fromarray(head).resize(
                    (head_edge, head_edge),
                    resample=Image.Resampling.LANCZOS,
                ),
                dtype=np.uint8,
            )
        if order == canonical_order:
            tile_edge = head_edge // 3
            if head_edge % 3:
                raise ValueError(
                    "four-view head camera edge must be divisible by three"
                )
            tiles = {
                role: np.asarray(
                    Image.fromarray(arrays[role]).resize(
                        (tile_edge, tile_edge),
                        resample=Image.Resampling.LANCZOS,
                    ),
                    dtype=np.uint8,
                )
                for role in ("left_wrist", "right_wrist", "overhead")
            }
            canvas = np.empty(
                (head_edge, tile_edge + head_edge, 3),
                dtype=np.uint8,
            )
            layout: dict[str, Mapping[str, int]] = {}
            for index, role in enumerate(
                ("left_wrist", "right_wrist", "overhead")
            ):
                y = index * tile_edge
                canvas[y : y + tile_edge, :tile_edge] = tiles[role]
                layout[role] = MappingProxyType(
                    {
                        "x": 0,
                        "y": y,
                        "width": tile_edge,
                        "height": tile_edge,
                    }
                )
            canvas[:, tile_edge:] = head
            layout["head"] = MappingProxyType(
                {
                    "x": tile_edge,
                    "y": 0,
                    "width": head_edge,
                    "height": head_edge,
                }
            )
            return canvas, MappingProxyType(layout)
        canvas = np.empty(
            (head_edge, wrist_edge + head_edge, 3),
            dtype=np.uint8,
        )
        canvas[:wrist_edge, :wrist_edge] = left
        canvas[wrist_edge:head_edge, :wrist_edge] = right
        canvas[:, wrist_edge:] = head
        return canvas, MappingProxyType(
            {
                "left_wrist": MappingProxyType(
                    {
                        "x": 0,
                        "y": 0,
                        "width": wrist_edge,
                        "height": wrist_edge,
                    }
                ),
                "right_wrist": MappingProxyType(
                    {
                        "x": 0,
                        "y": wrist_edge,
                        "width": wrist_edge,
                        "height": wrist_edge,
                    }
                ),
                "head": MappingProxyType(
                    {
                        "x": wrist_edge,
                        "y": 0,
                        "width": head_edge,
                        "height": head_edge,
                    }
                ),
            }
        )
    height = max(int(array.shape[0]) for array in arrays.values())
    width = sum(int(array.shape[1]) for array in arrays.values())
    canvas = np.full(
        (height, width, 3),
        COMPOSITE_PADDING_RGB,
        dtype=np.uint8,
    )
    layout: dict[str, Mapping[str, int]] = {}
    x = 0
    for role in order:
        array = arrays[role]
        frame_height, frame_width = map(int, array.shape[:2])
        y = (height - frame_height) // 2
        canvas[y : y + frame_height, x : x + frame_width] = array
        layout[role] = MappingProxyType(
            {
                "x": x,
                "y": y,
                "width": frame_width,
                "height": frame_height,
            }
        )
        x += frame_width
    return canvas, MappingProxyType(layout)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind)
    checksum = binascii.crc32(payload, checksum) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", checksum)
    )
