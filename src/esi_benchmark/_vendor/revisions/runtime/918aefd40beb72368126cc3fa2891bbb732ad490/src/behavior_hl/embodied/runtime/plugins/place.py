"""Observation-only ``place_inside_at`` action plugin.

The plugin deliberately stops at public sensor and controller boundaries.  It
uses an RGB-D-selected cavity patch to produce a conservative local arm
sequence.  Task predicates and research containment diagnostics belong to the
private verifier and are not accepted by this controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from behavior_hl.embodied.policy.observations import CAMERA_RESOLUTIONS
from behavior_hl.embodied.policy.protocol import (
    ActionEnvelope,
    PointActionArgs,
)
from behavior_hl.embodied.policy.rgbd_grounding import (
    RGBDGroundingError,
    GroundedPoint,
    ground_image_point,
    normalized_to_pixel,
)
from behavior_hl.embodied.runtime.dispatch import (
    ActionExecutionContext,
    ActionPluginResult,
    ActionPluginTick,
)


RGBD_SOURCE_SERVICE = "place.rgbd_source"
MOTION_SERVICE = "place.motion"
VISUAL_ASSESSOR_SERVICE = "place.visual_assessor"


class PlaceContractError(RuntimeError):
    """Raised before simulation when required plugin services are malformed."""


class CavityPatchError(ValueError):
    """Raised when public depth does not support a safe release patch."""


@dataclass(frozen=True, slots=True)
class PlaceConfig:
    """Frozen geometric and motion margins for one payload capability."""

    payload_radius_m: float = 0.06
    lateral_margin_m: float = 0.025
    minimum_cavity_recess_m: float = 0.025
    maximum_patch_depth_spread_m: float = 0.05
    rim_width_px: int = 4
    preplace_height_m: float = 0.18
    release_clearance_m: float = 0.02
    settle_steps: int = 60

    def __post_init__(self) -> None:
        for field in (
            "payload_radius_m",
            "lateral_margin_m",
            "minimum_cavity_recess_m",
            "maximum_patch_depth_spread_m",
            "preplace_height_m",
            "release_clearance_m",
        ):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0.0
            ):
                raise ValueError(f"{field} must be a positive finite number")
        for field in ("rim_width_px", "settle_steps"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")


@dataclass(frozen=True, slots=True)
class PlaceRGBDFrame:
    """Exactly the public camera fields consumed by placement."""

    camera: str
    rgb: np.ndarray
    depth_linear: np.ndarray
    cam_rel_poses: np.ndarray
    intrinsic_matrix: np.ndarray

    def __post_init__(self) -> None:
        if self.camera not in CAMERA_RESOLUTIONS:
            raise ValueError(f"unknown public camera: {self.camera!r}")
        height, width = CAMERA_RESOLUTIONS[self.camera]
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth_linear)
        poses = np.asarray(self.cam_rel_poses)
        intrinsic = np.asarray(self.intrinsic_matrix)
        if rgb.shape != (height, width, 3) or rgb.dtype.kind not in {"u", "i"}:
            raise ValueError("rgb must match the registered camera resolution")
        if depth.squeeze().shape != (height, width):
            raise ValueError("depth_linear must match the registered camera resolution")
        if poses.shape != (21,) or not np.isfinite(poses).all():
            raise ValueError("cam_rel_poses must be a finite 21D vector")
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("intrinsic_matrix must be finite and 3x3")
        for field, value in (
            ("rgb", rgb),
            ("depth_linear", depth),
            ("cam_rel_poses", poses),
            ("intrinsic_matrix", intrinsic),
        ):
            copied = np.array(value, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, field, copied)


class PlaceRGBDSource(Protocol):
    def snapshot(self, camera: str) -> PlaceRGBDFrame:
        """Return a current public onboard RGB-D frame."""


class PlaceMotionService(Protocol):
    """Physical controller boundary; all pose arguments are robot-base XYZ."""

    def reachable(self, hand: str, target_xyz_m: tuple[float, float, float]) -> bool:
        """Return whether a target lies in the calibrated arm workspace."""

    def solve_ik(
        self,
        hand: str,
        target_xyz_m: tuple[float, float, float],
        phase: str,
    ) -> bool:
        """Solve without moving the robot."""

    def move_to(
        self,
        hand: str,
        target_xyz_m: tuple[float, float, float],
        phase: str,
    ) -> int:
        """Execute a previously checked local arm target."""

    def open_gripper(self, hand: str) -> int:
        """Physically release the payload."""

    def settle(self, requested_steps: int) -> int:
        """Advance physics after release and retraction."""

    def abort_place(
        self,
        hand: str,
        *,
        released: bool,
        retract_xyz_m: tuple[float, float, float],
    ) -> None:
        """Hold or recover locally without moving the base."""


class PlaceVisualAssessor(Protocol):
    def release_consistent(
        self,
        *,
        camera: str,
        point: tuple[float, float],
        before_rgb: np.ndarray,
        after_rgb: np.ndarray,
    ) -> bool:
        """Use only the two public RGB frames to assess release evidence."""


@dataclass(frozen=True, slots=True)
class CavityPatch:
    grounded: GroundedPoint
    footprint_radius_px: int
    inner_depth_m: float
    rim_depth_m: float
    recess_m: float


@dataclass(slots=True)
class _PlaceState:
    args: PointActionArgs
    source: PlaceRGBDSource
    motion: PlaceMotionService
    visual_assessor: PlaceVisualAssessor
    before: PlaceRGBDFrame
    patch: CavityPatch | None = None
    preplace_xyz_m: tuple[float, float, float] | None = None
    release_xyz_m: tuple[float, float, float] | None = None
    phase: str = "preplace"
    terminal_status: str | None = None
    released: bool = False
    after: PlaceRGBDFrame | None = None


def cavity_patch_from_public_rgbd(
    frame: PlaceRGBDFrame,
    *,
    point: tuple[float, float],
    config: PlaceConfig,
) -> CavityPatch:
    """Validate that a full payload footprint lies below a visible rim."""

    observation = {
        "cameras": {
            frame.camera: {"depth_linear": frame.depth_linear},
        },
        "cam_rel_poses": frame.cam_rel_poses,
    }
    grounded = ground_image_point(
        observation,
        camera=frame.camera,
        point=point,
        intrinsic_matrix=frame.intrinsic_matrix,
        maximum_depth_spread_m=config.maximum_patch_depth_spread_m,
    )
    height, width = CAMERA_RESOLUTIONS[frame.camera]
    center_x, center_y = normalized_to_pixel(
        point,
        width=width,
        height=height,
    )
    focal_px = min(
        float(frame.intrinsic_matrix[0, 0]),
        float(frame.intrinsic_matrix[1, 1]),
    )
    footprint_m = config.payload_radius_m + config.lateral_margin_m
    initial_footprint_px = max(
        1,
        int(math.ceil(focal_px * footprint_m / grounded.depth_m)),
    )
    edge_limit_px = min(
        center_x,
        width - 1 - center_x,
        center_y,
        height - 1 - center_y,
    )
    search_limit_px = min(
        edge_limit_px,
        max(
            initial_footprint_px * 3,
            initial_footprint_px + 64,
        ),
    )
    if (
        search_limit_px
        < initial_footprint_px + config.rim_width_px
    ):
        raise CavityPatchError("payload footprint crosses the image boundary")

    depth = np.asarray(frame.depth_linear, dtype=np.float64).squeeze()
    outer_px = search_limit_px
    ys, xs = np.ogrid[
        center_y - outer_px : center_y + outer_px + 1,
        center_x - outer_px : center_x + outer_px + 1,
    ]
    squared = (xs - center_x) ** 2 + (ys - center_y) ** 2
    crop = depth[
        center_y - outer_px : center_y + outer_px + 1,
        center_x - outer_px : center_x + outer_px + 1,
    ]
    inner = crop[squared <= initial_footprint_px**2]
    inner = inner[np.isfinite(inner) & (inner >= 0.05) & (inner <= 5.0)]
    if inner.size == 0:
        raise CavityPatchError("cavity footprint has no reliable depth")
    inner_depth = float(np.median(inner))
    if float(np.max(np.abs(inner - inner_depth))) > config.maximum_patch_depth_spread_m:
        raise CavityPatchError("payload footprint is not a stable depth patch")

    # Search outward in fixed-width public-depth rings.  Assuming a rim at
    # exactly one payload radius would reject larger openings and could also
    # mistake cavity-bottom pixels for rim pixels.
    rim_depth: float | None = None
    cavity_radius_px: int | None = None
    ring_start = initial_footprint_px
    while ring_start + config.rim_width_px <= search_limit_px:
        ring_end = ring_start + config.rim_width_px
        rim = crop[
            (squared > ring_start**2) & (squared <= ring_end**2)
        ]
        rim = rim[
            np.isfinite(rim) & (rim >= 0.05) & (rim <= 5.0)
        ]
        if rim.size:
            candidate = float(np.median(rim))
            if (
                inner_depth - candidate
                >= config.minimum_cavity_recess_m
            ):
                rim_depth = candidate
                cavity_radius_px = ring_start
                break
        ring_start = ring_end
    if rim_depth is None or cavity_radius_px is None:
        raise CavityPatchError("selected patch is not visibly recessed below a rim")

    # Project at the nearer rim depth, not the farther cavity-bottom depth.
    # This is the conservative aperture size for the carried payload.
    footprint_px = max(
        1,
        int(math.ceil(focal_px * footprint_m / rim_depth)),
    )
    if footprint_px > cavity_radius_px:
        raise CavityPatchError("visible opening is too narrow for the payload margin")
    conservative_inner = crop[squared <= footprint_px**2]
    conservative_inner = conservative_inner[
        np.isfinite(conservative_inner)
        & (conservative_inner >= 0.05)
        & (conservative_inner <= 5.0)
    ]
    if conservative_inner.size == 0:
        raise CavityPatchError("payload footprint has no reliable depth")
    conservative_depth = float(np.median(conservative_inner))
    if (
        float(
            np.max(
                np.abs(conservative_inner - conservative_depth)
            )
        )
        > config.maximum_patch_depth_spread_m
    ):
        raise CavityPatchError("payload footprint is not a stable depth patch")
    recess = conservative_depth - rim_depth
    return CavityPatch(
        grounded=grounded,
        footprint_radius_px=footprint_px,
        inner_depth_m=conservative_depth,
        rim_depth_m=rim_depth,
        recess_m=recess,
    )


class PlaceInsideAtPlugin:
    """Bounded preplace → descend → release → retract → settle plugin."""

    controller_key = "manipulation.place_inside_at"

    def __init__(self, config: PlaceConfig | None = None) -> None:
        self.config = config or PlaceConfig()

    def begin(
        self,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> _PlaceState:
        if action.action != "place_inside_at" or not isinstance(
            action.args, PointActionArgs
        ):
            raise PlaceContractError(
                "PlaceInsideAtPlugin accepts only parsed place_inside_at actions"
            )
        source = _service(context, RGBD_SOURCE_SERVICE)
        motion = _service(context, MOTION_SERVICE)
        assessor = _service(context, VISUAL_ASSESSOR_SERVICE)
        before = source.snapshot(action.args.camera)
        if not isinstance(before, PlaceRGBDFrame) or before.camera != action.args.camera:
            raise PlaceContractError("RGB-D source returned the wrong public camera")
        state = _PlaceState(
            args=action.args,
            source=source,
            motion=motion,
            visual_assessor=assessor,
            before=before,
        )
        try:
            state.patch = cavity_patch_from_public_rgbd(
                before,
                point=action.args.point,
                config=self.config,
            )
        except RGBDGroundingError:
            state.terminal_status = "invalid_depth"
            return state
        except CavityPatchError:
            state.terminal_status = "cavity_uncertain"
            return state

        target = state.patch.grounded.robot_base_xyz_m
        release_height = (
            self.config.payload_radius_m
            + self.config.release_clearance_m
        )
        state.release_xyz_m = (
            target[0],
            target[1],
            target[2] + release_height,
        )
        state.preplace_xyz_m = (
            target[0],
            target[1],
            state.release_xyz_m[2] + self.config.preplace_height_m,
        )
        targets = (
            ("preplace", state.preplace_xyz_m),
            ("release", state.release_xyz_m),
        )
        if any(
            not motion.reachable(action.args.hand, xyz)
            for _, xyz in targets
        ):
            state.terminal_status = "out_of_reach"
            return state
        if any(
            not motion.solve_ik(action.args.hand, xyz, phase)
            for phase, xyz in targets
        ):
            state.terminal_status = "ik_failed"
        return state

    def tick(
        self,
        state: _PlaceState,
        context: ActionExecutionContext,
    ) -> ActionPluginTick:
        if state.terminal_status is not None:
            return ActionPluginTick(done=True, simulator_steps=0)
        assert state.preplace_xyz_m is not None
        assert state.release_xyz_m is not None
        try:
            if state.phase == "preplace":
                steps = state.motion.move_to(
                    state.args.hand,
                    state.preplace_xyz_m,
                    "preplace",
                )
                state.phase = "descend"
            elif state.phase == "descend":
                steps = state.motion.move_to(
                    state.args.hand,
                    state.release_xyz_m,
                    "release",
                )
                state.phase = "release"
            elif state.phase == "release":
                steps = state.motion.open_gripper(state.args.hand)
                state.released = True
                state.phase = "retract"
            elif state.phase == "retract":
                steps = state.motion.move_to(
                    state.args.hand,
                    state.preplace_xyz_m,
                    "retract",
                )
                state.phase = "settle"
            elif state.phase == "settle":
                steps = state.motion.settle(self.config.settle_steps)
                state.phase = "visual"
            elif state.phase == "visual":
                state.after = state.source.snapshot(state.args.camera)
                if (
                    not isinstance(state.after, PlaceRGBDFrame)
                    or state.after.camera != state.args.camera
                ):
                    state.terminal_status = "place_unstable"
                else:
                    consistent = state.visual_assessor.release_consistent(
                        camera=state.args.camera,
                        point=state.args.point,
                        before_rgb=state.before.rgb,
                        after_rgb=state.after.rgb,
                    )
                    state.terminal_status = (
                        "completed" if consistent else "place_unstable"
                    )
                steps = 0
            else:
                raise PlaceContractError(f"unknown place phase: {state.phase}")
        except Exception as exc:
            state.motion.abort_place(
                state.args.hand,
                released=state.released,
                retract_xyz_m=state.preplace_xyz_m,
            )
            state.terminal_status = "ik_failed"
            consumed = getattr(exc, "simulator_steps", 0)
            if (
                isinstance(consumed, bool)
                or not isinstance(consumed, int)
                or consumed < 0
            ):
                raise PlaceContractError(
                    "place motion failure reported invalid step accounting"
                ) from exc
            return ActionPluginTick(
                done=True,
                simulator_steps=consumed,
                public_feedback={"stage": state.phase},
            )
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            state.motion.abort_place(
                state.args.hand,
                released=state.released,
                retract_xyz_m=state.preplace_xyz_m,
            )
            state.terminal_status = "ik_failed"
            return ActionPluginTick(done=True, simulator_steps=0)
        return ActionPluginTick(
            done=state.terminal_status is not None,
            simulator_steps=steps,
            public_feedback={"stage": state.phase},
        )

    def finish(
        self,
        state: _PlaceState,
        context: ActionExecutionContext,
    ) -> ActionPluginResult:
        if state.terminal_status is None:
            raise PlaceContractError("place action finished before reaching a terminal state")
        feedback: dict[str, Any] = {
            "camera": state.args.camera,
            "hand": state.args.hand,
            # This is a public executor fact, not a containment predicate:
            # once the physical open-gripper phase ran, later retract,
            # settling, or visual-assessment failures must not leave the
            # runtime believing that it is still carrying a payload.
            "gripper_open_commanded": state.released,
        }
        if state.terminal_status in {"completed", "place_unstable"}:
            feedback["release_observation"] = (
                "visually_consistent"
                if state.terminal_status == "completed"
                else "uncertain"
            )
        return ActionPluginResult(state.terminal_status, feedback)

    def abort(
        self,
        state: _PlaceState,
        context: ActionExecutionContext,
        reason: str,
    ) -> ActionPluginResult:
        retract = state.preplace_xyz_m or (0.0, 0.0, 0.0)
        state.motion.abort_place(
            state.args.hand,
            released=state.released,
            retract_xyz_m=retract,
        )
        state.terminal_status = "ik_failed"
        return ActionPluginResult(
            "ik_failed",
            {"camera": state.args.camera, "hand": state.args.hand},
        )


def _service(context: ActionExecutionContext, name: str) -> Any:
    try:
        return context.services[name]
    except KeyError as exc:
        raise PlaceContractError(f"missing required service: {name}") from exc
