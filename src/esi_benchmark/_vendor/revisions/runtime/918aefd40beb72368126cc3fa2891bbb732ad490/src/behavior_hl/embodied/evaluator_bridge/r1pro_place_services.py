"""Live, observation-only R1Pro services for ``place_inside_at``."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from behavior_hl.embodied.policy.observations import CAMERA_RESOLUTIONS
from behavior_hl.embodied.policy.r1pro.action_layout import (
    R1PRO_ACTION_LAYOUT,
)
from behavior_hl.embodied.runtime.plugins.place import (
    MOTION_SERVICE,
    RGBD_SOURCE_SERVICE,
    VISUAL_ASSESSOR_SERVICE,
    PlaceRGBDFrame,
)


R1PRO_PLACE_SERVICES_VERSION = "r1pro-place-services-v1"
_FORBIDDEN_SNAPSHOT_FRAGMENTS = (
    "bddl",
    "goal",
    "held",
    "object_pose",
    "object_scope",
    "predicate",
    "success",
    "task_state",
)


class R1ProPlaceServiceError(RuntimeError):
    """Fail-closed place-service construction or execution error."""

    def __init__(self, message: str, *, simulator_steps: int = 0) -> None:
        super().__init__(message)
        if (
            isinstance(simulator_steps, bool)
            or not isinstance(simulator_steps, int)
            or simulator_steps < 0
        ):
            raise ValueError("simulator_steps must be non-negative")
        self.simulator_steps = simulator_steps


@dataclass(frozen=True, slots=True)
class R1ProPlaceMotionLimits:
    maximum_horizontal_reach_m: float = 1.30
    minimum_target_height_m: float = -0.25
    maximum_target_height_m: float = 1.80
    maximum_motion_actions: int = 240
    maximum_release_actions: int = 120
    maximum_settle_steps: int = 240

    def __post_init__(self) -> None:
        for name in (
            "maximum_horizontal_reach_m",
            "maximum_target_height_m",
        ):
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        minimum = _finite(
            self.minimum_target_height_m,
            "minimum_target_height_m",
        )
        if minimum >= self.maximum_target_height_m:
            raise ValueError("place target height interval is invalid")
        for name in (
            "maximum_motion_actions",
            "maximum_release_actions",
            "maximum_settle_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class LocalRGBPatchAssessment:
    radius_px: int = 18
    minimum_mean_absolute_change: float = 0.025
    minimum_changed_fraction: float = 0.08
    changed_pixel_threshold: float = 0.04

    def __post_init__(self) -> None:
        if (
            isinstance(self.radius_px, bool)
            or not isinstance(self.radius_px, int)
            or self.radius_px <= 0
        ):
            raise ValueError("radius_px must be a positive integer")
        for name in (
            "minimum_mean_absolute_change",
            "minimum_changed_fraction",
            "changed_pixel_threshold",
        ):
            value = _finite(getattr(self, name), name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class R1ProPlacePreflight:
    version: str
    available: bool
    cameras: tuple[str, ...]
    selected_hand: str
    maximum_motion_actions: int
    maximum_release_actions: int
    maximum_settle_steps: int
    evidence: Mapping[str, bool]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence",
            MappingProxyType(dict(self.evidence)),
        )


@dataclass(frozen=True, slots=True)
class R1ProPlaceServiceBundle:
    services: Mapping[str, Any]
    preflight: R1ProPlacePreflight

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "services",
            MappingProxyType(dict(self.services)),
        )


class FrozenPublicPlaceRGBDSource:
    """Copy frames from one admitted snapshot stream and fixed intrinsics."""

    def __init__(
        self,
        *,
        initial_snapshot: Mapping[str, Any],
        snapshot_provider: Callable[[], Mapping[str, Any]],
        camera_intrinsics: Mapping[str, Any],
    ) -> None:
        if not callable(snapshot_provider):
            raise R1ProPlaceServiceError(
                "snapshot_provider must be callable"
            )
        self._snapshot_provider = snapshot_provider
        self._initial = _freeze_snapshot(initial_snapshot)
        self._intrinsics = _freeze_intrinsics(camera_intrinsics)
        _validate_snapshot_cameras(self._initial, self._intrinsics)

    def snapshot(self, camera: str) -> PlaceRGBDFrame:
        if camera not in self._intrinsics:
            raise R1ProPlaceServiceError(
                f"camera {camera!r} has no admitted intrinsic"
            )
        if self._initial is not None:
            snapshot = self._initial
            self._initial = None
        else:
            snapshot = _freeze_snapshot(self._snapshot_provider())
            _validate_snapshot_cameras(snapshot, self._intrinsics)
        payload = snapshot["cameras"][camera]
        return PlaceRGBDFrame(
            camera=camera,
            rgb=np.asarray(payload["rgb"]),
            depth_linear=np.asarray(payload["depth_linear"]),
            cam_rel_poses=np.asarray(snapshot["cam_rel_poses"]),
            intrinsic_matrix=self._intrinsics[camera],
        )


@dataclass(frozen=True, slots=True)
class _IKPlan:
    hand: str
    target_xyz_m: tuple[float, float, float]
    pose: tuple[Any, Any] = field(repr=False)
    solution: Any = field(repr=False)


class StarterPlaceMotionService:
    """Bounded Starter generator execution with a fixed physical base."""

    def __init__(
        self,
        *,
        starter: Any,
        shared_control: Any,
        evaluator_step: Callable[[str], Any],
        target_pose_builder: Callable[
            [tuple[float, float, float], str], tuple[Any, Any]
        ],
        safe_hold_action: Callable[[str, bool], Sequence[float]],
        limits: R1ProPlaceMotionLimits,
    ) -> None:
        self._starter = starter
        self._control = shared_control
        self._evaluator_step = evaluator_step
        self._target_pose_builder = target_pose_builder
        self._safe_hold_action = safe_hold_action
        self._limits = limits
        self._plans: dict[
            tuple[str, tuple[float, float, float]], _IKPlan
        ] = {}
        self._released = False

    def reachable(
        self,
        hand: str,
        target_xyz_m: tuple[float, float, float],
    ) -> bool:
        target = _target(target_xyz_m)
        _require_hand(self._starter, hand)
        horizontal = math.hypot(target[0], target[1])
        return (
            horizontal <= self._limits.maximum_horizontal_reach_m
            and self._limits.minimum_target_height_m
            <= target[2]
            <= self._limits.maximum_target_height_m
        )

    def solve_ik(
        self,
        hand: str,
        target_xyz_m: tuple[float, float, float],
        phase: str,
    ) -> bool:
        del phase
        target = _target(target_xyz_m)
        _require_hand(self._starter, hand)
        pose = self._target_pose_builder(target, hand)
        _validate_pose(pose)
        solution = self._starter._ik_solver_cartesian_to_joint_space(
            pose,
            initial_joint_pos=None,
            skip_obstacle_update=False,
        )
        if solution is None:
            return False
        self._plans[(hand, target)] = _IKPlan(
            hand=hand,
            target_xyz_m=target,
            pose=pose,
            solution=solution,
        )
        return True

    def move_to(
        self,
        hand: str,
        target_xyz_m: tuple[float, float, float],
        phase: str,
    ) -> int:
        target = _target(target_xyz_m)
        try:
            plan = self._plans[(hand, target)]
        except KeyError as exc:
            raise R1ProPlaceServiceError(
                "move_to requires a successful non-mutating IK preflight"
            ) from exc
        generator = self._starter._move_hand(
            plan.pose,
            lock_auxiliary_arm=True,
        )
        return self._run_generator(
            generator,
            maximum=self._limits.maximum_motion_actions,
            label=f"place.{phase}",
        )

    def open_gripper(self, hand: str) -> int:
        _require_hand(self._starter, hand)
        steps = self._run_generator(
            self._starter._execute_release(),
            maximum=self._limits.maximum_release_actions,
            label="place.release_gripper",
        )
        self._released = True
        return steps

    def settle(self, requested_steps: int) -> int:
        if (
            isinstance(requested_steps, bool)
            or not isinstance(requested_steps, int)
            or requested_steps < 0
            or requested_steps > self._limits.maximum_settle_steps
        ):
            raise R1ProPlaceServiceError(
                "settle request exceeds its explicit maximum"
            )
        completed = 0
        for _ in range(requested_steps):
            self._control.hold()
            try:
                self._evaluator_step("place.settle")
            except Exception as exc:
                raise R1ProPlaceServiceError(
                    "place settle step failed",
                    simulator_steps=completed + 1,
                ) from exc
            completed += 1
        return requested_steps

    def abort_place(
        self,
        hand: str,
        *,
        released: bool,
        retract_xyz_m: tuple[float, float, float],
    ) -> None:
        _require_hand(self._starter, hand)
        _target(retract_xyz_m)
        # Abort cannot hide unaccounted simulator steps behind a void return.
        # Command a base-fixed safe hold now; the orchestrator owns any later
        # recovery rollout and its step accounting.
        action = _action23(self._safe_hold_action(hand, released))
        action[R1PRO_ACTION_LAYOUT.slice_for("base")] = 0.0
        self._control.command_direct_action(action)

    def _run_generator(
        self,
        value: Iterable[Any],
        *,
        maximum: int,
        label: str,
    ) -> int:
        try:
            generator = iter(value)
        except TypeError as exc:
            raise R1ProPlaceServiceError(
                "Starter motion did not return an iterator"
            ) from exc
        steps = 0
        while steps < maximum:
            try:
                raw_action = next(generator)
            except StopIteration:
                if steps == 0:
                    raise R1ProPlaceServiceError(
                        "Starter motion generated no physical action"
                    )
                return steps
            except Exception as exc:
                raise R1ProPlaceServiceError(
                    "Starter motion generator failed",
                    simulator_steps=steps,
                ) from exc
            try:
                action = _action23(raw_action)
            except Exception as exc:
                raise R1ProPlaceServiceError(
                    "Starter motion yielded an invalid action",
                    simulator_steps=steps,
                ) from exc
            action[R1PRO_ACTION_LAYOUT.slice_for("base")] = 0.0
            self._control.command_direct_action(action)
            try:
                self._evaluator_step(label)
            except Exception as exc:
                raise R1ProPlaceServiceError(
                    "Evaluator step failed during Starter motion",
                    simulator_steps=steps + 1,
                ) from exc
            steps += 1
        try:
            next(generator)
        except StopIteration:
            return steps
        except Exception as exc:
            raise R1ProPlaceServiceError(
                "Starter motion generator failed at its action limit",
                simulator_steps=steps,
            ) from exc
        raise R1ProPlaceServiceError(
            f"Starter motion exceeded explicit {maximum}-action limit",
            simulator_steps=steps,
        )


class LocalPatchPlaceVisualAssessor:
    """Release proxy using only corresponding before / after RGB patches."""

    def __init__(
        self,
        config: LocalRGBPatchAssessment | None = None,
    ) -> None:
        self._config = config or LocalRGBPatchAssessment()

    def release_consistent(
        self,
        *,
        camera: str,
        point: tuple[float, float],
        before_rgb: np.ndarray,
        after_rgb: np.ndarray,
    ) -> bool:
        if camera not in CAMERA_RESOLUTIONS:
            raise R1ProPlaceServiceError("unknown onboard camera")
        before = _rgb(before_rgb, camera)
        after = _rgb(after_rgb, camera)
        height, width = CAMERA_RESOLUTIONS[camera]
        x = int(round(float(point[0]) * (width - 1)))
        y = int(round(float(point[1]) * (height - 1)))
        radius = self._config.radius_px
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        if x1 <= x0 or y1 <= y0:
            return False
        difference = np.abs(
            before[y0:y1, x0:x1].astype(np.float32)
            - after[y0:y1, x0:x1].astype(np.float32)
        ) / 255.0
        per_pixel = difference.mean(axis=2)
        return (
            float(per_pixel.mean())
            >= self._config.minimum_mean_absolute_change
            and float(
                np.mean(
                    per_pixel
                    >= self._config.changed_pixel_threshold
                )
            )
            >= self._config.minimum_changed_fraction
        )


def preflight_r1pro_place_services(
    *,
    starter: Any,
    shared_control: Any,
    evaluator_step: Callable[[str], Any],
    target_pose_builder: Callable[
        [tuple[float, float, float], str], tuple[Any, Any]
    ],
    safe_hold_action: Callable[[str, bool], Sequence[float]],
    initial_frozen_public_snapshot: Mapping[str, Any],
    snapshot_provider: Callable[[], Mapping[str, Any]],
    camera_intrinsics: Mapping[str, Any],
    limits: R1ProPlaceMotionLimits,
) -> R1ProPlacePreflight:
    """Read-only capability check; no IK, generator, command, or step."""

    evidence = {
        "starter_selected_hand": getattr(starter, "arm", None)
        in {"left", "right"},
        "starter_non_mutating_ik": callable(
            getattr(starter, "_ik_solver_cartesian_to_joint_space", None)
        ),
        "starter_bounded_move_generator": callable(
            getattr(starter, "_move_hand", None)
        ),
        "starter_physical_release_generator": callable(
            getattr(starter, "_execute_release", None)
        ),
        "shared_direct_action": callable(
            getattr(shared_control, "command_direct_action", None)
        ),
        "shared_base_hold": callable(
            getattr(shared_control, "hold", None)
        ),
        "one_step_callback": callable(evaluator_step),
        "target_pose_builder": callable(target_pose_builder),
        "safe_hold_action": callable(safe_hold_action),
        "snapshot_provider": callable(snapshot_provider),
    }
    if not all(evidence.values()):
        missing = sorted(
            name for name, available in evidence.items() if not available
        )
        raise R1ProPlaceServiceError(
            f"place service preflight missing: {missing}"
        )
    snapshot = _freeze_snapshot(initial_frozen_public_snapshot)
    intrinsics = _freeze_intrinsics(camera_intrinsics)
    _validate_snapshot_cameras(snapshot, intrinsics)
    return R1ProPlacePreflight(
        version=R1PRO_PLACE_SERVICES_VERSION,
        available=True,
        cameras=tuple(sorted(intrinsics)),
        selected_hand=str(starter.arm),
        maximum_motion_actions=limits.maximum_motion_actions,
        maximum_release_actions=limits.maximum_release_actions,
        maximum_settle_steps=limits.maximum_settle_steps,
        evidence=evidence,
    )


def build_r1pro_place_service_bundle(
    *,
    starter: Any,
    shared_control: Any,
    evaluator_step: Callable[[str], Any],
    target_pose_builder: Callable[
        [tuple[float, float, float], str], tuple[Any, Any]
    ],
    safe_hold_action: Callable[[str, bool], Sequence[float]],
    initial_frozen_public_snapshot: Mapping[str, Any],
    snapshot_provider: Callable[[], Mapping[str, Any]],
    camera_intrinsics: Mapping[str, Any],
    limits: R1ProPlaceMotionLimits | None = None,
    visual_config: LocalRGBPatchAssessment | None = None,
) -> R1ProPlaceServiceBundle:
    frozen_limits = limits or R1ProPlaceMotionLimits()
    preflight = preflight_r1pro_place_services(
        starter=starter,
        shared_control=shared_control,
        evaluator_step=evaluator_step,
        target_pose_builder=target_pose_builder,
        safe_hold_action=safe_hold_action,
        initial_frozen_public_snapshot=initial_frozen_public_snapshot,
        snapshot_provider=snapshot_provider,
        camera_intrinsics=camera_intrinsics,
        limits=frozen_limits,
    )
    source = FrozenPublicPlaceRGBDSource(
        initial_snapshot=initial_frozen_public_snapshot,
        snapshot_provider=snapshot_provider,
        camera_intrinsics=camera_intrinsics,
    )
    motion = StarterPlaceMotionService(
        starter=starter,
        shared_control=shared_control,
        evaluator_step=evaluator_step,
        target_pose_builder=target_pose_builder,
        safe_hold_action=safe_hold_action,
        limits=frozen_limits,
    )
    return R1ProPlaceServiceBundle(
        services={
            RGBD_SOURCE_SERVICE: source,
            MOTION_SERVICE: motion,
            VISUAL_ASSESSOR_SERVICE: LocalPatchPlaceVisualAssessor(
                visual_config
            ),
        },
        preflight=preflight,
    )


def _freeze_snapshot(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R1ProPlaceServiceError("public snapshot must be a mapping")
    allowed = {"cameras", "cam_rel_poses", "proprio"}
    unknown = set(value).difference(allowed)
    if unknown:
        raise R1ProPlaceServiceError(
            f"public snapshot has undeclared fields: {sorted(unknown)}"
        )
    for key in value:
        folded = key.casefold()
        if any(
            fragment in folded
            for fragment in _FORBIDDEN_SNAPSHOT_FRAGMENTS
        ):
            raise R1ProPlaceServiceError(
                f"public snapshot contains private field {key!r}"
            )
    if set(value).intersection({"cameras", "cam_rel_poses"}) != {
        "cameras",
        "cam_rel_poses",
    }:
        raise R1ProPlaceServiceError(
            "public snapshot lacks cameras or cam_rel_poses"
        )
    poses = np.array(value["cam_rel_poses"], dtype=np.float64, copy=True)
    if poses.shape != (21,) or not np.isfinite(poses).all():
        raise R1ProPlaceServiceError(
            "cam_rel_poses must be finite and 21D"
        )
    poses.setflags(write=False)
    cameras = value["cameras"]
    if not isinstance(cameras, Mapping):
        raise R1ProPlaceServiceError("cameras must be a mapping")
    frozen_cameras = {}
    for camera, payload in cameras.items():
        if camera not in CAMERA_RESOLUTIONS or not isinstance(
            payload, Mapping
        ):
            raise R1ProPlaceServiceError(
                "snapshot camera payload is invalid"
            )
        if set(payload) != {"rgb", "depth_linear"}:
            raise R1ProPlaceServiceError(
                "snapshot camera modalities do not match"
            )
        rgb = np.array(payload["rgb"], copy=True)
        depth = np.array(payload["depth_linear"], copy=True)
        height, width = CAMERA_RESOLUTIONS[camera]
        if rgb.shape != (height, width, 3):
            raise R1ProPlaceServiceError("snapshot RGB shape does not match")
        if depth.squeeze().shape != (height, width):
            raise R1ProPlaceServiceError(
                "snapshot depth shape does not match"
            )
        rgb.setflags(write=False)
        depth.setflags(write=False)
        frozen_cameras[camera] = MappingProxyType(
            {"rgb": rgb, "depth_linear": depth}
        )
    return MappingProxyType(
        {
            "cameras": MappingProxyType(frozen_cameras),
            "cam_rel_poses": poses,
        }
    )


def _freeze_intrinsics(value: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    if not isinstance(value, Mapping) or not value:
        raise R1ProPlaceServiceError(
            "camera_intrinsics must be a non-empty mapping"
        )
    output = {}
    for camera, intrinsic in value.items():
        if camera not in CAMERA_RESOLUTIONS:
            raise R1ProPlaceServiceError(
                f"unsupported camera intrinsic {camera!r}"
            )
        matrix = np.array(intrinsic, dtype=np.float64, copy=True)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise R1ProPlaceServiceError(
                f"intrinsic for {camera!r} must be finite 3x3"
            )
        matrix.setflags(write=False)
        output[camera] = matrix
    return MappingProxyType(output)


def _validate_snapshot_cameras(
    snapshot: Mapping[str, Any],
    intrinsics: Mapping[str, np.ndarray],
) -> None:
    missing = set(intrinsics).difference(snapshot["cameras"])
    if missing:
        raise R1ProPlaceServiceError(
            f"snapshot lacks intrinsic cameras: {sorted(missing)}"
        )


def _target(value: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise R1ProPlaceServiceError("place target must be finite 3D")
    return tuple(float(item) for item in array)


def _require_hand(starter: Any, hand: str) -> None:
    if hand != getattr(starter, "arm", None):
        raise R1ProPlaceServiceError(
            "requested hand does not match initialized Starter arm"
        )


def _validate_pose(value: Any) -> None:
    if not isinstance(value, tuple) or len(value) != 2:
        raise R1ProPlaceServiceError(
            "target_pose_builder must return (position, quaternion)"
        )
    position = np.asarray(value[0], dtype=np.float64)
    quaternion = np.asarray(value[1], dtype=np.float64)
    if (
        position.shape != (3,)
        or quaternion.shape != (4,)
        or not np.isfinite(position).all()
        or not np.isfinite(quaternion).all()
    ):
        raise R1ProPlaceServiceError("Starter target pose is invalid")


def _action23(value: Sequence[float]) -> np.ndarray:
    action = np.asarray(value, dtype=np.float64)
    if action.shape != (23,) or not np.isfinite(action).all():
        raise R1ProPlaceServiceError(
            "Starter yielded a non-finite or non-23D action"
        )
    return np.array(action, copy=True)


def _rgb(value: np.ndarray, camera: str) -> np.ndarray:
    array = np.asarray(value)
    height, width = CAMERA_RESOLUTIONS[camera]
    if array.shape != (height, width, 3) or array.dtype.kind not in {
        "u",
        "i",
    }:
        raise R1ProPlaceServiceError(
            "public RGB patch source has invalid shape or dtype"
        )
    return array


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result
