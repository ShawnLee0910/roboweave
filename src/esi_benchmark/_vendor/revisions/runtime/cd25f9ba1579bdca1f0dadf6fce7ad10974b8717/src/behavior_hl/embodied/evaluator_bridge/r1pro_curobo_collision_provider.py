"""Fail-closed current-configuration collision audit for live R1Pro.

This adapter deliberately delegates collision semantics to the official
BEHAVIOR-1K 2026 ``CuRoboMotionGenerator``.  It refreshes the complete current
scene world, checks the robot's *current* full joint configuration with
``self_collision_check=True``, and includes the currently held object as an
attached CuRobo body.  The held object must remain in CuRobo's world model
during refresh: the official attachment implementation reads its mesh from
that model, disables the corresponding world obstacle while attached, and
re-enables it when detached.

The adapter performs no simulator step and exposes only one anonymous boolean:
``True`` means that either a world collision or a robot self-collision was
reported.  Missing official interfaces, malformed tensors, or CuRobo
exceptions raise and permanently poison the provider; they are never converted
to a benign ``False``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import Any, Callable

import numpy as np

from behavior_hl.navigation import PlanarPose


R1PRO_CUROBO_COLLISION_PROVIDER_VERSION = (
    "behavior2026.r1pro-curobo-current-q-collision/v2"
)
R1PRO_CUROBO_CANDIDATE_POSE_COLLISION_PROVIDER_VERSION = (
    "behavior2026.r1pro-curobo-candidate-pose-collision/"
    "v3-bounded-empty-hand-route-reuse"
)
OFFICIAL_CUROBO_COLLISION_SOURCE = (
    "OmniGibson/omnigibson/action_primitives/curobo.py:"
    "CuRoboMotionGenerator.check_collisions"
)


class R1ProCuRoboCollisionProviderError(RuntimeError):
    """Fail-closed official CuRobo binding or evaluation failure."""


@dataclass(frozen=True, slots=True)
class R1ProCuRoboCollisionAudit:
    """Private, identity-free diagnostic from the most recent successful call."""

    provider_version: str
    official_source: str
    sample_index: int
    held_object_attached: bool
    complete_world_refreshed_before_attach: bool
    self_collision_check: bool
    obstacle_update_skipped_by_check: bool
    predicted_collision: bool


@dataclass(frozen=True, slots=True)
class R1ProCuRoboCandidatePoseCollisionAudit:
    """Private diagnostic for one motionless world-base candidate check."""

    provider_version: str
    official_source: str
    sample_index: int
    candidate_world_x: float
    candidate_world_y: float
    candidate_world_yaw_rad: float
    held_object_attached: bool
    complete_world_refreshed_before_attach: bool
    world_cache_hit: bool
    self_collision_check: bool
    obstacle_update_skipped_by_check: bool
    predicted_collision: bool


@dataclass(frozen=True, slots=True)
class R1ProCuRoboCandidatePoseCollisionBatch:
    """Private result from one motionless multi-pose CuRobo query."""

    predicted_collisions: tuple[bool, ...]
    audits: tuple[R1ProCuRoboCandidatePoseCollisionAudit, ...]
    timing_private: Mapping[str, Any]


class R1ProCuRoboCurrentQCollisionProvider:
    """Read-only callable suitable for a private safety composition.

    Args:
        evaluator: Live evaluator whose ``robot`` owns the current joint state.
        starter: Live selected-arm Starter bound to the same robot.  Its
            official ``_motion_generator`` and ``_get_obj_in_hand`` interfaces
            are used directly.

    Calling the provider does not step physics, command the robot, or read task
    goal truth.  It does update CuRobo's private obstacle cache from the current
    scene before checking the current configuration.
    """

    api_contract_version = R1PRO_CUROBO_COLLISION_PROVIDER_VERSION
    official_source = OFFICIAL_CUROBO_COLLISION_SOURCE

    def __init__(
        self,
        *,
        evaluator: Any,
        starter: Any,
        held_object_provider: Callable[[], Any] | None = None,
    ) -> None:
        robot = getattr(evaluator, "robot", None)
        if robot is None or getattr(starter, "robot", None) is not robot:
            raise R1ProCuRoboCollisionProviderError(
                "collision provider requires one live evaluator/starter robot"
            )
        arm = getattr(starter, "arm", None)
        if arm not in {"left", "right"}:
            raise R1ProCuRoboCollisionProviderError(
                "Starter must expose one selected left/right arm"
            )
        motion_generator = getattr(starter, "_motion_generator", None)
        if motion_generator is None:
            raise R1ProCuRoboCollisionProviderError(
                "official Starter CuRobo motion generator is unavailable"
            )
        for owner, name in (
            (robot, "get_joint_positions"),
            (starter, "_get_obj_in_hand"),
            (motion_generator, "update_obstacles"),
            (motion_generator, "check_collisions"),
        ):
            if not callable(getattr(owner, name, None)):
                raise R1ProCuRoboCollisionProviderError(
                    f"official live collision interface is missing {name}"
                )
        eef_link_names = getattr(robot, "eef_link_names", None)
        if not isinstance(eef_link_names, Mapping) or arm not in eef_link_names:
            raise R1ProCuRoboCollisionProviderError(
                "official robot EEF link mapping is unavailable"
            )
        eef_link_name = eef_link_names[arm]
        if not isinstance(eef_link_name, str) or not eef_link_name:
            raise R1ProCuRoboCollisionProviderError(
                "official selected-arm EEF link name is invalid"
            )

        self._robot = robot
        self._starter = starter
        self._motion_generator = motion_generator
        self._arm = arm
        self._eef_link_name = eef_link_name
        if held_object_provider is not None and not callable(held_object_provider):
            raise R1ProCuRoboCollisionProviderError(
                "held object provider must be callable"
            )
        self._held_object_provider = held_object_provider
        self._poisoned = False
        self._sample_count = 0
        self._last_audit: R1ProCuRoboCollisionAudit | None = None

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def last_audit(self) -> R1ProCuRoboCollisionAudit | None:
        return self._last_audit

    def __call__(self) -> bool:
        if self._poisoned:
            raise R1ProCuRoboCollisionProviderError(
                "collision provider is poisoned"
            )
        try:
            return self._sample()
        except R1ProCuRoboCollisionProviderError:
            self._poisoned = True
            raise
        except Exception as exc:
            self._poisoned = True
            raise R1ProCuRoboCollisionProviderError(
                "official CuRobo current-q collision check failed"
            ) from exc

    def _sample(self) -> bool:
        current_q = self._robot.get_joint_positions()
        current_q_array = _finite_vector(current_q, "current joint positions")
        expected_joint_count = getattr(self._robot, "n_joints", None)
        if expected_joint_count is not None:
            if (
                isinstance(expected_joint_count, bool)
                or not isinstance(expected_joint_count, Integral)
                or expected_joint_count <= 0
                or current_q_array.size != expected_joint_count
            ):
                raise R1ProCuRoboCollisionProviderError(
                    "current joint vector does not match robot.n_joints"
                )
        q_batch = _single_configuration_batch(current_q)

        held = (
            self._starter._get_obj_in_hand()
            if self._held_object_provider is None
            else self._held_object_provider()
        )
        if held is None:
            attached_obj = None
        else:
            root_link = getattr(held, "root_link", None)
            if root_link is None:
                raise R1ProCuRoboCollisionProviderError(
                    "official held object has no root_link for CuRobo"
                )
            attached_obj = {self._eef_link_name: root_link}

        # CuRobo's official attach_objects_to_robot() resolves each collision
        # mesh by name from its current world model before temporarily disabling
        # that mesh as a world obstacle.  Excluding the held DatasetObject here
        # therefore produces None mesh entries and makes attachment impossible.
        # Refresh the complete world; the official attach/detach pair owns the
        # temporary disable/re-enable lifecycle during check_collisions().
        self._motion_generator.update_obstacles(ignore_objects=None)
        result = self._motion_generator.check_collisions(
            q_batch,
            initial_joint_pos=current_q,
            self_collision_check=True,
            skip_obstacle_update=True,
            attached_obj=attached_obj,
        )
        collision = _single_boolean(result)

        self._sample_count += 1
        self._last_audit = R1ProCuRoboCollisionAudit(
            provider_version=self.api_contract_version,
            official_source=self.official_source,
            sample_index=self._sample_count,
            held_object_attached=held is not None,
            complete_world_refreshed_before_attach=True,
            self_collision_check=True,
            obstacle_update_skipped_by_check=True,
            predicted_collision=collision,
        )
        return collision


class R1ProCuRoboCollisionProviderFactory:
    """Concrete ``(evaluator, starter) -> callable`` factory."""

    api_contract_version = R1PRO_CUROBO_COLLISION_PROVIDER_VERSION

    def __call__(
        self,
        evaluator: Any,
        starter: Any,
    ) -> R1ProCuRoboCurrentQCollisionProvider:
        return R1ProCuRoboCurrentQCollisionProvider(
            evaluator=evaluator,
            starter=starter,
        )


class R1ProCuRoboCandidatePoseCollisionProvider:
    """Check one proposed world base pose without moving or stepping R1Pro.

    This mirrors the official Starter ``_validate_poses`` base-collision
    branch, while using the corrected R1Pro world/root-frame conversion that
    is installed on the live Starter.  The current articulation is retained;
    only the virtual base coordinates are replaced.  A held object is attached
    to the selected EEF for the collision query.
    """

    api_contract_version = (
        R1PRO_CUROBO_CANDIDATE_POSE_COLLISION_PROVIDER_VERSION
    )
    official_source = OFFICIAL_CUROBO_COLLISION_SOURCE

    def __init__(
        self,
        *,
        evaluator: Any,
        starter: Any,
        held_object_provider: Callable[[], Any] | None = None,
    ) -> None:
        robot = getattr(evaluator, "robot", None)
        if robot is None or getattr(starter, "robot", None) is not robot:
            raise R1ProCuRoboCollisionProviderError(
                "candidate collision provider requires one live evaluator/starter robot"
            )
        arm = getattr(starter, "arm", None)
        if arm not in {"left", "right"}:
            raise R1ProCuRoboCollisionProviderError(
                "Starter must expose one selected left/right arm"
            )
        motion_generator = getattr(starter, "_motion_generator", None)
        if motion_generator is None:
            raise R1ProCuRoboCollisionProviderError(
                "official Starter CuRobo motion generator is unavailable"
            )
        for owner, name in (
            (robot, "get_joint_positions"),
            (starter, "_get_obj_in_hand"),
            (starter, "_behavior_hl_world_base_candidate_joint_position"),
            (motion_generator, "update_obstacles"),
            (motion_generator, "check_collisions"),
        ):
            if not callable(getattr(owner, name, None)):
                raise R1ProCuRoboCollisionProviderError(
                    f"official live candidate interface is missing {name}"
                )
        eef_link_names = getattr(robot, "eef_link_names", None)
        if not isinstance(eef_link_names, Mapping) or arm not in eef_link_names:
            raise R1ProCuRoboCollisionProviderError(
                "official robot EEF link mapping is unavailable"
            )
        eef_link_name = eef_link_names[arm]
        if not isinstance(eef_link_name, str) or not eef_link_name:
            raise R1ProCuRoboCollisionProviderError(
                "official selected-arm EEF link name is invalid"
            )

        self._robot = robot
        self._starter = starter
        self._motion_generator = motion_generator
        self._arm = arm
        self._eef_link_name = eef_link_name
        if held_object_provider is not None and not callable(held_object_provider):
            raise R1ProCuRoboCollisionProviderError(
                "held object provider must be callable"
            )
        self._held_object_provider = held_object_provider
        self._poisoned = False
        self._sample_count = 0
        self._last_audit: R1ProCuRoboCandidatePoseCollisionAudit | None = None
        self._last_batch: R1ProCuRoboCandidatePoseCollisionBatch | None = None
        self._invalid_world_signature = object()
        self._world_signature: object = self._invalid_world_signature
        self._has_refreshed_world = False

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def last_audit(self) -> R1ProCuRoboCandidatePoseCollisionAudit | None:
        return self._last_audit

    @property
    def last_batch(self) -> R1ProCuRoboCandidatePoseCollisionBatch | None:
        return self._last_batch

    def invalidate_world(self) -> None:
        """Force a complete world refresh before the next candidate query."""

        self._world_signature = self._invalid_world_signature

    def reuse_preflighted_empty_hand_world(self) -> bool:
        """Reuse the latest full-route snapshot only while no object is held."""

        held = (
            self._starter._get_obj_in_hand()
            if self._held_object_provider is None
            else self._held_object_provider()
        )
        if held is not None or not self._has_refreshed_world:
            return False
        self._world_signature = None
        return True

    def reuse_current_world(self) -> bool:
        """Reuse an uninvalidated world with the same held-object binding."""

        held = (
            self._starter._get_obj_in_hand()
            if self._held_object_provider is None
            else self._held_object_provider()
        )
        root_link = None if held is None else getattr(held, "root_link", None)
        if held is not None and root_link is None:
            return False
        signature = None if held is None else id(root_link)
        return bool(
            self._has_refreshed_world and self._world_signature == signature
        )

    def __call__(self, candidate: PlanarPose) -> bool:
        return self.check_many((candidate,)).predicted_collisions[0]

    def check_many(
        self,
        candidates: Sequence[PlanarPose],
    ) -> R1ProCuRoboCandidatePoseCollisionBatch:
        """Check a fixed candidate sequence in one official CuRobo call.

        All candidates share the same current articulation, held-object
        attachment, and obstacle-world snapshot.  No physics step or robot
        write occurs.  The caller remains responsible for choosing the first
        collision and therefore preserves the original safe-prefix policy.
        """

        if self._poisoned:
            raise R1ProCuRoboCollisionProviderError(
                "candidate collision provider is poisoned"
            )
        try:
            frozen_candidates = tuple(candidates)
        except TypeError as exc:
            self._poisoned = True
            raise R1ProCuRoboCollisionProviderError(
                "candidate collision batch must be iterable"
            ) from exc
        if not frozen_candidates:
            self._poisoned = True
            raise R1ProCuRoboCollisionProviderError(
                "candidate collision batch must not be empty"
            )
        if any(
            not isinstance(candidate, PlanarPose)
            for candidate in frozen_candidates
        ):
            self._poisoned = True
            raise R1ProCuRoboCollisionProviderError(
                "candidate collision batch requires PlanarPose values"
            )
        try:
            result = self._sample_many(frozen_candidates)
            self._last_batch = result
            self._last_audit = result.audits[-1]
            return result
        except R1ProCuRoboCollisionProviderError:
            self._poisoned = True
            raise
        except Exception as exc:
            self._poisoned = True
            raise R1ProCuRoboCollisionProviderError(
                "official CuRobo candidate-pose collision batch failed"
            ) from exc

    def _sample_many(
        self,
        candidates: tuple[PlanarPose, ...],
    ) -> R1ProCuRoboCandidatePoseCollisionBatch:
        total_started = perf_counter()
        joint_read_started = perf_counter()
        current_q = self._robot.get_joint_positions()
        current_q_array = _finite_vector(current_q, "current joint positions")
        joint_read_seconds = perf_counter() - joint_read_started
        expected_joint_count = getattr(self._robot, "n_joints", None)
        if expected_joint_count is not None and (
            isinstance(expected_joint_count, bool)
            or not isinstance(expected_joint_count, Integral)
            or expected_joint_count <= 0
            or current_q_array.size != expected_joint_count
        ):
            raise R1ProCuRoboCollisionProviderError(
                "current joint vector does not match robot.n_joints"
            )

        transform_started = perf_counter()
        new_tensor = getattr(current_q, "new_tensor", None)
        candidate_joint_positions: list[Any] = []
        for candidate in candidates:
            values = np.asarray(
                [candidate.x, candidate.y, candidate.yaw],
                dtype=np.float64,
            )
            if not np.isfinite(values).all():
                raise R1ProCuRoboCollisionProviderError(
                    "candidate world base pose must be finite"
                )
            pose_2d = (
                new_tensor(values.tolist())
                if callable(new_tensor)
                else values.astype(current_q_array.dtype, copy=False)
            )
            candidate_q = (
                self._starter._behavior_hl_world_base_candidate_joint_position(
                    pose_2d,
                    current_q,
                )
            )
            shape = tuple(getattr(candidate_q, "shape", ()))
            if shape and shape != (current_q_array.size,):
                raise R1ProCuRoboCollisionProviderError(
                    "candidate joint vector size changed"
                )
            candidate_joint_positions.append(candidate_q)
        q_batch = _configuration_batch(candidate_joint_positions)
        _finite_matrix(
            q_batch,
            name="candidate joint positions",
            rows=len(candidates),
            columns=current_q_array.size,
        )
        transform_seconds = perf_counter() - transform_started

        attachment_started = perf_counter()
        held = (
            self._starter._get_obj_in_hand()
            if self._held_object_provider is None
            else self._held_object_provider()
        )
        if held is None:
            attached_obj = None
            world_signature: object = None
        else:
            root_link = getattr(held, "root_link", None)
            if root_link is None:
                raise R1ProCuRoboCollisionProviderError(
                    "official held object has no root_link for CuRobo"
                )
            attached_obj = {self._eef_link_name: root_link}
            world_signature = id(root_link)
        attachment_seconds = perf_counter() - attachment_started

        world_cache_hit = self._world_signature == world_signature
        world_refresh_started = perf_counter()
        if not world_cache_hit:
            self._motion_generator.update_obstacles(ignore_objects=None)
            self._world_signature = world_signature
            self._has_refreshed_world = True
        world_refresh_seconds = perf_counter() - world_refresh_started
        collision_query_started = perf_counter()
        result = self._motion_generator.check_collisions(
            q_batch,
            initial_joint_pos=current_q,
            # A base-only candidate retains the current articulation.  This
            # matches official Starter._validate_poses: only new world/held
            # collisions need to be evaluated for the proposed base pose.
            self_collision_check=False,
            skip_obstacle_update=True,
            attached_obj=attached_obj,
        )
        collision_query_seconds = perf_counter() - collision_query_started
        result_decode_started = perf_counter()
        collisions = _boolean_vector(result, expected_size=len(candidates))
        result_decode_seconds = perf_counter() - result_decode_started

        first_sample_index = self._sample_count + 1
        audits = tuple(
            R1ProCuRoboCandidatePoseCollisionAudit(
                provider_version=self.api_contract_version,
                official_source=self.official_source,
                sample_index=first_sample_index + offset,
                candidate_world_x=float(candidate.x),
                candidate_world_y=float(candidate.y),
                candidate_world_yaw_rad=float(candidate.yaw),
                held_object_attached=held is not None,
                complete_world_refreshed_before_attach=(
                    not world_cache_hit and offset == 0
                ),
                world_cache_hit=(world_cache_hit or offset > 0),
                self_collision_check=False,
                obstacle_update_skipped_by_check=True,
                predicted_collision=collisions[offset],
            )
            for offset, candidate in enumerate(candidates)
        )
        self._sample_count += len(candidates)
        return R1ProCuRoboCandidatePoseCollisionBatch(
            predicted_collisions=collisions,
            audits=audits,
            timing_private={
                "candidate_count": len(candidates),
                "current_joint_read_seconds": joint_read_seconds,
                "candidate_transform_seconds": transform_seconds,
                "attachment_binding_seconds": attachment_seconds,
                "world_refresh_seconds": world_refresh_seconds,
                "collision_query_seconds": collision_query_seconds,
                "result_decode_seconds": result_decode_seconds,
                "total_seconds": perf_counter() - total_started,
                "world_refreshed": not world_cache_hit,
            },
        )


def _finite_vector(value: Any, name: str) -> np.ndarray:
    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    try:
        array = np.asarray(converted, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise R1ProCuRoboCollisionProviderError(
            f"{name} is not numeric"
        ) from exc
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise R1ProCuRoboCollisionProviderError(
            f"{name} must be a non-empty finite vector"
        )
    return array


def _single_configuration_batch(current_q: Any) -> Any:
    unsqueeze = getattr(current_q, "unsqueeze", None)
    if callable(unsqueeze):
        batch = unsqueeze(0)
        shape = tuple(getattr(batch, "shape", ()))
        if len(shape) != 2 or shape[0] != 1:
            raise R1ProCuRoboCollisionProviderError(
                "official current joint tensor could not form one batch"
            )
        return batch
    array = _finite_vector(current_q, "current joint positions")
    return array.reshape(1, -1)


def _configuration_batch(configurations: Sequence[Any]) -> Any:
    if not configurations:
        raise R1ProCuRoboCollisionProviderError(
            "candidate configurations must not be empty"
        )
    first = configurations[0]
    if callable(getattr(first, "unsqueeze", None)):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - live dependency gate
            raise R1ProCuRoboCollisionProviderError(
                "torch is required for official candidate tensors"
            ) from exc
        batch = torch.stack(tuple(configurations), dim=0)
    else:
        batch = np.stack(
            [np.asarray(value) for value in configurations],
            axis=0,
        )
    shape = tuple(getattr(batch, "shape", ()))
    if len(shape) != 2 or shape[0] != len(configurations):
        raise R1ProCuRoboCollisionProviderError(
            "candidate joint tensors could not form one batch"
        )
    return batch


def _finite_matrix(
    value: Any,
    *,
    name: str,
    rows: int,
    columns: int,
) -> np.ndarray:
    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    try:
        array = np.asarray(converted, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise R1ProCuRoboCollisionProviderError(
            f"{name} is not numeric"
        ) from exc
    if (
        array.shape != (rows, columns)
        or not np.isfinite(array).all()
    ):
        raise R1ProCuRoboCollisionProviderError(
            f"{name} must be one finite {rows}x{columns} matrix"
        )
    return array


def _single_boolean(value: Any) -> bool:
    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    try:
        array = np.asarray(converted)
    except (TypeError, ValueError) as exc:
        raise R1ProCuRoboCollisionProviderError(
            "official collision result is not array-like"
        ) from exc
    if array.size != 1 or array.dtype.kind != "b":
        raise R1ProCuRoboCollisionProviderError(
            "official collision result must contain exactly one boolean"
        )
    return bool(array.reshape(-1)[0])


def _boolean_vector(value: Any, *, expected_size: int) -> tuple[bool, ...]:
    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    try:
        array = np.asarray(converted)
    except (TypeError, ValueError) as exc:
        raise R1ProCuRoboCollisionProviderError(
            "official collision result is not array-like"
        ) from exc
    flattened = array.reshape(-1)
    if flattened.size != expected_size or flattened.dtype.kind != "b":
        if expected_size == 1:
            raise R1ProCuRoboCollisionProviderError(
                "official collision result must contain exactly one boolean"
            )
        raise R1ProCuRoboCollisionProviderError(
            "official collision result must match the candidate batch"
        )
    return tuple(bool(value) for value in flattened)
