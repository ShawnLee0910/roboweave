"""Bounded official-Starter driver for the R1Pro grasp binding.

The driver uses ``_ik_solver_cartesian_to_joint_space`` only for IK-only
preflight.  Physical motion remains in ``_move_hand``, ``_execute_release``,
and ``_execute_grasp`` generators, with at most one yielded 23D action sent by
each :meth:`tick_motion` call.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

from behavior_hl.embodied.evaluator_bridge.r1pro_grasp_binding import (
    R1PRO_GRASP_DRIVER_VERSION,
)
from behavior_hl.embodied.runtime.plugins.grasp import (
    GraspMotionPhase,
    GraspPhaseSample,
)


class StarterGraspDriverError(RuntimeError):
    """Fail-closed error for an unsafe or incomplete official binding."""


@dataclass(frozen=True, slots=True)
class StarterGraspPoseConfig:
    """Frozen world-frame pose offsets for the four-phase grasp chain."""

    top_down_quaternion_xyzw: tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        1.0,
    )
    pregrasp_offset_world_m: tuple[float, float, float] = (
        0.0,
        0.0,
        0.12,
    )
    lift_offset_world_m: tuple[float, float, float] = (
        0.0,
        0.0,
        0.08,
    )
    minimum_payload_lift_m: float = 0.05
    preflight_contact_pose_ik: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "top_down_quaternion_xyzw",
            _finite_vector(
                self.top_down_quaternion_xyzw,
                4,
                "top_down_quaternion_xyzw",
            ),
        )
        for name in (
            "pregrasp_offset_world_m",
            "lift_offset_world_m",
        ):
            object.__setattr__(
                self,
                name,
                _finite_vector(getattr(self, name), 3, name),
            )
        minimum_lift = _finite_scalar(
            self.minimum_payload_lift_m,
            "minimum_payload_lift_m",
        )
        if minimum_lift < 0.05:
            raise ValueError("minimum_payload_lift_m must be at least 0.05")
        object.__setattr__(
            self,
            "minimum_payload_lift_m",
            minimum_lift,
        )
        if self.lift_offset_world_m[2] < minimum_lift:
            raise ValueError(
                "lift_offset_world_m must command at least the minimum lift"
            )
        if not isinstance(self.preflight_contact_pose_ik, bool):
            raise ValueError("preflight_contact_pose_ik must be bool")


@dataclass(frozen=True, slots=True)
class PublicGripperWristEvidence:
    """Policy-eligible evidence sampled after one official action step."""

    qpos: tuple[float, float]
    qvel: tuple[float, float]
    wrist_visual_consistent: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "qpos",
            _finite_vector(self.qpos, 2, "qpos"),
        )
        object.__setattr__(
            self,
            "qvel",
            _finite_vector(self.qvel, 2, "qvel"),
        )
        if not isinstance(self.wrist_visual_consistent, bool):
            raise ValueError("wrist_visual_consistent must be bool")


@dataclass(frozen=True, slots=True)
class PrivateGraspEvidence:
    """Evaluator-only contact, held-object, and lift evidence."""

    qualified_contact: bool
    assisted_grasp_active: bool
    held_token: Any = field(default=None, repr=False)
    payload_lift_m: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.qualified_contact, bool):
            raise ValueError("qualified_contact must be bool")
        if not isinstance(self.assisted_grasp_active, bool):
            raise ValueError("assisted_grasp_active must be bool")
        object.__setattr__(
            self,
            "payload_lift_m",
            _finite_scalar(self.payload_lift_m, "payload_lift_m"),
        )


@dataclass(frozen=True, slots=True)
class StarterGraspPlan:
    """Private four-pose plan validated by official IK-only calls."""

    hand: str
    reach_pose: tuple[Any, Any] = field(repr=False)
    close_pose: tuple[Any, Any] = field(repr=False)
    lift_pose: tuple[Any, Any] = field(repr=False)
    safe_carry_retract_pose: tuple[Any, Any] = field(repr=False)
    ik_solutions: tuple[Any, Any, Any, Any] = field(repr=False)


class R1ProStarterGraspDriver:
    """Official Starter-backed implementation of ``R1ProGraspMotionDriver``."""

    api_contract_version = R1PRO_GRASP_DRIVER_VERSION

    def __init__(
        self,
        *,
        starter: Any,
        command_step: Callable[[np.ndarray, str], Any],
        public_evidence_provider: Callable[
            [str], PublicGripperWristEvidence
        ],
        private_audit_provider: Callable[
            [str, GraspMotionPhase], PrivateGraspEvidence
        ],
        safe_hold: Callable[[str], Any],
        safe_carry_pose_provider: Any = None,
        config: StarterGraspPoseConfig | None = None,
    ) -> None:
        self._starter = starter
        self._command_step = command_step
        self._public_evidence_provider = public_evidence_provider
        self._private_audit_provider = private_audit_provider
        self._safe_hold = safe_hold
        if not callable(safe_carry_pose_provider):
            raise StarterGraspDriverError(
                "safe_carry_pose_provider is required; no default carry pose "
                "is proven"
            )
        self._safe_carry_pose_provider = safe_carry_pose_provider
        self._config = config or StarterGraspPoseConfig()
        _validate_starter(starter)
        for value, name in (
            (command_step, "command_step"),
            (public_evidence_provider, "public_evidence_provider"),
            (private_audit_provider, "private_audit_provider"),
            (safe_hold, "safe_hold"),
        ):
            if not callable(value):
                raise StarterGraspDriverError(f"{name} must be callable")
        self._active_plan: StarterGraspPlan | None = None
        self._active_generators: dict[GraspMotionPhase, Iterator[Any]] = {}
        self._phase_action_counts: dict[GraspMotionPhase, int] = {}
        self._expected_phase = GraspMotionPhase.REACH
        self._last_private: PrivateGraspEvidence | None = None
        self._last_phase: GraspMotionPhase | None = None
        self._assisted_authorized = False
        self._contact_event_emitted = False
        self._last_preflight_diagnostics: tuple[dict[str, Any], ...] = ()

    @property
    def last_preflight_diagnostics(self) -> tuple[dict[str, Any], ...]:
        """Return the latest per-pose private IK admission diagnostics."""

        return tuple(dict(item) for item in self._last_preflight_diagnostics)

    def preflight_ik(
        self,
        *,
        world_point: tuple[float, float, float],
        hand: str,
    ) -> StarterGraspPlan | None:
        if hand != getattr(self._starter, "arm", None):
            raise StarterGraspDriverError(
                "requested hand does not match initialized Starter arm"
            )
        point = np.asarray(world_point, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise StarterGraspDriverError(
                "world_point must be a finite 3D point"
            )
        position_reference, quaternion_reference = (
            self._starter.robot.get_eef_pose(arm=hand)
        )
        quaternion = _tensor_like(
            quaternion_reference,
            self._config.top_down_quaternion_xyzw,
        )
        positions = {
            "reach": point
            + np.asarray(
                self._config.pregrasp_offset_world_m,
                dtype=np.float64,
            ),
            "close": point,
            "lift": point
            + np.asarray(
                self._config.lift_offset_world_m,
                dtype=np.float64,
            ),
        }
        poses = {
            name: (
                _tensor_like(position_reference, position),
                quaternion,
            )
            for name, position in positions.items()
        }
        raw_safe_carry_pose = self._safe_carry_pose_provider(
            world_point=tuple(float(value) for value in point),
            hand=hand,
            robot=self._starter.robot,
            eef_pose=(position_reference, quaternion_reference),
        )
        poses["safe_carry_retract"] = _validated_pose_like(
            raw_safe_carry_pose,
            position_reference=position_reference,
            quaternion_reference=quaternion_reference,
        )

        solutions = []
        diagnostics: list[dict[str, Any]] = []
        for index, name in enumerate(
            ("reach", "close", "lift", "safe_carry_retract")
        ):
            if name == "close" and not self._config.preflight_contact_pose_ik:
                diagnostics.append(
                    {
                        "pose": name,
                        "solution_present": None,
                        "position": [
                            float(value)
                            for value in _diagnostic_vector(poses[name][0], 3)
                        ],
                        "skipped_reason": (
                            "official_motion_constrained_assisted_contact_approach"
                        ),
                    }
                )
                solutions.append(None)
                continue
            solution = self._starter._ik_solver_cartesian_to_joint_space(
                poses[name],
                initial_joint_pos=None,
                skip_obstacle_update=index > 0,
            )
            diagnostics.append(
                {
                    "pose": name,
                    "solution_present": solution is not None,
                    "position": [
                        float(value)
                        for value in _diagnostic_vector(poses[name][0], 3)
                    ],
                }
            )
            if solution is None:
                self._last_preflight_diagnostics = tuple(diagnostics)
                return None
            solutions.append(solution)
        self._last_preflight_diagnostics = tuple(diagnostics)
        plan = StarterGraspPlan(
            hand=hand,
            reach_pose=poses["reach"],
            close_pose=poses["close"],
            lift_pose=poses["lift"],
            safe_carry_retract_pose=poses["safe_carry_retract"],
            ik_solutions=tuple(solutions),
        )
        begin_public_evidence = getattr(
            self._public_evidence_provider,
            "begin_grasp",
            None,
        )
        if callable(begin_public_evidence):
            begin_public_evidence(hand)
        self._active_plan = plan
        self._active_generators.clear()
        self._phase_action_counts.clear()
        self._expected_phase = GraspMotionPhase.REACH
        self._last_private = None
        self._last_phase = None
        self._assisted_authorized = False
        self._contact_event_emitted = False
        return plan

    def tick_motion(
        self,
        *,
        phase: GraspMotionPhase,
        plan_token: Any,
        hand: str,
    ) -> GraspPhaseSample:
        plan = self._require_plan(plan_token, hand)
        if phase is not self._expected_phase:
            raise StarterGraspDriverError(
                f"expected {self._expected_phase.value}, received {phase.value}"
            )
        if (
            phase in {
                GraspMotionPhase.LIFT,
                GraspMotionPhase.SAFE_CARRY_RETRACT,
            }
            and not self._assisted_authorized
        ):
            raise StarterGraspDriverError(
                "carry motion requires contact-authorized assisted grasp"
            )
        generator = self._active_generators.get(phase)
        if generator is None:
            generator = iter(self._generator_for_phase(phase, plan))
            self._active_generators[phase] = generator
            self._phase_action_counts[phase] = 0
        try:
            raw_action = next(generator)
        except StopIteration:
            if self._phase_action_counts[phase] == 0:
                raise StarterGraspDriverError(
                    "official physical phase generated no 23D action"
                )
            completed = True
            hold_result = self._safe_hold(hand)
            if hold_result != 1:
                raise StarterGraspDriverError(
                    "phase-completion safe_hold must advance exactly one step"
                )
        else:
            completed = False
            action = np.asarray(raw_action, dtype=np.float64)
            if action.shape != (23,) or not np.isfinite(action).all():
                raise StarterGraspDriverError(
                    "official generator yielded a non-finite or non-23D action"
                )
            step_result = self._command_step(
                np.array(action, copy=True),
                f"grasp.{phase.value}",
            )
            if step_result != 1:
                raise StarterGraspDriverError(
                    "command_step must advance exactly one simulator step"
                )
            self._phase_action_counts[phase] += 1

        public = self._public_evidence_provider(hand)
        private = self._private_audit_provider(hand, phase)
        if not isinstance(public, PublicGripperWristEvidence):
            raise StarterGraspDriverError(
                "public evidence provider returned an invalid value"
            )
        if not isinstance(private, PrivateGraspEvidence):
            raise StarterGraspDriverError(
                "private audit provider returned an invalid value"
            )
        self._last_private = private
        self._last_phase = phase

        if completed:
            self._active_generators.pop(phase, None)
            self._phase_action_counts.pop(phase, None)
            self._expected_phase = _next_phase(phase)

        contact_event = (
            phase is GraspMotionPhase.CLOSE
            and private.qualified_contact
            and private.assisted_grasp_active
            and private.held_token is not None
            and not self._contact_event_emitted
        )
        if contact_event:
            self._contact_event_emitted = True
        return GraspPhaseSample(
            completed=completed,
            simulator_steps=1,
            gripper_qpos=public.qpos,
            gripper_qvel=public.qvel,
            wrist_visual_consistent=public.wrist_visual_consistent,
            contact_event=contact_event,
            payload_lift_m=(
                private.payload_lift_m
                if phase is GraspMotionPhase.LIFT
                else 0.0
            ),
        )

    def enable_assisted_grasp(
        self,
        *,
        plan_token: Any,
        hand: str,
    ) -> bool:
        self._require_plan(plan_token, hand)
        private = self._last_private
        authorized = (
            self._last_phase is GraspMotionPhase.CLOSE
            and private is not None
            and private.qualified_contact
            and private.assisted_grasp_active
            and private.held_token is not None
        )
        self._assisted_authorized = authorized
        return authorized

    def hold_safe(self, *, hand: str) -> None:
        if hand != getattr(self._starter, "arm", None):
            raise StarterGraspDriverError(
                "safe hold hand does not match Starter arm"
            )
        result = self._safe_hold(hand)
        if result not in (None, 1):
            raise StarterGraspDriverError(
                "safe_hold must advance zero or one simulator step"
            )

    def _require_plan(
        self,
        plan_token: Any,
        hand: str,
    ) -> StarterGraspPlan:
        if (
            not isinstance(plan_token, StarterGraspPlan)
            or plan_token is not self._active_plan
        ):
            raise StarterGraspDriverError("unknown or stale grasp plan")
        if plan_token.hand != hand:
            raise StarterGraspDriverError("grasp plan hand changed")
        return plan_token

    def _generator_for_phase(
        self,
        phase: GraspMotionPhase,
        plan: StarterGraspPlan,
    ) -> Iterable[Any]:
        if phase is GraspMotionPhase.REACH:
            return itertools.chain(
                self._starter._execute_release(),
                self._starter._move_hand(
                    plan.reach_pose,
                    lock_auxiliary_arm=True,
                    # This is the collision-checked free-space pregrasp leg.
                    # R1Pro's torso has a measured steady-state tracking
                    # offset above the official strict threshold, so use the
                    # official Starter low-precision execution mode here only.
                    # The following contact approach remains strict.
                    low_precision=True,
                ),
                self._starter._move_hand(
                    plan.close_pose,
                    motion_constraint=[1, 1, 1, 1, 1, 0],
                    lock_auxiliary_arm=True,
                ),
            )
        if phase is GraspMotionPhase.CLOSE:
            return self._starter._execute_grasp()
        if phase is GraspMotionPhase.LIFT:
            return self._starter._move_hand(
                plan.lift_pose,
                lock_auxiliary_arm=True,
            )
        if phase is GraspMotionPhase.SAFE_CARRY_RETRACT:
            return self._starter._move_hand(
                plan.safe_carry_retract_pose,
                lock_auxiliary_arm=True,
            )
        raise StarterGraspDriverError(f"unsupported phase: {phase!r}")


def _validate_starter(starter: Any) -> None:
    missing = [
        name
        for name in (
            "_ik_solver_cartesian_to_joint_space",
            "_move_hand",
            "_execute_release",
            "_execute_grasp",
        )
        if not callable(getattr(starter, name, None))
    ]
    robot = getattr(starter, "robot", None)
    if robot is None or not callable(getattr(robot, "get_eef_pose", None)):
        missing.append("robot.get_eef_pose")
    if getattr(starter, "arm", None) not in {"left", "right"}:
        missing.append("selected arm")
    if missing:
        raise StarterGraspDriverError(
            f"Starter binding is missing capabilities: {missing}"
        )


def _next_phase(phase: GraspMotionPhase) -> GraspMotionPhase:
    transitions = {
        GraspMotionPhase.REACH: GraspMotionPhase.CLOSE,
        GraspMotionPhase.CLOSE: GraspMotionPhase.LIFT,
        GraspMotionPhase.LIFT: GraspMotionPhase.SAFE_CARRY_RETRACT,
        GraspMotionPhase.SAFE_CARRY_RETRACT: (
            GraspMotionPhase.SAFE_CARRY_RETRACT
        ),
    }
    return transitions[phase]


def _tensor_like(reference: Any, values: Sequence[float]) -> Any:
    if callable(getattr(reference, "new_tensor", None)):
        return reference.new_tensor(values)
    return np.asarray(values, dtype=np.float32)


def _validated_pose_like(
    value: Any,
    *,
    position_reference: Any,
    quaternion_reference: Any,
) -> tuple[Any, Any]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise StarterGraspDriverError(
            "safe_carry_pose_provider must return (position, quaternion)"
        )
    position = _finite_vector(value[0], 3, "safe_carry_position")
    quaternion = _finite_vector(value[1], 4, "safe_carry_quaternion")
    if float(np.linalg.norm(quaternion)) <= 1e-8:
        raise StarterGraspDriverError(
            "safe-carry quaternion has zero norm"
        )
    return (
        _tensor_like(position_reference, position),
        _tensor_like(quaternion_reference, quaternion),
    )


def _finite_vector(
    value: Sequence[float],
    size: int,
    name: str,
) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {size}D vector")
    return tuple(float(item) for item in array)


def _diagnostic_vector(value: Any, size: int) -> tuple[float, ...]:
    """Convert a CPU / torch tensor pose to a small JSON-safe vector."""

    item = value
    for method in ("detach", "cpu"):
        callback = getattr(item, method, None)
        if callable(callback):
            item = callback()
    numpy_method = getattr(item, "numpy", None)
    if callable(numpy_method):
        item = numpy_method()
    return _finite_vector(item, size, "diagnostic pose")


def _finite_scalar(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)
