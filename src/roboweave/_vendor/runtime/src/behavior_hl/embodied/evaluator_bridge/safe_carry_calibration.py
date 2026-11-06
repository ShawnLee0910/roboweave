"""Resident-scene safe-carry calibration record and provider bridge."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np


class SafeCarryCalibrationError(RuntimeError):
    """Fail-closed calibration or provider error."""


@dataclass(frozen=True, slots=True)
class SafeCarryCandidate:
    """Explicit candidate; it is not a validated safe-carry pose."""

    hand: str
    arm_target: tuple[float, ...]
    eef_world_position: tuple[float, ...]
    eef_world_quaternion_xyzw: tuple[float, ...]
    source: str

    def __post_init__(self) -> None:
        if self.hand not in {"left", "right"}:
            raise ValueError("hand must be left or right")
        object.__setattr__(
            self,
            "arm_target",
            _vector(self.arm_target, 7, "arm_target"),
        )
        object.__setattr__(
            self,
            "eef_world_position",
            _vector(
                self.eef_world_position,
                3,
                "eef_world_position",
            ),
        )
        quaternion = _vector(
            self.eef_world_quaternion_xyzw,
            4,
            "eef_world_quaternion_xyzw",
        )
        if _norm(quaternion) <= 1e-8:
            raise ValueError("EEF quaternion has zero norm")
        object.__setattr__(
            self,
            "eef_world_quaternion_xyzw",
            quaternion,
        )
        if (
            not isinstance(self.source, str)
            or not self.source.strip()
            or "reset" in self.source.casefold()
            or "default" in self.source.casefold()
            or "offset" in self.source.casefold()
        ):
            raise ValueError(
                "candidate source must identify a live calibration proposal"
            )


@dataclass(frozen=True, slots=True)
class SafeCarryCalibrationLimits:
    max_retract_steps: int
    g3b_stages: tuple[str, ...]
    samples_per_stage: int
    arm_tolerance_rad: float
    eef_position_tolerance_m: float
    eef_orientation_tolerance_deg: float
    minimum_object_lift_m: float
    maximum_payload_linear_speed_mps: float
    maximum_payload_angular_speed_radps: float
    maximum_base_z_drift_m: float
    maximum_abs_roll_pitch_deg: float

    def __post_init__(self) -> None:
        for name in ("max_retract_steps", "samples_per_stage"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        stages = tuple(self.g3b_stages)
        if not stages or len(set(stages)) != len(stages):
            raise ValueError("g3b_stages must be non-empty and unique")
        if any(not isinstance(stage, str) or not stage for stage in stages):
            raise ValueError("g3b stage names must be non-empty strings")
        object.__setattr__(self, "g3b_stages", stages)
        for name in (
            "arm_tolerance_rad",
            "eef_position_tolerance_m",
            "eef_orientation_tolerance_deg",
            "minimum_object_lift_m",
            "maximum_payload_linear_speed_mps",
            "maximum_payload_angular_speed_radps",
            "maximum_base_z_drift_m",
            "maximum_abs_roll_pitch_deg",
        ):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.minimum_object_lift_m < 0.05:
            raise ValueError("minimum_object_lift_m must be at least 0.05")


@dataclass(frozen=True, slots=True)
class SafeCarryLiveSample:
    """One real held-state sample; held token remains private and unrecorded."""

    hand: str
    arm_qpos: tuple[float, ...]
    torso_qpos: tuple[float, ...]
    arm_tracking_error_rad: float
    eef_world_position: tuple[float, ...]
    eef_world_quaternion_xyzw: tuple[float, ...]
    eef_body_position: tuple[float, ...]
    eef_body_quaternion_xyzw: tuple[float, ...]
    official_held: bool
    held_token: Any = field(repr=False)
    grasp_proxy_likely: bool
    object_lift_m: float
    payload_linear_speed_mps: float
    payload_angular_speed_radps: float
    base_z_drift_m: float
    roll_deg: float
    pitch_deg: float
    unsafe_contact: bool = False
    predicted_self_collision: bool = False
    joint_stalled: bool = False
    validation_stage: str | None = None

    def __post_init__(self) -> None:
        if self.hand not in {"left", "right"}:
            raise ValueError("sample hand must be left or right")
        for name, size in (
            ("arm_qpos", 7),
            ("torso_qpos", 4),
            ("eef_world_position", 3),
            ("eef_world_quaternion_xyzw", 4),
            ("eef_body_position", 3),
            ("eef_body_quaternion_xyzw", 4),
        ):
            object.__setattr__(
                self,
                name,
                _vector(getattr(self, name), size, name),
            )
        for name in (
            "official_held",
            "grasp_proxy_likely",
            "unsafe_contact",
            "predicted_self_collision",
            "joint_stalled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")
        for name in (
            "object_lift_m",
            "payload_linear_speed_mps",
            "payload_angular_speed_radps",
            "base_z_drift_m",
            "roll_deg",
            "pitch_deg",
            "arm_tracking_error_rad",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name),
            )
        if self.arm_tracking_error_rad < 0.0:
            raise ValueError("arm_tracking_error_rad must be non-negative")
        if (
            self.payload_linear_speed_mps < 0.0
            or self.payload_angular_speed_radps < 0.0
        ):
            raise ValueError("payload speeds must be non-negative")
        if self.validation_stage is not None and (
            not isinstance(self.validation_stage, str)
            or not self.validation_stage
        ):
            raise ValueError("validation_stage must be non-empty or None")


class SafeCarryCalibrationActuator(Protocol):
    def ik_only_valid(
        self,
        *,
        hand: str,
        eef_pose: tuple[tuple[float, ...], tuple[float, ...]],
    ) -> bool:
        """Run official IK-only without executing physics."""

    def step_toward(
        self,
        *,
        hand: str,
        arm_target: tuple[float, ...],
    ) -> SafeCarryLiveSample:
        """Advance one bounded physical retract step."""

    def step_g3b(
        self,
        *,
        hand: str,
        arm_target: tuple[float, ...],
        stage: str,
    ) -> SafeCarryLiveSample:
        """Advance one real held G3b validation step."""

    def safe_hold(self, *, hand: str) -> None:
        """Hold the last safe physical command."""


@dataclass(frozen=True, slots=True)
class G3BStageRecord:
    stage: str
    samples: int
    passed: bool


@dataclass(frozen=True, slots=True)
class SafeCarryCalibrationRecord:
    schema_version: str
    candidate: SafeCarryCandidate
    limits: SafeCarryCalibrationLimits
    ik_only_pass: bool
    retract_steps: int
    stage_records: tuple[G3BStageRecord, ...]
    held_continuous: bool
    grasp_proxy_continuous: bool
    body_eef_position: tuple[float, ...] | None
    body_eef_quaternion_xyzw: tuple[float, ...] | None
    passed: bool
    failure_reason: str | None

    def __post_init__(self) -> None:
        if self.schema_version != "behavior2026.safe-carry-calibration/v1":
            raise ValueError("unexpected calibration schema version")
        if not isinstance(self.candidate, SafeCarryCandidate):
            raise ValueError("candidate is invalid")
        if not isinstance(self.limits, SafeCarryCalibrationLimits):
            raise ValueError("limits are invalid")
        if self.passed:
            expected_stages = self.limits.g3b_stages
            actual_stages = tuple(item.stage for item in self.stage_records)
            if (
                not self.ik_only_pass
                or not self.held_continuous
                or not self.grasp_proxy_continuous
                or self.failure_reason is not None
                or actual_stages != expected_stages
                or any(
                    not item.passed
                    or item.samples != self.limits.samples_per_stage
                    for item in self.stage_records
                )
                or self.body_eef_position is None
                or self.body_eef_quaternion_xyzw is None
            ):
                raise ValueError(
                    "passing calibration record is internally inconsistent"
                )

    @property
    def sha256(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "hand": self.candidate.hand,
            "limits": {
                "max_retract_steps": self.limits.max_retract_steps,
                "g3b_stages": self.limits.g3b_stages,
                "samples_per_stage": self.limits.samples_per_stage,
                "arm_tolerance_rad": self.limits.arm_tolerance_rad,
                "eef_position_tolerance_m": (
                    self.limits.eef_position_tolerance_m
                ),
                "eef_orientation_tolerance_deg": (
                    self.limits.eef_orientation_tolerance_deg
                ),
            },
            "arm_target": self.candidate.arm_target,
            "candidate_world_pose": (
                self.candidate.eef_world_position,
                self.candidate.eef_world_quaternion_xyzw,
            ),
            "source": self.candidate.source,
            "ik_only_pass": self.ik_only_pass,
            "retract_steps": self.retract_steps,
            "stage_records": [
                (item.stage, item.samples, item.passed)
                for item in self.stage_records
            ],
            "held_continuous": self.held_continuous,
            "grasp_proxy_continuous": self.grasp_proxy_continuous,
            "body_eef_position": self.body_eef_position,
            "body_eef_quaternion_xyzw": (
                self.body_eef_quaternion_xyzw
            ),
            "passed": self.passed,
            "failure_reason": self.failure_reason,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()


def run_safe_carry_calibration(
    *,
    candidate: SafeCarryCandidate,
    limits: SafeCarryCalibrationLimits,
    actuator: SafeCarryCalibrationActuator,
    private_audit_writer: Callable[[Mapping[str, Any]], None],
) -> SafeCarryCalibrationRecord:
    """Run one bounded resident calibration and always return a record."""

    _validate_actuator(actuator)
    if not callable(private_audit_writer):
        raise SafeCarryCalibrationError(
            "private_audit_writer must be callable"
        )
    ik_pass = actuator.ik_only_valid(
        hand=candidate.hand,
        eef_pose=(
            candidate.eef_world_position,
            candidate.eef_world_quaternion_xyzw,
        ),
    )
    if not isinstance(ik_pass, bool):
        raise SafeCarryCalibrationError("ik_only_valid must return bool")
    retract_steps = 0
    stages: list[G3BStageRecord] = []
    held_token = None
    held_continuous = True
    proxy_continuous = True
    final_sample = None
    failure = None if ik_pass else "official_ik_only_failed"
    _audit(private_audit_writer, "ik_only", {"passed": ik_pass})

    try:
        if failure is None:
            for _ in range(limits.max_retract_steps):
                sample = actuator.step_toward(
                    hand=candidate.hand,
                    arm_target=candidate.arm_target,
                )
                retract_steps += 1
                held_token = _validate_sample(
                    sample,
                    candidate,
                    limits,
                    held_token,
                    require_stable_payload=False,
                    expected_stage=None,
                )
                final_sample = sample
                _audit(
                    private_audit_writer,
                    "retract_sample",
                    _sample_audit(sample, retract_steps),
                )
                if _at_candidate(sample, candidate, limits):
                    break
            else:
                failure = "retract_step_budget_exhausted"

        if failure is None:
            for stage in limits.g3b_stages:
                completed = 0
                for _ in range(limits.samples_per_stage):
                    sample = actuator.step_g3b(
                        hand=candidate.hand,
                        arm_target=candidate.arm_target,
                        stage=stage,
                    )
                    held_token = _validate_sample(
                        sample,
                        candidate,
                        limits,
                        held_token,
                        require_stable_payload=True,
                        expected_stage=stage,
                    )
                    if not _eef_at_candidate(
                        sample,
                        candidate,
                        limits,
                    ):
                        raise SafeCarryCalibrationError(
                            "safe-carry pose drifted during G3b"
                        )
                    final_sample = sample
                    completed += 1
                    _audit(
                        private_audit_writer,
                        "g3b_sample",
                        {
                            **_sample_audit(sample, completed),
                            "stage": stage,
                        },
                    )
                stages.append(G3BStageRecord(stage, completed, True))
    except SafeCarryCalibrationError as exc:
        failure = str(exc)
        held_continuous = (
            final_sample is not None and final_sample.official_held
        )
        proxy_continuous = (
            final_sample is not None
            and final_sample.grasp_proxy_likely
        )
    finally:
        actuator.safe_hold(hand=candidate.hand)

    passed = (
        failure is None
        and ik_pass
        and len(stages) == len(limits.g3b_stages)
        and final_sample is not None
    )
    record = SafeCarryCalibrationRecord(
        schema_version="behavior2026.safe-carry-calibration/v1",
        candidate=candidate,
        limits=limits,
        ik_only_pass=ik_pass,
        retract_steps=retract_steps,
        stage_records=tuple(stages),
        held_continuous=held_continuous and passed,
        grasp_proxy_continuous=proxy_continuous and passed,
        body_eef_position=(
            final_sample.eef_body_position
            if passed and final_sample is not None
            else None
        ),
        body_eef_quaternion_xyzw=(
            final_sample.eef_body_quaternion_xyzw
            if passed and final_sample is not None
            else None
        ),
        passed=passed,
        failure_reason=failure,
    )
    _audit(
        private_audit_writer,
        "final",
        {
            "passed": record.passed,
            "failure_reason": record.failure_reason,
            "record_sha256": record.sha256,
        },
    )
    return record


class ValidatedSafeCarryPoseProvider:
    """Body-relative provider constructible only from a passing record."""

    def __init__(self, record: SafeCarryCalibrationRecord) -> None:
        if (
            not isinstance(record, SafeCarryCalibrationRecord)
            or not record.passed
            or not record.ik_only_pass
            or not record.held_continuous
            or not record.grasp_proxy_continuous
            or record.body_eef_position is None
            or record.body_eef_quaternion_xyzw is None
            or not record.stage_records
            or not all(item.passed for item in record.stage_records)
            or tuple(item.stage for item in record.stage_records)
            != record.limits.g3b_stages
            or any(
                item.samples != record.limits.samples_per_stage
                for item in record.stage_records
            )
        ):
            raise SafeCarryCalibrationError(
                "safe-carry provider requires a fully validated record"
            )
        self.record = record
        self.record_sha256 = record.sha256

    def __call__(
        self,
        *,
        world_point: tuple[float, float, float],
        hand: str,
        robot: Any,
        eef_pose: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        del world_point, eef_pose
        if hand != self.record.candidate.hand:
            raise SafeCarryCalibrationError(
                "safe-carry provider hand does not match calibration"
            )
        read_pose = getattr(robot, "get_position_orientation", None)
        if not callable(read_pose):
            raise SafeCarryCalibrationError(
                "robot lacks body pose for calibrated provider"
            )
        body_position, body_quaternion = read_pose()
        world_position = _transform_position(
            body_position,
            body_quaternion,
            self.record.body_eef_position,
        )
        world_quaternion = _quat_multiply(
            body_quaternion,
            self.record.body_eef_quaternion_xyzw,
        )
        return world_position, world_quaternion


def provider_from_validated_calibration(
    record: SafeCarryCalibrationRecord,
) -> ValidatedSafeCarryPoseProvider:
    return ValidatedSafeCarryPoseProvider(record)


def _validate_sample(
    sample: Any,
    candidate: SafeCarryCandidate,
    limits: SafeCarryCalibrationLimits,
    held_token: Any,
    *,
    require_stable_payload: bool,
    expected_stage: str | None,
) -> Any:
    if not isinstance(sample, SafeCarryLiveSample):
        raise SafeCarryCalibrationError("actuator returned invalid sample")
    if sample.hand != candidate.hand:
        raise SafeCarryCalibrationError("active hand changed")
    if sample.validation_stage != expected_stage:
        raise SafeCarryCalibrationError(
            "calibration sample stage does not match requested G3b stage"
        )
    if not sample.official_held or sample.held_token is None:
        raise SafeCarryCalibrationError("official held state was lost")
    if held_token is not None and sample.held_token is not held_token:
        raise SafeCarryCalibrationError("held object identity changed")
    if not sample.grasp_proxy_likely:
        raise SafeCarryCalibrationError("public grasp proxy was lost")
    if sample.object_lift_m < limits.minimum_object_lift_m:
        raise SafeCarryCalibrationError("payload lift fell below threshold")
    if (
        sample.unsafe_contact
        or sample.predicted_self_collision
        or sample.joint_stalled
    ):
        raise SafeCarryCalibrationError("unsafe physical carry sample")
    if sample.arm_tracking_error_rad > limits.arm_tolerance_rad:
        raise SafeCarryCalibrationError(
            "active-arm compensation tracking bound exceeded"
        )
    if (
        abs(sample.base_z_drift_m) > limits.maximum_base_z_drift_m
        or abs(sample.roll_deg) > limits.maximum_abs_roll_pitch_deg
        or abs(sample.pitch_deg) > limits.maximum_abs_roll_pitch_deg
    ):
        raise SafeCarryCalibrationError("robot stability bound exceeded")
    if require_stable_payload and (
        sample.payload_linear_speed_mps
        > limits.maximum_payload_linear_speed_mps
        or sample.payload_angular_speed_radps
        > limits.maximum_payload_angular_speed_radps
    ):
        raise SafeCarryCalibrationError("payload stability bound exceeded")
    return sample.held_token


def _at_candidate(
    sample: SafeCarryLiveSample,
    candidate: SafeCarryCandidate,
    limits: SafeCarryCalibrationLimits,
) -> bool:
    return (
        max(
            abs(a - b)
            for a, b in zip(sample.arm_qpos, candidate.arm_target)
        )
        <= limits.arm_tolerance_rad
        and _eef_at_candidate(sample, candidate, limits)
    )


def _eef_at_candidate(
    sample: SafeCarryLiveSample,
    candidate: SafeCarryCandidate,
    limits: SafeCarryCalibrationLimits,
) -> bool:
    return (
        _distance(
            sample.eef_world_position,
            candidate.eef_world_position,
        )
        <= limits.eef_position_tolerance_m
        and _quat_angle_deg(
            sample.eef_world_quaternion_xyzw,
            candidate.eef_world_quaternion_xyzw,
        )
        <= limits.eef_orientation_tolerance_deg
    )


def _validate_actuator(actuator: Any) -> None:
    missing = [
        name
        for name in (
            "ik_only_valid",
            "step_toward",
            "step_g3b",
            "safe_hold",
        )
        if not callable(getattr(actuator, name, None))
    ]
    if missing:
        raise SafeCarryCalibrationError(
            f"calibration actuator missing capabilities: {missing}"
        )


def _sample_audit(
    sample: SafeCarryLiveSample,
    index: int,
) -> dict[str, Any]:
    return {
        "index": index,
        "hand": sample.hand,
        "official_held": sample.official_held,
        "grasp_proxy_likely": sample.grasp_proxy_likely,
        "object_lift_m": sample.object_lift_m,
        "arm_qpos": list(sample.arm_qpos),
        "torso_qpos": list(sample.torso_qpos),
        "arm_tracking_error_rad": sample.arm_tracking_error_rad,
        "validation_stage": sample.validation_stage,
        "eef_world_position": list(sample.eef_world_position),
        "eef_body_position": list(sample.eef_body_position),
    }


def _audit(
    writer: Callable[[Mapping[str, Any]], None],
    event: str,
    payload: Mapping[str, Any],
) -> None:
    writer(
        {
            "schema_version": "behavior2026.safe-carry-audit/v1",
            "event": event,
            **payload,
        }
    )


def _transform_position(
    body_position: Sequence[float],
    body_quaternion: Sequence[float],
    relative_position: Sequence[float],
) -> np.ndarray:
    return np.asarray(_vector(body_position, 3, "body_position")) + (
        _quat_matrix(body_quaternion)
        @ np.asarray(relative_position, dtype=np.float64)
    )


def _quat_matrix(value: Sequence[float]) -> np.ndarray:
    x, y, z, w = _normalized_quaternion(value)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat_multiply(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    ax, ay, az, aw = _normalized_quaternion(a)
    bx, by, bz, bw = _normalized_quaternion(b)
    return np.asarray(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _normalized_quaternion(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(_vector(value, 4, "quaternion"), dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-8:
        raise SafeCarryCalibrationError("quaternion has zero norm")
    return array / norm


def _quat_angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    dot = abs(float(np.dot(_normalized_quaternion(a), _normalized_quaternion(b))))
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(
        np.linalg.norm(
            np.asarray(a, dtype=np.float64)
            - np.asarray(b, dtype=np.float64)
        )
    )


def _norm(value: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


def _vector(value: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {size}D vector")
    return tuple(float(item) for item in array)


def _finite(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)
