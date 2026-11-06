"""Empty-hand, exact-candidate-pose public wrist negative calibration.

This CPU-importable bridge owns the final bootstrap loop before a live grasp:

``official IK-only -> move_hand -> empty close -> stable public samples
-> release``

Every yielded official 23D command is followed by exactly one evaluator step
and one freshly sanitized public snapshot.  The coordinator requires at least
three empty-close trials and replays each trial through the public-only proxy;
only zero false positives can produce a
``BootstrapPublicWristCalibrationRecord``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from behavior_hl.embodied.evaluator_bridge.r1pro_public_wrist_evidence import (
    BootstrapPublicGripperWristEvidenceProvider,
    BootstrapPublicWristCalibrationRecord,
    WristVisualThresholds,
    safe_carry_candidate_sha256,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_torso_release_binding import (
    LEFT_GRIPPER_QPOS,
    RIGHT_GRIPPER_QPOS,
)
from behavior_hl.embodied.evaluator_bridge.safe_carry_calibration import (
    SafeCarryCandidate,
)
from behavior_hl.embodied.policy.observations import (
    CAMERA_RESOLUTIONS,
    PROPRIO_DIM,
    validate_public_observation,
)


R1PRO_EMPTY_WRIST_CALIBRATION_ACTUATOR_VERSION = (
    "behavior2026.r1pro-empty-wrist-calibration-actuator/v1"
)
R1PRO_EMPTY_WRIST_CALIBRATION_RESULT_VERSION = (
    "behavior2026.r1pro-empty-wrist-calibration-result/v1"
)


class EmptyWristCalibrationError(RuntimeError):
    """Fail-closed calibration, motion, or observation error."""


class EmptyWristPhase(str, Enum):
    MOVE = "move_to_candidate"
    CLOSE = "empty_close"
    OPEN = "release"


@dataclass(frozen=True, slots=True)
class EmptyWristCalibrationConfig:
    """Explicit public thresholds and bounded physical calibration policy."""

    hand: str
    roi_xyxy: tuple[int, int, int, int]
    thresholds: WristVisualThresholds
    safe_carry_candidate_sha256: str
    trials: int = 3
    stable_samples_per_trial: int = 3
    max_steps_per_phase: int = 256
    empty_qpos_tolerance: float = 0.003
    eef_position_tolerance_m: float = 0.01
    eef_orientation_tolerance_deg: float = 2.0

    def __post_init__(self) -> None:
        if self.hand not in {"left", "right"}:
            raise ValueError("hand must be left or right")
        if not isinstance(self.thresholds, WristVisualThresholds):
            raise TypeError("thresholds must be WristVisualThresholds")
        if (
            not isinstance(self.safe_carry_candidate_sha256, str)
            or len(self.safe_carry_candidate_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.safe_carry_candidate_sha256
            )
        ):
            raise ValueError(
                "safe_carry_candidate_sha256 must be lowercase SHA256"
            )
        height, width = CAMERA_RESOLUTIONS[f"{self.hand}_wrist"]
        if (
            not isinstance(self.roi_xyxy, (tuple, list))
            or len(self.roi_xyxy) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.roi_xyxy
            )
        ):
            raise ValueError("roi_xyxy must contain four integers")
        x0, y0, x1, y1 = map(int, self.roi_xyxy)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("roi_xyxy is outside the native wrist frame")
        object.__setattr__(self, "roi_xyxy", (x0, y0, x1, y1))
        for name, minimum in (
            ("trials", 3),
            (
                "stable_samples_per_trial",
                self.thresholds.consecutive_frames,
            ),
            ("max_steps_per_phase", 1),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in (
            "empty_qpos_tolerance",
            "eef_position_tolerance_m",
            "eef_orientation_tolerance_deg",
        ):
            value = _positive(getattr(self, name), name)
            object.__setattr__(self, name, value)

    def to_manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "hand": self.hand,
                "roi_xyxy": self.roi_xyxy,
                "thresholds": dict(self.thresholds.to_manifest()),
                "safe_carry_candidate_sha256": (
                    self.safe_carry_candidate_sha256
                ),
                "trials": self.trials,
                "stable_samples_per_trial": (
                    self.stable_samples_per_trial
                ),
                "max_steps_per_phase": self.max_steps_per_phase,
                "empty_qpos_tolerance": self.empty_qpos_tolerance,
                "eef_position_tolerance_m": (
                    self.eef_position_tolerance_m
                ),
                "eef_orientation_tolerance_deg": (
                    self.eef_orientation_tolerance_deg
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class EmptyWristStepSample:
    phase: EmptyWristPhase
    completed: bool
    simulator_steps: int
    public_snapshot: Mapping[str, Any] = field(repr=False)
    at_candidate_pose: bool

    def __post_init__(self) -> None:
        if not isinstance(self.phase, EmptyWristPhase):
            raise ValueError("phase is invalid")
        if not isinstance(self.completed, bool):
            raise ValueError("completed must be bool")
        if self.simulator_steps != 1:
            raise ValueError("each actuator tick must advance exactly one step")
        if not isinstance(self.public_snapshot, Mapping):
            raise ValueError("public_snapshot must be a mapping")
        if not isinstance(self.at_candidate_pose, bool):
            raise ValueError("at_candidate_pose must be bool")


@dataclass(frozen=True, slots=True)
class EmptyWristTrialRecord:
    trial_index: int
    move_steps: int
    close_steps: int
    stable_samples: int
    open_steps: int
    public_false_positive: bool
    closed_sample_sha256: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EmptyWristCalibrationResult:
    schema_version: str
    passed: bool
    candidate_sha256: str
    ik_only_passed: bool
    trial_records: tuple[EmptyWristTrialRecord, ...]
    bootstrap_record: BootstrapPublicWristCalibrationRecord | None
    failure_reason: str | None
    config_manifest: Mapping[str, Any]
    actuator_poisoned: bool
    commands_issued: int
    steps_confirmed: int

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != R1PRO_EMPTY_WRIST_CALIBRATION_RESULT_VERSION
        ):
            raise ValueError("unexpected empty-wrist result version")
        if self.passed != (self.bootstrap_record is not None):
            raise ValueError(
                "passed result must contain exactly one bootstrap record"
            )
        for name in ("commands_issued", "steps_confirmed"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be non-negative")
        if self.passed and (
            not self.ik_only_passed
            or self.failure_reason is not None
            or self.actuator_poisoned
            or self.commands_issued != self.steps_confirmed
            or len(self.trial_records) < 3
            or any(
                trial.public_false_positive
                for trial in self.trial_records
            )
        ):
            raise ValueError("passing calibration result is inconsistent")
        object.__setattr__(
            self,
            "config_manifest",
            MappingProxyType(dict(self.config_manifest)),
        )

    def to_manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema_version": self.schema_version,
                "passed": self.passed,
                "candidate_sha256": self.candidate_sha256,
                "ik_only_passed": self.ik_only_passed,
                "trial_records": [
                    {
                        "trial_index": trial.trial_index,
                        "move_steps": trial.move_steps,
                        "close_steps": trial.close_steps,
                        "stable_samples": trial.stable_samples,
                        "open_steps": trial.open_steps,
                        "public_false_positive": (
                            trial.public_false_positive
                        ),
                        "closed_sample_sha256": (
                            trial.closed_sample_sha256
                        ),
                    }
                    for trial in self.trial_records
                ],
                "bootstrap_calibration_sha256": (
                    self.bootstrap_record.calibration_sha256
                    if self.bootstrap_record is not None
                    else None
                ),
                "failure_reason": self.failure_reason,
                "config": dict(self.config_manifest),
                "actuator_poisoned": self.actuator_poisoned,
                "commands_issued": self.commands_issued,
                "steps_confirmed": self.steps_confirmed,
            }
        )


@dataclass(slots=True)
class _PhaseToken:
    phase: EmptyWristPhase
    generator: Iterator[Any] = field(repr=False)
    candidate: SafeCarryCandidate = field(repr=False)
    yielded_actions: int = 0


class R1ProEmptyWristCalibrationActuator:
    """Official Starter-backed, one-command/one-step calibration actuator."""

    api_contract_version = (
        R1PRO_EMPTY_WRIST_CALIBRATION_ACTUATOR_VERSION
    )

    def __init__(
        self,
        *,
        starter: Any,
        control: Any,
        evaluator_step: Callable[[str], int],
        live_public_snapshot: Callable[[], Mapping[str, Any]],
        eef_position_tolerance_m: float,
        eef_orientation_tolerance_deg: float,
    ) -> None:
        self._starter = starter
        self._control = control
        self._evaluator_step = evaluator_step
        self._live_public_snapshot = live_public_snapshot
        self._position_tolerance = _positive(
            eef_position_tolerance_m,
            "eef_position_tolerance_m",
        )
        self._orientation_tolerance_deg = _positive(
            eef_orientation_tolerance_deg,
            "eef_orientation_tolerance_deg",
        )
        _validate_starter(starter)
        for name, callback in (
            ("evaluator_step", evaluator_step),
            ("live_public_snapshot", live_public_snapshot),
        ):
            if not callable(callback):
                raise TypeError(f"{name} must be callable")
        for method in ("command_direct_action", "hold"):
            if not callable(getattr(control, method, None)):
                raise EmptyWristCalibrationError(
                    f"control is missing {method}"
                )
        self._poisoned = False
        self._candidate_sha256: str | None = None
        self._expected_phase = EmptyWristPhase.MOVE
        self._active_token: _PhaseToken | None = None
        self._commands = 0
        self._steps = 0

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def step_accounting(self) -> tuple[int, int]:
        return self._commands, self._steps

    def ik_only_valid(self, candidate: SafeCarryCandidate) -> bool:
        self._require_operational()
        self._validate_candidate(candidate)
        pose = self._pose_like(candidate)
        result = self._starter._ik_solver_cartesian_to_joint_space(
            pose,
            initial_joint_pos=None,
            skip_obstacle_update=False,
        )
        passed = result is not None
        self._candidate_sha256 = (
            safe_carry_candidate_sha256(candidate) if passed else None
        )
        return passed

    def begin_phase(
        self,
        *,
        phase: EmptyWristPhase,
        candidate: SafeCarryCandidate,
    ) -> _PhaseToken:
        self._require_operational()
        self._validate_candidate(candidate)
        if phase is not self._expected_phase:
            raise EmptyWristCalibrationError(
                f"expected phase {self._expected_phase.value}, "
                f"received {phase.value}"
            )
        if (
            self._candidate_sha256 is None
            or safe_carry_candidate_sha256(candidate)
            != self._candidate_sha256
        ):
            raise EmptyWristCalibrationError(
                "phase candidate lacks matching official IK-only admission"
            )
        if self._active_token is not None:
            raise EmptyWristCalibrationError(
                "cannot begin a phase while another phase is active"
            )
        if phase is EmptyWristPhase.MOVE:
            generator = self._starter._move_hand(
                self._pose_like(candidate),
                lock_auxiliary_arm=True,
            )
        elif phase is EmptyWristPhase.CLOSE:
            generator = self._starter._execute_grasp()
        else:
            generator = self._starter._execute_release()
        token = _PhaseToken(
            phase=phase,
            generator=iter(generator),
            candidate=candidate,
        )
        self._active_token = token
        return token

    def tick_phase(self, token: _PhaseToken) -> EmptyWristStepSample:
        self._require_operational()
        if token is not self._active_token:
            raise EmptyWristCalibrationError("unknown or stale phase token")
        try:
            action = next(token.generator)
        except StopIteration:
            if token.yielded_actions == 0:
                self._fail("official calibration phase yielded no action")
            sample = self._hold_step(
                label=f"empty_wrist.{token.phase.value}.complete",
                candidate=token.candidate,
                phase=token.phase,
                completed=True,
            )
            if not sample.at_candidate_pose:
                self._fail(
                    "official phase completed away from candidate exact pose"
                )
            self._active_token = None
            self._expected_phase = {
                EmptyWristPhase.MOVE: EmptyWristPhase.CLOSE,
                EmptyWristPhase.CLOSE: EmptyWristPhase.OPEN,
                EmptyWristPhase.OPEN: EmptyWristPhase.MOVE,
            }[token.phase]
            return sample
        array = np.asarray(action, dtype=np.float64)
        if array.shape != (23,) or not np.isfinite(array).all():
            self._fail("official calibration action must be finite 23D")
        try:
            self._control.command_direct_action(
                np.array(array, dtype=np.float32, copy=True)
            )
            self._commands += 1
            self._step_once(f"empty_wrist.{token.phase.value}")
            snapshot = self._snapshot()
            at_candidate = self._at_candidate(token.candidate)
            if (
                token.phase in {EmptyWristPhase.CLOSE, EmptyWristPhase.OPEN}
                and not at_candidate
            ):
                self._fail(
                    "gripper phase drifted from candidate exact pose"
                )
        except EmptyWristCalibrationError:
            raise
        except Exception as exc:
            self._fail(
                "calibration command/step/snapshot failed: "
                f"{type(exc).__name__}: {exc}",
                exc,
            )
        token.yielded_actions += 1
        return EmptyWristStepSample(
            phase=token.phase,
            completed=False,
            simulator_steps=1,
            public_snapshot=snapshot,
            at_candidate_pose=at_candidate,
        )

    def hold_sample(
        self,
        *,
        candidate: SafeCarryCandidate,
        label: str,
    ) -> EmptyWristStepSample:
        self._require_operational()
        self._validate_candidate(candidate)
        if self._active_token is not None:
            raise EmptyWristCalibrationError(
                "stable hold cannot interrupt an active official phase"
            )
        if self._expected_phase is not EmptyWristPhase.OPEN:
            raise EmptyWristCalibrationError(
                "stable empty-close samples require completed close phase"
            )
        return self._hold_step(
            label=label,
            candidate=candidate,
            phase=EmptyWristPhase.CLOSE,
            completed=True,
        )

    def safe_hold(self) -> None:
        try:
            self._control.hold()
        except Exception as exc:
            self._poisoned = True
            raise EmptyWristCalibrationError("safe hold failed") from exc

    def abort(self, reason: str) -> None:
        del reason
        self._poisoned = True
        self._active_token = None
        try:
            self._control.hold()
        except Exception:
            pass

    def _hold_step(
        self,
        *,
        label: str,
        candidate: SafeCarryCandidate,
        phase: EmptyWristPhase,
        completed: bool,
    ) -> EmptyWristStepSample:
        try:
            self._control.hold()
            self._commands += 1
            self._step_once(label)
            snapshot = self._snapshot()
            at_candidate = self._at_candidate(candidate)
        except EmptyWristCalibrationError:
            raise
        except Exception as exc:
            self._fail("calibration hold/step/snapshot failed", exc)
        if not at_candidate:
            self._fail("stable sample is not at candidate exact pose")
        return EmptyWristStepSample(
            phase=phase,
            completed=completed,
            simulator_steps=1,
            public_snapshot=snapshot,
            at_candidate_pose=at_candidate,
        )

    def _step_once(self, label: str) -> None:
        result = self._evaluator_step(label)
        if isinstance(result, bool) or not isinstance(result, int) or result != 1:
            self._fail("evaluator_step must confirm exactly one step")
        self._steps += 1
        if self._steps != self._commands:
            self._fail(
                "calibration command/evaluator-step accounting diverged"
            )

    def _snapshot(self) -> Mapping[str, Any]:
        raw = self._live_public_snapshot()
        try:
            validate_public_observation(raw)
        except Exception as exc:
            self._fail("live snapshot is not sanitized public R1Pro RGB-D", exc)
        return _freeze_public_snapshot(raw)

    def _at_candidate(self, candidate: SafeCarryCandidate) -> bool:
        position, quaternion = self._starter.robot.get_eef_pose(
            arm=candidate.hand
        )
        position_error = float(
            np.linalg.norm(
                np.asarray(position, dtype=np.float64)
                - np.asarray(
                    candidate.eef_world_position,
                    dtype=np.float64,
                )
            )
        )
        orientation_error = _quat_angle_deg(
            quaternion,
            candidate.eef_world_quaternion_xyzw,
        )
        return (
            position_error <= self._position_tolerance
            and orientation_error <= self._orientation_tolerance_deg
        )

    def _pose_like(self, candidate: SafeCarryCandidate) -> tuple[Any, Any]:
        position_reference, quaternion_reference = (
            self._starter.robot.get_eef_pose(arm=candidate.hand)
        )
        return (
            _tensor_like(
                position_reference,
                candidate.eef_world_position,
            ),
            _tensor_like(
                quaternion_reference,
                candidate.eef_world_quaternion_xyzw,
            ),
        )

    def _validate_candidate(self, candidate: Any) -> None:
        if not isinstance(candidate, SafeCarryCandidate):
            raise EmptyWristCalibrationError(
                "candidate must be SafeCarryCandidate"
            )
        if candidate.hand != self._starter.arm:
            raise EmptyWristCalibrationError(
                "candidate hand does not match initialized Starter arm"
            )

    def _require_operational(self) -> None:
        if self._poisoned:
            raise EmptyWristCalibrationError(
                "empty-wrist calibration actuator is poisoned"
            )

    def _fail(
        self,
        message: str,
        cause: Exception | None = None,
    ) -> None:
        self._poisoned = True
        self._active_token = None
        try:
            self._control.hold()
        except Exception:
            pass
        if cause is None:
            raise EmptyWristCalibrationError(message)
        raise EmptyWristCalibrationError(message) from cause


def run_empty_wrist_negative_calibration(
    *,
    candidate: SafeCarryCandidate,
    config: EmptyWristCalibrationConfig,
    actuator: R1ProEmptyWristCalibrationActuator,
) -> EmptyWristCalibrationResult:
    """Run bounded empty-close trials and construct a bootstrap record."""

    candidate_sha256 = safe_carry_candidate_sha256(candidate)
    trials: list[EmptyWristTrialRecord] = []
    closed_snapshots: list[list[Mapping[str, Any]]] = []
    ik_pass = False
    failure: str | None = None
    bootstrap: BootstrapPublicWristCalibrationRecord | None = None
    try:
        if candidate.hand != config.hand:
            raise EmptyWristCalibrationError(
                "config hand and candidate hand differ"
            )
        if candidate_sha256 != config.safe_carry_candidate_sha256:
            raise EmptyWristCalibrationError(
                "config candidate hash does not match live candidate"
            )
        ik_pass = actuator.ik_only_valid(candidate)
        if not ik_pass:
            raise EmptyWristCalibrationError(
                "official collision-aware candidate IK-only failed"
            )
        for trial_index in range(config.trials):
            move_steps = _drain_phase(
                actuator,
                candidate=candidate,
                phase=EmptyWristPhase.MOVE,
                max_steps=config.max_steps_per_phase,
            )
            close_steps = _drain_phase(
                actuator,
                candidate=candidate,
                phase=EmptyWristPhase.CLOSE,
                max_steps=config.max_steps_per_phase,
            )
            samples = [
                actuator.hold_sample(
                    candidate=candidate,
                    label=(
                        f"empty_wrist.trial_{trial_index}.stable_{index}"
                    ),
                ).public_snapshot
                for index in range(config.stable_samples_per_trial)
            ]
            _validate_empty_qpos(samples, config)
            open_steps = _drain_phase(
                actuator,
                candidate=candidate,
                phase=EmptyWristPhase.OPEN,
                max_steps=config.max_steps_per_phase,
            )
            closed_snapshots.append(samples)
            trials.append(
                EmptyWristTrialRecord(
                    trial_index=trial_index,
                    move_steps=move_steps,
                    close_steps=close_steps,
                    stable_samples=len(samples),
                    open_steps=open_steps,
                    public_false_positive=False,
                    closed_sample_sha256=tuple(
                        _snapshot_sha256(snapshot)
                        for snapshot in samples
                    ),
                )
            )

        baseline_rgb, baseline_depth = _aggregate_empty_baseline(
            closed_snapshots,
            config,
        )
        provisional = BootstrapPublicWristCalibrationRecord(
            hand=config.hand,
            roi_xyxy=config.roi_xyxy,
            empty_rgb=baseline_rgb,
            empty_depth_m=baseline_depth,
            thresholds=config.thresholds,
            safe_carry_candidate_sha256=candidate_sha256,
            candidate_ik_only_passed=True,
            empty_close_trials=config.trials,
            empty_close_false_positives=0,
            baseline_context=(
                "ik_passed_safe_carry_candidate_exact_pose_empty_close"
            ),
        )
        false_positive_trials = _replay_negative_trials(
            provisional,
            closed_snapshots,
        )
        if false_positive_trials:
            trials = [
                replace_trial_false_positive(
                    trial,
                    trial.trial_index in false_positive_trials,
                )
                for trial in trials
            ]
            raise EmptyWristCalibrationError(
                "empty-close public proxy produced a false positive"
            )
        bootstrap = provisional
        actuator.safe_hold()
    except Exception as exc:
        failure = str(exc)
        actuator.abort(failure)

    commands_issued, steps_confirmed = actuator.step_accounting
    return EmptyWristCalibrationResult(
        schema_version=R1PRO_EMPTY_WRIST_CALIBRATION_RESULT_VERSION,
        passed=bootstrap is not None and failure is None,
        candidate_sha256=candidate_sha256,
        ik_only_passed=ik_pass,
        trial_records=tuple(trials),
        bootstrap_record=bootstrap if failure is None else None,
        failure_reason=failure,
        config_manifest=config.to_manifest(),
        actuator_poisoned=actuator.poisoned,
        commands_issued=commands_issued,
        steps_confirmed=steps_confirmed,
    )


def replace_trial_false_positive(
    trial: EmptyWristTrialRecord,
    value: bool,
) -> EmptyWristTrialRecord:
    return EmptyWristTrialRecord(
        trial_index=trial.trial_index,
        move_steps=trial.move_steps,
        close_steps=trial.close_steps,
        stable_samples=trial.stable_samples,
        open_steps=trial.open_steps,
        public_false_positive=value,
        closed_sample_sha256=trial.closed_sample_sha256,
    )


def _drain_phase(
    actuator: R1ProEmptyWristCalibrationActuator,
    *,
    candidate: SafeCarryCandidate,
    phase: EmptyWristPhase,
    max_steps: int,
) -> int:
    token = actuator.begin_phase(phase=phase, candidate=candidate)
    for count in range(1, max_steps + 1):
        sample = actuator.tick_phase(token)
        if sample.completed:
            return count
    actuator.abort(f"{phase.value} exceeded max_steps_per_phase")
    raise EmptyWristCalibrationError(
        f"{phase.value} exceeded max_steps_per_phase"
    )


def _validate_empty_qpos(
    snapshots: Sequence[Mapping[str, Any]],
    config: EmptyWristCalibrationConfig,
) -> None:
    qpos_slice = (
        LEFT_GRIPPER_QPOS
        if config.hand == "left"
        else RIGHT_GRIPPER_QPOS
    )
    expected = np.asarray(
        config.thresholds.empty_close_qpos,
        dtype=np.float64,
    )
    for snapshot in snapshots:
        proprio = np.asarray(snapshot["proprio"], dtype=np.float64)
        if proprio.shape != (PROPRIO_DIM,) or not np.isfinite(proprio).all():
            raise EmptyWristCalibrationError(
                "stable sample lacks finite public R1Pro proprio"
            )
        if float(np.max(np.abs(proprio[qpos_slice] - expected))) > (
            config.empty_qpos_tolerance
        ):
            raise EmptyWristCalibrationError(
                "observed empty-close qpos disagrees with explicit threshold"
            )


def _aggregate_empty_baseline(
    trials: Sequence[Sequence[Mapping[str, Any]]],
    config: EmptyWristCalibrationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = config.roi_xyxy
    camera = f"{config.hand}_wrist"
    rgb_stack = np.stack(
        [
            np.asarray(snapshot["cameras"][camera]["rgb"])[
                y0:y1,
                x0:x1,
            ]
            for trial in trials
            for snapshot in trial
        ]
    )
    depth_stack = np.stack(
        [
            np.asarray(
                _depth_2d(
                    snapshot["cameras"][camera]["depth_linear"]
                ),
                dtype=np.float32,
            )[y0:y1, x0:x1]
            for trial in trials
            for snapshot in trial
        ]
    )
    baseline_rgb = np.median(rgb_stack, axis=0).astype(np.uint8)
    with np.errstate(all="ignore"):
        baseline_depth = np.nanmedian(depth_stack, axis=0).astype(
            np.float32
        )
    if (
        baseline_depth.shape != (y1 - y0, x1 - x0)
        or np.isinf(baseline_depth).any()
        or not np.any(np.isfinite(baseline_depth) & (baseline_depth > 0.0))
    ):
        raise EmptyWristCalibrationError(
            "empty-close depth baseline is invalid"
        )
    return baseline_rgb, baseline_depth


def _replay_negative_trials(
    record: BootstrapPublicWristCalibrationRecord,
    trials: Sequence[Sequence[Mapping[str, Any]]],
) -> set[int]:
    false_positives: set[int] = set()
    current: dict[str, Mapping[str, Any]] = {}
    provider = BootstrapPublicGripperWristEvidenceProvider(
        snapshot_provider=lambda: current["snapshot"],
        calibrations={record.hand: record},
    )
    for trial_index, samples in enumerate(trials):
        provider.begin_grasp(record.hand)
        for snapshot in samples:
            current["snapshot"] = snapshot
            if provider(record.hand).wrist_visual_consistent:
                false_positives.add(trial_index)
    return false_positives


def _freeze_public_snapshot(
    snapshot: Mapping[str, Any],
) -> Mapping[str, Any]:
    cameras = {}
    for role, payload in snapshot["cameras"].items():
        cameras[role] = MappingProxyType(
            {
                modality: _read_only_array(value)
                for modality, value in payload.items()
            }
        )
    return MappingProxyType(
        {
            "schema_version": snapshot["schema_version"],
            "task_id": snapshot["task_id"],
            "proprio": _read_only_array(snapshot["proprio"]),
            "cam_rel_poses": _read_only_array(
                snapshot["cam_rel_poses"]
            ),
            "cameras": MappingProxyType(cameras),
        }
    )


def _snapshot_sha256(snapshot: Mapping[str, Any]) -> str:
    payload = hashlib.sha256()
    payload.update(
        np.ascontiguousarray(snapshot["proprio"]).tobytes()
    )
    for role in ("left_wrist", "right_wrist", "head"):
        payload.update(
            np.ascontiguousarray(
                snapshot["cameras"][role]["rgb"]
            ).tobytes()
        )
        payload.update(
            np.ascontiguousarray(
                snapshot["cameras"][role]["depth_linear"]
            ).tobytes()
        )
    return payload.hexdigest()


def _validate_starter(starter: Any) -> None:
    missing = [
        name
        for name in (
            "_ik_solver_cartesian_to_joint_space",
            "_move_hand",
            "_execute_grasp",
            "_execute_release",
        )
        if not callable(getattr(starter, name, None))
    ]
    robot = getattr(starter, "robot", None)
    if robot is None or not callable(getattr(robot, "get_eef_pose", None)):
        missing.append("robot.get_eef_pose")
    if getattr(starter, "arm", None) not in {"left", "right"}:
        missing.append("starter.arm")
    if missing:
        raise EmptyWristCalibrationError(
            f"Starter calibration capabilities missing: {missing}"
        )


def _tensor_like(reference: Any, values: Sequence[float]) -> Any:
    if callable(getattr(reference, "new_tensor", None)):
        return reference.new_tensor(values)
    return np.asarray(values, dtype=np.float32)


def _quat_angle_deg(first: Any, second: Any) -> float:
    lhs = np.asarray(first, dtype=np.float64)
    rhs = np.asarray(second, dtype=np.float64)
    if (
        lhs.shape != (4,)
        or rhs.shape != (4,)
        or not np.isfinite(lhs).all()
        or not np.isfinite(rhs).all()
    ):
        raise EmptyWristCalibrationError(
            "EEF orientation must be finite quaternion"
        )
    lhs_norm = float(np.linalg.norm(lhs))
    rhs_norm = float(np.linalg.norm(rhs))
    if lhs_norm <= 1e-8 or rhs_norm <= 1e-8:
        raise EmptyWristCalibrationError("EEF quaternion has zero norm")
    cosine = min(
        1.0,
        max(-1.0, abs(float(np.dot(lhs, rhs) / lhs_norm / rhs_norm))),
    )
    return math.degrees(2.0 * math.acos(cosine))


def _read_only_array(value: Any) -> np.ndarray:
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


def _depth_2d(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise EmptyWristCalibrationError(
            "public wrist depth must be two-dimensional"
        )
    return array


def _positive(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be positive")
    return float(value)
