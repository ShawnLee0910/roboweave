"""Single-scene bootstrap from physical lift to validated safe carry.

The scored grasp plugin intentionally requires an already validated
``safe_carry_pose_provider``.  This module breaks that bootstrap dependency
without ever executing an unvalidated retract:

1. run only the official physical reach / close / lift phases;
2. require real contact-authorized assisted grasp and a measured lift;
3. hand the still-held payload to the bounded safe-carry calibration;
4. admit the resulting pose only when the calibration record passes.

The caller may then reset the *resident* task instance (without reconstructing
the scene) and build the scored runtime from the validated record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from behavior_hl.embodied.evaluator_bridge.r1pro_starter_grasp_driver import (
    R1ProStarterGraspDriver,
)
from behavior_hl.embodied.evaluator_bridge.safe_carry_calibration import (
    SafeCarryCalibrationActuator,
    SafeCarryCalibrationLimits,
    SafeCarryCalibrationRecord,
    SafeCarryCandidate,
    run_safe_carry_calibration,
)
from behavior_hl.embodied.runtime.plugins.grasp import (
    GraspMotionPhase,
    GraspPhaseSample,
)


SAFE_CARRY_BOOTSTRAP_VERSION = "behavior2026.safe-carry-bootstrap/v1"


class SafeCarryBootstrapError(RuntimeError):
    """Raised after fail-closed bootstrap termination."""


@dataclass(frozen=True, slots=True)
class SafeCarryBootstrapLimits:
    """Physical budgets before the calibrated retract begins."""

    max_reach_steps: int
    max_close_steps: int
    max_lift_steps: int
    minimum_payload_lift_m: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "max_reach_steps",
            "max_close_steps",
            "max_lift_steps",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        lift = self.minimum_payload_lift_m
        if (
            isinstance(lift, bool)
            or not isinstance(lift, (int, float))
            or not math.isfinite(float(lift))
            or float(lift) < 0.05
        ):
            raise ValueError(
                "minimum_payload_lift_m must be finite and at least 0.05"
            )
        object.__setattr__(
            self,
            "minimum_payload_lift_m",
            float(lift),
        )


@dataclass(frozen=True, slots=True)
class SafeCarryBootstrapRecord:
    """Private accounting for one unscored resident-scene bootstrap."""

    candidate_sha256: str
    reach_steps: int
    close_steps: int
    lift_steps: int
    contact_observed: bool
    assisted_grasp_enabled: bool
    maximum_payload_lift_m: float
    calibration: SafeCarryCalibrationRecord
    schema_version: str = SAFE_CARRY_BOOTSTRAP_VERSION

    @property
    def passed(self) -> bool:
        return (
            self.contact_observed
            and self.assisted_grasp_enabled
            and self.maximum_payload_lift_m >= 0.05
            and self.calibration.passed
        )

    @property
    def simulator_steps(self) -> int:
        return (
            self.reach_steps
            + self.close_steps
            + self.lift_steps
            + self.calibration.retract_steps
            + sum(item.samples for item in self.calibration.stage_records)
        )


def run_safe_carry_bootstrap(
    *,
    world_point: tuple[float, float, float],
    candidate: SafeCarryCandidate,
    grasp_driver: R1ProStarterGraspDriver,
    calibration_limits: SafeCarryCalibrationLimits,
    calibration_actuator: SafeCarryCalibrationActuator,
    bootstrap_limits: SafeCarryBootstrapLimits,
    private_audit_writer: Callable[[Mapping[str, Any]], None],
) -> SafeCarryBootstrapRecord:
    """Physically lift once, then validate the candidate in the same scene."""

    if not isinstance(candidate, SafeCarryCandidate):
        raise TypeError("candidate must be SafeCarryCandidate")
    if not isinstance(grasp_driver, R1ProStarterGraspDriver):
        raise TypeError("grasp_driver must be R1ProStarterGraspDriver")
    if not isinstance(calibration_limits, SafeCarryCalibrationLimits):
        raise TypeError(
            "calibration_limits must be SafeCarryCalibrationLimits"
        )
    if not isinstance(bootstrap_limits, SafeCarryBootstrapLimits):
        raise TypeError("bootstrap_limits must be SafeCarryBootstrapLimits")
    if not callable(private_audit_writer):
        raise TypeError("private_audit_writer must be callable")

    _audit(
        private_audit_writer,
        "bootstrap_start",
        {
            "candidate_sha256": _candidate_sha256(candidate),
            "hand": candidate.hand,
        },
    )
    plan = grasp_driver.preflight_ik(
        world_point=world_point,
        hand=candidate.hand,
    )
    if plan is None:
        raise SafeCarryBootstrapError(
            "official grasp preflight IK rejected bootstrap plan"
        )

    step_counts = {
        GraspMotionPhase.REACH: 0,
        GraspMotionPhase.CLOSE: 0,
        GraspMotionPhase.LIFT: 0,
    }
    phase_budgets = {
        GraspMotionPhase.REACH: bootstrap_limits.max_reach_steps,
        GraspMotionPhase.CLOSE: bootstrap_limits.max_close_steps,
        GraspMotionPhase.LIFT: bootstrap_limits.max_lift_steps,
    }
    contact_observed = False
    assisted_enabled = False
    maximum_lift = 0.0

    try:
        for phase in (
            GraspMotionPhase.REACH,
            GraspMotionPhase.CLOSE,
            GraspMotionPhase.LIFT,
        ):
            for _ in range(phase_budgets[phase]):
                sample = grasp_driver.tick_motion(
                    phase=phase,
                    plan_token=plan,
                    hand=candidate.hand,
                )
                if not isinstance(sample, GraspPhaseSample):
                    raise SafeCarryBootstrapError(
                        "grasp driver returned an invalid phase sample"
                    )
                step_counts[phase] += sample.simulator_steps
                if phase is GraspMotionPhase.CLOSE and sample.contact_event:
                    contact_observed = True
                    if not assisted_enabled:
                        assisted_enabled = grasp_driver.enable_assisted_grasp(
                            plan_token=plan,
                            hand=candidate.hand,
                        )
                        if assisted_enabled is not True:
                            raise SafeCarryBootstrapError(
                                "contact did not authorize assisted grasp"
                            )
                if phase is GraspMotionPhase.LIFT:
                    maximum_lift = max(
                        maximum_lift,
                        sample.payload_lift_m,
                    )
                _audit(
                    private_audit_writer,
                    "bootstrap_phase_sample",
                    {
                        "phase": phase.value,
                        "completed": sample.completed,
                        "simulator_steps": sample.simulator_steps,
                        "phase_steps": step_counts[phase],
                        "contact_event": sample.contact_event,
                        "maximum_payload_lift_m": maximum_lift,
                    },
                )
                if sample.completed:
                    break
            else:
                raise SafeCarryBootstrapError(
                    f"{phase.value} step budget exhausted"
                )

            if phase is GraspMotionPhase.CLOSE and (
                not contact_observed or not assisted_enabled
            ):
                raise SafeCarryBootstrapError(
                    "physical close completed without authorized grasp"
                )
            if (
                phase is GraspMotionPhase.LIFT
                and maximum_lift
                < bootstrap_limits.minimum_payload_lift_m
            ):
                raise SafeCarryBootstrapError(
                    "payload lift did not reach bootstrap threshold"
                )

        calibration = run_safe_carry_calibration(
            candidate=candidate,
            limits=calibration_limits,
            actuator=calibration_actuator,
            private_audit_writer=private_audit_writer,
        )
        record = SafeCarryBootstrapRecord(
            candidate_sha256=_candidate_sha256(candidate),
            reach_steps=step_counts[GraspMotionPhase.REACH],
            close_steps=step_counts[GraspMotionPhase.CLOSE],
            lift_steps=step_counts[GraspMotionPhase.LIFT],
            contact_observed=contact_observed,
            assisted_grasp_enabled=assisted_enabled,
            maximum_payload_lift_m=maximum_lift,
            calibration=calibration,
        )
        _audit(
            private_audit_writer,
            "bootstrap_final",
            {
                "passed": record.passed,
                "simulator_steps": record.simulator_steps,
                "calibration_sha256": calibration.sha256,
            },
        )
        return record
    except Exception:
        grasp_driver.hold_safe(hand=candidate.hand)
        raise


def _candidate_sha256(candidate: SafeCarryCandidate) -> str:
    # The canonical calibration hash also covers this exact candidate.  This
    # pre-calibration identifier avoids serializing any held-object token.
    import hashlib
    import json

    payload = {
        "hand": candidate.hand,
        "arm_target": candidate.arm_target,
        "eef_world_position": candidate.eef_world_position,
        "eef_world_quaternion_xyzw": (
            candidate.eef_world_quaternion_xyzw
        ),
        "source": candidate.source,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def _audit(
    writer: Callable[[Mapping[str, Any]], None],
    event: str,
    payload: Mapping[str, Any],
) -> None:
    writer(
        {
            "schema_version": SAFE_CARRY_BOOTSTRAP_VERSION,
            "event": event,
            **dict(payload),
        }
    )
