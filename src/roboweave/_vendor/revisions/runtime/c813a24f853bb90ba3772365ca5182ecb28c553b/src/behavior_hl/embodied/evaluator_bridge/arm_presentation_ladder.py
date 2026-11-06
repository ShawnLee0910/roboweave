"""Graded arm-presentation ladder for base-motion safety.

Both arms always stow toward the chassis envelope while the base moves;
after the base settles, the highest presentation rung whose full-body
configuration is collision-free at the new pose is restored.  A rendered
action frame is therefore never captured with a wrist camera inside scene
geometry, and base motion is never vetoed solely by the presented arm span.

The ladder is pure joint-space policy: rung joint targets are linear blends
between the load-time presented arm pose and the official stowed arm pose,
so no per-step inverse kinematics or motion planning is required.  Whether
one rung is admissible at the current base pose is decided exclusively by
the injected collision query, which must already include any held object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ARM_PRESENTATION_LADDER_VERSION = "behavior2026.arm-presentation-ladder/v1"
ARM_JOINT_COUNT_PER_ARM = 7
# Blend 1.0 is the full load-time presentation; blend 0.0 is the official
# stowed arm pose.  Intermediate rungs keep some gripper visibility in the
# head frame when the full presentation does not fit.
LADDER_BLEND_FRACTIONS = (1.0, 0.66, 0.33, 0.0)
# One upgrade (toward fuller presentation) requires this many consecutive
# admissible evaluations of the same target rung, so a boundary pose cannot
# flap the arms between rungs on alternating frames.  Downgrades apply
# immediately: safety never waits.
UPGRADE_STABLE_EVALUATIONS = 2


class ArmPresentationLadderError(RuntimeError):
    """Raised when ladder construction or selection input is invalid."""


def _arm_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (ARM_JOINT_COUNT_PER_ARM,) or not np.isfinite(array).all():
        raise ArmPresentationLadderError(
            f"{name} must contain {ARM_JOINT_COUNT_PER_ARM} finite joint values"
        )
    return array


@dataclass(frozen=True, slots=True)
class ArmLadderRung:
    """One admissible-candidate arm pose, ordered from most presented."""

    rung_index: int
    blend_fraction: float
    left_arm_qpos: tuple[float, ...]
    right_arm_qpos: tuple[float, ...]

    @property
    def is_stowed(self) -> bool:
        return self.blend_fraction == 0.0

    @property
    def is_fully_presented(self) -> bool:
        return self.blend_fraction == 1.0


def build_arm_presentation_ladder(
    *,
    presented_left_arm_qpos: Any,
    presented_right_arm_qpos: Any,
    stowed_left_arm_qpos: Any,
    stowed_right_arm_qpos: Any,
    blend_fractions: Sequence[float] = LADDER_BLEND_FRACTIONS,
) -> tuple[ArmLadderRung, ...]:
    """Blend the presented and stowed arm poses into an ordered ladder."""

    presented_left = _arm_vector(
        presented_left_arm_qpos, "presented_left_arm_qpos"
    )
    presented_right = _arm_vector(
        presented_right_arm_qpos, "presented_right_arm_qpos"
    )
    stowed_left = _arm_vector(stowed_left_arm_qpos, "stowed_left_arm_qpos")
    stowed_right = _arm_vector(stowed_right_arm_qpos, "stowed_right_arm_qpos")

    fractions = tuple(float(value) for value in blend_fractions)
    if len(fractions) < 2:
        raise ArmPresentationLadderError(
            "ladder requires at least a presented and a stowed rung"
        )
    if any(not 0.0 <= value <= 1.0 for value in fractions):
        raise ArmPresentationLadderError(
            "ladder blend fractions must lie within [0.0, 1.0]"
        )
    if list(fractions) != sorted(fractions, reverse=True):
        raise ArmPresentationLadderError(
            "ladder blend fractions must strictly descend from presented"
        )
    if len(set(fractions)) != len(fractions):
        raise ArmPresentationLadderError(
            "ladder blend fractions must be unique"
        )
    if fractions[0] != 1.0 or fractions[-1] != 0.0:
        raise ArmPresentationLadderError(
            "ladder must start at the presented pose and end at the stowed pose"
        )

    rungs: list[ArmLadderRung] = []
    for index, fraction in enumerate(fractions):
        left = stowed_left + fraction * (presented_left - stowed_left)
        right = stowed_right + fraction * (presented_right - stowed_right)
        rungs.append(
            ArmLadderRung(
                rung_index=index,
                blend_fraction=fraction,
                left_arm_qpos=tuple(float(value) for value in left),
                right_arm_qpos=tuple(float(value) for value in right),
            )
        )
    return tuple(rungs)


@dataclass(frozen=True, slots=True)
class ArmLadderSelection:
    """One post-action ladder decision with full audit provenance."""

    selected_rung_index: int
    applied: bool
    no_collision_free_rung: bool
    upgrade_deferred: bool
    upgrade_streak: int
    attempts: tuple[Mapping[str, Any], ...]

    def report(self) -> dict[str, Any]:
        return {
            "version": ARM_PRESENTATION_LADDER_VERSION,
            "selected_rung_index": self.selected_rung_index,
            "applied": self.applied,
            "no_collision_free_rung": self.no_collision_free_rung,
            "upgrade_deferred": self.upgrade_deferred,
            "upgrade_streak": self.upgrade_streak,
            "attempts": [dict(item) for item in self.attempts],
        }


class ArmLadderSelector:
    """Own the rung state machine across one strategy's action stream.

    The selector never touches the simulator.  The caller supplies one
    collision query per rung evaluation; the query must evaluate the full
    current configuration (arms at the candidate rung, torso unchanged,
    any held object included) at the robot's current base pose.
    """

    def __init__(
        self,
        rungs: Sequence[ArmLadderRung],
        *,
        upgrade_stable_evaluations: int = UPGRADE_STABLE_EVALUATIONS,
    ) -> None:
        rung_list = tuple(rungs)
        if not rung_list:
            raise ArmPresentationLadderError("ladder selector requires rungs")
        if [item.rung_index for item in rung_list] != list(
            range(len(rung_list))
        ):
            raise ArmPresentationLadderError(
                "ladder rungs must be contiguously indexed from zero"
            )
        if int(upgrade_stable_evaluations) < 1:
            raise ArmPresentationLadderError(
                "upgrade_stable_evaluations must be at least 1"
            )
        self._rungs = rung_list
        self._upgrade_stable_evaluations = int(upgrade_stable_evaluations)
        # The load-time presentation applied the fully presented pose, so
        # the stream starts at rung zero.
        self._current_rung_index = 0
        self._upgrade_candidate_index: int | None = None
        self._upgrade_streak = 0

    @property
    def rungs(self) -> tuple[ArmLadderRung, ...]:
        return self._rungs

    @property
    def current_rung(self) -> ArmLadderRung:
        return self._rungs[self._current_rung_index]

    @property
    def stowed_rung(self) -> ArmLadderRung:
        return self._rungs[-1]

    def select_after_action(
        self,
        rung_is_collision_free: Callable[[ArmLadderRung], bool],
    ) -> ArmLadderSelection:
        """Pick the highest admissible rung, with flap-resistant upgrades."""

        if not callable(rung_is_collision_free):
            raise ArmPresentationLadderError(
                "rung_is_collision_free must be callable"
            )
        attempts: list[dict[str, Any]] = []
        admissible_index: int | None = None
        for rung in self._rungs:
            verdict = bool(rung_is_collision_free(rung))
            attempts.append(
                {
                    "rung_index": rung.rung_index,
                    "blend_fraction": rung.blend_fraction,
                    "collision_free": verdict,
                }
            )
            if verdict:
                admissible_index = rung.rung_index
                break

        if admissible_index is None:
            # Nothing is admissible, including the stowed pose.  The arms
            # stay stowed (they were stowed for the base motion itself) and
            # the caller records the flag for audit.
            self._current_rung_index = len(self._rungs) - 1
            self._upgrade_candidate_index = None
            self._upgrade_streak = 0
            return ArmLadderSelection(
                selected_rung_index=self._current_rung_index,
                applied=True,
                no_collision_free_rung=True,
                upgrade_deferred=False,
                upgrade_streak=0,
                attempts=tuple(attempts),
            )

        if admissible_index >= self._current_rung_index:
            # Same rung or a downgrade: apply immediately; safety and
            # stability never wait on a streak.
            self._current_rung_index = admissible_index
            self._upgrade_candidate_index = None
            self._upgrade_streak = 0
            return ArmLadderSelection(
                selected_rung_index=admissible_index,
                applied=True,
                no_collision_free_rung=False,
                upgrade_deferred=False,
                upgrade_streak=0,
                attempts=tuple(attempts),
            )

        # Upgrade toward fuller presentation: require consecutive stable
        # admissible evaluations of the same target rung.
        if self._upgrade_candidate_index == admissible_index:
            self._upgrade_streak += 1
        else:
            self._upgrade_candidate_index = admissible_index
            self._upgrade_streak = 1
        if self._upgrade_streak >= self._upgrade_stable_evaluations:
            self._current_rung_index = admissible_index
            self._upgrade_candidate_index = None
            self._upgrade_streak = 0
            return ArmLadderSelection(
                selected_rung_index=admissible_index,
                applied=True,
                no_collision_free_rung=False,
                upgrade_deferred=False,
                upgrade_streak=0,
                attempts=tuple(attempts),
            )
        return ArmLadderSelection(
            selected_rung_index=self._current_rung_index,
            applied=False,
            no_collision_free_rung=False,
            upgrade_deferred=True,
            upgrade_streak=self._upgrade_streak,
            attempts=tuple(attempts),
        )


__all__ = [
    "ARM_JOINT_COUNT_PER_ARM",
    "ARM_PRESENTATION_LADDER_VERSION",
    "ArmLadderRung",
    "ArmLadderSelection",
    "ArmLadderSelector",
    "ArmPresentationLadderError",
    "LADDER_BLEND_FRACTIONS",
    "UPGRADE_STABLE_EVALUATIONS",
    "build_arm_presentation_ladder",
]
