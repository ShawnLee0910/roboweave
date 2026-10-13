"""Pure integration helpers for the graded arm-presentation ladder.

This module owns everything about the ladder integration that does not need
a live simulator: endpoint resolution from the moderate-inward presentation
report, load-time failure grading for the ladder profile, the rung walk that
physically tries candidates through an injected joint writer, and the private
jsonl record shape.  It deliberately never imports OmniGibson or torch so the
rollout hooks remain unit-testable on CPU.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

import numpy as np

from behavior_hl.embodied.evaluator_bridge.arm_presentation_ladder import (
    ARM_JOINT_COUNT_PER_ARM,
    ArmLadderRung,
    ArmLadderSelection,
    ArmLadderSelector,
)
from behavior_hl.embodied.evaluator_bridge.moderate_inward_gripper_presentation import (
    MODERATE_INWARD_GRIPPER_PRESENTATION_VERSION,
)
from behavior_hl.embodied.evaluator_bridge.predeclared_camera_config import (
    R1PRO_OFFICIAL_UNTUCKED_LEFT_ARM,
    R1PRO_OFFICIAL_UNTUCKED_RIGHT_ARM,
)


ARM_PRESENTATION_LADDER_RUNTIME_VERSION = (
    "behavior2026.arm-presentation-ladder-runtime/v1"
)
# Private evaluator attribute threading the ladder endpoints and the profile
# flag from ``run_b1_single_load`` into the rollout phases, mirroring the
# existing ``_behavior_hl_camera_layout_profile`` channel.
ARM_PRESENTATION_LADDER_CONFIG_ATTRIBUTE = (
    "_behavior_hl_arm_presentation_ladder_config"
)
ARM_PRESENTATION_LADDER_JSONL_NAME = "arm_presentation_ladder.jsonl"
DEGRADED_TO_UNTUCKED_STATUS = "DEGRADED_TO_UNTUCKED"
PRESENTED_SOURCE_PRESENTATION_REPORT = "presentation_report_hands_after"
PRESENTED_SOURCE_UNTUCKED_FALLBACK = "official_untucked_constants_fallback"


class ArmPresentationLadderRuntimeError(RuntimeError):
    """Raised when ladder integration inputs are malformed."""


def _arm_joint_tuple(value: Any, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (ARM_JOINT_COUNT_PER_ARM,) or not np.isfinite(
        array
    ).all():
        raise ArmPresentationLadderRuntimeError(
            f"{name} must contain {ARM_JOINT_COUNT_PER_ARM} finite joints"
        )
    return tuple(float(item) for item in array)


def _presented_arms_from_pass_report(
    presentation_report: Mapping[str, Any],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    hands = presentation_report.get("hands")
    if not isinstance(hands, (list, tuple)) or len(hands) != 2:
        raise ArmPresentationLadderRuntimeError(
            "PASS presentation report must contain exactly two hands"
        )
    presented: dict[str, tuple[float, ...]] = {}
    for index, expected_hand in ((0, "left"), (1, "right")):
        entry = hands[index]
        if not isinstance(entry, Mapping):
            raise ArmPresentationLadderRuntimeError(
                "presentation report hand entry must be a mapping"
            )
        hand_name = entry.get("hand", expected_hand)
        if hand_name != expected_hand:
            raise ArmPresentationLadderRuntimeError(
                "presentation report hands must be ordered left then right"
            )
        after = entry.get("after")
        if not isinstance(after, Mapping):
            raise ArmPresentationLadderRuntimeError(
                "presentation report hand lacks an after state"
            )
        presented[expected_hand] = _arm_joint_tuple(
            after.get("arm_qpos_rad"),
            f"presented {expected_hand} arm_qpos_rad",
        )
    return presented["left"], presented["right"]


def resolve_arm_presentation_ladder_endpoints(
    *,
    presentation_report: Mapping[str, Any] | None,
    stowed_left_arm_qpos: Any,
    stowed_right_arm_qpos: Any,
) -> dict[str, Any]:
    """Resolve the ladder's presented and stowed arm endpoints.

    A successful presentation supplies both presented arms through
    ``hands[*]["after"]["arm_qpos_rad"]``.  A deferred, degraded, disabled,
    or missing presentation falls back to the official untucked constants so
    the ladder always has a well-defined most-presented rung.
    """

    stowed_left = _arm_joint_tuple(
        stowed_left_arm_qpos, "stowed_left_arm_qpos"
    )
    stowed_right = _arm_joint_tuple(
        stowed_right_arm_qpos, "stowed_right_arm_qpos"
    )
    status = (
        presentation_report.get("status")
        if isinstance(presentation_report, Mapping)
        else None
    )
    if status == "PASS":
        presented_left, presented_right = _presented_arms_from_pass_report(
            presentation_report
        )
        presented_source = PRESENTED_SOURCE_PRESENTATION_REPORT
    else:
        presented_left = _arm_joint_tuple(
            R1PRO_OFFICIAL_UNTUCKED_LEFT_ARM,
            "official untucked left arm",
        )
        presented_right = _arm_joint_tuple(
            R1PRO_OFFICIAL_UNTUCKED_RIGHT_ARM,
            "official untucked right arm",
        )
        presented_source = PRESENTED_SOURCE_UNTUCKED_FALLBACK
    return {
        "presented_left_arm_qpos": presented_left,
        "presented_right_arm_qpos": presented_right,
        "stowed_left_arm_qpos": stowed_left,
        "stowed_right_arm_qpos": stowed_right,
        "presented_source": presented_source,
        "presentation_status": status,
    }


def build_arm_presentation_ladder_config(
    *,
    profile_name: str,
    presentation_report: Mapping[str, Any] | None,
    stowed_left_arm_qpos: Any,
    stowed_right_arm_qpos: Any,
) -> dict[str, Any]:
    """Assemble the private evaluator-attribute ladder configuration."""

    if not isinstance(profile_name, str) or not profile_name:
        raise ArmPresentationLadderRuntimeError(
            "ladder config requires the active profile name"
        )
    endpoints = resolve_arm_presentation_ladder_endpoints(
        presentation_report=presentation_report,
        stowed_left_arm_qpos=stowed_left_arm_qpos,
        stowed_right_arm_qpos=stowed_right_arm_qpos,
    )
    return {
        "version": ARM_PRESENTATION_LADDER_RUNTIME_VERSION,
        "active": True,
        "profile": profile_name,
        **endpoints,
    }


def refresh_arm_presentation_ladder_presented_endpoints(
    config: Mapping[str, Any],
    *,
    presentation_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebind the presented endpoints after a later presentation applies.

    The deferred model-phase presentation runs long after load time; its PASS
    report carries the arms that are actually presented for the model lane.
    The stowed endpoints and the profile identity never change here.
    """

    if not isinstance(config, Mapping) or config.get("active") is not True:
        raise ArmPresentationLadderRuntimeError(
            "only an active ladder config can be refreshed"
        )
    return build_arm_presentation_ladder_config(
        profile_name=str(config["profile"]),
        presentation_report=presentation_report,
        stowed_left_arm_qpos=config["stowed_left_arm_qpos"],
        stowed_right_arm_qpos=config["stowed_right_arm_qpos"],
    )


def active_arm_presentation_ladder_config(
    evaluator: Any,
) -> Mapping[str, Any] | None:
    """Read the private ladder config attribute; None when inactive."""

    config = getattr(
        evaluator, ARM_PRESENTATION_LADDER_CONFIG_ATTRIBUTE, None
    )
    if not isinstance(config, Mapping) or config.get("active") is not True:
        return None
    return config


def apply_load_time_inward_presentation_with_ladder_grading(
    *,
    apply_presentation: Callable[[], Mapping[str, Any]],
    arm_motion_stow_ladder: bool,
) -> Mapping[str, Any]:
    """Run the load-time inward presentation with ladder-aware grading.

    Without the ladder flag every presentation failure propagates exactly as
    today.  With the flag a failure is graded to ``DEGRADED_TO_UNTUCKED``:
    the run proceeds and the ladder later presents from the official
    untucked constants instead.
    """

    if not callable(apply_presentation):
        raise ArmPresentationLadderRuntimeError(
            "apply_presentation must be callable"
        )
    if not isinstance(arm_motion_stow_ladder, bool):
        raise ArmPresentationLadderRuntimeError(
            "arm_motion_stow_ladder must be bool"
        )
    if not arm_motion_stow_ladder:
        return apply_presentation()
    try:
        return apply_presentation()
    except Exception as exc:
        return {
            "version": MODERATE_INWARD_GRIPPER_PRESENTATION_VERSION,
            "status": DEGRADED_TO_UNTUCKED_STATUS,
            "presentation_only": True,
            "direct_kinematic_joint_application": False,
            "arm_motion_stow_ladder": True,
            "presented_endpoint_fallback": (
                PRESENTED_SOURCE_UNTUCKED_FALLBACK
            ),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def select_and_apply_ladder_rung(
    *,
    selector: ArmLadderSelector,
    write_rung_joints: Callable[[ArmLadderRung], None],
    rung_collision_free: Callable[[ArmLadderRung], bool],
) -> tuple[ArmLadderSelection, float]:
    """Walk rungs by physically writing each candidate, then settle.

    ``write_rung_joints`` must kinematically write both arm joint groups and
    re-anchor any symbolic carry before ``rung_collision_free`` evaluates the
    robot's current configuration at its current base pose.  The ``finally``
    block guarantees the joints ultimately written equal the rung the
    selector owns, even when an upgrade is deferred (the walk probed a higher
    rung than the one retained) or a collision query raises.

    Returns the selection and the wall-clock milliseconds spent inside the
    walk and the settling write.
    """

    if not callable(write_rung_joints) or not callable(rung_collision_free):
        raise ArmPresentationLadderRuntimeError(
            "ladder rung walk requires writer and collision callables"
        )
    last_written_rung_index: int | None = None
    started = time.perf_counter()

    def probe(rung: ArmLadderRung) -> bool:
        nonlocal last_written_rung_index
        write_rung_joints(rung)
        last_written_rung_index = rung.rung_index
        return bool(rung_collision_free(rung))

    try:
        selection = selector.select_after_action(probe)
    finally:
        final_rung = selector.current_rung
        if last_written_rung_index != final_rung.rung_index:
            write_rung_joints(final_rung)
    ladder_query_ms = (time.perf_counter() - started) * 1000.0
    if selection.selected_rung_index != selector.current_rung.rung_index:
        raise ArmPresentationLadderRuntimeError(
            "ladder selection and selector rung state diverged"
        )
    return selection, float(ladder_query_ms)


def build_arm_presentation_ladder_step_record(
    *,
    lane: str,
    action_index: int,
    action_name: str,
    stowed_before_motion: bool,
    selection: ArmLadderSelection,
    ladder_query_ms: float,
) -> dict[str, Any]:
    """Shape one private jsonl line for a stow/restore cycle."""

    if isinstance(action_index, bool) or not isinstance(action_index, int):
        raise ArmPresentationLadderRuntimeError(
            "ladder record action_index must be an integer"
        )
    if not isinstance(action_name, str) or not action_name:
        raise ArmPresentationLadderRuntimeError(
            "ladder record action name must be a non-empty string"
        )
    if not isinstance(stowed_before_motion, bool):
        raise ArmPresentationLadderRuntimeError(
            "ladder record stowed_before_motion must be bool"
        )
    if not isinstance(selection, ArmLadderSelection):
        raise ArmPresentationLadderRuntimeError(
            "ladder record requires one ArmLadderSelection"
        )
    return {
        "schema_version": ARM_PRESENTATION_LADDER_RUNTIME_VERSION,
        "lane": str(lane),
        "action_index": int(action_index),
        "action": action_name,
        "stowed_before_motion": stowed_before_motion,
        "selection": selection.report(),
        "ladder_query_ms": float(ladder_query_ms),
    }


__all__ = [
    "ARM_PRESENTATION_LADDER_CONFIG_ATTRIBUTE",
    "ARM_PRESENTATION_LADDER_JSONL_NAME",
    "ARM_PRESENTATION_LADDER_RUNTIME_VERSION",
    "ArmPresentationLadderRuntimeError",
    "DEGRADED_TO_UNTUCKED_STATUS",
    "PRESENTED_SOURCE_PRESENTATION_REPORT",
    "PRESENTED_SOURCE_UNTUCKED_FALLBACK",
    "active_arm_presentation_ladder_config",
    "apply_load_time_inward_presentation_with_ladder_grading",
    "build_arm_presentation_ladder_config",
    "build_arm_presentation_ladder_step_record",
    "refresh_arm_presentation_ladder_presented_endpoints",
    "resolve_arm_presentation_ladder_endpoints",
    "select_and_apply_ladder_rung",
]
