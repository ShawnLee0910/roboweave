"""Opt-in camera and arm-presentation profiles for Action-23 evidence."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping


GRIPPER_VISUAL_PROFILE_DISABLED = "disabled"
GRIPPER_VISUAL_PROFILE_HEAD_DOMINANT = (
    "head_dominant_14p5_12_down10_v1"
)
GRIPPER_VISUAL_PROFILE_INWARD_12CM = (
    "inward_12cm_head_dominant_v1"
)
GRIPPER_VISUAL_PROFILE_INWARD_12CM_OFFICIAL_CAMERA = (
    "inward_12cm_head_dominant_official_camera_v1"
)
GRIPPER_VISUAL_PROFILE_LADDER_STOW = (
    "inward_12cm_ladder_stow_v1"
)


_PROFILES: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        GRIPPER_VISUAL_PROFILE_DISABLED: MappingProxyType(
            {
                "camera_tuning": False,
                "head_dominant_layout": False,
                "inward_arm_presentation": False,
                "arm_motion_stow_ladder": False,
            }
        ),
        GRIPPER_VISUAL_PROFILE_HEAD_DOMINANT: MappingProxyType(
            {
                "camera_tuning": True,
                "head_dominant_layout": True,
                "inward_arm_presentation": False,
                "arm_motion_stow_ladder": False,
            }
        ),
        GRIPPER_VISUAL_PROFILE_INWARD_12CM: MappingProxyType(
            {
                "camera_tuning": True,
                "head_dominant_layout": True,
                "inward_arm_presentation": True,
                "arm_motion_stow_ladder": False,
            }
        ),
        GRIPPER_VISUAL_PROFILE_INWARD_12CM_OFFICIAL_CAMERA: MappingProxyType(
            {
                "camera_tuning": False,
                "head_dominant_layout": True,
                "inward_arm_presentation": True,
                "arm_motion_stow_ladder": False,
            }
        ),
        # The ladder profile keeps the inward presentation for rendered
        # action frames but stows both arms to the chassis envelope for
        # every public base motion; the highest collision-free rung is
        # restored before each action_after frame is captured.
        GRIPPER_VISUAL_PROFILE_LADDER_STOW: MappingProxyType(
            {
                "camera_tuning": True,
                "head_dominant_layout": True,
                "inward_arm_presentation": True,
                "arm_motion_stow_ladder": True,
            }
        ),
    }
)


def gripper_visual_profile_contract(name: str) -> Mapping[str, Any]:
    if not isinstance(name, str) or name not in _PROFILES:
        raise ValueError(f"unknown gripper visual profile: {name!r}")
    return MappingProxyType({"name": name, **dict(_PROFILES[name])})


__all__ = [
    "GRIPPER_VISUAL_PROFILE_DISABLED",
    "GRIPPER_VISUAL_PROFILE_HEAD_DOMINANT",
    "GRIPPER_VISUAL_PROFILE_INWARD_12CM",
    "GRIPPER_VISUAL_PROFILE_INWARD_12CM_OFFICIAL_CAMERA",
    "GRIPPER_VISUAL_PROFILE_LADDER_STOW",
    "gripper_visual_profile_contract",
]
