"""Generate named action slices from the frozen controller order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


R1PRO_CONTROLLER_ORDER = (
    "base",
    "torso",
    "left_arm",
    "left_gripper",
    "right_arm",
    "right_gripper",
)
R1PRO_CONTROLLER_DIMS = {
    "base": 3,
    "torso": 4,
    "left_arm": 7,
    "left_gripper": 1,
    "right_arm": 7,
    "right_gripper": 1,
}


@dataclass(frozen=True, slots=True)
class ActionLayout:
    slices: Mapping[str, slice]
    dimension: int

    def slice_for(self, controller: str) -> slice:
        try:
            return self.slices[controller]
        except KeyError as exc:
            raise KeyError(f"unknown controller: {controller!r}") from exc


def build_action_layout(
    controller_dims: Mapping[str, int],
    *,
    controller_order: tuple[str, ...] = R1PRO_CONTROLLER_ORDER,
) -> ActionLayout:
    if set(controller_dims) != set(controller_order):
        missing = set(controller_order) - set(controller_dims)
        unknown = set(controller_dims) - set(controller_order)
        raise ValueError(
            f"controller dimensions mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    start = 0
    slices = {}
    for controller in controller_order:
        dimension = controller_dims[controller]
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise ValueError(f"{controller!r} dimension must be an integer")
        if dimension <= 0:
            raise ValueError(f"{controller!r} dimension must be positive")
        stop = start + dimension
        slices[controller] = slice(start, stop)
        start = stop
    return ActionLayout(slices=slices, dimension=start)


R1PRO_ACTION_LAYOUT = build_action_layout(R1PRO_CONTROLLER_DIMS)
