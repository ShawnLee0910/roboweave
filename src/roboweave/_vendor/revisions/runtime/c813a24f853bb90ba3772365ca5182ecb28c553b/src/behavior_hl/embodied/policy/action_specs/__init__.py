"""Built-in action plugins for the first R1Pro task profile."""

from behavior_hl.embodied.policy.action_specs.base_motion import base_motion_specs
from behavior_hl.embodied.policy.action_specs.grasp_at import (
    grasp_at_spec,
    visible_official_grasp_at_spec,
)
from behavior_hl.embodied.policy.action_specs.grasp_object import (
    visible_object_official_grasp_spec,
)
from behavior_hl.embodied.policy.action_specs.marker_symbolic_grasp import (
    marker_symbolic_action23_specs,
    marker_symbolic_grasp_spec,
    marker_symbolic_place_inside_spec,
    mark_visible_objects_spec,
    marker_symbolic_put_back_spec,
    marker_symbolic_place_on_floor_spec,
    marker_symbolic_release_spec,
    marker_symbolic_v34_specs,
)
from behavior_hl.embodied.policy.action_specs.place_inside_at import (
    place_inside_at_spec,
    visible_official_place_inside_at_spec,
)
from behavior_hl.embodied.policy.action_specs.release import release_spec
from behavior_hl.embodied.policy.action_specs.stop import stop_spec
from behavior_hl.embodied.policy.action_specs.torso_look import (
    torso_action23_grid_specs,
    torso_pitch_specs,
    torso_look_specs,
    torso_vertical_specs,
)
from behavior_hl.embodied.policy.action_specs.wait import (
    WAIT_SIMULATOR_STEPS,
    wait_spec,
)

__all__ = [
    "base_motion_specs",
    "grasp_at_spec",
    "visible_official_grasp_at_spec",
    "visible_object_official_grasp_spec",
    "marker_symbolic_grasp_spec",
    "marker_symbolic_action23_specs",
    "marker_symbolic_place_inside_spec",
    "mark_visible_objects_spec",
    "marker_symbolic_put_back_spec",
    "marker_symbolic_place_on_floor_spec",
    "marker_symbolic_release_spec",
    "marker_symbolic_v34_specs",
    "place_inside_at_spec",
    "visible_official_place_inside_at_spec",
    "release_spec",
    "stop_spec",
    "torso_action23_grid_specs",
    "torso_pitch_specs",
    "torso_look_specs",
    "torso_vertical_specs",
    "WAIT_SIMULATOR_STEPS",
    "wait_spec",
]
