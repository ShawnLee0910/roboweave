"""Explicit visual marking followed by instance-bound symbolic grasp."""

from behavior_hl.embodied.policy.protocol import (
    parse_marker_grasp_args,
    parse_marker_place_inside_args,
    parse_no_args,
    parse_marker_target_or_request_args,
    parse_marker_target_args,
)
from behavior_hl.embodied.policy.registry import ActionSpec


# Availability is enforced symmetrically by GraspProxyAvailability, so both
# halves of the carry precondition are stated in the model-visible text.
_EMPTY_HAND_SENTENCE = "Available only while your grippers are empty."
_HOLDING_SENTENCE = "Available only while you are holding an object."

MARKER_ARGS_SCHEMA = {
    "type": "object",
    "properties": {
        "marker": {
            "type": "string",
            "pattern": "^C[1-9][0-9]{0,2}$",
        }
    },
    "required": ["marker"],
    "additionalProperties": False,
}


def mark_visible_objects_spec() -> ActionSpec:
    return ActionSpec(
        name="mark_visible_objects",
        parse_args=parse_no_args,
        args_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        args_required=False,
        model_description=(
            "Request temporary C1, C2, ... labels on eligible objects visible "
            "in the current head-camera frame. This does not move the robot. "
            "The labels are valid only for the immediately following action."
        ),
        required_capabilities=frozenset(
            {"current_head_instance_segmentation", "official_instance_bbox"}
        ),
        max_simulator_steps=1,
        public_statuses=("markers_ready", "markers_unavailable"),
        invariants=frozenset(
            {"zero_simulator_steps", "no_object_identity_text", "no_teleport"}
        ),
        controller_key="perception.mark_visible_objects",
    )


def marker_symbolic_grasp_spec() -> ActionSpec:
    return ActionSpec(
        name="grasp",
        parse_args=parse_marker_grasp_args,
        args_schema=MARKER_ARGS_SCHEMA,
        args_required=True,
        model_description=(
            "Grasp exactly one temporary marker from the latest marked head "
            "frame, for example C3. It never searches for another instance, "
            "never performs long-range navigation, and accepts no category "
            "name or coordinates. It may physically close only the last small "
            "local gap to that same visible instance. Move near the target "
            "before marking it."
        ),
        required_capabilities=frozenset(
            {"fresh_visual_marker", "official_symbolic_grasp"}
        ),
        max_simulator_steps=600,
        public_statuses=("grasp_likely", "grasp_uncertain"),
        invariants=frozenset(
            {
                "same_frame_instance_binding",
                "one_shot_marker",
                "no_long_range_navigation",
                "bounded_physical_local_approach",
                "no_category_fallback",
                "no_textual_failure_truth",
            }
        ),
        controller_key="manipulation.marker_symbolic_grasp",
    )


def marker_symbolic_place_inside_spec() -> ActionSpec:
    return ActionSpec(
        name="place_inside",
        parse_args=parse_marker_place_inside_args,
        args_schema=MARKER_ARGS_SCHEMA,
        args_required=True,
        model_description=(
            "Place the currently held object inside exactly one container "
            "marker from the latest marked head frame, for example C1. It "
            "never searches for another container and never performs "
            "long-range navigation. It may physically close only the last "
            "small local gap to that same visible container. Move near the "
            "container before marking it."
        ),
        required_capabilities=frozenset(
            {"fresh_visual_marker", "official_symbolic_place_inside"}
        ),
        max_simulator_steps=1200,
        public_statuses=("completed", "placement_uncertain"),
        invariants=frozenset(
            {
                "same_frame_instance_binding",
                "one_shot_marker",
                "no_long_range_navigation",
                "bounded_physical_local_approach",
                "no_category_fallback",
                "no_textual_failure_truth",
            }
        ),
        controller_key="manipulation.marker_symbolic_place_inside",
    )


def _v34_marker_spec(
    *,
    name: str,
    description: str,
    capability: str,
    controller_key: str,
    max_simulator_steps: int = 600,
    selection_request: bool = False,
) -> ActionSpec:
    return ActionSpec(
        name=name,
        parse_args=(
            parse_marker_target_or_request_args
            if selection_request
            else parse_marker_target_args
        ),
        args_schema=(
            {
                "type": "object",
                "properties": dict(MARKER_ARGS_SCHEMA["properties"]),
                "additionalProperties": False,
            }
            if selection_request
            else MARKER_ARGS_SCHEMA
        ),
        args_required=not selection_request,
        model_description=description,
        required_capabilities=frozenset(
            {"fresh_visual_marker", capability}
        ),
        max_simulator_steps=max_simulator_steps,
        public_statuses=(
            (
                "selection_required",
                "markers_unavailable",
                "completed",
                "not_executed",
                "uncertain",
            )
            if selection_request
            else ("completed", "not_executed", "uncertain")
        ),
        invariants=frozenset(
            {
                "same_frame_instance_binding",
                "one_shot_marker",
                "no_long_range_navigation",
                "no_category_fallback",
                "no_textual_failure_truth",
                "kinematic_research",
            }
            | (
                {
                    "action_first_simple_spatial_selection",
                    "action_first_operation_marker_selection",
                    "selection_request_zero_step",
                }
                if selection_request
                else set()
            )
        ),
        controller_key=controller_key,
    )


def marker_symbolic_v34_specs(
    *,
    simple_spatial_selection: bool = False,
) -> tuple[ActionSpec, ...]:
    """Restricted ESI-style object actions for the v34 benchmark profile."""

    return (
        _v34_marker_spec(
            name="pick",
            description=(
                "With no marker, request currently visible pickable objects; "
                "with a C-marker, pick exactly that selection. A marker with "
                "connector lines selects the displayed objects as one carried "
                "group; an unconnected marker selects one object. "
                + _EMPTY_HAND_SENTENCE
                if simple_spatial_selection
                else "Pick exactly one C-marker from the latest marked onboard "
                "frame. A marker with connector lines selects the displayed "
                "objects as one carried group; an unconnected marker selects "
                "one object. The executor never searches for another instance "
                "or navigates to it. Move near it, mark again, then pick."
            ),
            capability="official_symbolic_grasp",
            controller_key="manipulation.marker_symbolic_pick",
            selection_request=simple_spatial_selection,
        ),
        _v34_marker_spec(
            name="place_inside",
            description=(
                "With no marker, request currently visible containers that "
                "are directly usable. Selecting a multi-compartment container "
                "returns only its currently visible usable compartments; "
                "select one of those C-markers to place the held object "
                "inside that region. " + _HOLDING_SENTENCE
                if simple_spatial_selection
                else "Put the currently held object inside exactly one container "
                "C-marker from the latest marked onboard frame."
            ),
            capability="official_symbolic_place_inside",
            controller_key="manipulation.marker_symbolic_place_inside",
            max_simulator_steps=1200,
            selection_request=simple_spatial_selection,
        ),
        _v34_marker_spec(
            name="place_on_top",
            description=(
                "With no marker, request currently visible supports; select "
                "one support C-marker. The executor tries stable, "
                "collision-checked positions across that selected support "
                "during execution. "
                + _HOLDING_SENTENCE
                if simple_spatial_selection
                else "Place the currently held object on top of exactly one "
                "support C-marker from the latest marked onboard frame."
            ),
            capability="official_symbolic_place_on_top",
            controller_key="manipulation.marker_symbolic_place_on_top",
            max_simulator_steps=1200,
            selection_request=simple_spatial_selection,
        ),
        _v34_marker_spec(
            name="place_next_to",
            description=(
                "With no marker, request visible reference objects. Select a "
                "reference to receive up to eight coarse nearby positions, "
                "then select one position and place the held object only "
                "there. " + _HOLDING_SENTENCE
                if simple_spatial_selection
                else "Place the currently held object next to exactly one target "
                "C-marker and verify canonical NextTo after settling."
            ),
            capability="canonical_next_to_default_region",
            controller_key="manipulation.marker_symbolic_place_next_to",
            max_simulator_steps=1200,
            selection_request=simple_spatial_selection,
        ),
        _v34_marker_spec(
            name="place_under",
            description=(
                "With no marker, request visible upper reference objects; "
                "select one C-marker and the executor chooses a feasible "
                "position underneath it for the held object. "
                + _HOLDING_SENTENCE
                if simple_spatial_selection
                else "Place the currently held object under exactly one target "
                "C-marker and verify canonical Under after settling."
            ),
            capability="canonical_under_default_region",
            controller_key="manipulation.marker_symbolic_place_under",
            max_simulator_steps=1200,
            selection_request=simple_spatial_selection,
        ),
        _v34_marker_spec(
            name="place_touching",
            description=(
                "With no marker, request visible reference choices. Select a "
                "reference C-marker to receive four coarse screen-direction "
                "choices, then select one C-marker to place the held object "
                "touching from only that direction. " + _HOLDING_SENTENCE
                if simple_spatial_selection
                else "Place the currently held object touching exactly one "
                "target C-marker and verify Touching after settling."
            ),
            capability="canonical_touching_default_region",
            controller_key="manipulation.marker_symbolic_place_touching",
            max_simulator_steps=1200,
            selection_request=simple_spatial_selection,
        ),
        _v34_marker_spec(
            name="open",
            description=(
                "With no marker, request currently visible openable objects "
                "or operable parts; with a C-marker, open exactly that "
                "selected choice. " + _EMPTY_HAND_SENTENCE
                if simple_spatial_selection
                else "Open exactly one openable C-marker from the latest marked "
                "onboard frame. This is unavailable while holding an object."
            ),
            capability="official_symbolic_open",
            controller_key="manipulation.marker_symbolic_open",
            selection_request=simple_spatial_selection,
        ),
        _v34_marker_spec(
            name="close",
            description=(
                "With no marker, request currently visible closeable objects "
                "or operable parts; with a C-marker, close exactly that "
                "selected choice. " + _EMPTY_HAND_SENTENCE
                if simple_spatial_selection
                else "Close exactly one openable C-marker from the latest marked "
                "onboard frame. This is unavailable while holding an object."
            ),
            capability="official_symbolic_close",
            controller_key="manipulation.marker_symbolic_close",
            selection_request=simple_spatial_selection,
        ),
        _v34_marker_spec(
            name="toggle_on",
            description=(
                "With no marker, request currently visible toggleable objects "
                "or operable parts; with a C-marker, turn on exactly that "
                "selected choice. " + _EMPTY_HAND_SENTENCE
                if simple_spatial_selection
                else "Turn on exactly one toggleable C-marker from the latest "
                "marked onboard frame. This is unavailable while holding an object."
            ),
            capability="official_symbolic_toggle_on",
            controller_key="manipulation.marker_symbolic_toggle_on",
            selection_request=simple_spatial_selection,
        ),
        _v34_marker_spec(
            name="toggle_off",
            description=(
                "With no marker, request currently visible toggleable objects "
                "or operable parts; with a C-marker, turn off exactly that "
                "selected choice. " + _EMPTY_HAND_SENTENCE
                if simple_spatial_selection
                else "Turn off exactly one toggleable C-marker from the latest "
                "marked onboard frame. This is unavailable while holding an object."
            ),
            capability="official_symbolic_toggle_off",
            controller_key="manipulation.marker_symbolic_toggle_off",
            selection_request=simple_spatial_selection,
        ),
    )


def marker_symbolic_action23_specs() -> tuple[ActionSpec, ...]:
    """Return the opt-in Action23 marker-bound object-action surface.

    This intentionally does not extend :func:`marker_symbolic_v34_specs` in
    place.  The historical v34 profile and its public manifest remain stable;
    only the explicitly selected Action23 profile receives the four spatial
    placement actions.
    """

    legacy = {spec.name: spec for spec in marker_symbolic_v34_specs()}
    spatial = {
        "place_on_top": _v34_marker_spec(
            name="place_on_top",
            description=(
                "Place the currently held object on top of exactly one "
                "support C-marker from the latest marked onboard frame."
            ),
            capability="canonical_spatial_place_on_top",
            controller_key="manipulation.marker_symbolic_place_on_top",
            max_simulator_steps=1200,
        ),
        "place_next_to": _v34_marker_spec(
            name="place_next_to",
            description=(
                "Place the currently held object next to exactly one target "
                "C-marker and verify canonical NextTo after settling."
            ),
            capability="canonical_next_to_default_region",
            controller_key="manipulation.marker_symbolic_place_next_to",
            max_simulator_steps=1200,
        ),
        "place_under": _v34_marker_spec(
            name="place_under",
            description=(
                "Place the currently held object under exactly one target "
                "C-marker and verify canonical Under after settling."
            ),
            capability="canonical_under_default_region",
            controller_key="manipulation.marker_symbolic_place_under",
            max_simulator_steps=1200,
        ),
        "place_touching": _v34_marker_spec(
            name="place_touching",
            description=(
                "Place the currently held object touching exactly one target "
                "C-marker and verify canonical Touching after settling."
            ),
            capability="canonical_touching_default_region",
            controller_key="manipulation.marker_symbolic_place_touching",
            max_simulator_steps=1200,
        ),
    }
    ordered = (
        "pick",
        "place_inside",
        "place_on_top",
        "place_next_to",
        "place_under",
        "place_touching",
        "open",
        "close",
        "toggle_on",
        "toggle_off",
    )
    combined = {**legacy, **spatial}
    return tuple(combined[name] for name in ordered)


def marker_symbolic_release_spec() -> ActionSpec:
    """Release the implicit-hand Symbolic carry without a marker target."""

    return ActionSpec(
        name="release",
        parse_args=parse_no_args,
        args_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        args_required=False,
        model_description=(
            "Release the currently held object at the current gripper pose. "
            "No marker, hand name, object identity, or coordinates are "
            "accepted. " + _HOLDING_SENTENCE
        ),
        required_capabilities=frozenset(
            {"held_object", "official_symbolic_release"}
        ),
        max_simulator_steps=600,
        public_statuses=("completed", "not_executed", "uncertain"),
        invariants=frozenset(
            {
                "held_object_only",
                "no_marker_target",
                "release_at_current_gripper_pose",
                "kinematic_research",
            }
        ),
        controller_key="manipulation.marker_symbolic_release",
    )


def marker_symbolic_place_on_floor_spec() -> ActionSpec:
    """Place the implicit-hand carry on a nearby safe floor region."""

    return ActionSpec(
        name="place_on_floor",
        parse_args=parse_no_args,
        args_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        args_required=False,
        model_description=(
            "Place the currently held object on a nearby safe floor region. "
            "No marker, object identity, or coordinates are accepted. The "
            "executor tries stable poses and keeps holding the object if no "
            "safe floor placement succeeds. " + _HOLDING_SENTENCE
        ),
        required_capabilities=frozenset(
            {"held_object", "canonical_spatial_place_on_floor"}
        ),
        max_simulator_steps=4096,
        public_statuses=("completed", "not_executed", "uncertain"),
        invariants=frozenset(
            {
                "held_object_only",
                "no_marker_target",
                "nearby_floor_region",
                "stable_placement_required",
                "recoverable_on_failure",
                "kinematic_research",
            }
        ),
        controller_key="manipulation.marker_symbolic_place_on_floor",
    )


def marker_symbolic_put_back_spec() -> ActionSpec:
    return ActionSpec(
        name="put_back",
        parse_args=parse_no_args,
        args_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        args_required=False,
        model_description=(
            "Return the currently held object to its executor-private "
            "pre-pick pose. No marker is accepted. This is a recorded "
            "kinematic-research operation."
        ),
        required_capabilities=frozenset(
            {"held_object", "private_pre_pick_pose"}
        ),
        max_simulator_steps=600,
        public_statuses=("completed", "not_executed", "uncertain"),
        invariants=frozenset(
            {
                "held_object_only",
                "no_marker_target",
                "direct_state_modification_recorded",
                "kinematic_research",
            }
        ),
        controller_key="manipulation.marker_symbolic_put_back",
    )
