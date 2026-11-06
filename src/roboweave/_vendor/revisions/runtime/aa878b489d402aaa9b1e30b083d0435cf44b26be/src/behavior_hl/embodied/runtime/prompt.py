"""Exact observation-only full-episode prompt serialization."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from behavior_hl.embodied.policy.observations import (
    HEAD_RGB_MINIMAL_CAMERA_PROFILE,
    THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
    THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
)
from behavior_hl.embodied.policy.registry import ActionRegistry
from behavior_hl.embodied.runtime.availability import ActionAvailability
from behavior_hl.embodied.runtime.episode import PublicTaskContext
from behavior_hl.embodied.runtime.public_action_feedback import (
    model_visible_action20_last_result,
    model_visible_action_result,
)


EPISODE_PROMPT_VERSION = "behavior-hl-episode-visual-self-assessment-v5"
ACTION20_EPISODE_PROMPT_VERSION = (
    "behavior-hl-action20-episode-visual-self-assessment-workspace17-v5-zero-physics-feedback"
)
# v185/v186 emitted "TokenPlan:<serving model>" here, so a lane could read its
# own identity out of its prompt (see the 2026-08-13 rynnbrain snapshot).  The
# fixed constant is now enforced rather than merely conventional.
ACTION20_RESIDENT_MODEL_PROFILE = "TokenPlan:resident"
ACTION23_SCHEMA_VERSION = "behavior2026-r1pro-esi-marker-symbolic-action23-v1"
ACTION20_SCHEMA_TOKENS = (
    "marker_symbolic_action20",
    "marker-symbolic-action20",
)
ACTION20_PUBLIC_PROFILE = "r1pro_esi_marker_symbolic_action20_v2_place_on_floor"


def build_episode_prompt(
    *,
    task: PublicTaskContext,
    registry: ActionRegistry,
    availability: ActionAvailability,
    frame_sha256: Mapping[str, str],
    public_history: Sequence[Mapping[str, Any]],
    ordered_observation_frame_sha256: Sequence[
        Mapping[str, str]
    ] = (),
    privileged_truth: Mapping[str, Any] | None = None,
    privileged_decision_brief: Mapping[str, Any] | None = None,
    navigation_repeat_enabled: bool = False,
    navigation_repeat_result: Mapping[str, Any] | None = None,
) -> str:
    if privileged_truth is not None and not isinstance(privileged_truth, Mapping):
        raise TypeError("privileged_truth must be a mapping or None")
    if privileged_decision_brief is not None and not isinstance(
        privileged_decision_brief, Mapping
    ):
        raise TypeError("privileged_decision_brief must be a mapping or None")
    if privileged_truth is not None and privileged_decision_brief is not None:
        raise ValueError("provide privileged truth or decision brief, not both")
    privileged = (
        privileged_decision_brief
        if privileged_decision_brief is not None
        else privileged_truth
    )
    for camera, digest in frame_sha256.items():
        if (
            not isinstance(camera, str)
            or not camera
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("camera hashes must be lowercase SHA256 digests")
    ordered_frames: list[dict[str, Any]] = []
    for order, frame_set in enumerate(ordered_observation_frame_sha256):
        verified: dict[str, str] = {}
        for camera, digest in frame_set.items():
            if (
                not isinstance(camera, str)
                or not camera
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in digest
                )
            ):
                raise ValueError(
                    "ordered camera hashes must be lowercase SHA256 digests"
                )
            verified[camera] = digest
        ordered_frames.append(
            {
                "order": order,
                "camera_sha256": dict(sorted(verified.items())),
            }
        )
    available = set(availability.available_actions)
    if available.difference(task.enabled_actions):
        raise ValueError("availability contains actions disabled by task")
    action20 = any(
        token in registry.schema_version for token in ACTION20_SCHEMA_TOKENS
    )
    if action20:
        if task.episode.action_profile != ACTION20_PUBLIC_PROFILE:
            raise ValueError("Action20 prompt action profile identity mismatch")
        if task.episode.observation_profile not in {
            HEAD_RGB_MINIMAL_CAMERA_PROFILE,
            THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
            THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
        }:
            raise ValueError(
                "Action20 prompt observation profile identity mismatch"
            )
        if task.episode.model_profile != ACTION20_RESIDENT_MODEL_PROFILE:
            raise ValueError(
                "Action20 prompt must not expose the serving model identity"
            )
    v34_symbolic = "pick" in registry.names
    simple_spatial_selection = any(
        "action_first_simple_spatial_selection" in registry.spec(name).invariants
        for name in ("place_on_top", "place_touching")
        if name in registry.names
    )
    if v34_symbolic:
        marker_consumers = tuple(
            name
            for name in (
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
            if name in registry.names
        )
        marker_consumer_text = ", ".join(marker_consumers)
        explicit_mark_action = "mark_visible_objects" in registry.names
        action_constraints = [
            "Every marker-targeted action acts only on that visible marked "
            "instance. It never searches by category, "
            "navigate to another instance, or expose coordinates.",
            "Move close enough using onboard images before requesting choices. "
            "If an operation is not executed, inspect the new image and choose "
            "movement or request fresh choices yourself.",
        ]
        if explicit_mark_action:
            action_constraints[0:0] = [
                "mark_visible_objects labels only the latest onboard head frame; "
                "C-labels are one-shot and valid only for the immediately next "
                f"one of these actions: {marker_consumer_text}.",
                "Any move, turn, look, wait, object action, or new mark "
                "invalidates all older C-labels.",
            ]
        else:
            action_constraints.insert(
                0,
                "Marker selection is internal to each target-bound action; "
                "mark_visible_objects is not a public action.",
            )
        if "put_back" in registry.names:
            action_constraints.append(
                "put_back accepts no marker and only returns the currently "
                "held object to its pre-pick pose."
            )
        if "place_on_floor" in registry.names:
            action_constraints.append(
                "place_on_floor accepts no marker and automatically tries "
                "nearby collision-free floor poses. It succeeds only after "
                "official relation readback; otherwise you keep holding the object."
            )
        if "look_up" in registry.names:
            action_constraints.append(
                "look_up and look_down change onboard camera pitch by one "
                "bounded grid step while preserving the selected camera height."
            )
        if "wait" in registry.names:
            action_constraints.append(
                (
                    "wait refreshes the onboard observation without advancing "
                    "simulator time or changing world state."
                    if action20
                    else "wait advances one fixed simulated second (exactly 30 "
                    "simulator steps) and reveals only the resulting fresh "
                    "onboard observation."
                )
            )
        if simple_spatial_selection:
            action_constraints.extend(
                (
                    "Action-first Marker selection is enabled. Call any target-"
                    "bound object action without args to request only the "
                    "choices relevant to that action in the current head frame; "
                    "the request changes no object state and takes zero simulator "
                    "steps. Call the same action with one returned C-marker to "
                    "select or execute.",
                    "Requested choices are delivered only as C1, C2, ... labels "
                    "drawn onto the next onboard head image you receive. No "
                    "textual list of choices, object name, or coordinate is "
                    "ever sent, so read the labels from that image before "
                    "selecting. A repeated no-args request only re-renders "
                    "the choices for the current view. When no valid choice "
                    "exists, last_action_result reports the request blocked "
                    "and no label appears; move or change view, then request "
                    "again.",
                    "A returned C-marker is valid only for the immediately "
                    "next action, and only for the same action name that "
                    "produced it. Anything else in between -- moving, "
                    "turning, looking, waiting, or a different object action "
                    "-- discards the whole set and you must request the "
                    "choices again. Naming a label that was not in the set "
                    "you just received also discards that set.",
                    "For every marker-targeted placement action the selected "
                    "C-marker is the "
                    "destination (container, compartment, support, "
                    "reference, or nearby position); the object that moves is "
                    "always the one already in your gripper. A local workspace "
                    "bound never applies to what you carry: place_on_top "
                    "measures it to each generated future object footprint, "
                    "while the other placements use their selected "
                    "destination.",
                    "open, close, and toggle choices may represent one visible "
                    "whole object or one visible operable part; hidden parts are "
                    "never invented.",
                )
            )
            if action20:
                action_constraints.append(
                    "Selection: place_on_top = request supports, select a support, "
                    "select a marked surface; place_next_to = request references, "
                    "select a reference, select a position; place_inside adds a "
                    "compartment only when shown. Other target actions use request, "
                    "select."
                )
    else:
        action_constraints = [
            "grasp_at and place_inside_at points refer to the current named camera.",
            "grasp object and selector refer only to instances visible in the "
            "current head image; omit selector only when one instance matches.",
            "mark_visible_objects labels only the current head frame; C-labels "
            "are one-shot and valid only for the immediately following grasp "
            "or place_inside action.",
            "grasp(marker) and place_inside(marker) never search by category "
            "and never perform long-range navigation to another instance. "
            "They can physically close only a small final local gap to the "
            "same marked instance.",
            "Before marking, use move/turn/look until the intended object or "
            "container is clearly visible, reasonably centered, and appears "
            "within roughly one arm's reach (about 0.9 m). If you move after "
            "marking, request fresh markers again.",
        ]
    executor_feedback_constraint = (
        "After every public action, last_action_result reports the "
        "immediately preceding action's status=completed/blocked; base "
        "motion additionally includes bounded finite body-relative "
        "odometry. It is null only before the first action. A blocked "
        "result means that request did not produce its stated effect; "
        "inspect the fresh images and choose the next public action "
        "yourself. A blocked result may include one reliable coarse "
        "blocked_category. No coordinates, route, contact truth, "
        "predicate truth, Q, or simulator identity is supplied."
        if action20
        else "The only executor reason code you may receive is "
        "TARGET_OUTSIDE_LOCAL_WORKSPACE with status=not_executed. It "
        "means the local skill did not act; choose a public movement "
        "action yourself from onboard images. No distance, direction, route, "
        "contact truth, grasp truth, or placement truth is supplied."
    )
    payload = {
        "prompt_version": (
            ACTION20_EPISODE_PROMPT_VERSION
            if action20
            else EPISODE_PROMPT_VERSION
        ),
        "task": task.to_dict(),
        # A digest carries no visual meaning for the policy and costs roughly
        # twenty tokens per frame, so Action20 publishes the camera identity
        # and chronological position instead.  The digests stay in the
        # provider request and the run's own evidence records.
        "observations": {
            "onboard_rgb_frames": (
                sorted(frame_sha256)
                if action20
                else dict(sorted(frame_sha256.items()))
            ),
            "ordered_action_observation_frames": (
                [
                    {
                        "order": entry["order"],
                        "cameras": sorted(entry["camera_sha256"]),
                    }
                    for entry in ordered_frames
                ]
                if action20
                else ordered_frames
            ),
            **(
                {
                    "frame_order": (
                        "oldest first; the last entry is the current view"
                    )
                }
                if action20
                else {}
            ),
            "coordinate_convention": (
                "normalized [x,y], [0,0] top-left, [1,1] bottom-right"
            ),
        },
        "constraints": [
            "Choose exactly one available registered action.",
            (
                "Use the chronologically ordered onboard images, prior actions, "
                "and the privileged decision data. Treat current live truth as authoritative."
                if privileged is not None
                else "Use only the chronologically ordered onboard images and your "
                "own prior actions."
            ),
            (
                "Local movement uses collision-checked symbolic endpoints and "
                "may be blocked."
                if action20
                else "Local movement is physical and may be blocked."
            ),
            (
                "Use supplied coordinates, predicates, collision diagnostics, and "
                "navigation hints to select registered actions; teleport remains disabled."
                if privileged is not None
                else "Never assume maps, object coordinates, task predicates, or teleport."
            ),
            *action_constraints,
            executor_feedback_constraint,
            *(
                ("stop before official success is a task failure.",)
                if "stop" in registry.names
                else ()
            ),
        ],
        "available_actions": sorted(available),
        "action_manifest": {
            "schema_version": registry.schema_version,
            "actions": [
                registry.spec(name).model_manifest()
                for name in sorted(available)
            ],
        },
        **(
            {
                "registered_action_manifest": {
                    "schema_version": registry.schema_version,
                    "actions": [
                        registry.spec(name).model_manifest()
                        for name in registry.names
                    ],
                }
            }
            if registry.schema_version == ACTION23_SCHEMA_VERSION
            else {}
        ),
        "prior_actions": _model_visible_action_history(
            public_history,
            expose_legacy_result=not action20,
        ),
        **(
            {
                "last_action_result": _action20_last_action_result(
                    public_history
                )
            }
            if action20
            else {}
        ),
        "output": (
            "Return exactly one strict JSON action envelope with fields: "
            "'action' (required action name), 'args' (JSON object, required "
            "only when the action takes arguments), 'reasoning' (optional "
            "string, 1 to 2000 characters; longer responses are rejected). "
            "No other fields are accepted."
        ),
        **(
            {
                (
                    "privileged_decision_brief"
                    if privileged_decision_brief is not None
                    else "privileged_truth"
                ): dict(privileged)
            }
            if privileged is not None
            else {}
        ),
    }
    if navigation_repeat_enabled:
        from .resident_navigation_repeat import NAVIGATION_REPEAT_CONTRACT
        payload["navigation_repeat"] = NAVIGATION_REPEAT_CONTRACT
        payload["output"] = payload["output"].replace(
            "No other fields are accepted.",
            "Optional top-level 'repeat' is accepted for navigation only (integer 1..10). "
            "No other fields are accepted.",
        )
        if navigation_repeat_result is not None:
            payload["last_navigation_repeat"] = dict(navigation_repeat_result)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _model_visible_action_history(
    history: Sequence[Mapping[str, Any]],
    *,
    expose_legacy_result: bool = True,
) -> list[dict[str, Any]]:
    """Expose actions plus one tightly allowlisted controller precondition."""

    visible: list[dict[str, Any]] = []
    for expected_sequence, item in enumerate(history):
        if not isinstance(item, Mapping):
            raise ValueError("public history entries must be mappings")
        sequence = item.get("sequence")
        action = item.get("action")
        if sequence != expected_sequence or not isinstance(action, Mapping):
            raise ValueError("public history action contract is invalid")
        public_item: dict[str, Any] = {
            "sequence": sequence,
            "action": dict(action),
        }
        if expose_legacy_result:
            safe_result = model_visible_action_result(item.get("result"))
            if safe_result is not None:
                public_item["result"] = safe_result
        visible.append(public_item)
    return visible


def _action20_last_action_result(
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Expose the immediately preceding bounded public-action outcome."""

    if not history:
        return None
    last = history[-1]
    action = last.get("action")
    return model_visible_action20_last_result(
        last.get("result"),
        action=(action.get("action") if isinstance(action, Mapping) else None),
    )


def episode_prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
