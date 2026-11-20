"""Lower frozen Oracle DAGs into privacy-audited Action20 templates.

This module is deliberately CPU-only.  It certifies that every deterministic
node in ``prerender_dag_audit`` has a structural lowering to the public
Action20 request/selection protocol.  It does *not* invent navigation, bind a
live marker, resolve geometry, execute an action, or claim simulator success.

The generated action sequence is Oracle certification evidence.  Although its
payloads use only the public action protocol, the sequence itself must never be
placed in a model prompt.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from behavior_hl.embodied.policy.profiles import WIPE_ENABLED


WITNESS_SCHEMA = "behavior2026.action20-oracle-protocol-witness/v1"
SOURCE_AUDIT_SCHEMA = "behavior2026.prerender-dag-audit/v1"
MARKER_PLACEHOLDER = "C_CURRENT_FRAME"

STRICT_ACTION20 = (
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
    "pick",
    "place_on_floor",
    "place_inside",
    "place_on_top",
    "place_next_to",
    "place_under",
    "place_touching",
    "open",
    "close",
    "toggle_on",
    "toggle_off",
    "wait",
    *(("wipe",) if WIPE_ENABLED else ()),
)
REMOVED_PUBLIC_ACTIONS = frozenset(
    {"move_up", "move_down", "mark_visible_objects"}
)
MODEL_HIDDEN_ACTIONS = frozenset({"place_touching"})
MODEL_VISIBLE_ACTION20 = tuple(
    action for action in STRICT_ACTION20 if action not in MODEL_HIDDEN_ACTIONS
)
PLACEMENT_ACTIONS = frozenset(
    {
        "place_inside",
        "place_on_top",
        "place_next_to",
        "place_under",
        "place_touching",
    }
)
SUPPORTED_DAG_ACTIONS = PLACEMENT_ACTIONS | {
    "open",
    "close",
    "toggle_on",
    "toggle_off",
} | ({"wipe"} if WIPE_ENABLED else set())

_TERMINAL_PUBLIC_STATUSES = ("completed", "not_executed", "uncertain")
_SELECTION_PUBLIC_STATUSES = ("selection_required", "markers_unavailable")
_PRIVATE_PUBLIC_KEY_DENYLIST = frozenset(
    {
        "target",
        "destination",
        "object_name",
        "instance_id",
        "simulator_id",
        "coordinates",
        "pose",
        "predicate_truth",
        "q_score",
        "selected_option_index",
    }
)


class Action20OracleWitnessError(ValueError):
    """Raised when a frozen DAG cannot be safely lowered."""


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _request_phase(action: str) -> str:
    return {
        "pick": "request_visible_pickable_choices",
        "place_inside": "request_visible_container_choices",
        "place_on_top": "request_visible_support_choices",
        "place_next_to": "request_visible_reference_choices",
        "place_under": "request_visible_upper_reference_choices",
        "place_touching": "request_visible_reference_choices",
        "open": "request_visible_openable_object_or_part_choices",
        "close": "request_visible_openable_object_or_part_choices",
        "toggle_on": "request_visible_toggleable_object_or_part_choices",
        "toggle_off": "request_visible_toggleable_object_or_part_choices",
        "wipe": "request_visible_wipeable_choices",
    }[action]


def _final_phase(action: str) -> str:
    return {
        "pick": "execute_current_frame_target",
        "place_inside": "execute_current_frame_compartment_or_shelf_level",
        "place_on_top": "execute_current_frame_support",
        "place_next_to": "execute_current_frame_nearby_position",
        "place_under": "execute_current_frame_upper_reference",
        "place_touching": "execute_current_frame_contact_direction",
        "open": "execute_current_frame_object_or_part",
        "close": "execute_current_frame_object_or_part",
        "toggle_on": "execute_current_frame_object_or_part",
        "toggle_off": "execute_current_frame_object_or_part",
        "wipe": "execute_current_frame_target",
    }[action]


def _selection_step(
    *,
    action: str,
    phase: str,
    args: Mapping[str, Any],
    produces_next_stage: bool,
) -> dict[str, Any]:
    marker_input = "none" if not args else "current_frame_only"
    statuses = (
        _SELECTION_PUBLIC_STATUSES
        if produces_next_stage
        else _TERMINAL_PUBLIC_STATUSES
    )
    return {
        "action": action,
        "args": dict(args),
        "protocol_phase": phase,
        "marker_input": marker_input,
        "expected_public_statuses": list(statuses),
        "continue_only_on": (
            "selection_required" if produces_next_stage else "completed"
        ),
        "marker_lifecycle": {
            "prior_namespaces_valid_after_action": False,
            "output_namespace": (
                "fresh_current_frame" if produces_next_stage else "not_reusable"
            ),
        },
    }


def action_protocol_template(action: str) -> tuple[dict[str, Any], ...]:
    """Return the structural public turns for one Action20 object action."""

    if action not in SUPPORTED_DAG_ACTIONS | {"pick"}:
        raise Action20OracleWitnessError(
            f"unsupported Action20 object action: {action!r}"
        )
    request = _selection_step(
        action=action,
        phase=_request_phase(action),
        args={},
        produces_next_stage=True,
    )
    request["binding_semantics"] = "operation_specific_live_candidates"
    if action in {"place_inside", "place_next_to", "place_touching"}:
        middle_phase = {
            "place_inside": "select_current_frame_container",
            "place_next_to": "select_current_frame_reference",
            "place_touching": "select_current_frame_reference",
        }[action]
        middle = _selection_step(
            action=action,
            phase=middle_phase,
            args={"marker": MARKER_PLACEHOLDER},
            produces_next_stage=True,
        )
        middle["binding_semantics"] = {
            "place_inside": (
                "current_frame_container_then_publish_visible_usable_"
                "compartment_or_shelf_level_candidates"
            ),
            "place_next_to": (
                "current_frame_reference_then_publish_nearby_position_candidates"
            ),
            "place_touching": (
                "current_frame_reference_then_publish_contact_direction_candidates"
            ),
        }[action]
        final = _selection_step(
            action=action,
            phase=_final_phase(action),
            args={"marker": MARKER_PLACEHOLDER},
            produces_next_stage=False,
        )
        final["binding_semantics"] = {
            "place_inside": (
                "preserve_current_frame_compartment_or_shelf_level_selector_"
                "through_execution"
            ),
            "place_next_to": "preserve_current_frame_nearby_position_selector",
            "place_touching": "preserve_current_frame_contact_direction_selector",
        }[action]
        return request, middle, final
    final = _selection_step(
        action=action,
        phase=_final_phase(action),
        args={"marker": MARKER_PLACEHOLDER},
        produces_next_stage=False,
    )
    final["binding_semantics"] = {
        "pick": "exact_current_frame_visible_pickable_binding",
        "place_on_top": "exact_current_frame_visible_support_binding",
        "place_under": "exact_current_frame_upper_reference_binding",
        "open": "exact_current_frame_visible_openable_object_or_part_binding",
        "close": "exact_current_frame_visible_openable_object_or_part_binding",
        "toggle_on": "exact_current_frame_visible_toggleable_object_or_part_binding",
        "toggle_off": "exact_current_frame_visible_toggleable_object_or_part_binding",
        "wipe": "exact_current_frame_visible_wipeable_binding",
    }[action]
    return request, final


def _private_identity_values(nodes: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    values: set[str] = set()
    for node in nodes:
        for key in ("target", "destination"):
            value = node.get(key)
            if isinstance(value, str) and value:
                values.add(value)
    return frozenset(values)


def _public_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _public_keys(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _public_keys(child)


def _validate_public_privacy(
    public: Mapping[str, Any],
    *,
    private_identities: Iterable[str],
) -> None:
    public_text = _stable_json(public)
    leaked = sorted(
        identity for identity in private_identities if identity in public_text
    )
    if leaked:
        raise Action20OracleWitnessError(
            "private target/reference identity leaked into public template"
        )
    denied_keys = _PRIVATE_PUBLIC_KEY_DENYLIST & set(_public_keys(public))
    if denied_keys:
        raise Action20OracleWitnessError(
            f"private keys leaked into public template: {sorted(denied_keys)}"
        )


def _validate_source_profile(profile: Mapping[str, Any]) -> None:
    if profile.get("compile_error") is not None:
        raise Action20OracleWitnessError("source profile did not compile")
    checks = profile.get("checks")
    if not isinstance(checks, Mapping):
        raise Action20OracleWitnessError("source profile checks are missing")
    if checks.get("cycle_or_non_topological") is not False:
        raise Action20OracleWitnessError("source DAG is not topological")
    if checks.get("unknown_dependency_node_ids") != []:
        raise Action20OracleWitnessError("source DAG has unknown dependencies")
    nodes = profile.get("deterministic_nodes")
    if not isinstance(nodes, list):
        raise Action20OracleWitnessError("source deterministic nodes are missing")
    node_ids = [node.get("node_id") for node in nodes]
    if any(not isinstance(node_id, str) or not node_id for node_id in node_ids):
        raise Action20OracleWitnessError("source DAG node identity is invalid")
    if len(node_ids) != len(set(node_ids)):
        raise Action20OracleWitnessError("source DAG node identities are not unique")
    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    for index, node in enumerate(nodes):
        action = node.get("action")
        if action not in SUPPORTED_DAG_ACTIONS:
            raise Action20OracleWitnessError(
                f"DAG action has no Action20 lowering: {action!r}"
            )
        for dependency in node.get("dependencies", ()):
            if dependency not in positions or positions[dependency] >= index:
                raise Action20OracleWitnessError(
                    "DAG dependency ordering drifted"
                )


def lower_profile(
    profile: Mapping[str, Any],
    *,
    profile_ordinal: int,
) -> Mapping[str, Any]:
    """Lower one audited private DAG to a marker-placeholder template."""

    _validate_source_profile(profile)
    nodes = profile["deterministic_nodes"]
    public_turns: list[dict[str, Any]] = []
    private_nodes: list[dict[str, Any]] = []
    for node_ordinal, source_node in enumerate(nodes):
        start = len(public_turns)
        action = str(source_node["action"])
        lowered_actions = (
            ("pick", action) if action in PLACEMENT_ACTIONS else (action,)
        )
        for lowered_action in lowered_actions:
            for raw_step in action_protocol_template(lowered_action):
                step = deepcopy(raw_step)
                step["turn_index"] = len(public_turns)
                step["source_node_ordinal"] = node_ordinal
                step["lowering_role"] = (
                    "acquire_placement_subject"
                    if lowered_action == "pick"
                    else "execute_dag_operation"
                )
                public_turns.append(step)
        stop = len(public_turns)
        private_nodes.append(
            {
                "source_node_ordinal": node_ordinal,
                "node_id": source_node["node_id"],
                "action": action,
                "target": source_node["target"],
                "destination": source_node.get("destination"),
                "dependencies": list(source_node.get("dependencies", ())),
                "achieves": list(source_node.get("achieves", ())),
                "public_turn_start_inclusive": start,
                "public_turn_stop_exclusive": stop,
                "inserted_pick_protocol": action in PLACEMENT_ACTIONS,
                "action_protocol_depth": len(action_protocol_template(action)),
            }
        )
    public = {
        "template_id": f"oracle63_profile_{profile_ordinal:03d}_action20",
        "artifact_only_not_model_prompt": True,
        "turn_count": len(public_turns),
        "turns": public_turns,
    }
    _validate_public_privacy(
        public,
        private_identities=_private_identity_values(nodes),
    )
    return {
        "profile_ordinal": profile_ordinal,
        "public_protocol_template": public,
        "private_certification": {
            "catalog": profile["catalog"],
            "task_name": profile["task_name"],
            "task_id": profile["task_id"],
            "scene_model": profile["scene_model"],
            "split": profile["split"],
            "instance_id": profile["instance_id"],
            "selected_option_index": profile["selected_option_index"],
            "selected_option_leaf_ids": list(profile["selected_option_leaf_ids"]),
            "source_checks": deepcopy(profile["checks"]),
            "requires_live_geometry": bool(profile["requires_live_geometry"]),
            "live_geometry_reasons": list(profile["live_geometry_reasons"]),
            "deterministic_nodes": private_nodes,
        },
    }


def compile_oracle63_witness(audit: Mapping[str, Any]) -> Mapping[str, Any]:
    """Compile the complete frozen 63-profile static witness report."""

    if audit.get("schema_version") != SOURCE_AUDIT_SCHEMA:
        raise Action20OracleWitnessError("unsupported source audit schema")
    if audit.get("cpu_only") is not True or audit.get("simulator_imported") is not False:
        raise Action20OracleWitnessError("source audit is not CPU-only")
    profiles = audit.get("profiles")
    if not isinstance(profiles, list) or audit.get("profile_count") != len(profiles):
        raise Action20OracleWitnessError("source profile count drifted")
    if len(profiles) != 63 or audit.get("compile_failure_count") != 0:
        raise Action20OracleWitnessError("Oracle63 source audit is incomplete")

    lowered = [
        lower_profile(profile, profile_ordinal=index)
        for index, profile in enumerate(profiles)
    ]
    dag_action_counts: Counter[str] = Counter()
    emitted_action_counts: Counter[str] = Counter()
    for source_profile, output_profile in zip(profiles, lowered, strict=True):
        dag_action_counts.update(
            str(node["action"])
            for node in source_profile["deterministic_nodes"]
        )
        emitted_action_counts.update(
            step["action"]
            for step in output_profile["public_protocol_template"]["turns"]
        )
    emitted_actions = set(emitted_action_counts)
    if not emitted_actions <= set(STRICT_ACTION20):
        raise Action20OracleWitnessError("lowering emitted a non-Action20 action")
    if emitted_actions & REMOVED_PUBLIC_ACTIONS:
        raise Action20OracleWitnessError("lowering emitted a removed public action")

    public_contract = {
        "strict_action20": list(STRICT_ACTION20),
        "marker_placeholder": MARKER_PLACEHOLDER,
        "request_args": {},
        "marker_binding_rule": (
            "Substitute only a Cn from the current action-specific selection frame."
        ),
        "marker_invalidation_rule": (
            "Executing any action invalidates every older Cn namespace; a staged "
            "selection_required result creates a fresh current-frame namespace."
        ),
        "operation_specific_candidates": (
            "Each empty-argument request exposes only candidates for that action "
            "and protocol stage, never every movable scene object."
        ),
        "removed_actions": sorted(REMOVED_PUBLIC_ACTIONS),
        "oracle_sequence_prompt_policy": (
            "Certification artifact only; never include these Oracle turns in a "
            "model prompt or observation."
        ),
    }
    execution_boundary = {
        "promotion_label": "STATIC_CONTRACT_ONLY",
        "navigation": (
            "No move or turn sequence is inferred. Live policy navigation and "
            "camera alignment are required before each selection when needed."
        ),
        "visibility": (
            "Candidate existence, current-frame visibility, and Cn binding are "
            "resolved only from the live onboard observation."
        ),
        "geometry": (
            "Compartments, shelf levels, support placement, nearby positions, under "
            "feasibility, contact directions, collision, and stability require "
            "the live canonical runtime and verifier."
        ),
        "failure_recovery": (
            "The static happy-path template does not certify carry recovery or "
            "retry branches after markers_unavailable, not_executed, or uncertain."
        ),
        "live_success_claimed": False,
        "model_rollout_claimed": False,
    }
    report = {
        "schema_version": WITNESS_SCHEMA,
        "cpu_only": True,
        "simulator_imported": False,
        "model_called": False,
        "source_audit_sha256": _sha256_json(audit),
        "profile_count": len(lowered),
        "source_dag_node_count": sum(dag_action_counts.values()),
        "public_protocol_turn_count": sum(emitted_action_counts.values()),
        "inserted_pick_protocol_count": sum(
            count for action, count in dag_action_counts.items() if action in PLACEMENT_ACTIONS
        ),
        "source_dag_action_counts": dict(sorted(dag_action_counts.items())),
        "emitted_public_action_counts": dict(sorted(emitted_action_counts.items())),
        "public_protocol_contract": public_contract,
        "public_execution_boundary": execution_boundary,
        "checks": {
            "all_source_dags_topological": True,
            "all_dag_actions_lowered": True,
            "all_emitted_actions_in_strict_action20": True,
            "removed_action_occurrence_count": 0,
            "private_identity_leak_count": 0,
            "marker_current_frame_placeholder_only": True,
            "older_marker_namespaces_invalidated_after_every_action": True,
            "navigation_sequence_invented": False,
        },
        "profiles": lowered,
    }
    return report


def write_witness(report: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
