"""Replay recorded predicate truth through the frozen CPU verifiers.

Records provide truth, not simulator geometry. This API cannot attest that a
record matches a scene; Live uses the original evaluator in the scene process.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import json
from typing import Any

from .bootstrap import activate

SCHEMA = "esi-benchmark/score-record-v1"


def _truth_reader(rows):
    if not isinstance(rows, list):
        raise ValueError("truth must be a list of predicate records")
    values = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"predicate", "target", "destination", "value"}:
            raise ValueError("invalid predicate truth record")
        if not all(isinstance(row[k], str) and row[k] for k in ("predicate", "target")):
            raise ValueError("predicate and target must be nonempty strings")
        if row["destination"] is not None and not isinstance(row["destination"], str):
            raise ValueError("destination must be a string or null")
        if type(row["value"]) is not bool:
            raise ValueError("predicate truth must be boolean")
        key = (row["predicate"], row["target"], row["destination"])
        if key in values:
            raise ValueError("duplicate predicate truth")
        values[key] = row["value"]
    # Missing evidence deliberately raises; ProjectedGoalVerifier fails closed.
    return lambda predicate, target, destination: values[(predicate, target, destination)]


def _options(value):
    if not isinstance(value, list) or not value:
        raise ValueError("official options must be a nonempty list")
    if any(not isinstance(option, list) or not option or any(type(v) is not bool for v in option)
           for option in value):
        raise ValueError("official option leaves must be nonempty boolean lists")
    return tuple(tuple(option) for option in value)


def _atom_truth(rows):
    if not isinstance(rows, list):
        raise ValueError("process atom truth must be a list")
    values = {}
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"atom", "value"}
                or not isinstance(row["atom"], dict) or type(row["value"]) is not bool):
            raise ValueError("invalid process atom truth")
        key = json.dumps(row["atom"], sort_keys=True, separators=(",", ":"))
        if key in values:
            raise ValueError("duplicate process atom truth")
        values[key] = row["value"]
    return values


def _role_list(value, domain, name):
    if (not isinstance(value, list) or any(not isinstance(r, str) or r not in domain for r in value)
            or len(value) != len(set(value))):
        raise ValueError(f"{name} must be a list of distinct declared roles")


def _bool_list(value, name, count=None):
    if (not isinstance(value, list) or any(type(v) is not bool for v in value)
            or (count is not None and len(value) != count)):
        raise ValueError(f"{name} must contain the required boolean readbacks")


def _process_snapshot(snapshot, requirements, roles, name):
    if not isinstance(snapshot, dict):
        raise ValueError(f"{name} must be an available process snapshot")
    _role_list(snapshot.get("held_roles"), roles, name + ".held_roles")
    if "selected_roles" in snapshot:
        _role_list(snapshot["selected_roles"], requirements["no_manipulation_roles"], name + ".selected_roles")
    count = len(requirements["before_placement"])
    _bool_list(snapshot.get("prerequisites"), name + ".prerequisites", count)
    _bool_list(snapshot.get("protected_relations"), name + ".protected_relations")
    if count or "placement_goals" in snapshot:
        _bool_list(snapshot.get("placement_goals"), name + ".placement_goals", count)
    states = requirements.get("state_protection_roles", [])
    if states or "protected_states" in snapshot:
        value = snapshot.get("protected_states")
        _bool_list(value, name + ".protected_states")
        if len(value) < len(states):
            raise ValueError(f"{name}.protected_states lacks declared state protection readbacks")
    references = requirements.get("current_reference_contracts", [])
    if references or "current_references" in snapshot:
        value = snapshot.get("current_references")
        if not isinstance(value, list) or len(value) != len(references):
            raise ValueError(f"{name}.current_references lacks declared reference readbacks")
        for matches, rule in zip(value, references):
            if not isinstance(matches, dict) or set(matches) != {"source_matches", "target_matches"}:
                raise ValueError(f"{name}.current_references has invalid match fields")
            for field, matches_roles in matches.items():
                _role_list(matches_roles, rule["domain_roles"], name + ".current_references." + field)
    if "process_program" in requirements:
        from behavior_hl.embodied.evaluator_bridge.l1_am_process_programs import EXTENSIONS
        program = requirements["process_program"]
        keys = {stage["milestone_id"] for stage in program["milestones"]}
        unary = any(k in program for k in EXTENSIONS)
        values = snapshot.get("milestones")
        if not isinstance(values, dict) or set(values) != keys | ({"_unary"} if unary else set()):
            raise ValueError(f"{name}.milestones lacks declared milestone readbacks")
        if any(type(values[k]) is not bool for k in keys):
            raise ValueError(f"{name}.milestones must contain boolean readbacks")
        if unary:
            domain = {a["subject_role"] for stage in program["milestones"] for a in stage["requirements"]}
            if (not isinstance(values["_unary"], dict) or set(values["_unary"]) != domain
                    or any(type(v) is not bool for v in values["_unary"].values())):
                raise ValueError(f"{name}.milestones._unary lacks declared state readbacks")


def replay_process(record: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    """Replay existing begin/before/after/end evidence with the original audit."""
    activate()
    from behavior_hl.embodied.evaluator_bridge.l1_process_audit import (
        ProcessAudit, validate_process_requirements,
    )
    if not isinstance(record, Mapping) or not {"requirements", "roles", "initial_truth", "final_truth", "events"} <= set(record):
        raise ValueError("process record lacks requirements, roles, truth or events")
    requirements = record["requirements"]
    roles = record["roles"]
    if not isinstance(roles, list) or any(not isinstance(r, str) or not r for r in roles):
        raise ValueError("process roles must be nonempty strings")
    _role_list(roles, roles, "process roles")
    try:
        validate_process_requirements(requirements, set(roles))
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("invalid process requirements") from exc
    if "history_program" in requirements:
        raise ValueError("history_program replay requires per-event predicate truth absent from score-record-v1")
    current = _atom_truth(record["initial_truth"])
    final_truth = _atom_truth(record["final_truth"])
    events = record["events"]
    if not isinstance(events, list):
        raise ValueError("process events must be a list")
    evidence_counts = None
    for event in events:
        if not isinstance(event, dict) or set(event) != {"action", "before", "after", "status", "action_index"}:
            raise ValueError("invalid process event")
        if any(not isinstance(event[k], str) or not event[k].strip() for k in ("action", "status")):
            raise ValueError("process action and status must be nonempty strings")
        index = event["action_index"]
        if index is not None and (type(index) is not int or index < 0):
            raise ValueError("process action index must be nonnegative or null")
        for phase in ("before", "after"):
            snapshot = event[phase]
            _process_snapshot(snapshot, requirements, set(roles), phase)
            counts = (len(snapshot["protected_relations"]), len(snapshot.get("protected_states", [])))
            if evidence_counts is not None and counts != evidence_counts:
                raise ValueError("process protection readback counts changed")
            evidence_counts = counts

    def truth(atom, _role_object):
        return current[json.dumps(atom, sort_keys=True, separators=(",", ":"))]

    directory = Path(directory)
    if directory.exists():
        raise ValueError("process replay directory must be fresh")
    try:
        audit = ProcessAudit(requirements, role_object=lambda role: role, recipe_truth=truth,
                             protected_relations=(), directory=directory)
    except KeyError as exc:
        raise ValueError("process initialization lacks required predicate truth") from exc
    for event in events:
        audit.observe(**event)
    current = final_truth
    try:
        return audit.report()
    except KeyError as exc:
        raise ValueError("process final readback lacks required predicate truth") from exc


def score_record(record: Mapping[str, Any], *, process_directory: Path | None = None) -> dict[str, Any]:
    """Score supplied records without connecting to a simulator or model."""
    if not isinstance(record, Mapping) or record.get("schema_version") != SCHEMA:
        raise ValueError(f"score record must use {SCHEMA}")
    activate()
    frames = record.get("snapshots")
    if not isinstance(frames, list) or not frames or any(not isinstance(f, dict) for f in frames):
        raise ValueError("at least one snapshot is required")
    route = record.get("route")
    snapshots = []
    if route == "official_full":
        from behavior_hl.embodied.evaluator_bridge.official_predicate_verifier import OfficialPredicateVerifier
        initial = _options(record["initial_options"])
        verifier = OfficialPredicateVerifier(require_single_option=len(initial) == 1)
        for frame in frames:
            if frame.get("integrity_ok", True) is not True:
                snapshots.append({"operational": False, "success": False, "q_score": None,
                                  "errors": ["recorded execution integrity failure"]})
                continue
            current = _options(frame["options"])
            if tuple(map(len, current)) != tuple(map(len, initial)):
                raise ValueError("official goal option structure changed")
            success, q = verifier._score_options(current, initial)
            snapshots.append({"operational": True, "success": success, "q_score": q, "errors": []})
    elif route in {"official_projection", "canonical_fallback"}:
        from behavior_hl.embodied.evaluator_bridge.l0_verifier_adapter import ProjectedGoalVerifier
        spec = record["goal_spec"]
        if spec.get("route") != route:
            raise ValueError("record route and goal_spec route differ")
        verifier = ProjectedGoalVerifier(spec, _truth_reader(record["initial_truth"]),
                                         stability_window=record.get("stability_window", 1))
        snapshots = [verifier.snapshot(_truth_reader(frame["truth"]),
                                       done_requested=frame.get("done_requested", False))
                     for frame in frames]
    else:
        raise ValueError("unsupported canonical scoring route")
    canonical = dict(snapshots[-1])
    final = dict(canonical)
    process = None
    required = record.get("requires_process", False)
    if type(required) is not bool:
        raise ValueError("requires_process must be boolean")
    if required or "process" in record:
        if "process" not in record:
            process = {"operational": False, "public_process_satisfied": False,
                       "violations": [], "errors": ["required begin/step/end process evidence is missing"]}
        else:
            if process_directory is None:
                raise ValueError("process replay requires an explicit fresh output directory")
            process = replay_process(record["process"], process_directory)
        final["success"] = bool(canonical["success"] and process["public_process_satisfied"])
        if not process["operational"]:
            final.update(operational=False, failure_class="VERIFIER_ERROR")
        elif process["violations"] or (canonical["success"] and not process["public_process_satisfied"]):
            final["failure_class"] = "PUBLIC_PROCESS_REQUIREMENT_FAILED"
    return {"schema_version": "esi-benchmark/score-result-v1", "mode": "cpu-record-replay",
            "route": route, "truth_source": "supplied-records", "snapshots": snapshots,
            "canonical": canonical, "process": process, "final": final}
