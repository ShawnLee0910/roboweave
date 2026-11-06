"""Replay recorded predicate truth through the frozen CPU verifiers.

Records provide truth, not simulator geometry. This API cannot attest that a
record matches a scene; Live uses the original evaluator in the scene process.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import json
from types import SimpleNamespace
from typing import Any

from ..profiles import DEFAULT_PROFILE as PROFILE
from ..scoring import SCHEMA, SCHEMA_V2, canonical_record
from .bootstrap import activate


def _check_unary_truth(states, predicate, subject, destination, value):
    if (isinstance(predicate, str) and predicate in {"open", "closed", "toggled_on", "toggled_off"}
            and destination is None):
        # Native readback obtains each complementary pair from one boolean state.
        positive = "open" if predicate in {"open", "closed"} else "toggled_on"
        key = json.dumps([positive, subject], sort_keys=True)
        observed = value if predicate == positive else not value
        if key in states and states[key] != observed:
            raise ValueError("contradictory native state readbacks")
        states[key] = observed


def _truth_reader(rows):
    if not isinstance(rows, list):
        raise ValueError("truth must be a list of predicate records")
    values = {}
    unary_states = {}
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
        _check_unary_truth(unary_states, *key, row["value"])
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
    unary_states = {}
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"atom", "value"}
                or not isinstance(row["atom"], dict) or type(row["value"]) is not bool):
            raise ValueError("invalid process atom truth")
        key = json.dumps(row["atom"], sort_keys=True, separators=(",", ":"))
        if key in values:
            raise ValueError("duplicate process atom truth")
        values[key] = row["value"]
        atom = row["atom"]
        _check_unary_truth(unary_states, atom.get("predicate"), atom.get("subject_role"),
                           atom.get("object_role"), row["value"])
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
        from behavior_hl.embodied.evaluator_bridge import l1_am_process_programs
        program = requirements["process_program"]
        keys = {stage["milestone_id"] for stage in program["milestones"]}
        unary = any(k in program for k in getattr(l1_am_process_programs, "EXTENSIONS", ()))
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
            polarity = {"open": True, "closed": False, "toggled_on": True, "toggled_off": False}
            readback = l1_am_process_programs.MilestoneProgress(program,
                lambda atom: values["_unary"][atom["subject_role"]] == polarity[atom["predicate"]])
            if readback.read() != values:
                raise ValueError(f"{name} has contradictory milestone and unary state readbacks")


def _atom_roles(atom, roles):
    if (not isinstance(atom, dict) or not isinstance(atom.get("predicate"), str)
            or not atom["predicate"] or not isinstance(atom.get("subject_role"), str)
            or atom["subject_role"] not in roles
            or (atom.get("object_role") is not None and
                (not isinstance(atom["object_role"], str) or atom["object_role"] not in roles))):
        raise ValueError("process predicate atom has an invalid role or predicate identity")


def _replay_process_v2(record, directory, audit_class):
    """Reconstruct native snapshots from the truth available at each phase."""
    fields = {"requirements", "roles", "initial_truth", "final_truth", "events",
              "protected_relations", "protected_state_relations", "placement_targets"}
    if set(record) != fields:
        raise ValueError("v2 process requires the original audit's private relation lists")
    roles = set(record["roles"])
    requirements = record["requirements"]

    def readbacks(rows):
        values = _atom_truth(rows)
        for row in rows:
            _atom_roles(row["atom"], roles)
        return values

    for field in ("protected_relations", "protected_state_relations", "placement_targets"):
        atoms = record[field]
        if not isinstance(atoms, list):
            raise ValueError(f"{field} must be the original private predicate list")
        for atom in atoms:
            _atom_roles(atom, roles)
    if any(a["subject_role"] not in requirements["no_manipulation_roles"] for a in record["protected_relations"]):
        raise ValueError("protected relation identity differs from the protected roles")
    states = record["protected_state_relations"]
    if ({a["subject_role"] for a in states} != set(requirements.get("state_protection_roles", []))
            or any(a["predicate"] not in {"open", "closed", "toggled_on", "toggled_off"}
                   or a.get("object_role") is not None for a in states)):
        raise ValueError("protected state identity differs from the state protection contract")
    targets = record["placement_targets"]
    if (len(targets) != len(requirements["before_placement"])
            or any(a["subject_role"] != rule["subject_role"] for a, rule in
                   zip(targets, requirements["before_placement"]))):
        raise ValueError("placement target identity differs from the placement contract")
    initial = readbacks(record["initial_truth"])
    final = readbacks(record["final_truth"])
    events = record["events"]
    if not isinstance(events, list):
        raise ValueError("process events must be a list")
    evidence = []
    for ordinal, event in enumerate(events, 1):
        if (not isinstance(event, dict)
                or set(event) != {"event_id", "action", "before", "after", "status", "action_index"}
                or type(event["event_id"]) is not int or event["event_id"] != ordinal):
            raise ValueError("v2 process event identity/order must be consecutive from 1")
        if any(not isinstance(event[k], str) or not event[k].strip() for k in ("action", "status")):
            raise ValueError("process action and status must be nonempty strings")
        index = event["action_index"]
        if index is not None and (type(index) is not int or index < 0):
            raise ValueError("process action index must be nonnegative or null")
        phases = []
        for phase in ("before", "after"):
            snapshot = event[phase]
            if (not isinstance(snapshot, dict) or not {"held_roles", "truth"} <= set(snapshot)
                    or set(snapshot) - {"held_roles", "truth", "selected_roles"}):
                raise ValueError(f"v2 {phase} requires held_roles and predicate truth only")
            _role_list(snapshot["held_roles"], roles, phase + ".held_roles")
            if len(snapshot["held_roles"]) > 1:
                raise ValueError("v2 readback has more than one held object identity")
            if "selected_roles" in snapshot:
                _role_list(snapshot["selected_roles"], requirements["no_manipulation_roles"], phase + ".selected_roles")
                if phase != "before" or event["action"] != "pick":
                    raise ValueError("selected_roles is a pre-pick receipt")
            phases.append(readbacks(snapshot["truth"]))
        evidence.append(phases)

    directory = Path(directory)
    if directory.exists():
        raise ValueError("process replay directory must be fresh")
    objects = {role: object() for role in roles}
    current = initial
    phase_name = "initial"
    missing = []

    def truth(atom, _role_object):
        key = json.dumps(atom, sort_keys=True, separators=(",", ":"))
        if key not in current:
            missing.append(phase_name)
            raise ValueError(f"{phase_name} lacks required predicate truth")
        return current[key]

    audit = audit_class(requirements, role_object=objects.__getitem__, recipe_truth=truth,
                        protected_relations=record["protected_relations"],
                        protected_state_relations=states, placement_targets=targets, directory=directory)
    for event, phases in zip(events, evidence):
        snapshots = []
        for phase, values in zip(("before", "after"), phases):
            current = values
            phase_name = f"event {event['event_id']} {phase}"
            raw = event[phase]
            held = objects[raw["held_roles"][0]] if raw["held_roles"] else None
            runtime = SimpleNamespace(symbolic_primitives=SimpleNamespace(_get_obj_in_hand=lambda: held))
            snapshot = audit.snapshot(runtime)
            if snapshot is None or missing:
                raise ValueError(f"{phase_name} lacks valid required predicate truth")
            if "selected_roles" in raw:
                snapshot["selected_roles"] = list(raw["selected_roles"])
            snapshots.append(snapshot)
        # observe() re-reads history predicates after the post-action snapshot.
        audit.observe(action=event["action"], before=snapshots[0], after=snapshots[1],
                      status=event["status"], action_index=event["action_index"])
    current = final
    phase_name = "final"
    report = audit.report()
    if missing:
        raise ValueError(f"{missing[0]} lacks required predicate truth")
    return report


def replay_process(record: Mapping[str, Any], directory: Path, *, schema_version: str = SCHEMA,
                   profile: str = PROFILE) -> dict[str, Any]:
    """Replay existing begin/before/after/end evidence with the original audit."""
    activate(profile)
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
    if schema_version == SCHEMA_V2:
        return _replay_process_v2(record, directory, ProcessAudit)
    if schema_version != SCHEMA:
        raise ValueError("unsupported process record schema")
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


def score_record(record: Mapping[str, Any], *, process_directory: Path | None = None,
                 profile: str | None = None) -> dict[str, Any]:
    """Score supplied records without connecting to a simulator or model."""
    record = canonical_record(record)
    if profile is not None and "profile" in record and profile != record["profile"]:
        raise ValueError("score profile differs from the record profile")
    profile = record.get("profile", PROFILE) if profile is None else profile
    if not isinstance(profile, str) or not profile:
        raise ValueError("score profile must be a nonempty string")
    activate(profile)
    frames = record.get("snapshots")
    if not isinstance(frames, list) or not frames or any(not isinstance(f, dict) for f in frames):
        raise ValueError("at least one snapshot is required")
    route = record.get("route")
    snapshots = []
    if route == "official_full":
        from behavior_hl.embodied.evaluator_bridge.official_predicate_verifier import OfficialPredicateVerifier
        initial = _options(record["initial_options"])
        verifier = OfficialPredicateVerifier(require_single_option=len(initial) == 1)
        integrity_ok = True
        for frame in frames:
            integrity_ok = integrity_ok and frame.get("integrity_ok", True) is True
            if not integrity_ok:
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
            process = replay_process(record["process"], process_directory,
                                     schema_version=record["schema_version"], profile=profile)
        final["success"] = bool(canonical["success"] and process["public_process_satisfied"])
        if not process["operational"]:
            final.update(operational=False, failure_class="VERIFIER_ERROR")
        elif process["violations"] or (canonical["success"] and not process["public_process_satisfied"]):
            final["failure_class"] = "PUBLIC_PROCESS_REQUIREMENT_FAILED"
    return {"schema_version": "roboweave/score-result-v1", "mode": "cpu-record-replay",
            "route": route, "truth_source": "supplied-records", "snapshots": snapshots,
            "canonical": canonical, "process": process, "final": final}
