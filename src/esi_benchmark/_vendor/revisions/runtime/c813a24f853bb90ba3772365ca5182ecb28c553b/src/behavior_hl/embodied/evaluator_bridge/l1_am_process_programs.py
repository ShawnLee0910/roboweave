"""Validation and lowering for finite, explicitly ordered L1 milestones.

The resulting program is private runtime input.  A milestone is latched only
after a distinct verified execution event; list order never creates an
implicit scoring constraint.
"""
from copy import deepcopy
import hashlib
import json

SCHEMA = "l1-public-process-program/v1"
CAPABILITY = "L1_ORDERED_MILESTONES_V1"
_BINARY = {"ontop", "inside", "under", "nextto", "next_to"}
_UNARY = {"open", "closed", "toggled_on", "toggled_off"}
_OPPOSITE = {"open": "closed", "closed": "open", "toggled_on": "toggled_off", "toggled_off": "toggled_on"}
_OTHER = {"no_manipulation_roles", "state_protection_roles", "before_placement",
          "reference_paths_are_not_scoring_rules", "reference_strategies",
          "identity_interchangeability", "public_requirements", "executor_boundary"}


def validate_runtime_program(program, roles):
    """The implemented Oracle path handles finite unary milestones only."""
    fields = {'schema_version', 'milestones', 'final_requirements', 'final_placement_after',
              'completion_rule', 'condition_evaluation', 'milestone_memory',
              'source_process', 'public_instruction_evidence'}
    if not isinstance(program, dict) or set(program) != fields or program['schema_version'] != SCHEMA:
        raise ValueError('unsupported milestone program schema')
    if (program['final_placement_after'] is not None
            or program['completion_rule'] != 'ALL_DECLARED_MILESTONES_AND_CURRENT_FINAL_REQUIREMENTS'
            or program['condition_evaluation'] != 'CANONICAL_READBACK_AFTER_VERIFIED_EXECUTION_OR_INITIAL_READBACK'
            or program['milestone_memory'] != 'LATCH_ONCE_AFTER_DECLARED_PREDECESSORS_SATISFIED'):
        raise ValueError('unsupported milestone program semantics')
    evidence = program['public_instruction_evidence']
    if not isinstance(evidence, dict) or not evidence or any(not isinstance(v, str) or not v.strip() for v in evidence.values()):
        raise ValueError('milestone public instructions missing')
    milestones = program['milestones']
    if not isinstance(milestones, list) or len(milestones) != 1:
        raise ValueError('runtime supports one unary intermediate milestone before final requirements')
    known = set()
    folded = set()
    for stage in milestones:
        if not isinstance(stage, dict) or set(stage) != {'milestone_id', 'after', 'requirements'}:
            raise ValueError('invalid milestone fields')
        name, after, atoms = stage['milestone_id'], stage['after'], stage['requirements']
        if (not isinstance(name, str) or not name or name.casefold() in folded
                or not isinstance(after, list) or len(after) != len(set(after))
                or any(not isinstance(dep, str) or dep not in known for dep in after)
                or not isinstance(atoms, list) or not 0 < len(atoms) <= 64):
            raise ValueError('invalid ordered milestone identity or dependencies')
        known.add(name)
        folded.add(name.casefold())
        for atom in atoms:
            _atom(atom, set(roles), 'milestones')
            if atom['predicate'] not in _UNARY:
                raise ValueError('milestone runtime currently requires unary states')
        _consistent(atoms, 'milestones')
    final = program['final_requirements']
    if not isinstance(final, list) or not final:
        raise ValueError('missing milestone final requirements')
    for atom in final:
        _atom(atom, set(roles), 'final_requirements')
    _consistent(final, 'final_requirements')


def validate_program_final(program, goal_clauses):
    def atom_key(atom):
        return (atom['predicate'].replace('next_to', 'nextto'), atom['subject_role'],
                atom.get('object_role'), atom.get('desired', True))
    if {atom_key(a) for a in program['final_requirements']} != {atom_key(a) for a in goal_clauses}:
        raise ValueError('milestone final requirements differ from canonical goals')


class MilestoneProgress:
    def __init__(self, program, truth):
        self.program, self.truth = program, truth
        self.reached = {}

    def read(self):
        return {stage['milestone_id']: all(self.truth(a) for a in stage['requirements'])
                for stage in self.program['milestones']}

    def observe(self, values, event_id):
        # Predecessors must have been established before this event.  One
        # readback cannot invent several ordered historical transitions.
        prior = set(self.reached)
        for stage in self.program['milestones']:
            name = stage['milestone_id']
            if name not in prior and set(stage['after']) <= prior and values[name]:
                self.reached[name] = event_id

    @property
    def complete(self):
        return len(self.reached) == len(self.program['milestones'])

    def execute_pending(self, execute_unary):
        """Use the existing phase's executor and budgets, with live stage checks."""
        actions = {'open': 'open', 'closed': 'close', 'toggled_on': 'toggle_on', 'toggled_off': 'toggle_off'}
        for stage in self.program['milestones']:
            name = stage['milestone_id']
            if name in self.reached:
                continue
            if not set(stage['after']) <= set(self.reached):
                raise RuntimeError('milestone predecessor has not been verified')
            for atom in stage['requirements']:
                if not self.truth(atom):
                    execute_unary(actions[atom['predicate']], atom['subject_role'])
            if name not in self.reached:
                raise RuntimeError('milestone lacks a verified execution/readback receipt')


class ProcessCompletionVerifier:
    """Keep raw canonical Q/leaves intact; defer termination until milestones."""
    def __init__(self, canonical, progress):
        self.canonical, self.progress = canonical, progress

    def evaluate(self, episode, evaluator):
        from dataclasses import replace
        snapshot = self.canonical.evaluate(episode, evaluator)
        return replace(snapshot, official_success=snapshot.official_success and self.progress.complete,
                       research_diagnostics={**dict(snapshot.research_diagnostics),
                           'canonical_final_success': snapshot.official_success,
                           'milestones_reached': dict(self.progress.reached)})


class UnsupportedProcessProgram(ValueError):
    def __init__(self, code, path, detail):
        self.code, self.path, self.detail = code, path, detail
        super().__init__(f"{code} at {path}: {detail}")


def _require(ok, code, path, detail):
    if not ok:
        raise UnsupportedProcessProgram(code, path, detail)


def _atom(value, roles, path):
    _require(isinstance(value, dict), "INVALID_PROCESS_ATOM", path, "expected object")
    predicate = value.get("predicate")
    allowed = {"predicate", "subject_role", "desired"} | ({"object_role"} if predicate in _BINARY else set())
    _require(predicate in _BINARY | _UNARY and not set(value) - allowed and value.get("desired", True) is True,
             "UNSUPPORTED_PROCESS_ATOM", path, value)
    _require(value.get("subject_role") in roles and (predicate not in _BINARY or value.get("object_role") in roles),
             "UNKNOWN_PROCESS_ROLE", path, value)
    return dict(value, desired=True)


def _consistent(atoms, path):
    for atom in atoms:
        opposite = _OPPOSITE.get(atom["predicate"])
        _require(not opposite or not any(other["subject_role"] == atom["subject_role"] and other["predicate"] == opposite for other in atoms),
                 "CONTRADICTORY_MILESTONE_STATE", path, atoms)
    return atoms


def _condition(value, roles, path):
    _require(isinstance(value, dict), "INVALID_MILESTONE_CONDITION", path, "expected object")
    if "predicate" in value:
        return [_atom(value, roles, path)]
    if value.get("op") == "AND":
        _require(set(value) == {"op", "args"} and isinstance(value["args"], list) and value["args"],
                 "INVALID_MILESTONE_AND", path, value)
        return _consistent([a for i, child in enumerate(value["args"]) for a in _condition(child, roles, f"{path}/args/{i}")], path)
    if value.get("op") in {"BINDING_EXISTS", "MAP_LOOKUP", "RELATION_MAP"}:
        raise UnsupportedProcessProgram("DYNAMIC_EVENT_BINDING_REQUIRED", path, value)
    raise UnsupportedProcessProgram("FINITE_CONDITION_CAPABILITY_REQUIRED", path, value)


def _topological(items):
    by_id = {m["milestone_id"]: m for m in items}
    for m in items:
        for dep in m["after"]:
            _require(dep in by_id, "UNKNOWN_MILESTONE_DEPENDENCY", m["milestone_id"], dep)
    done, ordered = set(), []
    while len(done) < len(items):
        ready = [m for m in items if m["milestone_id"] not in done and set(m["after"]) <= done]
        _require(ready, "CYCLIC_MILESTONES", "ordered_milestones", "no acyclic next stage")
        ordered.extend(ready)
        done.update(m["milestone_id"] for m in ready)
    return ordered


def lower_process_program(process, *, final_goals, roles, public_instructions):
    _require(isinstance(process, dict) and not set(process) - (_OTHER | {"ordered_milestones", "final_placement_after"}),
             "UNKNOWN_PROCESS_FIELD", "goal.process", process)
    _require(isinstance(public_instructions, dict) and public_instructions and
             all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in public_instructions.items()),
             "PUBLIC_PROCESS_DISCLOSURE_MISSING", "public_instructions", "actual delivered instructions required")
    roles = set(roles)
    _require(roles and all(isinstance(r, str) and r for r in roles), "INVALID_PROCESS_ROLES", "roles", roles)
    _require(isinstance(final_goals, list) and final_goals, "MISSING_FINAL_REQUIREMENTS", "final_goals", final_goals)
    final = _consistent([_atom(a, roles, "final_goals") for a in final_goals], "final_goals")
    declared = process.get("ordered_milestones")
    _require(isinstance(declared, list) and declared, "MISSING_TYPED_MILESTONES", "goal.process", "ordered milestones required")
    milestones, ids = [], set()
    for index, source in enumerate(declared):
        path = f"ordered_milestones/{index}"
        _require(isinstance(source, dict) and not set(source) - {"milestone_id", "condition", "after"}, "UNKNOWN_MILESTONE_FIELD", path, source)
        identity = source.get("milestone_id")
        _require(isinstance(identity, str) and identity and identity.casefold() not in ids and identity != "__final_requirements__", "DUPLICATE_OR_INVALID_MILESTONE", path, identity)
        ids.add(identity.casefold())
        after = source.get("after", [])
        _require(isinstance(after, list) and len(after) == len(set(after)) and all(isinstance(v, str) and v for v in after), "INVALID_MILESTONE_DEPENDENCIES", path, after)
        _require(index == 0 or "after" in source, "MILESTONE_ORDER_NOT_EXPLICIT", path, "explicit after is required")
        milestones.append({"milestone_id": identity, "after": deepcopy(after), "requirements": _condition(source.get("condition"), roles, path + "/condition")})
    milestones = _topological(milestones)
    gate = deepcopy(process.get("final_placement_after"))
    if gate is not None:
        _require(isinstance(gate, dict) and set(gate) == {"roles", "milestone_id", "scope"}, "UNKNOWN_FINAL_PLACEMENT_GATE", "final_placement_after", gate)
        _require(gate["scope"] == "DECLARED_FINAL_GOAL_RELATIONS", "UNSUPPORTED_PLACEMENT_GATE_SCOPE", "final_placement_after", gate)
        _require(isinstance(gate["roles"], list) and gate["roles"] and len(gate["roles"]) == len(set(gate["roles"])) and all(r in roles for r in gate["roles"]), "INVALID_PLACEMENT_GATE_ROLES", "final_placement_after", gate["roles"])
        _require(gate["milestone_id"] in ids, "UNKNOWN_PLACEMENT_MILESTONE", "final_placement_after", gate["milestone_id"])
        _require(all(any(a["subject_role"] == r and a["predicate"] in _BINARY for a in final) for r in gate["roles"]), "MISSING_GATED_FINAL_RELATION", "final_placement_after", gate["roles"])
    program = {"schema_version": SCHEMA, "milestones": milestones, "final_requirements": final,
               "final_placement_after": gate, "completion_rule": "ALL_DECLARED_MILESTONES_AND_CURRENT_FINAL_REQUIREMENTS",
               "condition_evaluation": "CANONICAL_READBACK_AFTER_VERIFIED_EXECUTION_OR_INITIAL_READBACK",
               "milestone_memory": "LATCH_ONCE_AFTER_DECLARED_PREDECESSORS_SATISFIED",
               "source_process": deepcopy(process), "public_instruction_evidence": deepcopy(public_instructions)}
    digest = hashlib.sha256(json.dumps(program, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return {"process_program": program, "process_program_sha256": digest,
            "stage_goals": [{"stage_id": m["milestone_id"], "after": deepcopy(m["after"]), "goal": {"operator": "AND", "clauses": deepcopy(m["requirements"])}} for m in milestones] +
                           [{"stage_id": "__final_requirements__", "after": [m["milestone_id"] for m in milestones], "goal": {"operator": "AND", "clauses": deepcopy(final)}}],
            "remaining_process_requirements": {k: deepcopy(v) for k, v in process.items() if k not in {"ordered_milestones", "final_placement_after"}},
            "required_runtime_capabilities": [CAPABILITY], "public_disclosure_status": "REQUIRES_ADAPTER_REVIEW_NOT_INFERRED_FROM_TEXT", "initial_state_materialized": False}
