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
EXTENSIONS = {'global_unary_phases': 'L1_GLOBAL_UNARY_PHASES_V1',
              'exact_cycles_per_role': 'L1_EXACT_CYCLE_LEDGER_V1',
              'cycle_order_final': 'L1_CYCLE_ORDER_FINAL_V1'}
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
    if not isinstance(program, dict) or not fields <= set(program) or set(program) - fields - {'last_cycle_final'} - set(EXTENSIONS) or program['schema_version'] != SCHEMA:
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
    if not isinstance(milestones, list) or not 1 <= len(milestones) <= 64:
        raise ValueError('runtime supports one milestone or at most 64 unary transition-ledger stages')
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
    if not isinstance(final, list) or not final and 'cycle_order_final' not in program:
        raise ValueError('missing milestone final requirements')
    for atom in final:
        _atom(atom, set(roles), 'final_requirements')
    _consistent(final, 'final_requirements')
    if 'global_unary_phases' in program:
        if program['global_unary_phases'] is not True or any(k in program for k in ('exact_cycles_per_role', 'cycle_order_final', 'last_cycle_final')):
            raise ValueError('global unary phases cannot mix with cycle ledgers')
        previous = None
        for stage in milestones:
            atoms = stage['requirements']
            if len({a['subject_role'] for a in atoms}) != len(atoms):
                raise ValueError('global phase requires one state per role')
            if stage['after'] != ([] if previous is None else [previous['milestone_id']]):
                raise ValueError('global phases require an explicit linear phase chain')
            if previous:
                old = {a['subject_role']: a['predicate'] for a in previous['requirements']}
                if {a['subject_role']: _OPPOSITE[a['predicate']] for a in atoms} != old:
                    raise ValueError('global phases must reverse the same declared devices')
            previous = stage
    elif len(milestones) > 1:
        # Finite per-device cycle ledger, not arbitrary stage composition.
        previous = {}
        for stage in milestones:
            if len(stage['requirements']) != 1:
                raise ValueError('transition ledger requires one unary atom per stage')
            atom = stage['requirements'][0]
            role = atom['subject_role']
            old = previous.get(role)
            if stage['after'] != ([] if old is None else [old['milestone_id']]):
                raise ValueError('transition ledger must depend only on its own prior device stage')
            if old and _OPPOSITE[old['requirements'][0]['predicate']] != atom['predicate']:
                raise ValueError('transition ledger requires alternating device states')
            previous[role] = stage
    if 'exact_cycles_per_role' in program:
        counts = program['exact_cycles_per_role']
        chains = {}
        for stage in milestones:
            atom = stage['requirements'][0]
            chains.setdefault(atom['subject_role'], []).append(atom)
        if (not isinstance(counts, dict) or set(counts) != set(chains)
                or any(type(n) is not int or n < 1 or len(chains[r]) != 2*n for r, n in counts.items())):
            raise ValueError('exact cycle counts must match every complete device transition chain')
    if 'cycle_order_final' in program:
        from .l1_last_cycle_final import validate_last_cycle_final
        rule = program['cycle_order_final']
        if not isinstance(rule, dict) or set(rule) != {'cycle_end_milestones', 'active_predicate', 'neutral_predicate', 'ordinal'}:
            raise ValueError('invalid cycle-order final fields')
        ordinal = rule['ordinal']
        if not (ordinal == 'last' or type(ordinal) is int and 1 <= ordinal <= len(rule['cycle_end_milestones'])):
            raise ValueError('cycle-order ordinal outside completed event set')
        validate_last_cycle_final({k:v for k,v in rule.items() if k != 'ordinal'}, milestones)
        if 'last_cycle_final' in program or 'exact_cycles_per_role' not in program:
            raise ValueError('cycle-order final requires exact ledger and one final selector')
        checked = set(program['exact_cycles_per_role'])
        if any(a['subject_role'] in checked and a['predicate'] in _UNARY for a in final):
            raise ValueError('static final state must not freeze the dynamic cycle-order answer')
    if 'last_cycle_final' in program:
        from .l1_last_cycle_final import validate_last_cycle_final
        validate_last_cycle_final(program['last_cycle_final'], milestones)
        checked = {a['subject_role'] for stage in milestones for a in stage['requirements']}
        if any(a['subject_role'] in checked and a['predicate'] in _UNARY for a in final):
            raise ValueError('static final state must not freeze the dynamic last-cycle answer')


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
        self.last_event = 0
        self.violations = []
        self.cycles = {r: 0 for r in program.get('exact_cycles_per_role', {})}
        self.cycle_started = set()

    def read(self):
        values = {stage['milestone_id']: all(self.truth(a) for a in stage['requirements'])
                  for stage in self.program['milestones']}
        if any(k in self.program for k in EXTENSIONS):
            values['_unary'] = {a['subject_role']: bool(self.truth(dict(a, predicate=
                'open' if a['predicate'] in {'open','closed'} else 'toggled_on')))
                for stage in self.program['milestones'] for a in stage['requirements']}
        return values

    def observe(self, values, event_id, before=None, action=None):
        # Predecessors must have been established before this event.  One
        # readback cannot invent several ordered historical transitions.
        if self.program.get('global_unary_phases') and event_id == 0 and not self.reached:
            first = self.program['milestones'][0]
            if values.get(first['milestone_id']) is True:
                self.reached[first['milestone_id']] = 0
            return
        prior = set(self.reached)
        ledger = len(self.program['milestones']) > 1 or any(k in self.program for k in EXTENSIONS)
        if ledger:
            if type(event_id) is not int or event_id <= self.last_event or before is None:
                return
            self.last_event = event_id
        if any(k in self.program for k in EXTENSIONS):
            if before is None or event_id <= 0:
                return
            transitions = {r:(old, values.get('_unary', {}).get(r)) for r,old in before.get('_unary', {}).items()
                if type(old) is bool and type(values.get('_unary', {}).get(r)) is bool and old != values['_unary'][r]}
            if len(transitions) > 1:
                self.violations.append(dict(event_id=event_id,kind='AMBIGUOUS_MULTI_DEVICE_TRANSITION'))
                return
            for role, (_, new) in transitions.items():
                atoms = [a for s in self.program['milestones'] for a in s['requirements'] if a['subject_role']==role]
                positive = 'open' if atoms[0]['predicate'] in {'open','closed'} else 'toggled_on'
                predicate = positive if new else _OPPOSITE[positive]
                expected = {'open':'open','closed':'close','toggled_on':'toggle_on','toggled_off':'toggle_off'}[predicate]
                if action != expected:
                    self.violations.append(dict(event_id=event_id,kind='UNVERIFIED_UNARY_TRANSITION',subject_role=role))
                    continue
                if self.program.get('global_unary_phases'):
                    pending = next((s for s in self.program['milestones'] if s['milestone_id'] not in prior), None)
                    if pending and any(a['subject_role']==role and a['predicate']!=predicate for a in pending['requirements']):
                        self.violations.append(dict(event_id=event_id,kind='PREMATURE_UNARY_PHASE',subject_role=role))
                if role in self.cycles:
                    if predicate == atoms[0]['predicate']:
                        if (self.cycles[role] >= self.program['exact_cycles_per_role'][role]
                                and len(prior) < len(self.program['milestones'])):
                            self.violations.append(dict(event_id=event_id,kind='PREMATURE_CYCLE_FINAL'
                                if 'cycle_order_final' in self.program else 'EXACT_CYCLE_LIMIT_EXCEEDED',subject_role=role))
                        self.cycle_started.add(role)
                    elif role in self.cycle_started:
                        self.cycle_started.remove(role)
                        self.cycles[role] += 1
                        if self.cycles[role] > self.program['exact_cycles_per_role'][role]:
                            self.violations.append(dict(event_id=event_id,kind='EXACT_CYCLE_LIMIT_EXCEEDED',subject_role=role))
        for stage in self.program['milestones']:
            name = stage['milestone_id']
            if self.program.get('global_unary_phases') and (before.get(name) is not False or not transitions):
                continue
            if ledger and not self.program.get('global_unary_phases'):
                expected = {'open':'open', 'closed':'close', 'toggled_on':'toggle_on', 'toggled_off':'toggle_off'}[stage['requirements'][0]['predicate']]
                if before.get(name) is not False or action != expected:
                    continue
            if name not in prior and set(stage['after']) <= prior and values[name]:
                self.reached[name] = event_id

    @property
    def complete(self):
        if self.violations:
            return False
        if self.cycles != self.program.get('exact_cycles_per_role', {}):
            return False
        if len(self.reached) != len(self.program['milestones']):
            return False
        atoms = self.dynamic_final_atoms()
        return atoms is not None and all(self.truth(a) for a in atoms)

    def dynamic_final_atoms(self):
        if 'cycle_order_final' in self.program:
            rule = self.program['cycle_order_final']
            names = rule['cycle_end_milestones']
            events = [self.reached.get(n) for n in names]
            if any(type(e) is not int or e <= 0 for e in events) or len(set(events)) != len(events):
                return None
            ordered = sorted(names, key=self.reached.__getitem__)
            selected = ordered[-1 if rule['ordinal']=='last' else rule['ordinal']-1]
            by_name = {s['milestone_id']:s for s in self.program['milestones']}
            return [dict(predicate=rule['active_predicate'] if n==selected else rule['neutral_predicate'],
                subject_role=by_name[n]['requirements'][0]['subject_role'],desired=True) for n in names]
        if 'last_cycle_final' not in self.program:
            return []
        from .l1_last_cycle_final import last_cycle_final_atoms
        return last_cycle_final_atoms(self.program['last_cycle_final'], self.program['milestones'], self.reached)

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
        atoms = self.dynamic_final_atoms()
        if atoms is None:
            raise RuntimeError('last-cycle final lacks complete verified history')
        for atom in atoms:
            if not self.truth(atom):
                execute_unary(actions[atom['predicate']], atom['subject_role'])
        if not self.complete:
            raise RuntimeError('dynamic last-cycle final state was not achieved')


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
                           'milestones_reached': dict(self.progress.reached),
                           **({'process_violations': list(self.progress.violations),
                               'verified_cycle_counts': dict(self.progress.cycles),
                               'cycle_order_final_requirements': self.progress.dynamic_final_atoms()}
                              if any(k in self.progress.program for k in EXTENSIONS) else {}),
                           **({'last_cycle_final_requirements': self.progress.dynamic_final_atoms()}
                              if 'last_cycle_final' in self.progress.program else {})})


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
    _require(isinstance(process, dict) and not set(process) - (_OTHER | {"ordered_milestones", "final_placement_after", "last_cycle_final"} | set(EXTENSIONS)),
             "UNKNOWN_PROCESS_FIELD", "goal.process", process)
    _require(isinstance(public_instructions, dict) and public_instructions and
             all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in public_instructions.items()),
             "PUBLIC_PROCESS_DISCLOSURE_MISSING", "public_instructions", "actual delivered instructions required")
    roles = set(roles)
    _require(roles and all(isinstance(r, str) and r for r in roles), "INVALID_PROCESS_ROLES", "roles", roles)
    _require(isinstance(final_goals, list) and (final_goals or 'cycle_order_final' in process), "MISSING_FINAL_REQUIREMENTS", "final_goals", final_goals)
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
    if 'last_cycle_final' in process:
        program['last_cycle_final'] = deepcopy(process['last_cycle_final'])
    for key in EXTENSIONS:
        if key in process:
            program[key] = deepcopy(process[key])
    digest = hashlib.sha256(json.dumps(program, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return {"process_program": program, "process_program_sha256": digest,
            "stage_goals": [{"stage_id": m["milestone_id"], "after": deepcopy(m["after"]), "goal": {"operator": "AND", "clauses": deepcopy(m["requirements"])}} for m in milestones] +
                           [{"stage_id": "__final_requirements__", "after": [m["milestone_id"] for m in milestones], "goal": {"operator": "AND", "clauses": deepcopy(final)}}],
            "remaining_process_requirements": {k: deepcopy(v) for k, v in process.items() if k not in {"ordered_milestones", "final_placement_after", "last_cycle_final"} | set(EXTENSIONS)},
            "required_runtime_capabilities": [CAPABILITY] + [v for k,v in EXTENSIONS.items() if k in process], "public_disclosure_status": "REQUIRES_ADAPTER_REVIEW_NOT_INFERRED_FROM_TEXT", "initial_state_materialized": False}
