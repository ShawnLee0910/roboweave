"""Episode-scoped L2 V sidecars; v1 remains the original MV canary."""
from contextvars import ContextVar
from copy import deepcopy
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path


ACTIVE = ContextVar('l2_mv_session', default=None)
SCHEMA = 'action20.l2-mv-sidecar/v1'
V2_SCHEMA = 'action20.l2-v-sidecar/v2'
_UNARY = frozenset({'open', 'closed', 'toggled_on', 'toggled_off'})
_BINARY = frozenset({'ontop', 'inside', 'under', 'nextto', 'next_to'})
_ACTIONS = frozenset({'open', 'close', 'toggle_on', 'toggle_off'})


def read_ref(ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('invalid MV reference')
    path = Path(ref['path'])
    if not path.is_absolute() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('invalid MV reference path')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('MV reference hash mismatch')
    return json.loads(raw)


def _base_member(config, spec, *, v2=False):
    base = read_ref(config['base_package'])
    specs_path = Path(base['task_specs']['path'])
    if hashlib.sha256(specs_path.read_bytes()).hexdigest() != base['task_specs']['sha256']:
        raise ValueError('Base TaskSpec hash mismatch')
    matches = [json.loads(line) for line in specs_path.read_text().splitlines()
               if json.loads(line)['task_spec_id'] == spec['task_spec_id']]
    if len(matches) != 1:
        raise ValueError('MV Base membership mismatch')
    prior = matches[0]
    composition = prior.get('composition_contract', {})
    axes = composition.get('axes')
    if not v2:
        if axes != ['M', 'V'] or composition.get('v_stage') != 'BASE':
            raise ValueError('MV source must be an MV Base, not D0/D1')
    elif config['execution_mode'] == 'Physical':
        if axes != ['M', 'P'] or not isinstance(prior.get('physical_template'), dict):
            raise ValueError('Physical V source must be an MP Base with a physical template')
        if prior.get('physical_template') != spec.get('physical_template'):
            raise ValueError('Physical V changed Base physical template')
    elif axes == ['A', 'M']:
        if spec.get('composition_contract', {}).get('axes') not in (['A','M'], ['A','M','V']):
            raise ValueError('AM Base requires AMV derived composition')
    elif axes != ['M', 'V'] or composition.get('v_stage') != 'BASE':
        raise ValueError('Symbolic V source must be an MV Base, not D0/D1')
    for key in ('task_contract', 'goal_contract', 'role_bindings', 'construction_contract',
                'private_runtime', 'public_instruction', 'public_instruction_zh'):
        if prior.get(key) != spec.get(key):
            raise ValueError('MV changed Base ' + key)
    return base, prior


def _validate_receipt(config, prior):
    receipt = read_ref(config['base_receipt'])
    if receipt.get('privileged_validation') or receipt.get('promotable') is False:
        raise ValueError('test-only or privileged evidence cannot promote a V package')
    if (receipt.get('task_spec_id') != config['task_spec_id'] or receipt.get('status') != 'PASS'
            or type(receipt.get('official_q')) not in (int, float) or receipt['official_q'] != 1
            or receipt.get('verification_completed') is not True
            or receipt.get('seed') != config['seed']
            or not receipt.get('public_process_report', {}).get('public_process_satisfied')):
        raise ValueError('MV needs a fresh matching full Base PASS/Q1 receipt')
    expected = hashlib.sha256(json.dumps(prior['task_contract'], sort_keys=True,
                                         separators=(',', ':')).encode()).hexdigest()
    identity = receipt.get('l2_base_identity', {})
    if (identity.get('stage') != 'BASE' or identity.get('process_contract_sha256') != expected
            or identity.get('public_instruction_sha256') != hashlib.sha256(
                prior['public_instruction'].encode()).hexdigest()):
        raise ValueError('Base receipt contract/instruction identity mismatch')
    if (config.get('execution_mode') == 'Physical'
            and receipt.get('physical_template_review', {}).get('qualification_status') != 'PASS'):
        raise ValueError('Physical V needs a Base physical template review PASS')


def _validate_v1(config, spec):
    required = {'schema_version', 'branch', 'task_spec_id', 'trigger', 'target_role', 'predicate',
                'base_package', 'base_receipt', 'seed', 'execution_mode', 'checkpoint_policy'}
    if (set(config) != required or config['schema_version'] != SCHEMA
            or config['branch'] not in {'D0', 'D1'}
            or config['task_spec_id'] != spec['task_spec_id']
            or config['trigger'] != 'AFTER_FULL_BASE_SUCCESS'
            or config['predicate'] != 'open' or config['execution_mode'] != 'Symbolic'
            or config['checkpoint_policy'] != 'NO_MID_EPISODE_RESTORE'
            or type(config['seed']) is not int):
        raise ValueError('unsupported MV sidecar')
    base, prior = _base_member(config, spec)
    _validate_receipt(config, prior)
    if base.get('requested_execution_policy', {}).get('action_mode') != 'Symbolic':
        raise ValueError('Physical V not supported')
    contract = spec['task_contract']
    process = contract['process_requirements']
    program = process.get('process_program')
    # First canary supports latched unary milestones; exact-cycle recovery needs v2.
    if (not program or process.get('history_program') or program.get('exact_cycles_per_role')
            or process.get('no_manipulation_roles') or process.get('state_protection_roles')):
        raise ValueError('MV supports only unprotected unary milestone Base')
    final = program['final_requirements']
    target = config['target_role']
    if any(atom['predicate'] not in {'open', 'closed'} for atom in final):
        raise ValueError('MV door goals only')
    if not any(atom['predicate'] == 'open' and atom['subject_role'] == target for atom in final):
        raise ValueError('target must restore open state')
    if not any(atom['predicate'] == 'open' and atom['subject_role'] == target
               for atom in contract['initial_assertions']):
        raise ValueError('target must initially be open')
    return config


def _roles(spec):
    bindings = spec.get('role_bindings', {})
    roles = set(bindings) if isinstance(bindings, dict) else set()
    roles.update(spec.get('private_runtime', {}).get('role_to_scope', {}))
    return roles


def _milestones(spec):
    requirements = spec.get('task_contract', {}).get('process_requirements', {})
    program = requirements.get('process_program', {})
    history = requirements.get('history_program', {}).get('process', {})
    return {stage['milestone_id'] for stage in program.get('milestones', [])} | {
        stage['milestone_id'] for stage in history.get('ordered_milestones', [])
    }


def _history_bindings(spec):
    program = spec.get('task_contract', {}).get('process_requirements', {}).get('history_program', {})
    return {binding['binding_id']: binding for binding in program.get('bindings', [])}


def _validate_trigger(value, spec):
    if value == 'AFTER_FULL_BASE_SUCCESS':
        return
    if not isinstance(value, dict) or set(value) != {'kind', 'milestone_id'}:
        raise ValueError('invalid V2 trigger')
    if (value['kind'] != 'AFTER_PROCESS_MILESTONE' or not isinstance(value['milestone_id'], str)
            or value['milestone_id'] not in _milestones(spec)):
        raise ValueError('V2 trigger must name one declared process milestone')


def _validate_atom(atom, roles, *, bindings):
    if not isinstance(atom, dict):
        raise ValueError('invalid V2 recovery atom')
    predicate = atom.get('predicate')
    allowed = {'predicate', 'subject_role', 'subject_ref', 'desired'}
    if predicate in _BINARY:
        allowed |= {'object_role', 'object_ref'}
    if (predicate not in _UNARY | _BINARY or set(atom) - allowed
            or atom.get('desired', True) is not True
            or ('subject_role' in atom) == ('subject_ref' in atom)
            or (predicate in _BINARY and ('object_role' in atom) == ('object_ref' in atom))):
        raise ValueError('invalid V2 recovery atom')
    subject = atom.get('subject_role')
    obj = atom.get('object_role')
    if subject is not None and subject not in roles:
        raise ValueError('unknown V2 recovery subject')
    if obj is not None and obj not in roles:
        raise ValueError('unknown V2 recovery object')
    subject_ref = atom.get('subject_ref')
    if subject_ref is not None:
        if set(subject_ref) != {'binding_ref'} or subject_ref['binding_ref'] not in bindings:
            raise ValueError('unknown V2 dynamic recovery subject')
    object_ref = atom.get('object_ref')
    if object_ref is not None:
        direct = set(object_ref) == {'binding_ref'} and object_ref['binding_ref'] in bindings
        mapped = (set(object_ref) == {'map_ref', 'key_role'} and isinstance(object_ref['map_ref'], dict)
                  and set(object_ref['map_ref']) == {'binding_ref'}
                  and object_ref['map_ref']['binding_ref'] in bindings
                  and object_ref['key_role'] in roles)
        if not direct and not mapped:
            raise ValueError('unknown V2 dynamic recovery object')


def _validate_recovery_conditions(value, spec, effect):
    if value is None:
        if effect['kind'] == 'RESTORE_STAGING_POSE':
            raise ValueError('staging V effect requires explicit recovery conditions')
        return
    if not isinstance(value, list) or not value or len(value) > 16:
        raise ValueError('invalid V2 recovery conditions')
    bindings = _history_bindings(spec)
    for atom in value:
        _validate_atom(atom, _roles(spec), bindings=bindings)


def _validate_guard(value, spec):
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != {'before_agent_action', 'subject_role', 'requires_current'}:
        raise ValueError('invalid V2 continuation guard')
    if (value['before_agent_action'] not in _ACTIONS or value.get('subject_role') not in _roles(spec)
            or not isinstance(value['requires_current'], list)):
        raise ValueError('invalid V2 continuation guard')
    if not value['requires_current']:
        raise ValueError('V2 continuation guard needs a current precondition')
    for atom in value['requires_current']:
        if not isinstance(atom, dict) or set(atom) != {'predicate', 'subject_role', 'desired'}:
            raise ValueError('invalid V2 continuation precondition')
        if atom['predicate'] not in _UNARY or atom.get('desired') is not True or atom['subject_role'] not in _roles(spec):
            raise ValueError('invalid V2 continuation precondition')


def validate_v2_plan(config, spec):
    """Validate a draft V2 sidecar without granting it runnable-package status."""
    return _validate_v2(config, spec, require_receipt=False)


def _validate_v2(config, spec, *, require_receipt):
    required = {'schema_version', 'branch', 'task_spec_id', 'trigger', 'effect', 'base_package',
                'base_receipt', 'seed', 'execution_mode', 'checkpoint_policy'}
    optional = {'continuation_guard', 'recovery_conditions', 'validation_mode', 'preserved_conditions'}
    if (not isinstance(config, dict) or set(config) - optional != required
            or config['schema_version'] != V2_SCHEMA or config['branch'] not in {'D0', 'D1'}
            or config['task_spec_id'] != spec['task_spec_id'] or type(config['seed']) is not int
            or config['execution_mode'] not in {'Symbolic', 'Physical'}
            or config['checkpoint_policy'] != 'NO_MID_EPISODE_RESTORE'):
        raise ValueError('unsupported V2 sidecar')
    _validate_trigger(config['trigger'], spec)
    effect = config['effect']
    roles = _roles(spec)
    if not isinstance(effect, dict) or effect.get('target_role') not in roles:
        raise ValueError('invalid V2 effect target')
    if effect.get('kind') == 'UNARY_STATE':
        if (set(effect) != {'kind', 'target_role', 'predicate', 'value'}
                or effect['predicate'] not in {'open', 'toggled_on'}
                or type(effect['value']) is not bool):
            raise ValueError('invalid unary V2 effect')
    elif effect.get('kind') == 'RESTORE_STAGING_POSE':
        if (set(effect) != {'kind', 'target_role', 'staging_role'}
                or effect.get('staging_role') not in roles
                or effect['staging_role'] == effect['target_role']):
            raise ValueError('invalid staging V2 effect')
    else:
        raise ValueError('unsupported V2 effect')
    _validate_recovery_conditions(config.get('recovery_conditions'), spec, effect)
    if 'preserved_conditions' in config:
        if not isinstance(config['preserved_conditions'], list) or not config['preserved_conditions']:
            raise ValueError('V preserved conditions must be a nonempty list')
        _validate_recovery_conditions(config['preserved_conditions'], spec, effect)
    if (isinstance(config['trigger'], dict)
            and any(atom.get('predicate') in _BINARY
                    for atom in config.get('recovery_conditions') or ())):
        raise ValueError('V2 mid-process recovery supports unary atoms only')
    _validate_guard(config.get('continuation_guard'), spec)
    base, prior = _base_member(config, spec, v2=True)
    action_mode = base.get('requested_execution_policy', {}).get('action_mode')
    if ((config['execution_mode'] == 'Symbolic' and action_mode != 'Symbolic')
            or (config['execution_mode'] == 'Physical' and action_mode not in {'Physical', 'PhysicalPlacement'})):
        raise ValueError('V2 execution mode differs from Base')
    test_only = config.get('validation_mode') == 'PRIVILEGED_TEST_ONLY'
    if 'validation_mode' in config and not test_only:
        raise ValueError('unsupported V validation mode')
    if test_only and config['base_receipt'] is not None:
        raise ValueError('test-only V must not claim a qualified Base receipt')
    if require_receipt and config['base_receipt'] is None and not test_only:
        raise ValueError('runnable V2 sidecar needs a fresh matching full Base PASS/Q1 receipt')
    if config['base_receipt'] is not None:
        _validate_receipt(config, prior)
    return config


def validate_sidecar(ref, spec):
    config = read_ref(ref)
    if config.get('schema_version') == SCHEMA:
        return _validate_v1(config, spec)
    if config.get('schema_version') == V2_SCHEMA:
        return _validate_v2(config, spec, require_receipt=True)
    raise ValueError('unsupported MV sidecar')


class MVSession:
    def __init__(self, config, audit, directory):
        self.config, self.audit, self.directory = config, audit, Path(directory)
        self.v2 = config.get('schema_version') == V2_SCHEMA
        self.consumed = self.delivered = self.failed = self.recovered = False
        self.trigger_action = None
        self.generation = 0
        self.frame = self.event = None
        self.seen = set()
        self.effect_capture = self.effect_mutation = None
        self.checkpoint_violations = []
        self.pending_guards = {}
        self.model_delivery_proven = False

    def save(self, name, value):
        from .resident_scene_batch import write_json_atomic
        write_json_atomic(self.directory / name, value)

    def ledger(self):
        if not self.v2:
            progress = self.audit.milestones
            return deepcopy(dict(reached=progress.reached, cycles=progress.cycles,
                                 cycle_started=sorted(progress.cycle_started), last_event=progress.last_event))
        milestones = getattr(self.audit, 'milestones', None)
        history = getattr(self.audit, 'history', None)
        return deepcopy(dict(
            milestones=None if milestones is None else dict(
                reached=milestones.reached, cycles=milestones.cycles,
                cycle_started=sorted(milestones.cycle_started), last_event=milestones.last_event),
            history=None if history is None else dict(reached=history.reached, bindings=history.bindings,
                binding_events=history.binding_events, events=sorted(history.events)),
        ))

    def state(self):
        """Return a private, read-only protocol snapshot for model diagnostics."""
        recovery = []
        recovery_error = None
        if self.v2 and self.config['branch'] == 'D1' and self.consumed:
            try:
                recovery = [dict(atom=deepcopy(atom), satisfied=bool(
                    self.audit.recipe_truth(atom, self.audit.role_object)))
                    for atom in self._resolved_recovery_conditions()]
            except Exception as exc:
                recovery_error = type(exc).__name__ + ': ' + str(exc)
        return deepcopy(dict(
            branch=self.config.get('branch'), schema_version=self.config.get('schema_version'),
            consumed=self.consumed, delivered=self.delivered, failed=self.failed,
            recovered=self.recovered, trigger_action=self.trigger_action,
            generation=self.generation, model_delivery_proven=self.model_delivery_proven,
            trigger=self.config.get('trigger'), effect=self.config.get('effect'),
            recovery_conditions=recovery, recovery_read_error=recovery_error,
            checkpoint_violations=self.checkpoint_violations, ledger=self.ledger(),
        ))

    def mutate(self, runtime):
        # Original v1 setter/cache sequence stays isolated for compatibility.
        from .spatial_symbolic_actions import state_by_name
        target = self.audit.role_object(self.config['target_role'])
        state = state_by_name(target, 'Open')
        both_sides, _, _ = state.relevant_joints_info
        if both_sides:
            raise RuntimeError('MV bidirectional door requires explicit direction')
        if state.set_value(False, fully=True) is not True:
            raise RuntimeError('MV setter failed')
        for atom in self.audit.milestones.program['final_requirements']:
            self.audit.role_object(atom['subject_role']).clear_states_cache()
        runtime.store.clear()

    def _base_ok(self, snapshot):
        current_goal = (self._physical_goal_ok() if self.config.get('execution_mode') == 'Physical'
                        else snapshot.official_success and float(snapshot.official_q) == 1)
        return bool(snapshot.ok and current_goal
                    and self.audit.report()['public_process_satisfied'])

    def _physical_goal_ok(self):
        from .physical_template_runtime import ACTIVE, goal_truth
        physical = ACTIVE.get()
        return physical is not None and goal_truth(physical.task['goal'], physical.facts()) is True

    def _check_preserved_conditions(self):
        if not self.consumed:
            return
        for atom in self.config.get('preserved_conditions', ()):
            resolved = deepcopy(atom)
            if 'subject_ref' in atom or 'object_ref' in atom:
                resolved = self.audit.history.resolve(resolved)
            if resolved is None or not self.audit.recipe_truth(resolved, self.audit.role_object):
                self.failed = True
                violation = dict(kind='V_PRESERVED_CONDITION_LOST', atom=deepcopy(atom))
                if violation not in self.checkpoint_violations:
                    self.checkpoint_violations.append(violation)
                self.save('checkpoint_violations.json', self.checkpoint_violations)

    def _v2_capture(self, runtime):
        from .l2_v_effects import capture
        record = capture(runtime, self.audit, self.config['effect'])
        if record is not None:
            if not isinstance(record, dict):
                raise RuntimeError('V2 staging capture is not auditable')
            self.effect_capture = deepcopy(record)
            self.save('effect_capture.json', self.effect_capture)

    def _v2_mutate(self, runtime):
        from .l2_v_effects import apply
        record = apply(runtime, self.audit, self.config['effect'], self.effect_capture)
        if not isinstance(record, dict):
            raise RuntimeError('V2 mutation is not auditable')
        self.effect_mutation = deepcopy(record)
        return record

    def _effect_truth(self):
        effect = self.config['effect']
        if effect['kind'] == 'UNARY_STATE':
            atom = dict(predicate=effect['predicate'], subject_role=effect['target_role'], desired=True)
        else:
            atom = dict(predicate='ontop', subject_role=effect['target_role'],
                        object_role=effect['staging_role'], desired=True)
        return bool(self.audit.recipe_truth(atom, self.audit.role_object))

    def _trigger_reached(self, base_ok):
        trigger = self.config['trigger']
        if trigger == 'AFTER_FULL_BASE_SUCCESS':
            return base_ok
        name = trigger['milestone_id']
        milestones = getattr(self.audit, 'milestones', None)
        history = getattr(self.audit, 'history', None)
        return bool((milestones is not None and name in milestones.reached)
                    or (history is not None and name in history.reached))

    def _resolved_recovery_conditions(self):
        conditions = self.config.get('recovery_conditions')
        if conditions is None:
            effect = self.config['effect']
            if effect['kind'] != 'UNARY_STATE':
                return []
            return [dict(predicate=effect['predicate'] if not effect['value'] else (
                'closed' if effect['predicate'] == 'open' else 'toggled_off'),
                subject_role=effect['target_role'], desired=True)]
        history = getattr(self.audit, 'history', None)
        resolved = []
        for condition in conditions:
            atom = deepcopy(condition)
            if 'subject_ref' in atom or 'object_ref' in atom:
                if history is None:
                    raise RuntimeError('V2 recovery needs unavailable history binding')
                atom = history.resolve(atom)
                if atom is None:
                    raise RuntimeError('V2 recovery history binding is unresolved')
            resolved.append(atom)
        return resolved

    def pending_recovery_atoms(self):
        """Return only currently-unmet canonical recovery atoms after public feedback."""
        if (not self.v2 or self.config['branch'] != 'D1' or not self.consumed
                or not self.delivered or self.failed
                or self.recovered):
            return []
        pending = [atom for atom in self._resolved_recovery_conditions()
                   if not self.audit.recipe_truth(atom, self.audit.role_object)]
        self.recovered = not pending
        return pending

    def _recovery_satisfied(self):
        return not self.pending_recovery_atoms()

    def _leaf_changes(self, before, after):
        if before.leaf_truth.keys() != after.leaf_truth.keys():
            raise RuntimeError('MV readback changed goal set')
        return [key for key in before.leaf_truth if before.leaf_truth[key] != after.leaf_truth[key]]

    @staticmethod
    def _action_marker(action):
        args = getattr(action, 'args', None)
        marker = getattr(args, 'marker', None)
        if marker is not None:
            return marker
        if isinstance(args, dict):
            return args.get('marker')
        to_dict = getattr(args, 'to_dict', None)
        values = to_dict() if callable(to_dict) else {}
        return values.get('marker') if isinstance(values, dict) else None

    def _action_targets_role(self, runtime, action, role):
        marker = self._action_marker(action)
        snapshot = getattr(getattr(runtime, 'store', None), 'active_snapshot', None)
        bindings = getattr(snapshot, 'bindings', {})
        binding = bindings.get(marker) if isinstance(bindings, Mapping) else None
        if binding is None:
            return False
        token = getattr(binding, 'target_token', None)
        names = {getattr(binding, 'object_name', None)}
        names.update(getattr(token, 'member_names', ()) or ())
        return getattr(self.audit.role_object(role), 'name', None) in names

    @staticmethod
    def _action_effect_atom(action, role):
        predicate = {'open': 'open', 'close': 'closed', 'toggle_on': 'toggled_on',
                     'toggle_off': 'toggled_off'}[action.action]
        return dict(predicate=predicate, subject_role=role, desired=True)

    def _complete_guard(self, action, result, action_index):
        pending = self.pending_guards.pop(action_index, None)
        if pending is None or result.status != 'completed':
            return
        if not self.audit.recipe_truth(pending['action_effect'], self.audit.role_object):
            return
        if pending['preconditions_valid']:
            self.recovered = True
            return
        record = dict(actor='AGENT', action_index=action_index, action=action.action,
                      kind='V_CURRENT_PRECONDITION_NOT_HELD', guard=pending['guard'], valid=False)
        self.checkpoint_violations.append(record)
        violations = getattr(self.audit, 'violations', None)
        if isinstance(violations, list):
            violations.append(record)
        append = getattr(self.audit, 'append', None)
        if callable(append):
            append('l2_v_checkpoint_events.jsonl', record)
        self.save('checkpoint_violations.json', self.checkpoint_violations)

    def _apply_v1(self, runtime, action, action_index, read_snapshot, current, base_ok):
        if self.consumed:
            if (self.delivered and action_index > self.trigger_action and action.action in {'open', 'close'} and base_ok):
                self.recovered = True
            return False
        if not base_ok:
            return False
        before_ledger = self.ledger()
        self.consumed = True
        self.trigger_action = action_index
        self.generation = 1
        event = dict(actor='INTERVENTION', generation=1, branch=self.config['branch'], status='STARTED',
                     after_agent_action=action_index, before=current.to_private_record(), history_before=before_ledger)
        self.event = event
        self.save('event.json', event)
        try:
            if self.config['branch'] == 'D1':
                self.mutate(runtime)
            after = read_snapshot()
            if not after.ok:
                raise RuntimeError('MV verifier unavailable')
            changed = self._leaf_changes(current, after)
            target = dict(predicate='open', subject_role=self.config['target_role'], desired=True)
            target_open = self.audit.recipe_truth(target, self.audit.role_object)
            if self.config['branch'] == 'D1':
                if len(changed) != 1 or not current.leaf_truth[changed[0]] or after.leaf_truth[changed[0]] or target_open:
                    raise RuntimeError('MV expected exactly one target regression')
            elif changed:
                raise RuntimeError('MV D0 changed world')
            if before_ledger != self.ledger():
                raise RuntimeError('MV intervention polluted M ledger')
            event.update(status='APPLIED', after=after.to_private_record(), changed_leaves=changed,
                         history_after=self.ledger())
            self.save('event.json', event)
            return True
        except BaseException as exc:
            self.failed = True
            event.update(status='FAILED', error=str(exc))
            self.save('event.json', event)
            raise

    def _apply_v2(self, runtime, action, action_index, read_snapshot, current, base_ok):
        if self.consumed:
            self._check_preserved_conditions()
            if self.delivered and action_index > self.trigger_action:
                if self.config.get('recovery_conditions') is not None:
                    self._recovery_satisfied()
                elif base_ok or self._recovery_satisfied():
                    self.recovered = True
            return False
        if not self._trigger_reached(base_ok):
            return False
        before_ledger = self.ledger()
        before_effect = self._effect_truth()
        self.consumed = True
        self.trigger_action = action_index
        self.generation = 1
        event = dict(actor='INTERVENTION', generation=1, branch=self.config['branch'], status='STARTED',
                     trigger=deepcopy(self.config['trigger']), effect=deepcopy(self.config['effect']),
                     after_agent_action=action_index, before=current.to_private_record(),
                     effect_before_truth=before_effect, history_before=before_ledger,
                     staging_capture=deepcopy(self.effect_capture))
        self.event = event
        self.save('event.json', event)
        try:
            mutation = None
            self._check_preserved_conditions()
            if self.failed:
                raise RuntimeError('V preserved condition missing at trigger')
            if self.config['branch'] == 'D1':
                mutation = self._v2_mutate(runtime)
            after = read_snapshot()
            if not after.ok:
                raise RuntimeError('V2 verifier unavailable')
            changed = self._leaf_changes(current, after)
            after_effect = self._effect_truth()
            if self.config['branch'] == 'D1':
                desired = self.config['effect']['value'] if self.config['effect']['kind'] == 'UNARY_STATE' else True
                if before_effect == desired or after_effect != desired:
                    raise RuntimeError('V2 effect did not create its declared world change')
                if self.config['trigger'] == 'AFTER_FULL_BASE_SUCCESS' and self._base_ok(after):
                    raise RuntimeError('V2 full-goal effect did not regress the Base task')
            elif changed or before_effect != after_effect:
                raise RuntimeError('V2 D0 changed world')
            if before_ledger != self.ledger():
                raise RuntimeError('V2 intervention polluted M ledger')
            self._check_preserved_conditions()
            if self.failed:
                raise RuntimeError('V intervention changed a preserved condition')
            event.update(status='APPLIED', after=after.to_private_record(), changed_leaves=changed,
                         effect_after_truth=after_effect, effect_mutation=mutation,
                         history_after=self.ledger())
            self.save('event.json', event)
            return True
        except BaseException as exc:
            self.failed = True
            event.update(status='FAILED', error=str(exc))
            self.save('event.json', event)
            raise

    def after_action(self, runtime, action, result, action_index, read_snapshot):
        self._check_preserved_conditions()
        if self.failed:
            raise RuntimeError('MV session invalid')
        if action_index in self.seen:
            return False
        self.seen.add(action_index)
        if result.status != 'completed' or read_snapshot is None:
            self._complete_guard(action, result, action_index)
            return False
        current = read_snapshot()
        if not current.ok:
            raise RuntimeError('MV verifier unavailable')
        base_ok = self._base_ok(current)
        if self.v2:
            self._v2_capture(runtime)
            triggered = self._apply_v2(runtime, action, action_index, read_snapshot, current, base_ok)
            self._complete_guard(action, result, action_index)
            return triggered
        return self._apply_v1(runtime, action, action_index, read_snapshot, current, base_ok)

    def before_action(self, runtime, action, action_index, read_snapshot=None):
        """Stage a guard; only a verified completed target action can violate it."""
        if (not self.v2 or not self.consumed or not self.delivered
                or self.recovered or action_index in self.pending_guards):
            return
        guard = self.config.get('continuation_guard')
        if (guard is None or action.action != guard['before_agent_action']
                or not self._action_targets_role(runtime, action, guard['subject_role'])):
            return
        valid = all(self.audit.recipe_truth(atom, self.audit.role_object)
                    for atom in guard['requires_current'])
        self.pending_guards[action_index] = dict(guard=deepcopy(guard), preconditions_valid=bool(valid),
                                                 action_effect=self._action_effect_atom(action, guard['subject_role']))

    def attach_frame(self, record):
        if not self.event or self.failed:
            raise RuntimeError('MV frame without valid event')
        self.frame = record
        self.save('feedback_frame.json', dict(generation=self.generation, **record))

    def observation_delivered(self, action_index, *, consumer='PUBLIC_ACTION20_ORACLE_TURN',
                              request_sequence=None, request_path=None,
                              delivery_evidence=None):
        if self.consumed and not self.delivered:
            if action_index != self.trigger_action or self.frame is None:
                raise RuntimeError('MV fresh feedback missing')
            evidence = {}
            if consumer == 'MODEL_REQUEST' and delivery_evidence is None:
                request_path = Path(request_path)
                request = json.loads(request_path.read_text())
                hashes = request['png_sha256s']
                paths = request['png_paths']
                if (type(request_sequence) is not int or request_sequence < 0
                        or not hashes or len(paths) != len(hashes)
                        or hashes[-1] != self.frame.get('png_sha256')
                        or any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h
                               for p, h in zip(paths, hashes))):
                    raise RuntimeError('MV model request lacks current intervention frame')
                evidence = dict(request_sequence=request_sequence, request_path=str(request_path),
                    request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
                    call_index=request['call_index'], frame_sha256=hashes[-1])
                self.model_delivery_proven = True
            elif consumer == 'MODEL_REQUEST':
                expected = {
                    'request_id', 'observation_id', 'sequence', 'composite_sha256',
                    'source_rgb_sha256', 'request_manifest_sha256', 'image_url_sha256',
                }
                if (not isinstance(delivery_evidence, Mapping)
                        or set(delivery_evidence) != expected
                        or not isinstance(delivery_evidence['request_id'], str)
                        or not delivery_evidence['request_id']
                        or not isinstance(delivery_evidence['observation_id'], str)
                        or not delivery_evidence['observation_id']
                        or type(delivery_evidence['sequence']) is not int
                        or delivery_evidence['sequence'] < 0
                        or delivery_evidence['source_rgb_sha256'] != (
                            self.frame.get('camera_sha256') or {}
                        ).get('head')
                        or any(
                            not isinstance(delivery_evidence[field], str)
                            or len(delivery_evidence[field]) != 64
                            for field in (
                                'composite_sha256', 'request_manifest_sha256',
                                'image_url_sha256',
                            )
                        )):
                    raise RuntimeError('MV Bench model request lacks current intervention image')
                evidence = dict(delivery_evidence)
                self.model_delivery_proven = True
            elif consumer == 'HUMAN_BROWSER':
                expected = {
                    'request_id', 'observation_id', 'sequence', 'composite_sha256',
                    'source_rgb_sha256',
                }
                if (not isinstance(delivery_evidence, Mapping)
                        or set(delivery_evidence) != expected
                        or not isinstance(delivery_evidence['request_id'], str)
                        or not delivery_evidence['request_id']
                        or not isinstance(delivery_evidence['observation_id'], str)
                        or not delivery_evidence['observation_id']
                        or type(delivery_evidence['sequence']) is not int
                        or delivery_evidence['sequence'] < 0
                        or delivery_evidence['source_rgb_sha256'] != (
                            self.frame.get('camera_sha256') or {}
                        ).get('head')
                        or not isinstance(delivery_evidence['composite_sha256'], str)
                        or len(delivery_evidence['composite_sha256']) != 64):
                    raise RuntimeError('MV browser delivery lacks the current intervention image')
                evidence = dict(delivery_evidence)
            elif consumer != 'PUBLIC_ACTION20_ORACLE_TURN':
                raise ValueError('unsupported MV feedback consumer')
            self.delivered = True
            self.save('feedback_delivery.json', dict(generation=self.generation, action_index=action_index,
                                                     frame=self.frame, consumer=consumer,
                                                     model_delivery_proven=self.model_delivery_proven, **evidence))

    def restore_checkpoint(self, *args, **kwargs):
        raise RuntimeError('MV mid-episode checkpoint restoration unsupported')

    def finish(self, result):
        self._check_preserved_conditions()
        if self.v2 and self.config['branch'] == 'D1':
            self.pending_recovery_atoms()
        goal = (result.get('status') == 'PASS'
                and (self._physical_goal_ok() if self.config.get('execution_mode') == 'Physical'
                     else result.get('official_q') == 1)
                and self.audit.report()['public_process_satisfied'])
        complete = bool(goal and self.consumed and self.delivered and not self.failed
                        and (self.config['branch'] == 'D0' or self.recovered)
                        and not self.checkpoint_violations)
        report = dict(branch=self.config['branch'], trigger_reached=self.consumed,
                      intervention_valid=bool(self.event and self.event['status'] == 'APPLIED'),
                      fresh_feedback_delivered=self.delivered, agent_recovery_completed=self.recovered,
                      v_protocol_complete=complete, canonical_goal_success=goal,
                      feedback_necessity='NOT_ESTABLISHED', model_delivery_proven=self.model_delivery_proven,
                      checkpoint_policy='NO_MID_EPISODE_RESTORE')
        if self.v2:
            report.update(schema_version=V2_SCHEMA, trigger=deepcopy(self.config['trigger']),
                          effect=deepcopy(self.config['effect']),
                          checkpoint_violations=deepcopy(self.checkpoint_violations),
                          recovery_conditions=self.pending_recovery_atoms())
        self.save('final.json', report)
        return report
