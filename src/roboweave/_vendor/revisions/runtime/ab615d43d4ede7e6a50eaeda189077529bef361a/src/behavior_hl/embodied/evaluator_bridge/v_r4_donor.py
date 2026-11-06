"""Use already validated V donor preparation without changing its full goal."""
from copy import deepcopy
from itertools import islice
import json
from pathlib import Path

from .v_intervention import _read_ref, _require


def donor_details(path, *, task_names, seed):
    from .v_intervention import load_package
    from .v_r4_intervention import _goal_atoms
    raw = json.loads(Path(path).read_text())
    _require(raw['schema_version'] in {'behavior.v-r1-package/v1','behavior.v-r3-package/v1','behavior.v-r4-package/v1'}
             and 'donor_v_package' not in raw and 'variant_source' not in raw,
             'R4 donor must be an existing R1/R3 source, not another R4 wrapper')
    if raw['schema_version']=='behavior.v-r1-package/v1' and 'native_world' in raw:
        from .v_r4_native_donor import load_r1_package
        loaded, spec_path, overlay, policy = load_r1_package(Path(path),deepcopy(raw),task_names=task_names,seed=seed)
    else:
        loaded, spec_path, overlay, policy = load_package(path, task_names=task_names, seed=seed)
    actual_spec_path, data = _read_ref(raw['task_specs'], Path(path).parent)
    specs = [json.loads(line) for line in data.splitlines() if line.strip()]
    spec = next(s for s in specs if s['task_spec_id'] == raw['task_spec_id'])
    adapter = loaded['_source_adapter']
    registration = None
    if adapter == 'registered_l0':
        base_path, data = _read_ref(raw['source']['base_package'], Path(path).parent)
        base = json.loads(data)
        catalog_path, data = _read_ref(base['runtime']['registration_catalog'], base_path.parent)
        registration = next(r for r in json.loads(data)['tasks'] if r['task_spec_id']==spec['task_spec_id'])
        _, data = _read_ref(registration['template_contract'], catalog_path.parent)
        bindings = json.loads(data)['scope_bindings']
        atoms = _goal_atoms(spec, base, base_path, adapter)
    elif adapter == 'historical_full_household':
        from .canonical_bddl_grounder import object_map_from_scope, iter_ground_options
        bindings = spec['private']['scope_bindings']
        options = list(islice(iter_ground_options(spec['private']['goal_spec']['full_goal_ast'],
                                                  object_map_from_scope(bindings)),4097))
        _require(len(options)<=4096, 'full goal grounding exceeds donor limit')
        atoms = set()
        for option in options:
            for leaf in option:
                desired = leaf[0] != 'not'
                atom = leaf if desired else leaf[1]
                atoms.add((atom[0],desired,atom[1],atom[2] if len(atom)>2 else None))
    else:
        bindings = spec['private']['scope_bindings']
        base_path, data = _read_ref(raw['source']['base_package'], Path(path).parent)
        atoms = _goal_atoms(spec,json.loads(data),base_path,adapter)
    return loaded, spec_path, overlay, policy, spec, bindings, atoms, registration


def load_r4_donor(path, config, *, task_names, seed):
    from .v_r4_intervention import SCHEMA, POLICY, SCORING_FIELDS, _action_effect, _requested
    _require(set(config)=={'schema_version','package_id','task_spec_id','seed','donor_v_package',
                          'trigger_mode','trigger','blocked_operation'}, 'unknown R4 donor wrapper fields')
    _require(config['schema_version']==SCHEMA and tuple(task_names)==(config['task_spec_id'],)
             and type(config['seed']) is int and config['seed']==seed, 'R4 donor identity mismatch')
    _require(config['trigger_mode'] in {'AFTER_OPEN_THEN_TARGET_PICK','BEFORE_SELECTED_PLACE_INSIDE'}, 'R4 donor requires held-target timing')
    donor_path, _ = _read_ref(config['donor_v_package'], path.parent)
    loaded, spec_path, overlay, policy, spec, bindings, atoms, _ = donor_details(
        donor_path, task_names=task_names, seed=seed)
    _require(policy==POLICY, 'R4 donor execution mode mismatch')
    trigger, blocked = _action_effect(config['trigger']), _action_effect(config['blocked_operation'])
    _require(trigger['action']=='open' and not trigger['destination_scopes'] and trigger['desired'] is True,
             'R4 donor trigger must open a container')
    _require(blocked['action']=='place_inside' and blocked['desired'] is True and _requested(blocked)<=atoms,
             'R4 cannot change or invent the original Inside goal')
    _require(bool(trigger['target_scopes']) and set(trigger['target_scopes'])<=set(blocked['destination_scopes']),
             'one declared destination required per R4 configuration')
    _require(not (_requested(trigger)&atoms), 'opening a final-goal target is not this R4 template')
    if config['trigger_mode']=='BEFORE_SELECTED_PLACE_INSIDE':
        legal = {d for p,v,s,d in atoms if p=='inside' and v and s in blocked['target_scopes']}
        _require(set(trigger['target_scopes']) == set(blocked['destination_scopes']) == legal,
                 'selected-reference R4 must cover every legal destination for its targets')
    _require(all(s in bindings for s in blocked['target_scopes']+blocked['destination_scopes']),
             'R4 donor binding missing')
    container = bindings[trigger['target_scopes'][0]]
    _require(container.get('actual_name') and container.get('model'), 'R4 destination identity missing')
    # Existing loader validates the original receipt, complete goal, registry,
    # native-state or registered-state restore and prepared bundle separately.
    result = deepcopy(loaded)
    result.pop('landing',None)
    result.pop('later_trigger',None)
    result.update(config)
    result.update(_task_spec=spec, _r4_donor_bindings=bindings, scoring_fields=list(SCORING_FIELDS),
                  classification={'r_class':'R4','strength':'BASIC_PRECONDITION_RESET','live':'NOT_RUN'})
    result['intervention']={'kind':'close_same_container','allow_object_pose_change':False,
        'allow_robot_pose_change':False,'allow_mass_or_collider_change':False,
        'container_actual_name':container['actual_name'],'observed_joint':'LIVE_OPEN_STATE_JOINTS'}
    return result,spec_path,overlay,policy
