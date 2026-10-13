"""R4: revoke an established access condition without moving objects.

Open is a means, not a goal leaf. Closing the same container after a completed
Open does not regress an inherited goal atom. D0 consumes the same one-shot
event without mutation. Pose landing is forbidden.
"""
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import json
import numpy as np

from .v_intervention import PoseInterventionSession, _read_ref, _require, _vector
from .v_r1_intervention import ACTION_EFFECTS, STATE_NAMES, _validate_trigger_goal

SCHEMA = 'behavior.v-r4-package/v1'
NATIVE_WORLD_SCHEMA = 'behavior.v-native-world-identity/v1'
ROBOT_CAMERA_REPLAY = 'SUCCESS_PRE_ACTION_BUNDLE_APPLIED_AFTER_NATIVE_WORLD'
PUBLIC_CLOSED_CONTAINER_ERROR = 'Open the container before placing an object inside it.'
POLICY = {'action_mode': 'Symbolic', 'physics_steps_per_action': 0,
          'scene_load': 'Full', 'oracle_timeout_seconds_per_task': 1800}
SCORING_FIELDS = (
    'task_success', 'event_exposure', 'intervention_validity',
    'explicit_precondition_repair', 'original_goal_operation_completed',
    'full_goal_reverification', 'alternative_legal_path', 'executor_automatic_repair',
)
PACKAGE_FIELDS = {
    'schema_version', 'package_id', 'task_spec_id', 'seed', 'public_notice',
    'task_specs', 'reset_pose_overlay', 'requested_execution_policy', 'trigger',
    'blocked_operation', 'intervention', 'source', 'prepared_state', 'native_world',
    'classification', 'scoring_fields',
}


def _scope_list(values, label):
    _require(isinstance(values, list) and values and all(isinstance(s, str) and s for s in values)
             and len(set(values)) == len(values), 'invalid R4 ' + label)
    return values


def _action_effect(spec):
    _require(set(spec) == {'action', 'target_scopes', 'destination_scopes', 'predicate', 'desired'},
             'invalid R4 action fields')
    _require(spec['action'] in ACTION_EFFECTS and type(spec['desired']) is bool and
             (spec['predicate'], spec['desired']) == ACTION_EFFECTS[spec['action']],
             'unsupported or mismatched R4 action effect')
    _scope_list(spec['target_scopes'], 'target_scopes')
    dest = spec['destination_scopes']
    _require(isinstance(dest, list) and all(isinstance(s, str) and s for s in dest)
             and len(set(dest)) == len(dest), 'invalid R4 destination_scopes')
    placement = spec['action'].startswith('place_')
    _require(bool(dest) == placement, 'R4 state effects have no destination; placements need a destination')
    return spec


def _goal_atoms(spec, base, base_path, adapter):
    def normalized(predicate, desired):
        aliases = {'closed': 'open', 'toggled_off': 'toggled_on'}
        if desired is None:
            desired = True
        return (aliases[predicate], not desired) if predicate in aliases else (predicate, desired)
    atoms = set()
    if adapter == 'registered_l0':
        catalog_path, raw = _read_ref(base['runtime']['registration_catalog'], base_path.parent)
        records = [r for r in json.loads(raw)['tasks'] if r['task_spec_id'] == spec['task_spec_id']]
        _require(len(records) == 1, 'R4 registration identity missing or duplicate')
        _, raw = _read_ref(records[0]['template_contract'], catalog_path.parent)
        roles = json.loads(raw)['role_to_scope']
        for clause in spec['goal_contract']['goal_clauses']:
            predicate, desired = normalized(clause['predicate'], clause.get('desired', True))
            subject = roles[clause['subject_role']]
            destinations = clause.get('destination_equivalence_roles') or [clause.get('object_role')]
            atoms.update((predicate, desired, subject, None if d is None else roles[d]) for d in destinations)
    else:
        for clause in spec['private']['goal_spec'].get('clauses', []):
            predicate, desired = normalized(clause['canonical_predicate'], clause.get('desired', True))
            destinations = clause.get('destination_equivalence_members') or clause.get('destination_scopes') or [None]
            atoms.update((predicate, desired, subject, destination)
                         for subject in clause['target_scopes'] for destination in destinations)
    return atoms


def _requested(effect):
    destinations = effect['destination_scopes'] or [None]
    return {(effect['predicate'], effect['desired'], subject, destination)
            for subject in effect['target_scopes'] for destination in destinations}


def _verify_native_world(ref, path, spec, source_receipt_ref):
    world_path, raw = _read_ref(ref, path.parent)
    world = json.loads(raw)
    _require(world.get('schema_version') == NATIVE_WORLD_SCHEMA, 'unsupported R4 native world schema')
    _require(world.get('task_spec_id') == spec['task_spec_id'], 'R4 native world task mismatch')
    _require(world.get('world_replay') == 'UNCHANGED_FIXED_NATIVE_TEMPLATE_AND_TRO',
             'R4 cannot merge or rebuild a world snapshot')
    _require(world.get('robot_camera_replay') == ROBOT_CAMERA_REPLAY,
             'R4 native robot/camera must be the success pre-action bundle applied after the donor world')
    artifacts = spec['private']['source_artifacts']
    for key in ('template', 'tro_state', 'bddl'):
        field = {'template': 'source_template', 'tro_state': 'source_tro_state', 'bddl': 'source_bddl'}[key]
        _require(world[field] == artifacts[key], 'R4 native ' + key + ' identity changed')
        _read_ref(world[field], world_path.parent)
    _require(world['successful_receipt'] == source_receipt_ref, 'R4 native receipt identity changed')
    _read_ref(world['successful_receipt'], world_path.parent)
    for field in ('source_pool', 'reset_pose_overlay', 'successful_prepared_state',
                  'successful_run_manifest', 'successful_task_protocol'):
        _read_ref(world[field], world_path.parent)
    identity = spec['private']['source_task_identity']
    _require(world.get('source_task_identity') == identity, 'R4 native source identity changed')
    return world


def load_r4_package(path, config, *, task_names, seed):
    if 'donor_v_package' in config:
        from .v_r4_donor import load_r4_donor
        return load_r4_donor(path,config,task_names=task_names,seed=seed)
    if 'variant_source' in config:
        return load_r4_destination_variant(path, config, task_names=task_names, seed=seed)
    _require(set(config) - {'trigger_mode'} == PACKAGE_FIELDS, 'unknown or missing R4 fields')
    _require(config.get('schema_version') == SCHEMA, 'unsupported R4 package schema')
    _require(tuple(task_names) == (config['task_spec_id'],), 'R4 requires one exact source task')
    _require(type(config['seed']) is int and config['seed'] == seed, 'R4 seed mismatch')
    for key in ('package_id', 'task_spec_id', 'public_notice'):
        _require(isinstance(config[key], str) and config[key].strip(), 'invalid R4 ' + key)
    _require(config.get('trigger_mode', 'AFTER_OPEN') in {'AFTER_OPEN','AFTER_OPEN_THEN_TARGET_PICK'},
             'unsupported R4 trigger mode')
    trigger, blocked = _action_effect(config['trigger']), _action_effect(config['blocked_operation'])
    _require(trigger['action'] == 'open' and trigger['desired'] is True and not trigger['destination_scopes'],
             'R4 access trigger must be a completed Open of the bound container')
    _require(blocked['action'] == 'place_inside' and blocked['desired'] is True and blocked['destination_scopes'],
             'this R4 template requires an inherited place_inside goal operation')
    _require(set(trigger['target_scopes']) <= set(blocked['destination_scopes']),
             'R4 may only close a container that the blocked goal operation uses')
    intervention = config['intervention']
    _require(set(intervention) == {
        'kind', 'allow_object_pose_change', 'allow_robot_pose_change',
        'allow_mass_or_collider_change', 'observed_joint', 'container_actual_name',
    }, 'invalid R4 intervention fields')
    _require(intervention['kind'] == 'close_same_container', 'unsupported R4 intervention')
    _require(intervention['allow_object_pose_change'] is False and
             intervention['allow_robot_pose_change'] is False and
             intervention['allow_mass_or_collider_change'] is False,
             'R4 may not move objects, the robot, or change colliders')
    _require(isinstance(intervention['observed_joint'], str) and intervention['observed_joint']
             and isinstance(intervention['container_actual_name'], str)
             and intervention['container_actual_name'], 'R4 articulation identity missing')
    _require(config['requested_execution_policy'] == POLICY, 'unsupported R4 policy')
    _require(tuple(config['scoring_fields']) == SCORING_FIELDS, 'R4 scoring fields changed')
    classification = config['classification']
    _require(set(classification) >= {
        'r_class', 'strength', 'template', 'same_base_group', 'independent_physical_base',
    } and classification['r_class'] == 'R4'
             and classification['template'] == 'ACCESS_CONDITION_REVOKED_AFTER_OPEN',
             'invalid R4 classification')
    _require(classification['strength'] in {
        'R1_R4_BOUNDARY', 'BASIC_PRECONDITION_RESET', 'CONFIRMED_STRONG_R4',
    }, 'unsupported R4 strength label')
    source = config['source']
    _require(set(source) == {'base_package', 'base_task_spec_id', 'receipt'} and
             source['base_task_spec_id'] == config['task_spec_id'], 'R4 source identity mismatch')
    base_path, raw = _read_ref(source['base_package'], path.parent)
    base = json.loads(raw)
    _, receipt_bytes = _read_ref(source['receipt'], path.parent)
    receipt = json.loads(receipt_bytes)
    _require(receipt.get('status') == 'PASS' and receipt.get('official_q') == 1
             and receipt.get('official_success') is True and receipt.get('task_spec_id') == config['task_spec_id']
             and receipt.get('oracle_status') == 'COMPLETED'
             and receipt.get('oracle_execution_status') == 'COMPLETED'
             and receipt.get('verification_completed') is True, 'R4 source receipt is not complete PASS/Q1')
    adapter = base.get('runtime', {}).get('adapter')
    _require(adapter in {'fixed_l0_pool', 'registered_l0'}, 'unsupported R4 source adapter')
    _require(config['task_spec_id'] in base.get('task_spec_ids', []), 'R4 source not in frozen base')
    config['_base_package_path'], config['_source_adapter'] = str(base_path), adapter
    _require(config['task_specs'] is not None, 'R4 requires exact source TaskSpecs')
    spec_path, data = _read_ref(config['task_specs'], path.parent)
    selected = [json.loads(line) for line in data.splitlines() if line.strip()]
    selected = [s for s in selected if s['task_spec_id'] == config['task_spec_id']]
    _require(len(selected) == 1, 'missing or duplicate R4 TaskSpec')
    spec = selected[0]
    source_specs = base.get('task_specs')
    _require(source_specs is not None, 'R4 base TaskSpecs missing')
    _, original = _read_ref(source_specs, base_path.parent)
    originals = [json.loads(line) for line in original.splitlines() if line.strip()]
    originals = [s for s in originals if s['task_spec_id'] == config['task_spec_id']]
    _require(len(originals) == 1 and spec == originals[0], 'R4 must preserve complete source TaskSpec')
    scopes = spec.get('private', {}).get('scope_bindings') or {}
    if scopes:
        used = trigger['target_scopes'] + blocked['target_scopes'] + blocked['destination_scopes']
        _require(all(s in scopes for s in used), 'R4 scope absent from source')
    atoms = _goal_atoms(spec, base, base_path, adapter)
    _require(not (_requested(trigger) & atoms),
             'R4 Open trigger is an inherited goal atom; that is R1, not this template')
    _validate_trigger_goal(spec, blocked, base, base_path, adapter)
    _verify_native_world(config['native_world'], path, spec, source['receipt'])
    if adapter != 'registered_l0':
        config['max_action_decisions'] = spec['public']['max_action_decisions']
    else:
        config['_task_specs_path'] = str(spec_path)
        spec_path = None
    overlay = None
    if config['reset_pose_overlay'] is not None:
        overlay, _ = _read_ref(config['reset_pose_overlay'], path.parent)
    _require(adapter != 'fixed_l0_pool' or overlay is not None,
             'native R4 world restore requires the donor overlay')
    _require(config['prepared_state'] is not None, 'R4 requires the success pre-action robot/camera bundle')
    _, data = _read_ref(config['prepared_state'], path.parent)
    bundle = json.loads(data)
    _require(isinstance(bundle, dict), 'invalid R4 prepared robot/camera bundle')
    from .prepared_model_state import validate_bundle
    from behavior_hl.embodied.policy.observations import THREE_RGB_HEAD_EVAL_CAMERA_PROFILE
    validate_bundle(bundle, instruction_id=config['task_spec_id'], seed=seed,
                    observation_profile=THREE_RGB_HEAD_EVAL_CAMERA_PROFILE)
    config['_task_spec'] = spec
    config['_prepared_states'] = {config['task_spec_id']: bundle}
    config['_world_restore'] = 'NATIVE_TEMPLATE_TRO_OVERLAY'
    config['_native_prepared_replay'] = ROBOT_CAMERA_REPLAY
    return config, spec_path, overlay, config['requested_execution_policy']


def load_r4_destination_variant(path, config, *, task_names, seed):
    """Explicit user-authored goal change; original PASS attests the donor only."""
    from .canonical_bddl_grounder import parse_bddl, problem_goal, find_section
    from .l0_taskspec_live_adapter import L0TaskSpecLiveAdapter, canonical_task_spec_sha256
    from .action23_outcome_verifier import Action23OutcomeVerifier
    _require(set(config) - {'variant_source','trigger_mode'} == PACKAGE_FIELDS,
             'unknown destination variant envelope fields')
    _require(config['schema_version']==SCHEMA and tuple(task_names)==(config['task_spec_id'],)
             and type(config['seed']) is int and config['seed']==seed, 'variant identity mismatch')
    _require(config.get('trigger_mode')=='AFTER_OPEN_THEN_TARGET_PICK', 'variant requires held recovery timing')
    _,raw=_read_ref(config['variant_source'],path.parent);variant=json.loads(raw)
    substitution = variant.get('target_substitution')
    _require(set(variant) - {'target_substitution'}=={'schema_version','original_r4_package','variant_task_spec','old_destination_scope',
        'new_destination_scope','new_destination_actual_name','new_destination_joint','candidate_contract','goal_mode','new_goal_success'},
        'unknown destination variant source fields')
    _require(variant['schema_version']=='behavior.r4-destination-variant-source/v1' and
             variant['goal_mode']==('SUBSTITUTED_TARGET_INSIDE_EXISTING_MICROWAVE' if substitution else 'ANY_ORIGINAL_PIZZA_INSIDE_EXISTING_MICROWAVE') and
             variant['new_goal_success']=='NOT_RUN', 'variant cannot inherit new goal success')
    original_path,raw=_read_ref(variant['original_r4_package'],path.parent);original=json.loads(raw)
    _require('variant_source' not in original,'nested destination variants forbidden')
    loaded,_,_,_=load_r4_package(original_path,deepcopy(original),task_names=[original['task_spec_id']],seed=seed)
    _require(config['task_spec_id']!=original['task_spec_id'] and config['source']==original['source'] and
             config['native_world']==original['native_world'] and config['requested_execution_policy']==original['requested_execution_policy']
             and config['scoring_fields']==original['scoring_fields'], 'variant changed donor evidence/policy')
    _require(config['task_specs']==variant['variant_task_spec'],'variant TaskSpec identity mismatch')
    sp,raw=_read_ref(config['task_specs'],path.parent);specs=[json.loads(l) for l in raw.splitlines() if l.strip()]
    _require(len(specs)==1 and specs[0]['task_spec_id']==config['task_spec_id'],'variant requires one new TaskSpec')
    spec=specs[0];oldspec=loaded['_task_spec'];old=variant['old_destination_scope'];new=variant['new_destination_scope'];actual=variant['new_destination_actual_name']
    _require(original['blocked_operation']['action']=='place_inside' and original['blocked_operation']['destination_scopes']==[old],
             'variant donor must be a single-destination Inside task')
    expected_trigger=deepcopy(original['trigger']);expected_trigger['target_scopes']=[new]
    expected_blocked=deepcopy(original['blocked_operation']);expected_blocked['destination_scopes']=[new]
    if substitution:
        _require(set(substitution)=={'source_scope','target_scope','actual_name','category','model','support_scope',
                                    'source_metadata','target_metadata'}, 'invalid target substitution fields')
        _require(substitution['source_scope'] in original['blocked_operation']['target_scopes'], 'replacement source is not a donor target')
        _require(substitution['target_scope'].rsplit('_',1)[0]=='durian.n.01' and
                 substitution['category']=='durian' and substitution['model']=='mttfyh', 'unreviewed target asset')
        expected_blocked['target_scopes']=[substitution['target_scope']]
    _require(config['trigger']==expected_trigger and config['blocked_operation']==expected_blocked,
             'variant target set or action contract changed')
    expected_intervention=deepcopy(original['intervention']);expected_intervention.update(container_actual_name=actual,observed_joint=variant['new_destination_joint'])
    _require(config['intervention']==expected_intervention,'variant mutation permissions changed')
    old_art=oldspec['private']['source_artifacts'];new_art=spec['private']['source_artifacts']
    originals={};modified={}
    for key in ('template','tro_state','bddl'):
        _,raw=_read_ref(old_art[key],path.parent);originals[key]=raw
        _,raw=_read_ref(new_art[key],path.parent);modified[key]=raw
    template=json.loads(originals['template']);changed=json.loads(modified['template'])
    oldtro=json.loads(originals['tro_state'])
    oldexpr=parse_bddl(originals['bddl'].decode())[0]
    if substitution:
        from .v_target_substitution import substitute_target
        metadata=[]
        for key in ('source_metadata','target_metadata'):
            _, raw = _read_ref(substitution[key], path.parent)
            metadata.append(json.loads(raw))
        template,oldtro,oldexpr=substitute_target(template,oldtro,oldexpr,substitution,*metadata)
    _require(actual in template['objects_info']['init_info'] and actual in template['state']['registry']['object_registry'],
             'microwave is not present in original physical scene')
    _require(new not in template['metadata']['task']['inst_to_name'], 'variant scope unexpectedly already exists')
    expected=deepcopy(template);expected['metadata']['task']['inst_to_name'][new]=actual
    expected['objects_info']['init_info'][actual]['args']['bddl_object_scope']=new
    _require(changed==expected,'variant changed physical scene or unrelated metadata')
    newtro=json.loads(modified['tro_state'])
    expected=deepcopy(oldtro);expected[new]=template['state']['registry']['object_registry'][actual]
    _require(newtro==expected,'variant TRO changed any original physical state')
    newexpr=parse_bddl(modified['bddl'].decode())[0]
    target_scopes=original['blocked_operation']['target_scopes']
    _require(all(scope.startswith('pizza.n.01_') for scope in target_scopes),'variant donor is not the original pizza choice set')
    goal=['exists',['?pizza.n.01','-','pizza.n.01'],['inside','?pizza.n.01','?'+new]]
    if substitution:
        goal=['inside','?'+substitution['target_scope'],'?'+new]
    expected=deepcopy(oldexpr);expected[1]=['problem',config['task_spec_id']+'-0'];find_section(expected,':objects').extend([new,'-','microwave.n.02'])
    find_section(expected,':init').append(['inroom',new,'kitchen']);find_section(expected,':goal')[:]=[':goal',goal]
    _require(newexpr==expected,'variant BDDL changed outside the explicit destination delta')
    expected_goal=deepcopy(oldspec['private']['goal_spec'])
    _require(len(expected_goal['clauses'])==1,'variant cannot delete companion source goals')
    clause=expected_goal['clauses'][0]
    for key in ('destination_scopes','destination_equivalence_members'):clause[key]=[new]
    clause['destination_equivalence_policy']='NONE'
    if substitution:
        clause['target_scopes']=[substitution['target_scope']]
        clause['target_equivalence_policy']='NONE'
    _require(spec['private']['goal_spec']==expected_goal and spec['private']['verifier']['clauses']==expected_goal['clauses'],
             'variant goal changed beyond single destination substitution')
    expected_identity=deepcopy(oldspec['private']['source_task_identity'])
    expected_identity.update(task_id=94013002 if substitution else 94013001,task_name=config['task_spec_id'])
    _require(spec['private']['source_task_identity']==expected_identity, 'variant activity identity must be new and keep source scene/split/instance')
    expected_bindings=deepcopy(oldspec['private']['scope_bindings']);args=template['objects_info']['init_info'][actual]['args']
    _require(args['category']=='microwave' and 'kitchen_0' in args['in_rooms'],'variant destination is not the existing kitchen microwave')
    expected_bindings[new]={'actual_name':actual,'category':'microwave','model':args['model'],'in_rooms':args['in_rooms'],'binding_source':'EXISTING_SCENE_VARIANT_BINDING'}
    if substitution:
        binding=expected_bindings.pop(substitution['source_scope'])
        binding.update(actual_name=substitution['actual_name'],category=substitution['category'],model=substitution['model'],
                       binding_source='USER_AUTHORED_TARGET_SUBSTITUTION_NOT_Q1')
        expected_bindings[substitution['target_scope']]=binding
    _require(spec['private']['scope_bindings']==expected_bindings,'variant changed original object bindings')
    membership=spec['private']['verifier']['official_goal_membership']
    _require(membership['bddl']==new_art['bddl'] and membership['pattern_identity_sha256']==canonical_task_spec_sha256(goal),
             'variant goal membership still points to donor goal')
    _read_ref(membership['grounder'],path.parent)
    _,raw=_read_ref(variant['candidate_contract'],path.parent);contract=json.loads(raw)
    _require(spec['candidate_contract_sha256']==canonical_task_spec_sha256(contract) and contract['goal_spec']==expected_goal
             and contract['source_artifacts']==new_art,'variant candidate contract mismatch')
    _require(spec['evaluation'] == oldspec['evaluation'] and
             spec['public']['max_action_decisions'] == oldspec['public']['max_action_decisions'],
             'variant changed original evaluation or action budget')
    _require(spec['public']['instruction']==('Put the durian inside the microwave in the kitchen.' if substitution else
                                           'Put either of the two pizzas inside the microwave in the kitchen.'),
             'variant public instruction must preserve either-pizza choice')
    L0TaskSpecLiveAdapter(spec,stability_window=1,predicate_mode='action20_symbolic',
                         action_outcome_verifier=Action23OutcomeVerifier(lambda *a:False,lambda:None))
    overlay,raw=_read_ref(config['reset_pose_overlay'],path.parent);records=[json.loads(l) for l in raw.splitlines() if l.strip()]
    _,raw=_read_ref(original['reset_pose_overlay'],path.parent)
    expected=next(json.loads(l) for l in raw.splitlines() if l.strip() and json.loads(l)['task_spec_id']==original['task_spec_id'])
    expected.update(task_spec_id=config['task_spec_id'],task_name=config['task_spec_id'],source_tro_state=new_art['tro_state'])
    _require(records==[expected],'variant robot reset changed physical preparation')
    _,raw=_read_ref(config['prepared_state'],path.parent);bundle=json.loads(raw)
    _,raw=_read_ref(original['prepared_state'],path.parent);expected=json.loads(raw)
    expected['instruction_id']=config['task_spec_id'];expected['prepared']['task_name']=config['task_spec_id']
    sources=deepcopy(bundle['sources']);sources.pop('variant_task_rebinding',None)
    _require(sources == expected['sources'], 'variant changed original preparation provenance')
    _require(bundle.get('sources',{}).get('variant_task_rebinding',{}).get('new_goal_oracle_success') is False,
             'variant prepared provenance missing')
    comparable=deepcopy(bundle);comparable['sources']=sources
    _require(comparable==expected,'variant changed recorded robot/camera state')
    from .prepared_model_state import validate_bundle
    from behavior_hl.embodied.policy.observations import THREE_RGB_HEAD_EVAL_CAMERA_PROFILE
    validate_bundle(bundle,instruction_id=config['task_spec_id'],seed=seed,observation_profile=THREE_RGB_HEAD_EVAL_CAMERA_PROFILE)
    config.update(_base_package_path=loaded['_base_package_path'],_source_adapter='fixed_l0_pool',_task_spec=spec,
        _prepared_states={config['task_spec_id']:bundle},_world_restore='VARIANT_SCOPE_ONLY_NATIVE_WORLD',
        _native_prepared_replay=ROBOT_CAMERA_REPLAY,_source_evidence_mode='DONOR_INITIALIZATION_PASS_NEW_GOAL_NOT_RUN',
        _variant_task_spec=spec,_variant_template_path=new_art['template']['path'],_variant_tro_path=new_art['tro_state']['path'],
        _variant_bddl_path=new_art['bddl']['path'],max_action_decisions=spec['public']['max_action_decisions'])
    if substitution:
        config['_target_substitution']=substitution
        config['_world_restore']='USER_AUTHORED_TARGET_SUBSTITUTION_NOT_Q1'
    return config,sp,overlay,config['requested_execution_policy']


def native_r4_initial_semantics_ok(*, evaluator, task, config):
    """Confirm the donor world is present. Do not search a layout or move objects."""
    spec = config.get('_task_spec') or {}
    live_task = getattr(getattr(evaluator, 'env', None), 'task', None)
    scope = getattr(live_task, 'object_scope', None)
    if not isinstance(scope, Mapping):
        return False
    trigger = config['trigger']
    blocked = config['blocked_operation']
    needed = list(dict.fromkeys(trigger['target_scopes'] + blocked['target_scopes']
                                + blocked['destination_scopes']))
    bindings = spec.get('private', {}).get('scope_bindings') or {}
    for name in needed:
        obj = scope.get(name)
        if obj is None:
            return False
        expected = bindings.get(name, {}).get('actual_name')
        if expected and getattr(obj, 'name', None) != expected:
            return False
    from .spatial_symbolic_actions import read_action_relation, state_by_name
    substitution=config.get('_target_substitution')
    if substitution:
        child=scope.get(substitution['target_scope']);support=scope.get(substitution['support_scope'])
        if child is None or support is None or not read_action_relation(child,'OnTop',support,predicate_mode='action20_symbolic'):
            return False
    for subject in blocked['target_scopes']:
        child = scope.get(subject)
        if child is None or state_by_name(child, 'Inside') is None:
            continue
        for destination in blocked['destination_scopes']:
            container = scope.get(destination)
            if container is None:
                continue
            if read_action_relation(child, 'Inside', container, predicate_mode='action20_symbolic'):
                return False
    return True


def make_native_r4_world_verification_callback(config):
    """Reuse the native overlay world. Never reconstruct or search object poses."""

    def callback(*, evaluator, control, bindings, task):
        accepted = native_r4_initial_semantics_ok(evaluator=evaluator, task=task, config=config)
        return {
            'construction_pass': accepted,
            'object_placement_executed': False,
            'robot_pose_selection_executed': False,
            'pose_search': False,
            'initialization_mode': 'NATIVE_TRO_OVERLAY_WORLD_PLUS_PREPARED_ROBOT_CAMERA',
            'world_replay': ('FROZEN_TARGET_SUBSTITUTION_TEMPLATE_AND_TRO' if config.get('_target_substitution')
                             else 'UNCHANGED_FIXED_NATIVE_TEMPLATE_AND_TRO'),
            'robot_camera_replay': ROBOT_CAMERA_REPLAY,
        }

    return callback


def _joint_name(joint):
    for attr in ('joint_name', 'name', 'prim_path', 'body1'):
        value = getattr(joint, attr, None)
        if isinstance(value, str) and value:
            return value
    return str(joint)


def _joint_position(joint):
    native = getattr(joint, 'get_state', None)
    if callable(native):
        position, _, _ = native(normalized=False)
        values = _vector(position)
        _require(len(values) == 1, 'R4 Open joint must have one degree of freedom')
        return values[0]
    getter = getattr(joint, 'get_position', None)
    if callable(getter):
        value = getter()
        if hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
            value = value[0]
        return float(value)
    value = getattr(joint, 'position', None)
    _require(value is not None, 'articulated joint position unavailable')
    return float(value)


def _articulation_snapshot(state):
    info = getattr(state, 'relevant_joints_info', None)
    _require(isinstance(info, (tuple, list)) and len(info) == 3, 'articulation info unavailable')
    both_sides, joints, directions = info
    rows = [{'name': _joint_name(joint), 'position': _joint_position(joint),
             'direction': int(direction)}
            for joint, direction in zip(joints, directions)]
    return bool(both_sides), tuple(joints), tuple(directions), rows


class R4InterventionSession(PoseInterventionSession):
    def reset_episode(self, output_dir):
        super().reset_episode(output_dir)
        self.objects = {}
        self.repair_open_seen = False
        self.goal_operation_completed = False
        self.alternative_legal_path = False
        self.executor_automatic_repair = False
        self.blocked_operation_rejected = False
        self.closed_scope = None
        self.robot_pose_before = None
        self.container_base_pose_before = None
        self.open_completion = None
        self.held_target_scope = None
        self.post_event_actions = []
        self.unexplained_goal_completion = False

    def _refresh(self):
        for obj in self.objects.values():
            obj.clear_states_cache()

    def _effect(self, target_scope, destination_scope):
        from .spatial_symbolic_actions import read_action_relation
        self._refresh()
        trigger = self.config['trigger']
        value = read_action_relation(self.objects[target_scope], STATE_NAMES[trigger['predicate']],
            None if destination_scope is None else self.objects[destination_scope],
            predicate_mode='action20_symbolic')
        return bool(value) == trigger['desired']

    def _blocked_effect(self, target_scope, destination_scope):
        from .spatial_symbolic_actions import read_action_relation
        blocked = self.config['blocked_operation']
        value = read_action_relation(self.objects[target_scope], STATE_NAMES[blocked['predicate']],
            self.objects[destination_scope], predicate_mode='action20_symbolic')
        return bool(value) == blocked['desired']

    def _robot(self):
        robots = getattr(self.evaluator.env, 'robots', None) or []
        return robots[0] if robots else None

    def _pose(self, obj):
        position, orientation = obj.get_position_orientation()
        return {'position': _vector(position), 'orientation_xyzw': _vector(orientation)}

    def before_action(self, runtime, read_snapshot, action_index=None):
        _require(not self.failed, 'R4 session invalid after intervention failure')
        _require(runtime.zero_physics and not runtime.placement_physical_consequences_enabled,
                 'R4 requires inherited Symbolic executor')
        snapshot = read_snapshot()
        _require(snapshot.ok, 'R4 verifier unavailable')
        trigger = self.config['trigger']
        blocked = self.config['blocked_operation']
        if not self.initialized:
            scopes = list(dict.fromkeys(trigger['target_scopes'] + blocked['target_scopes']
                                        + blocked['destination_scopes']))
            self.objects = {s: self._object(s) for s in scopes}
            _require(len({id(self.objects[s]) for s in trigger['target_scopes']}) == len(trigger['target_scopes']),
                     'R4 container aliases require an explicit contract')
            self._save('initial.json', {'config': self.config, 'branch': self.branch,
                       'initial_snapshot': snapshot.to_private_record()})
            self.initialized = True
        if self.consumed:
            self.delivered = False
        self.pre_effects = {(s, None): self._effect(s, None) for s in trigger['target_scopes']}
        held = runtime.symbolic_primitives._get_obj_in_hand()
        self.held = held
        marker_snapshot = getattr(runtime.store, 'active_snapshot', None)
        self.pre_bindings = {} if marker_snapshot is None else dict(marker_snapshot.bindings)
        return snapshot

    def _selected_container(self, action):
        args = action.args
        marker = args.get('marker') if isinstance(args, Mapping) else getattr(args, 'marker', None)
        binding = self.pre_bindings.get(marker)
        if binding is None:
            return None
        name = getattr(binding, 'object_name', None)
        return next((s for s in self.config['trigger']['target_scopes']
                     if self.objects[s].name == name), None)

    def _safe_to_mutate(self, runtime, container, event):
        owner = runtime.symbolic_primitives
        held = owner._get_obj_in_hand()
        closure = getattr(owner, '_logical_inventory_snapshot', None)
        if self.config.get('trigger_mode', 'AFTER_OPEN') == 'AFTER_OPEN_THEN_TARGET_PICK':
            expected = self.objects.get(self.held_target_scope)
            if held is None or held is not expected:
                event['deferred'] = 'held_target_identity_mismatch'
                return False
            if closure is not None and (getattr(closure, 'root', None) is not held
                    or getattr(closure, 'descendant_count', None) != 0):
                event['deferred'] = 'unsupported_carry_contents_or_unknown_closure'
                return False
            event['held_target_scope'] = self.held_target_scope
            event['held_object_name'] = held.name
            event['carry_descendant_count'] = 0
        elif held is not None or closure is not None:
            event['deferred'] = 'robot_still_holding'
            return False
        owner.require_marker_visual_projection_inactive()
        event['live_container_name'] = container.name
        event['donor_container_name'] = self.config['intervention']['container_actual_name']
        event['contact_check'] = 'NOT_AVAILABLE_CPU'
        return True

    def _record_poses(self, container):
        robot = self._robot()
        self.robot_pose_before = None if robot is None else self._pose(robot)
        self.container_base_pose_before = self._pose(container)
        self.object_poses_before = {scope: self._pose(obj) for scope, obj in self.objects.items()
                                    if obj is not container}

    def _require_poses_unchanged(self, container, event):
        # Joint-only mutation invokes no root pose setters. Numerical readback
        # drift is diagnostic; held identity and canonical leaves own admission.
        event['container_base_pose_after'] = self._pose(container)
        pairs = [('container', self.container_base_pose_before, event['container_base_pose_after'])]
        robot = self._robot()
        if robot is not None and self.robot_pose_before is not None:
            event['robot_pose_after'] = self._pose(robot)
            pairs.append(('robot', self.robot_pose_before, event['robot_pose_after']))
        pairs.extend((scope, pose, self._pose(self.objects[scope])) for scope, pose in self.object_poses_before.items())
        event['pose_deltas_diagnostic'] = [{'object': name,
            'position_delta': (np.asarray(after['position']) - before['position']).tolist(),
            'orientation_delta': (np.asarray(after['orientation_xyzw']) - before['orientation_xyzw']).tolist()}
            for name, before, after in pairs]
        event['moved_unrelated_objects'] = [name for name, before, after in pairs if before != after]
        event['pose_admission_policy'] = 'JOINT_ONLY_SETTER_PLUS_CANONICAL_LEAVES_AND_HELD_IDENTITY'

    def _close_container(self, container, event):
        from .spatial_symbolic_actions import state_by_name
        state = state_by_name(container, 'Open')
        _require(state is not None and callable(getattr(state, 'set_value', None)),
                 'R4 Open setter unavailable')
        both_sides, joints, _, before = _articulation_snapshot(state)
        event['joints_before'] = before
        event['state_before'] = bool(state.get_value())
        observed = self.config['intervention']['observed_joint']
        if observed == 'LIVE_OPEN_STATE_JOINTS':
            _require(bool(before), 'live Open state has no articulated joints')
            event['joint_binding_mode'] = 'LIVE_OPEN_STATE_JOINTS'
        else:
            _require(any(observed == row['name'] or observed.endswith(row['name'])
                     or row['name'].endswith(observed.split(':')[-1]) for row in before),
                 'frozen observed joint is absent from the live asset')
        event['bidirectional'] = both_sides
        event['mutation_started'] = True
        event['state_setter_mode'] = 'fully_closed_same_container'
        accepted = state.set_value(False, fully=True)
        _require(accepted is True, 'R4 state setter did not confirm closure')
        self._refresh()
        event['state_after'] = bool(state.get_value())
        _, _, _, after = _articulation_snapshot(state)
        event['joints_after'] = after
        _require(event['state_after'] is False, 'R4 container is still open after D1')
        event['articulated_final_collision'] = {
            'status': 'NOT_RUN',
            'reason': 'USER_DISABLED_R4_WHOLE_OBJECT_POST_INTERVENTION_CHECK',
        }

    def after_action(self, runtime, action, result, action_index, read_snapshot):
        self._seen_actions.add(action_index)
        self.action_count = len(self._seen_actions)
        if self.consumed:
            self._score_post_event(action, result, runtime=runtime, action_index=action_index)
            return False
        if self.config.get('trigger_mode', 'AFTER_OPEN') == 'AFTER_OPEN_THEN_TARGET_PICK':
            if result.status != 'completed':
                return False
            if action.action == 'open':
                scope = self._selected_container(action)
                if scope is not None and not self.pre_effects[(scope, None)] and self._effect(scope, None):
                    self.open_completion = {'container_scope': scope, 'open_action_index': action_index,
                                            'open_receipt_status': result.status}
                    self._save('open_completed.json', self.open_completion)
                return False
            if action.action != 'pick' or self.open_completion is None or action_index <= self.open_completion['open_action_index']:
                return False
            args = action.args
            marker = args.get('marker') if isinstance(args, Mapping) else getattr(args, 'marker', None)
            binding = self.pre_bindings.get(marker)
            actual = runtime.symbolic_primitives._get_obj_in_hand()
            scope = next((s for s in self.config['blocked_operation']['target_scopes']
                          if self.objects[s] is actual and getattr(binding, 'object_name', None) == actual.name), None)
            container_scope = self.open_completion['container_scope']
            if scope is None or self.held is actual or not self._effect(container_scope, None):
                return False
            self.held_target_scope = scope
            return self.apply_event(runtime, action, result, action_index, read_snapshot, container_scope)
        if result.status != 'completed' or action.action != 'open':
            return False
        container_scope = self._selected_container(action)
        if container_scope is None or self.pre_effects[(container_scope, None)] or not self._effect(container_scope, None):
            return False
        if self.held is not None:
            return False
        return self.apply_event(runtime, action, result, action_index, read_snapshot, container_scope)

    def apply_event(self, runtime, action, result, action_index, read_snapshot, container_scope):
        owner = runtime.symbolic_primitives
        owner.require_marker_visual_projection_inactive()
        before = read_snapshot()
        _require(before.ok, 'R4 pre-event verifier unavailable')
        container = self.objects[container_scope]
        event = {'branch': self.branch, 'status': 'STARTED', 'action_index': action_index,
                 'action_receipt_status': result.status, 'target_scope': container_scope,
                 'destination_scope': None, 'before': before.to_private_record(),
                 'physics_steps': 0, 'mutation_started': False, 'event_kind': 'human_intervention_not_action_receipt'}
        if self.config.get('trigger_mode', 'AFTER_OPEN') == 'AFTER_OPEN_THEN_TARGET_PICK':
            event.update(self.open_completion, pick_action_index=action_index, pick_receipt_status=result.status,
                         trigger_mode='AFTER_OPEN_THEN_TARGET_PICK')
        if not self._safe_to_mutate(runtime, container, event):
            self._save('deferred.json', event)
            return False
        self.consumed, self.delivered = True, False
        self.closed_scope = container_scope
        self._event_before = before
        self._save('event.json', event)
        try:
            self._record_poses(container)
            if self.branch == 'D1':
                self._close_container(container, event)
            runtime.store.clear()
            self._refresh()
            after = read_snapshot()
            _require(after.ok and before.leaf_truth.keys() == after.leaf_truth.keys(),
                     'R4 goal readback changed shape')
            changed = [key for key in before.leaf_truth if before.leaf_truth[key] != after.leaf_truth[key]]
            if self.branch == 'D0':
                _require(not changed and self._effect(container_scope, None), 'R4 no-op changed a goal or closed access')
            else:
                _require(not self._effect(container_scope, None), 'R4 D1 left the container open')
                _require(not changed, 'R4 D1 changed a goal leaf; Open is not the goal')
            self._require_poses_unchanged(container, event)
            if self.held_target_scope is not None:
                _require(owner._get_obj_in_hand() is self.objects[self.held_target_scope], 'R4 changed held target identity')
                event['held_identity_preserved'] = True
            self.generation += 1
            event.update(status='APPLIED', after=after.to_private_record(), changed_goal_leaves=changed,
                         generation=self.generation,
                         access_revoked=(self.branch == 'D1' and not self._effect(container_scope, None)),
                         blocked_operation=self.config['blocked_operation']['action'],
                         public_closed_container_error=PUBLIC_CLOSED_CONTAINER_ERROR,
                         event_is_not_action_receipt=True)
            self.event = event
            self._save('event.json', event)
            return True
        except BaseException as exc:
            self.failed = True
            event.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
            self._save('event.json', event)
            from behavior_hl.embodied.runtime.action_integrity import invalidate_execution
            invalidate_execution(self.evaluator, exc)
            raise

    def _score_post_event(self, action, result, *, runtime=None, action_index=None):
        held = None if runtime is None else runtime.symbolic_primitives._get_obj_in_hand()
        self.post_event_actions.append({
            "action_index": action_index, "action": action.action, "status": result.status,
            "held_before": None if self.held is None else self.held.name,
            "held_after": None if held is None else held.name})
        self._save("post_event_actions.json", {"actions": self.post_event_actions})
        if action.action == self.config['blocked_operation']['action'] and result.status in {'blocked','failed','uncertain'}:
            self.blocked_operation_rejected = True
        if result.status != 'completed':
            return
        if (action.action == 'open' and self._selected_container(action) == self.closed_scope
                and self._effect(self.closed_scope, None)):
            self.repair_open_seen = True
        if action.action != self.config['blocked_operation']['action']:
            return
        args = action.args
        marker = args.get('marker') if isinstance(args, Mapping) else getattr(args, 'marker', None)
        binding = self.pre_bindings.get(marker)
        name = None if binding is None else getattr(binding, 'object_name', None)
        destination = next((s for s in self.config['blocked_operation']['destination_scopes']
                            if self.objects[s].name == name), None)
        if destination is None:
            return
        if self.config.get('trigger_mode', 'AFTER_OPEN') == 'AFTER_OPEN_THEN_TARGET_PICK':
            subject = next((s for s in self.config['blocked_operation']['target_scopes']
                            if self.objects[s] is self.held), None)
            if subject is None or held is not None or not self._blocked_effect(subject, destination):
                self.unexplained_goal_completion = True
                return
        self.goal_operation_completed = True
        if destination != self.closed_scope:
            self.alternative_legal_path = True
        elif self.branch == 'D1' and not self.repair_open_seen:
            before_open = self.pre_effects.get((self.closed_scope, None))
            after_open = self._effect(self.closed_scope, None)
            if before_open is False and after_open is True:
                self.executor_automatic_repair = True
            else:
                self.unexplained_goal_completion = True

    def finish(self, reason, snapshot):
        report = super().finish(reason, snapshot)
        success = bool(snapshot.ok and snapshot.official_success)
        lane = getattr(self.evaluator, '_v_execution_lane', 'unknown')
        if self.repair_open_seen:
            channel = 'public_open_action'
        elif self.executor_automatic_repair:
            channel = 'executor_internal'
        else:
            channel = 'none'
        chain = bool(success and self.repair_open_seen and self.goal_operation_completed
                     and not self.alternative_legal_path and not self.executor_automatic_repair
                     and not self.failed)
        model_feedback = bool(chain and lane == 'model')
        scoring = {
            'task_success': success,
            'event_exposure': bool(self.consumed),
            'intervention_validity': bool(self.event is not None and not self.failed),
            'explicit_precondition_repair': self.repair_open_seen,
            'original_goal_operation_completed': self.goal_operation_completed,
            'full_goal_reverification': success,
            'alternative_legal_path': self.alternative_legal_path,
            'executor_automatic_repair': self.executor_automatic_repair,
            'repair_open_channel': channel,
            'repair_open_actor': 'none' if channel != 'public_open_action' else lane,
            'blocked_operation_rejected': self.blocked_operation_rejected,
            'recovery_chain_solvable': chain,
            'model_feedback_use': model_feedback,
        }
        if success and self.alternative_legal_path:
            scoring['r4_recovery_evidence'] = False
            scoring['r4_recovery_evidence_reason'] = 'ALTERNATIVE_LEGAL_PATH'
        elif success and self.executor_automatic_repair and not self.repair_open_seen:
            scoring['r4_recovery_evidence'] = False
            scoring['r4_recovery_evidence_reason'] = 'EXECUTOR_AUTOMATIC_REPAIR'
        elif chain and lane == 'native_oracle':
            scoring['r4_recovery_evidence'] = False
            scoring['r4_recovery_evidence_reason'] = 'NATIVE_ORACLE_SOLVABILITY_NOT_MODEL_FEEDBACK'
        elif model_feedback:
            scoring['r4_recovery_evidence'] = True
        else:
            scoring['r4_recovery_evidence'] = False
            scoring['r4_recovery_evidence_reason'] = 'PENDING_OR_INCOMPLETE'
        scoring['unexplained_goal_completion'] = self.unexplained_goal_completion
        scoring['automatic_repair_evidence'] = ('OBSERVED_INTERNAL_OPEN_CHANGE' if self.executor_automatic_repair
            else 'UNKNOWN' if self.unexplained_goal_completion else 'NONE')
        scoring['post_event_actions'] = self.post_event_actions
        report['scoring'] = scoring
        report['r4_strength'] = self.config['classification']['strength']
        self._save('final.json', report)
        return report
