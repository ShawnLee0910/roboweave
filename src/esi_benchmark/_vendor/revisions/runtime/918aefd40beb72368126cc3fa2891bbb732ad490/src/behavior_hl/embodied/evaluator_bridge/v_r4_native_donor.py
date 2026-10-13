"""Frozen native donor validators reused from R1 wave2 2ae7f708.

Only intake/evidence validation is reused; no R1 execution or intervention is run.
"""
import csv
import io
import json
import math
from pathlib import Path
from .v_intervention import _require, _read_ref
from .v_r1_intervention import ACTION_EFFECTS, _validate_trigger_goal

def load_r1_package(path, config, *, task_names, seed):
    _require(set(config) - {'native_world'} == {'schema_version', 'package_id', 'task_spec_id', 'seed', 'public_notice',
        'task_specs', 'reset_pose_overlay', 'requested_execution_policy', 'trigger', 'landing',
        'source', 'prepared_state'}, 'unknown or missing R1 fields')
    _require(tuple(task_names) == (config['task_spec_id'],), 'R1 requires one exact source task')
    _require(type(config['seed']) is int and config['seed'] == seed, 'R1 seed mismatch')
    for key in ('package_id', 'task_spec_id', 'public_notice'):
        _require(isinstance(config[key], str) and config[key].strip(), 'invalid R1 ' + key)
    trigger = config['trigger']
    _require(set(trigger) == {'action', 'target_scopes', 'destination_scopes', 'predicate', 'desired'},
             'invalid R1 trigger fields')
    _require(trigger['action'] in ACTION_EFFECTS and type(trigger['desired']) is bool and
             (trigger['predicate'], trigger['desired']) == ACTION_EFFECTS[trigger['action']],
             'unsupported or mismatched R1 action effect')
    for key in ('target_scopes', 'destination_scopes'):
        values = trigger[key]
        _require(isinstance(values, list) and all(isinstance(s, str) and s for s in values)
                 and len(set(values)) == len(values), 'invalid R1 scope list')
    placement = trigger['action'].startswith('place_')
    _require(trigger['target_scopes'] and bool(trigger['destination_scopes']) == placement,
             'R1 state effects have no destination; placements need a destination')
    landing = config['landing']
    _require(set(landing) == {'max_candidates', 'max_distance_m', 'clearance_m'}, 'invalid R1 landing fields')
    _require(type(landing['max_candidates']) is int and 1 <= landing['max_candidates'] <= 25,
             'invalid R1 candidate budget')
    for key, upper, positive in [('max_distance_m', 1., True), ('clearance_m', .01, False)]:
        value = landing[key]
        _require(type(value) in (int, float) and math.isfinite(value) and
                 (value > 0 if positive else value >= 0) and value <= upper, 'invalid R1 ' + key)
    policy = config['requested_execution_policy']
    _require(policy == {'action_mode': 'Symbolic', 'physics_steps_per_action': 0,
                        'scene_load': 'Full', 'oracle_timeout_seconds_per_task': 1800}, 'unsupported R1 policy')
    source = config['source']
    _require(set(source) in ({'base_package', 'base_task_spec_id', 'receipt'},
                            {'base_package', 'base_task_spec_id', 'history_attestation'}) and
             source['base_task_spec_id'] == config['task_spec_id'], 'R1 source identity mismatch')
    base_path, raw = _read_ref(source['base_package'], path.parent)
    base = json.loads(raw)
    if 'receipt' in source:
        _, receipt_bytes = _read_ref(source['receipt'], path.parent)
        receipt = json.loads(receipt_bytes)
        _require(receipt.get('status') == 'PASS' and receipt.get('official_q') == 1
                 and receipt.get('official_success') is True and receipt.get('task_spec_id') == config['task_spec_id']
                 and receipt.get('oracle_status') == 'COMPLETED'
                 and receipt.get('oracle_execution_status') == 'COMPLETED'
                 and receipt.get('verification_completed') is True, 'R1 source receipt is not complete PASS/Q1')
        config['_source_evidence_mode'] = 'RAW_RECEIPT_REOPENED_PASS_Q1'
    else:
        _require(base.get('runtime', {}).get('adapter') == 'fixed_l0_pool', 'history evidence is native L0 only')
        _validate_history_attestation(config, path.parent)
        _require(config.get('native_world') is not None, 'history evidence requires explicit native world replay contract')
    adapter = base.get('runtime', {}).get('adapter')
    _require(adapter in {'fixed_l0_pool', 'registered_l0'}, 'unsupported R1 source adapter')
    _require(config['task_spec_id'] in base.get('task_spec_ids', []), 'R1 source not in frozen base')
    config['_base_package_path'], config['_source_adapter'] = str(base_path), adapter
    spec_path = None
    if config['task_specs'] is not None:
        spec_path, data = _read_ref(config['task_specs'], path.parent)
        selected = [json.loads(line) for line in data.splitlines() if line.strip()]
        selected = [s for s in selected if s['task_spec_id'] == config['task_spec_id']]
        _require(len(selected) == 1, 'missing or duplicate R1 TaskSpec')
        spec = selected[0]
        source_specs = base.get('task_specs')
        if source_specs is not None:
            _, original = _read_ref(source_specs, base_path.parent)
            originals = [json.loads(line) for line in original.splitlines() if line.strip()]
            originals = [s for s in originals if s['task_spec_id'] == config['task_spec_id']]
            _require(len(originals) == 1, 'missing source TaskSpec')
            _require(spec == originals[0], 'R1 must preserve complete source TaskSpec')
        scopes = spec.get('private', {}).get('scope_bindings')
        if scopes:
            _require(all(s in scopes for s in trigger['target_scopes'] + trigger['destination_scopes']),
                     'R1 trigger scope absent from source')
        _validate_trigger_goal(spec, trigger, base, base_path, adapter)
        if adapter == 'fixed_l0_pool':
            config['_task_specs_path'] = str(spec_path)
            if config.get('prepared_state') is not None:
                _require(config.get('native_world') is not None, 'native prepared robot state requires separate world identity')
            if config.get('native_world') is not None:
                _validate_native_world(config, spec, base, path.parent)
        else:
            _require('native_world' not in config, 'registered R1 cannot use native world identity')
        if adapter != 'registered_l0':
            config['max_action_decisions'] = spec['public']['max_action_decisions']
        else:
            config['_task_specs_path'] = str(spec_path)
            spec_path = None
    else:
        raise ValueError('R1 requires exact source TaskSpecs for goal membership validation')
    overlay = None
    if config['reset_pose_overlay'] is not None:
        overlay, _ = _read_ref(config['reset_pose_overlay'], path.parent)
    if config['prepared_state'] is not None:
        _, data = _read_ref(config['prepared_state'], path.parent)
        bundle = json.loads(data)
        _require(isinstance(bundle, dict), 'invalid R1 prepared robot/camera bundle')
        config['_prepared_states'] = {config['task_spec_id']: bundle}
    return config, spec_path, overlay, policy

def _validate_history_attestation(config, root):
    """Verify the authoritative success index, never impersonate a raw receipt."""
    _, raw = _read_ref(config['source']['history_attestation'], root)
    attestation = json.loads(raw)
    _require(set(attestation) == {'schema_version', 'mode', 'catalog', 'record_number', 'config_id',
        'task_spec_id', 'source_package', 'receipt_identity'}, 'invalid historical evidence fields')
    _require(attestation['schema_version'] == 'behavior.r1-history-attestation/v1' and
             attestation['mode'] == 'HISTORY_ATTESTED_NOT_REOPENED', 'invalid historical evidence mode')
    _require(type(attestation['record_number']) is int and attestation['record_number'] > 0,
             'invalid historical catalog record number')
    _, raw = _read_ref(attestation['catalog'], root)
    rows = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    row = next((row for number, row in enumerate(rows, 1) if number == attestation['record_number']), None)
    _require(row is not None and row['config_id'] == attestation['config_id'] and
             row['task_spec_id'] == attestation['task_spec_id'] == config['task_spec_id'],
             'historical catalog task/config identity mismatch')
    _require(row['level'] == 'L0' and row['status'] == 'PASS' and float(row['q']) == 1.0 and
             row['oracle_q1'] == 'True' and row['q1_recorded'] == 'True' and
             row['success_scope'] == 'ORACLE_EXECUTION_PASS' and row['diagnostic_only'] == 'False',
             'historical catalog is not a full L0 Oracle PASS/Q1 attestation')
    package = config['source']['base_package']
    _require(attestation['source_package'] == package and row['package_path'] == package['path'] and
             row['package_sha256'] == package['sha256'], 'historical source package identity mismatch')
    receipt = attestation['receipt_identity']
    _require(receipt == {'path':row['receipt_path'], 'host':row['receipt_host'], 'sha256':row['receipt_sha256']} and
             isinstance(receipt['sha256'], str) and len(receipt['sha256']) == 64 and
             all(c in '0123456789abcdef' for c in receipt['sha256']), 'historical receipt identity mismatch')
    _require(int(row['seed']) == config['seed'], 'historical seed mismatch')
    config['_source_evidence_mode'] = 'HISTORY_ATTESTED_NOT_REOPENED'
    config['_history_receipt_identity'] = receipt
    config['_history_config_id'] = attestation['config_id']

def _validate_native_world(config, spec, base, root):
    """Bind native world inputs; distinguish exact prepared replay from standard reset."""
    _, raw = _read_ref(config['native_world'], root)
    world = json.loads(raw)
    fields = {'schema_version', 'task_spec_id', 'source_config_id', 'source_template',
        'source_tro_state', 'source_bddl', 'source_pool', 'reset_pose_overlay', 'successful_receipt',
        'successful_prepared_state', 'successful_run_manifest', 'successful_task_protocol',
        'source_task_identity', 'world_replay', 'robot_camera_replay'}
    historical = world['schema_version'] == 'behavior.v-r1-native-world/v2'
    if historical:
        fields = (fields - {'successful_receipt'}) | {'history_attestation', 'reset_mode'}
    _require(set(world) == fields, 'invalid native world identity fields')
    _require(world['schema_version'] in {'behavior.v-r1-native-world/v1','behavior.v-r1-native-world/v2'} and
        world['task_spec_id'] == config['task_spec_id'] and
        world['world_replay'] == 'UNCHANGED_FIXED_NATIVE_TEMPLATE_AND_TRO', 'invalid native world replay contract')
    artifacts = spec['private']['source_artifacts']
    for key, source_key in [('source_template','template'),('source_tro_state','tro_state'),('source_bddl','bddl')]:
        _require(world[key] == artifacts[source_key], 'native world source differs from original TaskSpec')
        _read_ref(world[key], root)
    _require(world['source_pool'] == base['runtime']['pool'], 'native source pool changed')
    _read_ref(world['source_pool'], root)
    _require(world['reset_pose_overlay'] == base['runtime'].get('reset_pose_overlay') == config['reset_pose_overlay'],
             'native reset overlay differs from original source')
    _require(world['source_task_identity'] == spec['private']['source_task_identity'], 'native task identity changed')
    if historical:
        _require(config['_source_evidence_mode'] == 'HISTORY_ATTESTED_NOT_REOPENED' and
                 world['history_attestation'] == config['source']['history_attestation'] and
                 world['source_config_id'] == config['_history_config_id'], 'historical native evidence differs')
        mode = world['reset_mode']
        _require(mode in {'EXACT_PREPARED_ROBOT_CAMERA','STANDARD_NATIVE_RESET'}, 'unsupported native reset mode')
    else:
        _require(world['successful_receipt'] == config['source'].get('receipt'), 'native success receipt identity changed')
        mode = 'EXACT_PREPARED_ROBOT_CAMERA'
    config['_native_reset_mode'] = mode
    if mode == 'STANDARD_NATIVE_RESET':
        _require(config['prepared_state'] is None and world['successful_prepared_state'] is None and
                 world['successful_run_manifest'] is None and world['successful_task_protocol'] is None and
                 world['robot_camera_replay'] == 'STANDARD_NATIVE_GRIPPER_PREPARATION',
                 'standard native reset cannot claim exact prepared replay')
        return
    _require(world['robot_camera_replay'] == 'SUCCESS_PRE_ACTION_BUNDLE_NOT_WORLD_SNAPSHOT' and
             config['prepared_state'] is not None, 'exact native replay requires genuine prepared state')
    _, raw = _read_ref(world['successful_task_protocol'], root)
    protocol = json.loads(raw)
    _require(protocol['task_names'].count(config['task_spec_id']) == 1, 'native source task missing or duplicate in successful run')
    index = protocol['task_names'].index(config['task_spec_id'])
    _require(protocol['task_identities'][index] == world['source_task_identity'], 'native source run task identity mismatch')
    _, raw = _read_ref(world['successful_run_manifest'], root)
    _require(json.loads(raw)['seed'] == config['seed'] == protocol['seed'], 'native successful seed mismatch')
    _, raw = _read_ref(world['successful_prepared_state'], root)
    prepared = json.loads(raw)
    _, raw = _read_ref(config['prepared_state'], root)
    bundle = json.loads(raw)
    _require(bundle['prepared'] == prepared and prepared['task_name'] == config['task_spec_id'] and
             prepared['oracle_actions_executed'] == prepared['physics_steps_executed'] == 0,
             'native robot/camera state differs from original pre-action capture')
    if historical:
        source_run = str(Path(config['_history_receipt_identity']['path']).parents[3])
        _require(prepared['first_frame'].startswith(source_run + '/'), 'prepared capture is not from catalog-attested run')
