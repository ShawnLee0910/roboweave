"""Frozen AMV inputs, with vendor imports restricted to the private CPU worker."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROFILES = frozenset({'l1-a30', 'l1-m30-public-v2', 'l1-v30'})


def resolve_inputs(config, task, manifest, rows, root):
    """Resolve immutable private files without activating any source profile."""
    from ..assets import _path, _suffix
    if task['profile'] not in PROFILES or manifest['profile'] != task['profile']:
        raise ValueError('AMV input profile differs from task')
    by_key = {row['key']: row for row in rows}
    metadata = manifest['inputs_by_task'][task['task_id']]
    result = {key: str(_path(root, by_key[value])) if key.endswith('_json') else value
              for key, value in metadata.items()}
    release = json.loads(Path(result['task_json']).read_text())
    selected = [row for row in release if row['instruction_id'] == task['task_id']]
    if len(selected) != 1:
        raise ValueError('Frozen AMV public task identity is absent or ambiguous')
    raw = selected[0]
    if (raw['runtime_instruction_id'] != task['runtime_instruction_id']
            or raw['instruction_en'] != task['instruction_en'] or raw['budget'] != task['budget']):
        raise ValueError('AMV public instruction, native identity or frozen budget differs')
    review = json.loads(Path(result['instruction_review_json']).read_text())
    public_id = task['source']['row_index']
    approved = [row for row in review['rewrites'] if row['task_id'] == public_id]
    if (review.get('status') != 'APPROVED' or len(approved) != 1
            or approved[0]['task_name'] != task['task_id']
            or approved[0]['instruction'] != task['instruction_en']):
        raise ValueError('AMV public instruction lacks its exact approved review')
    if hashlib.sha256(task['instruction_en'].encode()).hexdigest() != task['instruction_sha256']:
        raise ValueError('AMV public instruction hash differs')
    model_task = {'task_id': public_id,
                  'task_name': task['task_id'] if result['entrypoint'] == 'v_collection' else task['runtime_instruction_id'],
                  'task_seed': result['seed'], 'instruction': task['instruction_en'],
                  **task['budget'], 'budget': dict(task['budget']),
                  'instruction_provenance': {'profile': task['profile'],
                      'source_sha256': by_key[metadata['task_json']]['sha256'],
                      'review_sha256': by_key[metadata['instruction_review_json']]['sha256'],
                      'instruction_sha256': task['instruction_sha256']}}
    return {**result, 'profile': task['profile'], 'asset_root': str(root),
            'instruction_id': task['runtime_instruction_id'], 'public_task_id': task['task_id'],
            'scene_model': task['scene_model'], 'model_task': model_task,
            'mapping': {_suffix(row): str(_path(root, row)) for row in rows},
            'source_provenance': {'profile': task['profile'],
                'task_source_sha256': by_key[metadata['task_json']]['sha256'],
                'instruction_review_sha256': by_key[metadata['instruction_review_json']]['sha256']}}


def inspect_inputs(profile, inputs):
    """Run the frozen static contracts after the caller activates this profile.

    This function restores no scene, stages no evidence and makes no connections.
    """
    if profile not in PROFILES or inputs.get('profile') != profile:
        raise ValueError('AMV static inspection profile mismatch')
    native = inputs['instruction_id']
    if inputs['entrypoint'] == 'prepared':
        from instruction_handoff import compile_handoff, native_instruction_arguments
        from prepared_state_inputs import load_prepared
        matrix = compile_handoff(Path(inputs['handoff_json']), {'model': 1})
        instruction = matrix['instructions'][native]
        identity, seed = instruction['source_task_identity'], instruction['seed']
        bundle = load_prepared(Path(inputs['prepared_index_json']), matrix['handoff_sha256'], native)
        native_args = native_instruction_arguments(Path(inputs['handoff_json']), native)
        if '--nearby-initial-pose-search' in native_args:
            raise ValueError('AMV frozen preparation cannot use a fallback pose search')
        if '--runtime-registration' in native_args:
            from behavior_hl.embodied.evaluator_bridge.registered_l0_task_runtime import load_registered_l0_tasks
            registry = Path(native_args[native_args.index('--runtime-registration') + 1])
            if native not in load_registered_l0_tasks(registry):
                raise ValueError('AMV registration does not contain its native task')
        package = instruction.get('runtime_package', instruction['package'])
        extra = {'benchmark_package_json': package['path'],
                 'task_specs_jsonl': json.loads(Path(package['path']).read_text())['task_specs']['path']}
    elif inputs['entrypoint'] == 'v_collection':
        from v_collection_handoff import load_binding, native_arguments, native_source_identity, verify_runtime
        binding, row, package, _, prepared = load_binding(Path(inputs['v_collection_binding_json']))
        runtime_module = verify_runtime(binding)
        if (row['task_spec_id'] != native or row['collection_config_key'] != inputs['public_config_id']
                or inputs['model_task']['task_name'] != row['collection_config_key']):
            raise ValueError('V collection binding identity differs')
        identity = native_source_identity(Path(inputs['v_collection_binding_json']))
        seed = package['seed']
        loaded, _, _, _ = runtime_module.load_package(Path(row['v_package']['path']), task_names=[native], seed=seed)
        if loaded.get('_source_adapter') == 'registered_l0':
            from behavior_hl.embodied.evaluator_bridge.registered_l0_task_runtime import load_registered_l0_tasks
            base_path = Path(loaded['_base_package_path'])
            base = json.loads(base_path.read_text())
            registry = Path(base['runtime']['registration_catalog']['path'])
            if not registry.is_absolute():
                registry = base_path.parent / registry
            if native not in load_registered_l0_tasks(registry):
                raise ValueError('V registration does not contain its native task')
        inherited = loaded.get('_prepared_states', {}).get(native)
        if prepared is not None and inherited is not None and prepared != inherited:
            raise ValueError('V prepared bundle differs from its frozen inherited package')
        bundle = prepared if prepared is not None else inherited
        if bundle is None:
            raise ValueError('V configuration has no explicit prepared bundle')
        native_args = native_arguments(Path(inputs['v_collection_binding_json']),
                                       ['--registered-public-action20-marker-se2-navigation'])
        extra = {'v_package_json': row['v_package']['path'],
                 'task_specs_jsonl': row['source_task_spec']['path'],
                 'v_identity': {'collection_config_key': row['collection_config_key'],
                    'task_spec_id': native, 'package_id': row['package_id'], 'branch': row['branch'],
                    'runtime_revision': binding['runtime']['commit'],
                    'historical_oracle_runtime_revision': (row['source_runtime']['recorded_revision']
                        if row['oracle_success'] else None),
                    'source_oracle_success': row['oracle_success'],
                    'runtime_validation_only': binding.get('purpose') == 'VALIDATION_ONLY',
                    'v_package_sha256': row['v_package']['sha256']}}
    else:
        raise ValueError('Unsupported AMV entrypoint')
    from behavior_hl.embodied.evaluator_bridge.prepared_model_state import validate_bundle
    validate_bundle(bundle, instruction_id=native, seed=seed,
                    observation_profile='r1pro_three_rgb_head_eval')
    if (identity['scene_model'] != inputs['scene_model'] or seed != inputs['model_task']['task_seed']):
        raise ValueError('AMV native scene or seed differs from frozen public metadata')
    model_task = {**inputs['model_task'], 'task_split': identity['split'],
                  'task_instance_id': str(identity['instance_id'])}
    if inputs['entrypoint'] == 'v_collection':
        model_task.update(collection_config_key=inputs['public_config_id'], v_identity=dict(extra['v_identity']))
    model_task['instruction_provenance'] = {**model_task['instruction_provenance'],
                                            'source_task_identity': dict(identity)}
    return {**inputs, **extra, 'model_task': model_task, 'native_args': native_args,
            'static_contract_verified': True, 'cpu_admission': 'PASS'}
