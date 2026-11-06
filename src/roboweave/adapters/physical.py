"""Frozen P30/L2 bindings; runtime-specific admission stays in its native profile."""
import hashlib
import json
from pathlib import Path

def _read(path):
    path = Path(path)
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError('Oversized task metadata')
    return json.loads(path.read_text())


def resolve_inputs(config, task, manifest, rows, root):
    """Resolve hash-checked assets without importing any historical runtime module."""
    from ..assets import _path
    if manifest['profile'] != task['profile'] or config['profile'] != task['profile']:
        raise ValueError('Frozen task belongs to a different version profile')
    entry = manifest['inputs_by_task'][task['task_id']]
    instruction = task['runtime_instruction_id']
    if (entry['instruction_id'] != instruction or entry['instruction_en'] != task['instruction_en']
            or entry['action_mode'] != task['action_mode'] or entry['budget'] != task['budget']
            or hashlib.sha256(task['instruction_en'].encode()).hexdigest() != task['instruction_sha256']):
        raise ValueError('Frozen task public identity, instruction or budget changed')
    if entry['entrypoint'] not in ('prepared', 'physical_collection'):
        raise ValueError('Unknown frozen task entrypoint')
    if (entry['entrypoint'] == 'physical_collection') != (task['family'] == 'L2' and entry['action_mode'] == 'Physical'):
        raise ValueError('Physical reconstruction route differs from the task contract')
    by_key = {row['key']: row for row in rows}
    paths = {}
    for field, key in entry.items():
        if field.endswith(('_json', '_jsonl')):
            if key not in by_key or task['task_id'] not in by_key[key]['task_ids']:
                raise ValueError('Input reference is outside the selected asset manifest')
            paths[field] = str(_path(root, by_key[key]))
    review = _read(paths['instruction_review_json'])
    selected = [row for row in review.get('rewrites', [])
                if row.get('task_id') == entry['public_task_id'] and row.get('task_name') == instruction]
    if review.get('status') != 'APPROVED' or len(selected) != 1 or selected[0].get('instruction') != entry['instruction_en']:
        raise ValueError('Frozen public English overlay is not approved or differs from task')
    identity = entry['source_task_identity']
    model_task = dict(task_id=entry['public_task_id'], task_name=instruction,
        task_split=identity['split'], task_instance_id=str(identity['instance_id']), task_seed=entry['seed'],
        instruction=entry['instruction_en'], budget=dict(entry['budget']), **entry['budget'],
        instruction_provenance=dict(profile=task['profile'], instruction_sha256=task['instruction_sha256'],
            source_task_identity=dict(identity), review_sha256=by_key[entry['instruction_review_json']]['sha256'],
            source_profile_sha256=entry.get('source_profile_sha256')))
    return dict(profile=task['profile'], entrypoint=entry['entrypoint'], action_mode=entry['action_mode'],
        asset_root=str(root), instruction_id=instruction, model_task=model_task, history_turns=entry['history_turns'],
        mapping={row['source_root']+'/'+row['source_path']:str(_path(root,row)) for row in rows},
        source_task_identity=dict(identity), **paths)


def inspect_inputs(profile, inputs):
    """Invoke native metadata validators only after the caller activates this profile.

    L2's approved English is an overlay, so its source TaskSpec text is retained
    rather than required to equal the reviewed model-facing instruction.
    """
    from lab.instruction_handoff import compile_handoff, native_instruction_arguments
    from lab.prepared_state_inputs import load_prepared
    from ..tasks import profile_for
    definition = profile_for(profile) if isinstance(profile, str) else profile
    if definition['profile'] != inputs['profile']:
        raise ValueError('Active profile differs from resolved inputs')
    instruction = inputs['instruction_id']
    handoff = Path(inputs['handoff_json'])
    if inputs['entrypoint'] == 'physical_collection':
        from lab.physical_collection_handoff import load_physical_instruction, native_physical_instruction_arguments
        loaded = load_physical_instruction(handoff, instruction)
        matrix, row, package = loaded['matrix'], loaded['instruction'], loaded['package']
        native = native_physical_instruction_arguments(handoff, instruction)
        strategy = loaded['initial_state_strategy']
    else:
        matrix = compile_handoff(handoff, {'model': 1})
        row = matrix['instructions'][instruction]
        native = native_instruction_arguments(handoff, instruction)
        package = _read(row.get('runtime_package', row['package'])['path'])
        strategy = package.get('initial_state_compilation', {}).get('reset_mode')
    if row['runtime_commit'] != definition['runtime_revision']:
        raise ValueError('Handoff runtime revision differs from frozen profile')
    if row['source_task_identity'] != inputs['source_task_identity']:
        raise ValueError('Native task identity differs from frozen binding')
    conditions = matrix['model_conditions']
    if (row['seed'] != inputs['model_task']['task_seed']
            or conditions['action_mode'] != inputs['action_mode']):
        raise ValueError('Native action mode or seed differs from frozen binding')
    bundle = load_prepared(Path(inputs['prepared_index_json']), matrix['handoff_sha256'], instruction)
    if (bundle['seed'] != row['seed'] or any(bundle['execution_contract'].get(k) != conditions.get(k)
            for k in ('action_mode','physics_steps_per_action','scene_load','observation_profile','overhead_view_mode'))):
        raise ValueError('Prepared evidence execution contract differs from handoff')
    if inputs['profile'].startswith('l1-p-') and row['instruction'] != inputs['model_task']['instruction']:
        raise ValueError('P public instruction differs from its frozen TaskSpec')
    if inputs['profile'].startswith('l1-p-'):
        from lab.l1p_model_entry import admit_l1p_model_start, model_scene_physical_kwargs
        handoff_record = _read(handoff)
        admit_l1p_model_start(package=package, task_spec=row['task_spec'],
            budgets=handoff_record.get('budgets'), preparation_only=handoff_record.get('run_class')
            in {'restore_validation_canary', 'prepared_model_evaluation'})
        model_scene_physical_kwargs(matrix, package)
    return {**inputs, 'native_args': native, 'admission': {
        'status': 'STATIC_INPUTS_VALIDATED', 'live_verified': False,
        'initial_state_strategy': strategy, 'model_prepared_states': None if inputs['entrypoint']=='physical_collection' else 'prepared_index',
        'source_package_live_run_ready': package.get('live_run_ready'),
        'source_package_promotion_label': package.get('promotion_label'),
        'source_task_spec_instruction_sha256': hashlib.sha256(row['instruction'].encode()).hexdigest(),
        'public_overlay_retained': inputs['profile']=='l2-collection37'}}
