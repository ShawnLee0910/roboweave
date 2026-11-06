"""Serve one handed-off Instruction after the native initial-state preparation.

Run inside the existing isolated Isaac environment. The parent owns the slot and
AppData lease; this entry point neither claims capacity nor starts a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from instruction_handoff import compile_handoff, read_json, model_profile


def registered_task_for_name(instruction_id):
    from behavior_hl.embodied.evaluator_bridge.registered_l0_task_runtime import _ACTIVE_REGISTERED_L0_TASKS
    return _ACTIVE_REGISTERED_L0_TASKS.get(instruction_id)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--instruction-id', required=True)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--public-task-id', type=int)
    parser.add_argument('--prepared-state-index', type=Path, required=True)
    parser.add_argument('--instruction-group')
    parser.add_argument('--public-task-ids')
    args, native_args = parser.parse_known_args(argv)
    matrix = compile_handoff(args.handoff, {'model': 1})
    from instruction_scene_groups import continuation, scene_groups
    group = [args.instruction_id]
    handles = {args.instruction_id: args.public_task_id}
    if args.instruction_group:
        declared = args.instruction_group.split(',')
        full_group = scene_groups(matrix['instructions'], maximum=None)[declared[0]]
        if declared != [key for key in full_group if key in declared]:
            raise ValueError('resident group is not the frozen compatible group')
        group = continuation(declared, args.instruction_id)
        handles = json.loads(args.public_task_ids)
        if set(handles) != set(declared) or handles[args.instruction_id] != args.public_task_id:
            raise ValueError('Instruction handle mapping changed')
        offset = native_args.index('--task-names') + 1
        if native_args[offset] != args.instruction_id:
            raise ValueError('native primary Instruction mismatch')
        native_args[offset:offset + 1] = group
    instruction = matrix['instructions'][args.instruction_id]
    from lab.prepared_state_inputs import load_prepared
    oracle_prepared = {key: load_prepared(args.prepared_state_index, matrix['handoff_sha256'], key) for key in group}
    from behavior_hl.embodied.evaluator_bridge.prepared_model_state import validate_bundle
    for key, bundle in oracle_prepared.items():
        validate_bundle(bundle, instruction_id=key, seed=matrix['instructions'][key]['seed'],
                        observation_profile='r1pro_three_rgb_head_eval')
    if '--nearby-initial-pose-search' in native_args:
        raise ValueError('Oracle prepared replay forbids a pose-search fallback')
    package = read_json(Path(instruction.get('package', matrix.get('package'))['path']))
    ref = package['task_specs']
    spec_path = Path(ref['path'])
    if spec_path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('oversized TaskSpec file')
    contents = spec_path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != ref['sha256']:
        raise ValueError('package TaskSpecs changed')
    specs = [json.loads(line) for line in contents.splitlines() if line.strip()]
    selected = {s['task_spec_id']: s for s in specs if s['task_spec_id'] in group}
    if set(selected) != set(group):
        raise ValueError('resident group TaskSpecs missing')
    for key, spec in selected.items():
        if (spec.get('public', {}).get('instruction') or spec.get('public_instruction')) != matrix['instructions'][key]['instruction']:
            raise ValueError('Instruction binding mismatch')
    from lab.l1p_model_entry import admit_l1p_model_start, model_scene_physical_kwargs
    handoff_raw = read_json(args.handoff)
    restore_canary = handoff_raw.get('run_class') in ('restore_validation_canary', 'prepared_model_evaluation')
    if restore_canary and (len(group) != 1 or len(matrix['instructions']) != 1):
        raise ValueError('restore canary requires one Instruction')
    if package.get('required_qualification') == 'physical_template_review':
        admit_l1p_model_start(
            package=package,
            task_spec=selected[args.instruction_id],
            budgets=handoff_raw.get('budgets'),
            preparation_only=restore_canary,
        )
    from scene_server import make_scene_server_callback, _resolve_task_profile
    from scene_protocol import control_socket_path
    from behavior_hl.embodied.evaluator_bridge.launch_resident_scene_oracle_batch import main as prepare

    def serve(*, item_id, profile, prepared, **runtime):
        initial_state_validator = runtime.pop('initial_state_validator', None)
        if item_id not in group:
            raise ValueError('native preparation selected a different Instruction')
        instruction = matrix['instructions'][item_id]
        task_spec = selected[item_id]
        actual = {k: getattr(profile, k) for k in ('task_id', 'scene_model', 'split', 'instance_id')}
        expected = instruction['source_task_identity']
        if expected is not None and actual != {k: expected[k] for k in actual}:
            raise ValueError('native source identity differs from handoff')
        if expected is None and actual['scene_model'] != instruction['scene_model']:
            raise ValueError('registered scene mismatch')
        profile = model_profile(profile, _resolve_task_profile)
        state = prepared['prepared']
        if state['oracle_actions_executed'] != 0 or state['physics_steps_executed'] != 0:
            raise ValueError('model must start before Oracle execution')
        registered_task = registered_task_for_name(item_id)
        if task_spec.get('task_contract', {}).get('process_requirements') is not None and registered_task is None:
            raise ValueError('declared process contract lacks its registered runtime task')
        physical_kwargs = model_scene_physical_kwargs(matrix, package)
        if restore_canary and package.get('required_qualification') == 'physical_template_review':
            report_path = Path(runtime['run_dir']) / 'private/resident_tasks' / item_id / 'prepared_view_restore.json'
            report = read_json(report_path)
            admit_l1p_model_start(package=package, task_spec=task_spec,
                budgets=handoff_raw.get('budgets'), live_restore_report=report)
            args.runtime_root.mkdir(parents=True, exist_ok=True)
            (args.runtime_root / 'l1p_restore_admission.json').write_text(json.dumps({
                'status': 'LIVE_RESTORE_PASS', 'instruction_id': item_id,
                'report_path': str(report_path), 'handoff_sha256': matrix['handoff_sha256'],
                'scope': 'current_episode_restore_only_not_package_promotion'}, indent=2))
        callback = make_scene_server_callback(
            runtime_root=args.runtime_root, socket_path=control_socket_path(args.runtime_root),
            ready_path=args.runtime_root / 'ready.json', seed=instruction['seed'],
            profile=profile, fixture_rules=None, scene_load_mode='full',
            l0_task_spec=task_spec if task_spec.get('task_spec_version') == 'action20-capability-benchmark.l0-task/v1' else None,
            registered_task=registered_task,
            public_task_id=handles[item_id],
            public_instruction_id=item_id,
            preserve_prepared_initial_state=True,
            oracle_prepared_state=oracle_prepared[item_id],
            initial_state_validator=initial_state_validator,
            next_instruction_id=(group[group.index(item_id) + 1]
                                 if group.index(item_id) + 1 < len(group) else None),
            **physical_kwargs,
        )
        receipt = dict(callback(**runtime))
        receipt['stop_batch'] = receipt.get('advance_instruction') is None
        return receipt

    return prepare(native_args, model_callback=serve, model_prepared_states=oracle_prepared)


if __name__ == '__main__':
    raise SystemExit(main())
