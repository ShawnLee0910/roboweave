"""L1-P model admission and public-only context. No scheduler."""
from __future__ import annotations

import json
from pathlib import Path

from lab.instruction_handoff import compile_handoff, read_json

BUDGET_FIELDS = (
    'model_keys',
    'repeats_per_model',
    'max_concurrency',
    'max_action_decisions',
    'wall_clock_seconds',
)
PRIVATE_MARKERS = (
    'oracle_prefix',
    'canonical_goal_atoms',
    'canonical_goal_leaves',
    'oracle_chosen',
    'goal_clauses',
    'private_runtime',
    'construction_contract',
    'source_artifacts',
    'selected_witness_private',
    'complete_goal_leaf_truth_private',
)


def admit_l1p_model_start(*, package, task_spec=None, budgets=None, english_status=None, preparation_only=False, live_restore_report=None):
    """Refuse Live/model start until English, restore, model list and budgets are set."""
    if package.get('required_qualification') != 'physical_template_review':
        return package
    if package.get('model_rollout_allowed') is not True:
        raise ValueError('package does not admit model rollout')
    if package.get('native_q_alone_is_success') is not False:
        raise ValueError('native Q cannot be the P success rule')
    if package.get('required_qualification') != 'physical_template_review':
        raise ValueError('Physical template qualification is required')
    status = english_status
    if status is None and task_spec is not None:
        status = task_spec.get('public_instruction_en_status')
    if status != 'APPROVED':
        raise ValueError('formal English Instruction is missing')
    if not isinstance(budgets, dict):
        raise ValueError('model or budget is unset')
    for key in BUDGET_FIELDS:
        value = budgets.get(key)
        if value in (None, '', []) or (key == 'model_keys' and not value):
            raise ValueError('model or budget is unset')
        if key != 'model_keys' and value in (0, 0.0):
            raise ValueError('model or budget is unset')
    frozen = package.get('initial_state_compilation', {})
    if preparation_only:
        if frozen.get('reset_mode') != 'RESTORE_AND_VERIFY_NO_RECONSTRUCTION':
            raise ValueError('restore contract is missing')
        return package  # Scene preparation only; caller must recheck before publishing READY.
    if live_restore_report is not None:
        if (live_restore_report.get('status') != 'PASS'
                or live_restore_report.get('instruction_id') != task_spec.get('task_spec_id')
                or live_restore_report.get('policy') != 'ORACLE_PREPARED_ONLY'):
            raise ValueError('live restore evidence differs')
        if frozen.get('reset_mode') != 'RESTORE_AND_VERIFY_NO_RECONSTRUCTION':
            raise ValueError('restore contract is missing')
        return package
    if frozen.get('status') != 'FROZEN' or frozen.get('reset_mode') != 'RESTORE_AND_VERIFY_NO_RECONSTRUCTION':
        raise ValueError('restore is not verified; refuse model start')
    return package


def public_model_context(handoff_path, instruction_id):
    """Public observation binding only. Private goals, Oracle choice and prefix stay out."""
    matrix = compile_handoff(Path(handoff_path), {'model': 1})
    row = matrix['instructions'][instruction_id]
    spec = row['task_spec']
    instruction = spec.get('public', {}).get('instruction') or spec.get('public_instruction')
    context = {
        'schema_version': 'l1p-public-model-context/v1',
        'instruction_id': instruction_id,
        'instruction': instruction,
        'instruction_en_status': spec.get('public_instruction_en_status'),
        'observation_profile': matrix['model_conditions']['observation_profile'],
        'action_mode': matrix['model_conditions']['action_mode'],
        'restore_initial_state_before_model': matrix['model_conditions']['restore_initial_state_before_model'],
    }
    encoded = json.dumps(context, ensure_ascii=False)
    for marker in PRIVATE_MARKERS:
        if marker in encoded:
            raise ValueError('public model context leaked private field: ' + marker)
    if 'oracle_prefix' in spec or spec.get('physical_template', {}).get('oracle_prefix'):
        # Presence on the TaskSpec is expected; it must not be copied here.
        pass
    return context


def model_scene_physical_kwargs(matrix, package):
    """Carry Physical execution into the scene server. Native Q is not success."""
    physical = matrix['model_conditions'].get('action_mode') == 'Physical'
    if not physical:
        return {
            'placement_physical_consequences_enabled': False,
            'native_q_alone_is_success': True,
        }
    if package.get('native_q_alone_is_success') is not False:
        raise ValueError('native Q cannot be the P success rule')
    if package.get('required_qualification') != 'physical_template_review':
        raise ValueError('Physical template qualification is required')
    if matrix['model_conditions'].get('physics_steps_per_action') is not None:
        raise ValueError('Physical model physics_steps_per_action must stay action-dependent')
    return {
        'placement_physical_consequences_enabled': True,
        'native_q_alone_is_success': False,
    }


def restore_blocks_model_start(exc):
    name = type(exc).__name__
    text = str(exc)
    return name in {'PreparedModelStateError', 'ValueError'} or 'restore' in text.lower()


def physical_model_may_auto_end(*, official_success, native_q_alone_is_success, qualification_status):
    """Native Q cannot end a P episode. Qualification PASS is required."""
    if not official_success:
        return False
    if native_q_alone_is_success:
        return True
    return qualification_status == 'PASS'
