"""Opt-in L2 composition identity, using the existing M process runtime."""
import hashlib
import json
from pathlib import Path

CAPABILITY = 'L2_AM_COMPOSITION_V1'
MV_CAPABILITY = 'L2_MV_COMPOSITION_V1'
MP_CAPABILITY = 'L2_MP_COMPOSITION_V1'
SCHEMA = 'action20.l2-composition-contract/v1'


def validate_composition(package, spec, record, capabilities):
    value = spec.get('composition_contract')
    if package.get('benchmark_level') != 'L2':
        if value is not None or record.get('composition_contract') is not None or spec.get('benchmark_level')=='L2':
            raise ValueError('L2 composition cannot be hidden in another level')
        return
    fields={'schema_version','axes','candidate_id','backend','execution_mode','intake'}
    if isinstance(value,dict) and value.get('axes') in (['M','V'], ['M','V','P'], ['A','M','V']):
        fields.add('v_stage')
        if value.get('v_stage') in {'D0','D1'}:fields.add('v_sidecar')
    axes = value.get('axes') if isinstance(value, dict) else None
    required_capability = (
        MV_CAPABILITY if axes in (['M','V'], ['M','V','P'], ['A','M','V']) else
        MP_CAPABILITY if axes == ['M','P'] else CAPABILITY
    )
    physical_axes = axes in (['M','P'], ['M','V','P'])
    expected_mode = 'Physical' if physical_axes else 'Symbolic'
    if (required_capability not in capabilities or spec.get('benchmark_level') != 'L2'
            or not isinstance(value,dict) or value != record.get('composition_contract')
            or set(value) != fields
            or value['schema_version'] != SCHEMA or value['axes'] not in [['A','M'],['M','V'],['M','P'],['M','V','P'],['A','M','V']]
            or value['backend'] != 'L1_HISTORY_PROCESS' or value['execution_mode'] != expected_mode):
        raise ValueError('unsupported or mismatched L2 composition')
    policy=package.get('requested_execution_policy',{})
    if physical_axes:
        if MP_CAPABILITY not in capabilities:
            raise ValueError('Physical composition capability missing')
        if policy.get('action_mode') not in {'Physical', 'PhysicalPlacement'}:
            raise ValueError('MP composition requires declared Physical execution')
    elif policy.get('action_mode')!='Symbolic' or policy.get('physics_steps_per_action')!=0:
        raise ValueError('AM composition requires declared Symbolic execution')
    ref=value['intake']
    if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:raise ValueError('invalid L2 intake reference')
    p=Path(ref['path'])
    if not p.is_absolute() or p.stat().st_size>16*1024*1024:raise ValueError('invalid L2 intake path')
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('L2 intake hash mismatch')
    envelope=json.loads(raw)
    if (envelope.get('schema_version')!='action20.l2-composition-intake/v1'
            or envelope.get('benchmark_level')!='L2' or envelope.get('axes')!=value['axes']
            or envelope.get('candidate_id')!=value['candidate_id'] or envelope.get('execution_mode')!=expected_mode):
        raise ValueError('L2 intake identity mismatch')
    contract=spec.get('task_contract',{})
    expected_axis = 'P' if physical_axes else 'M'
    if contract.get('axis')!=expected_axis or contract.get('proposal_id')!=value['candidate_id']:
        raise ValueError('L2 component identity mismatch')
    process=contract.get('process_requirements',{})
    if not (process.get('history_program') or process.get('process_program')):
        raise ValueError('L2 AM requires executable memory process')
    if spec.get('scene_model') and spec['scene_model']!=envelope['primary_donor']['scene']:
        raise ValueError('L2 primary world mismatch')
    if physical_axes:
        physical = spec.get('physical_template')
        if not isinstance(physical, dict) or not physical or physical != record.get('physical_template'):
            raise ValueError('MP requires matching physical template contracts')
    if value['axes'] in (['M','V'], ['M','V','P'], ['A','M','V']):
        if value['v_stage'] not in {'BASE','D0','D1'}:raise ValueError('invalid MV stage')
        if physical_axes and value['v_stage']=='BASE':
            raise ValueError('Physical V requires explicit sidecar; use MP for Base')
        if 'A' not in axes and envelope.get('information_contract') is not None:raise ValueError('MV must not pretend A declaration')
        if value['v_stage']!='BASE':
            from .l2_mv_session import validate_sidecar
            config=validate_sidecar(value['v_sidecar'],spec)
            if config['branch']!=value['v_stage']:raise ValueError('MV branch mismatch')
            if config.get('validation_mode') == 'PRIVILEGED_TEST_ONLY':
                if package.get('validation_mode') != 'PRIVILEGED_TEST_ONLY' or package.get('promotable') is not False:
                    raise ValueError('test-only V requires an explicitly non-promotable package')
        if 'A' not in axes:
            return
        if CAPABILITY not in capabilities:
            raise ValueError('AMV requires the A/M composition capability')
    if value['axes']==['M','P']:
        if envelope.get('information_contract') is not None:
            raise ValueError('MP must not pretend A information declaration')
        return
    info=envelope.get('information_contract',{})
    if (set(info)!={'missing_fact','legal_acquisition','dependent_decision','evidence_status'}
            or any(not isinstance(v,str) or not v.strip() for v in info.values())
            or info['evidence_status']!='PENDING_VISUAL_AND_BEHAVIOR_CHECK'):
        raise ValueError('L2 A information declaration missing or falsely certified')
    if spec.get('scene_model') and spec['scene_model']!=envelope['primary_donor']['scene']:
        raise ValueError('L2 primary world mismatch')
