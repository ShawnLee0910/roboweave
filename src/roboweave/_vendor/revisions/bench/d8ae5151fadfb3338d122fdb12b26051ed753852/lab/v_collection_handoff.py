"""Explicit V collection identity and native launch binding; no protocol upgrade."""
from __future__ import annotations
import hashlib
import importlib
import json
from pathlib import Path
import subprocess

SCHEMA = 'human-v-collection-binding/v1'
CONFLICTING = {'--v-package','--v-branch','--v-suite','--benchmark-package',
               '--runtime-registration','--l0-taskspecs','--reset-pose-overlay',
               '--compile-initial-state','--construct-and-oracle','--nearby-initial-pose-search',
               '--partial-scene-load'}


def read_ref(ref):
    path=Path(ref['path'])
    if not path.is_file() or path.stat().st_size>32*1024*1024:
        raise ValueError('V referenced file missing or oversized: '+str(path))
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ref['sha256']:
        raise ValueError('V referenced file changed: '+str(path))
    return raw


def load_binding(path):
    data=json.loads(Path(path).read_text())
    if data.get('schema_version')!=SCHEMA:
        raise ValueError('invalid V collection binding schema')
    index=json.loads(read_ref(data['success_index']))
    rows=[r for r in index['rows'] if r['collection_config_key']==data['collection_config_key']]
    if len(rows)!=1:
        raise ValueError('V collection key absent or duplicate')
    row=rows[0]
    experimental_r4 = (
        row.get('oracle_success') is False
        and data.get('purpose') == 'VALIDATION_ONLY'
        and data.get('validation_kind') == 'NEW_EXPLICIT_R4_WITHOUT_CURRENT_Q1'
        and row.get('class') == 'R4'
        and row.get('trigger_mode') == 'BEFORE_SELECTED_PLACE_INSIDE'
        and data['runtime']['commit'] == row['source_runtime']['recorded_revision']
    )
    if row['branch']!='D1' or (not row['oracle_success'] and not experimental_r4):
        raise ValueError('collection source is not a D1 Oracle success')
    runtime=data['runtime']
    if runtime['commit']!=row['source_runtime']['recorded_revision']:
        ref=data.get('compatibility_report')
        if ref is None or data.get('purpose')!='VALIDATION_ONLY':
            raise ValueError('V historical runtime cannot be upgraded by rebinding')
        report=json.loads(read_ref(ref))
        expected_protocol=row['class']+'_'+row['trigger_mode'] if row['class']=='R4' else row['trigger_mode']
        entry=report.get('configurations',{}).get(row['collection_config_key'],{})
        goal_sha=hashlib.sha256(json.dumps(row['complete_goal'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if (report.get('schema_version')!='v-runtime-cpu-compatibility/v1'
                or report.get('runtime_commit')!=runtime['commit']
                or report.get('success_index_sha256')!=data['success_index']['sha256']
                or entry.get('historical_runtime')!=row['source_runtime']['recorded_revision']
                or entry.get('v_package_sha256')!=row['v_package']['sha256']
                or entry.get('trigger_mode')!=row['trigger_mode']
                or entry.get('protocol')!=expected_protocol
                or entry.get('complete_goal_sha256')!=goal_sha
                or entry.get('cpu_package_pass') is not True):
            raise ValueError('V unified Runtime compatibility binding mismatch')
    package=json.loads(read_ref(row['v_package']))
    if experimental_r4 and not package.get('donor_v_package'):
        raise ValueError('new R4 validation requires its explicit frozen donor package')
    expected_schema={'R1':'behavior.v-r1-package/v1','R3':'behavior.v-r3-package/v1','R4':'behavior.v-r4-package/v1'}[row['class']]
    if package.get('schema_version')!=expected_schema:
        raise ValueError('V class and package schema differ')
    if package['package_id']!=row['package_id'] or package['task_spec_id']!=row['task_spec_id']:
        raise ValueError('V package identity mismatch')
    mode=package.get('trigger_mode',row['class']+'_LEGACY')
    if mode!=row['trigger_mode']:
        raise ValueError('V intervention protocol changed')
    specs=[json.loads(l) for l in read_ref(row['source_task_spec']).splitlines() if l.strip()]
    selected=[s for s in specs if s['task_spec_id']==row['task_spec_id']]
    if len(selected)!=1:
        raise ValueError('V original TaskSpec missing or duplicate')
    spec=selected[0]
    goal=spec.get('goal_contract') or spec.get('private',{}).get('goal_spec')
    if goal!=row['complete_goal']:
        raise ValueError('V original complete goal differs from success index')
    instruction=spec.get('public_instruction') or spec.get('public',{}).get('instruction')
    if instruction!=row['instruction_en']:
        raise ValueError('V original instruction changed')
    prepared=json.loads(read_ref(row['prepared_state'])) if row.get('prepared_state') else None
    return data,row,package,spec,prepared


def native_arguments(path, extra=()):
    _,row,_,_,_=load_binding(path)
    if any(arg.split('=',1)[0] in CONFLICTING for arg in extra):
        raise ValueError('V launch cannot mix donor, suite or alternative V flags')
    return list(extra)+['--v-package',row['v_package']['path'],'--v-branch',row['branch']]


def verify_runtime(binding):
    from esi_benchmark.profiles import verify_bundle
    import sys
    root = Path(binding['runtime']['root']).resolve()
    verify_bundle(root, binding['runtime']['commit'])
    module = importlib.import_module('behavior_hl.embodied.evaluator_bridge.v_intervention')
    for name, loaded in tuple(sys.modules.items()):
        if name == 'behavior_hl' or name.startswith('behavior_hl.'):
            origin = getattr(loaded, '__file__', None)
            if origin is None or not Path(origin).resolve().is_relative_to(root):
                raise ValueError('mixed V Runtime import roots')
    return module



def serve(args, extra):
    """Explicit alternate entry within package_scene_server, never a donor-only fallback."""
    binding,row,package,spec,prepared=load_binding(args.v_collection_binding)
    public_config_id=getattr(args,'public_config_id',None)
    if public_config_id is not None and public_config_id!=row['collection_config_key']:
        raise ValueError('V public configuration identity differs')
    if args.instruction_id!=row['task_spec_id'] or args.instruction_group or args.handoff:
        raise ValueError('one exact V collection configuration per resident required')
    if args.prepared_state_index is not None or args.public_task_ids is not None:
        raise ValueError('V collection uses its indexed prepared state and a single public handle')
    module=verify_runtime(binding)
    config,_,_,_=module.load_package(Path(row['v_package']['path']),
                                   task_names=[row['task_spec_id']],seed=package['seed'])
    inherited=config.get('_prepared_states',{}).get(row['task_spec_id'])
    if prepared is None:
        prepared=inherited
    if inherited is not None and prepared!=inherited:
        raise ValueError('V prepared bundle differs from the inherited V package')
    if prepared is None:
        raise ValueError('V configuration has no explicit prepared bundle; native reset compatibility remains pending')
    from behavior_hl.embodied.evaluator_bridge.prepared_model_state import validate_bundle
    validate_bundle(prepared,instruction_id=row['task_spec_id'],seed=package['seed'],
                    observation_profile='r1pro_three_rgb_head_eval')
    from behavior_hl.embodied.evaluator_bridge.launch_resident_scene_oracle_batch import main as prepare
    from scene_server import make_scene_server_callback,_resolve_task_profile
    from instruction_handoff import model_profile
    from scene_protocol import control_socket_path
    from package_scene_server import registered_task_for_name
    argv=native_arguments(args.v_collection_binding,extra)
    if any(arg.startswith(('--seed=', '--task-names=')) for arg in argv) or argv.count('--seed')>1 or argv.count('--task-names')>1:
        raise ValueError('V seed/task selectors must be unique explicit arguments')
    if '--task-names' not in argv:
        argv+=['--task-names',row['task_spec_id']]
    else:
        offset=argv.index('--task-names')+1
        names=[]
        while offset<len(argv) and not argv[offset].startswith('--'):
            names.append(argv[offset]);offset+=1
        if names!=[row['task_spec_id']]:raise ValueError('V task selection changed')
    if '--seed' in argv and int(argv[argv.index('--seed')+1])!=package['seed']:
        raise ValueError('V seed changed')
    if '--seed' not in argv:argv+=['--seed',str(package['seed'])]

    def callback(*,item_id,profile,prepared,**runtime):
        if item_id!=row['task_spec_id']:raise ValueError('native V preparation identity mismatch')
        session=module.get_session(runtime['evaluator'])
        if session is None or session.config['package_id']!=row['package_id'] or session.branch!=row['branch']:
            raise ValueError('native V session missing before human service')
        validator=runtime.pop('initial_state_validator',None)
        state=prepared['prepared']
        if state['oracle_actions_executed'] or state['physics_steps_executed']:
            raise ValueError('human V service must precede Oracle execution')
        frozen_prepared=json.loads(read_ref(row['prepared_state'])) if row.get('prepared_state') else inherited
        inner=make_scene_server_callback(runtime_root=args.runtime_root,
            socket_path=control_socket_path(args.runtime_root),ready_path=args.runtime_root/'ready.json',
            seed=package['seed'],profile=model_profile(profile,_resolve_task_profile),fixture_rules=None,
            scene_load_mode='full',l0_task_spec=spec if spec.get('task_spec_version')=='action20-capability-benchmark.l0-task/v1' else None,
            registered_task=registered_task_for_name(item_id),public_task_id=args.public_task_id,
            # The prepared bundle is keyed by the native V task identity.  The
            # public collection key may be an alias (e.g. VHARD100_*:D1).
            public_instruction_id=item_id,preserve_prepared_initial_state=True,
            oracle_prepared_state=frozen_prepared,initial_state_validator=validator,
            v_collection_binding={'collection_config_key':row['collection_config_key'],
                'task_spec_id':item_id,'package_id':row['package_id'],'branch':row['branch'],
                'runtime_revision':binding['runtime']['commit'],
                'historical_oracle_runtime_revision':(row['source_runtime']['recorded_revision']
                    if row['oracle_success'] else None),
                'source_oracle_success':row['oracle_success'],
                'runtime_validation_only':binding.get('purpose')=='VALIDATION_ONLY',
                'v_package_sha256':row['v_package']['sha256']})
        return dict(inner(**runtime),stop_batch=True)
    return prepare(argv,model_callback=callback,model_prepared_states={row['task_spec_id']:prepared})


def load_lane_job(job):
    """Hash-bound V collection job used by the normal resident launcher."""
    required={'v_collection_binding','v_collection_binding_sha256','instruction_id'}
    if not isinstance(job,dict) or not required<=set(job) or set(job)-required-{'collection_config_key'}:
        raise ValueError('invalid V lane binding fields')
    read_ref({'path':job['v_collection_binding'],'sha256':job['v_collection_binding_sha256']})
    result=load_binding(Path(job['v_collection_binding']))
    if result[1]['task_spec_id']!=job['instruction_id']:
        raise ValueError('V lane task identity differs from binding')
    if 'collection_config_key' in job and job['collection_config_key']!=result[1]['collection_config_key']:
        raise ValueError('V lane configuration differs from binding')
    return result


def model_instruction_job(binding_path):
    """Use the very same hash-bound job as the human console; no donor fallback."""
    path=Path(binding_path).resolve()
    _,row,_,_,_=load_binding(path)
    return {'v_collection_binding':str(path),
            'v_collection_binding_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'instruction_id':row['task_spec_id']}

def native_source_identity(binding_path):
    """Resolve the same frozen native profile used by the V launcher, without Kit."""
    from behavior_hl.embodied.evaluator_bridge.v_intervention import load_package
    binding,row,package,_,_=load_binding(Path(binding_path))
    config,spec_path,_,_=load_package(Path(row['v_package']['path']),task_names=[row['task_spec_id']],seed=package['seed'])
    if config.get('_source_adapter')=='registered_l0':
        base_path=Path(config['_base_package_path'])
        base=json.loads(base_path.read_text())
        ref=dict(base['runtime']['registration_catalog'])
        if not Path(ref['path']).is_absolute():ref['path']=str(base_path.parent/ref['path'])
        catalog=json.loads(read_ref(ref))
        matches=[t for t in catalog['tasks'] if t.get('task_spec_id',t.get('candidate_id'))==row['task_spec_id']]
        if len(matches)!=1:raise ValueError('V native catalog identity is not unique')
        return {k:str(matches[0]['profile'][k]) for k in ['scene_model','split','instance_id']}
    if spec_path is None:raise ValueError('V native identity source unavailable')
    rows=[json.loads(line) for line in Path(spec_path).read_text().splitlines() if line.strip()]
    selected=[s for s in rows if s['task_spec_id']==row['task_spec_id']]
    if len(selected)!=1:raise ValueError('V native identity is not unique')
    identity=selected[0]['private']['source_task_identity']
    return {k:str(identity[k]) for k in ['scene_model','split','instance_id']}
