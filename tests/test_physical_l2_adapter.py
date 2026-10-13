"""Small distributable contract fixtures; no private assets or simulator imports."""
import hashlib
import json
from pathlib import Path
from types import ModuleType
import sys

import pytest

from esi_benchmark.adapters.physical import resolve_inputs, inspect_inputs


def fixture_inputs(tmp_path, *, family='L2', mode='Physical'):
    name = 'synthetic-task'
    profile = 'l2-collection37' if family=='L2' else 'l1-p-inside5'
    instruction = 'Move the synthetic object.'
    budget = dict(maximum_action_decisions=12, maximum_model_calls=36, wall_clock_seconds=7200)
    task = dict(task_id=name, runtime_instruction_id=name, family=family, profile=profile,
        instruction_en=instruction, instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
        action_mode=mode, budget=budget)
    keys = ('handoff_json','prepared_index_json','benchmark_package_json','task_specs_jsonl',
            'registration_catalog_json','prepared_bundle_json','instruction_review_json')
    rows = [dict(key=k, source_root='synthetic', source_path=k+'.json', task_ids=[name]) for k in keys]
    for row in rows:
        p=tmp_path/'files'/row['source_root']/row['source_path'];p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('{}')
    review = dict(status='APPROVED', rewrites=[dict(task_id=0,task_name=name,instruction=instruction)])
    (tmp_path/'files/synthetic/instruction_review_json.json').write_text(json.dumps(review))
    entry = dict(instruction_id=name, instruction_en=instruction, budget=budget, action_mode=mode,
        entrypoint='physical_collection' if family=='L2' and mode=='Physical' else 'prepared',
        public_task_id=0, seed=19506, history_turns=5,
        source_task_identity=dict(task_id=100,task_name=name,scene_model='synthetic',split='train',instance_id=0),
        **{k:k for k in keys})
    for row in rows:
        row['sha256']=hashlib.sha256((tmp_path/'files'/row['source_root']/row['source_path']).read_bytes()).hexdigest()
    manifest=dict(profile=profile,inputs_by_task={name:entry})
    return task,manifest,rows,dict(profile=profile),entry


def test_resolve_separates_public_handle_and_native_identity(tmp_path):
    task,manifest,rows,config,entry=fixture_inputs(tmp_path)
    result=resolve_inputs(config,task,manifest,rows,tmp_path)
    assert result['model_task']['task_id']==0
    assert result['source_task_identity']['task_id']==100
    assert result['model_task']['task_seed']==19506
    assert result['entrypoint']=='physical_collection'
    assert result['model_task']['budget']==task['budget']
    assert result['model_task']['instruction_provenance']['review_sha256']==rows[-1]['sha256']
    assert Path(result['handoff_json']).is_relative_to(tmp_path)


@pytest.mark.parametrize('change',[{'instruction_en':'changed'},{'action_mode':'Symbolic'},
    {'budget':{}},{'instruction_sha256':'wrong'}])
def test_resolve_rejects_changed_catalog(tmp_path,change):
    task,manifest,rows,config,entry=fixture_inputs(tmp_path)
    with pytest.raises(ValueError,match='identity'):
        resolve_inputs(config,{**task,**change},manifest,rows,tmp_path)


def test_p_must_not_reconstruct(tmp_path):
    task,manifest,rows,config,entry=fixture_inputs(tmp_path,family='L1-P')
    entry['entrypoint']='physical_collection'
    with pytest.raises(ValueError,match='reconstruction'):
        resolve_inputs(config,task,manifest,rows,tmp_path)


def test_review_must_be_approved_and_bind_exact_instruction(tmp_path):
    task,manifest,rows,config,entry=fixture_inputs(tmp_path)
    path=tmp_path/'files/synthetic/instruction_review_json.json'
    review=json.loads(path.read_text());review['status']='DRAFT';path.write_text(json.dumps(review))
    with pytest.raises(ValueError,match='approved'):
        resolve_inputs(config,task,manifest,rows,tmp_path)


def test_reference_must_belong_to_selected_assets(tmp_path):
    task,manifest,rows,config,entry=fixture_inputs(tmp_path)
    with pytest.raises(ValueError,match='outside'):
        resolve_inputs(config,task,manifest,rows[:-1],tmp_path)


def native_metadata(monkeypatch, inputs, *, text='Frozen source-language instruction'):
    instruction=inputs['instruction_id']
    conditions=dict(action_mode=inputs['action_mode'],physics_steps_per_action=None,
        scene_load='Full',observation_profile='r1pro_three_rgb_head_eval',overhead_view_mode='off')
    row=dict(runtime_commit='frozen',source_task_identity=inputs['source_task_identity'],
        seed=inputs['model_task']['task_seed'],instruction=text,package={'path':inputs['benchmark_package_json']},
        task_spec={'task_spec_id':instruction})
    package=dict(initial_state_compilation={'reset_mode':'RESTORE_AND_VERIFY_NO_RECONSTRUCTION'})
    Path(inputs['benchmark_package_json']).write_text(json.dumps(package))
    matrix=dict(model_conditions=conditions,instructions={instruction:row},handoff_sha256='handoff')
    bundle=dict(seed=row['seed'],execution_contract=conditions)
    module=ModuleType('lab.instruction_handoff');module.compile_handoff=lambda *a:matrix
    module.native_instruction_arguments=lambda *a:['--prepared-model-replay']
    prepared=ModuleType('lab.prepared_state_inputs');prepared.load_prepared=lambda *a:bundle
    physical=ModuleType('lab.physical_collection_handoff')
    physical.load_physical_instruction=lambda *a:dict(matrix=matrix,instruction=row,package=package,
        initial_state_strategy='RECONSTRUCT_AND_VERIFY_DONOR_PHYSICAL_CONTRACT')
    physical.native_physical_instruction_arguments=lambda *a:['--physical-human-service']
    p_entry=ModuleType('lab.l1p_model_entry')
    p_entry.admit_l1p_model_start=lambda **kwargs:kwargs['package']
    p_entry.model_scene_physical_kwargs=lambda *a:{'placement_physical_consequences_enabled':True,'native_q_alone_is_success':False}
    for mod in (module,prepared,physical,p_entry):monkeypatch.setitem(sys.modules,mod.__name__,mod)
    return row,bundle


def test_l2_overlay_preserved_and_physical_uses_reconstruction(tmp_path,monkeypatch):
    task,manifest,rows,config,entry=fixture_inputs(tmp_path)
    inputs=resolve_inputs(config,task,manifest,rows,tmp_path)
    native_metadata(monkeypatch,inputs)
    result=inspect_inputs(dict(profile=task['profile'],runtime_revision='frozen'),inputs)
    assert result['admission']['public_overlay_retained'] is True
    assert result['admission']['model_prepared_states'] is None
    assert result['admission']['live_verified'] is False
    assert result['model_task']['instruction']==task['instruction_en']
    assert result['native_args']==['--physical-human-service']


def test_p_text_and_runtime_revision_are_still_exact(tmp_path,monkeypatch):
    task,manifest,rows,config,entry=fixture_inputs(tmp_path,family='L1-P')
    inputs=resolve_inputs(config,task,manifest,rows,tmp_path)
    row,bundle=native_metadata(monkeypatch,inputs,text=task['instruction_en'])
    definition=dict(profile=task['profile'],runtime_revision='frozen')
    assert inspect_inputs(definition,inputs)['admission']['model_prepared_states']=='prepared_index'
    row['instruction']='wrong'
    with pytest.raises(ValueError,match='instruction differs'):
        inspect_inputs(definition,inputs)
    row['runtime_commit']='other'
    with pytest.raises(ValueError,match='revision'):
        inspect_inputs(definition,inputs)


@pytest.mark.parametrize('gate',['admit_l1p_model_start','model_scene_physical_kwargs'])
def test_p_original_entry_gate_can_reject_inspection(tmp_path,monkeypatch,gate):
    task,manifest,rows,config,entry=fixture_inputs(tmp_path,family='L1-P')
    inputs=resolve_inputs(config,task,manifest,rows,tmp_path)
    native_metadata(monkeypatch,inputs,text=task['instruction_en'])
    budgets={'max_action_decisions':12,'wall_clock_seconds':7200}
    Path(inputs['handoff_json']).write_text(json.dumps({'run_class':'prepared_model_evaluation','budgets':budgets}))
    def reject(*args,**kwargs):
        if gate=='admit_l1p_model_start':
            assert kwargs['preparation_only'] is True
            assert kwargs['budgets']==budgets
            assert 'live_restore_report' not in kwargs
        raise ValueError('frozen entry gate rejected')
    monkeypatch.setattr(sys.modules['lab.l1p_model_entry'],gate,reject)
    with pytest.raises(ValueError,match='frozen entry gate rejected'):
        inspect_inputs(dict(profile=task['profile'],runtime_revision='frozen'),inputs)


def test_bundled_manifest_totals_and_release_boundaries():
    from importlib.resources import files
    root=files('esi_benchmark').joinpath('data')
    total=0
    for profile,count in [('l1-p-inside5',5),('l1-p-ontop5',5),('l1-p-additions20',20),('l2-collection37',37)]:
        manifest=json.loads(root.joinpath(f'assets/{profile}.json').read_text())
        assert len(manifest['inputs_by_task'])==count
        assert all('selection' in row for row in manifest['files'])
        assert not any(row['source_path'].endswith('/profile.json') for row in manifest['files'])
        assert all(len(row['sha256'])==64 and row['bytes']>0 for row in manifest['files'])
        total+=count
    assert total==67
    catalog=json.loads(root.joinpath('tasks/l2-collection37.json').read_text())['tasks']
    assert len(catalog)==37
    assert sum(t['action_mode']=='Physical' for t in catalog)==14
    assert sum(t['action_mode']=='Symbolic' for t in catalog)==23
