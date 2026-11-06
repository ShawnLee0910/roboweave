"""Distributable AMV public binding and execution-closure contract checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from roboweave.adapters.amv import PROFILES, inspect_inputs, resolve_inputs
from roboweave.tasks import list_tasks


def test_import_does_not_activate_vendor_sources():
    result = subprocess.run([sys.executable, '-c',
        'import sys; before=list(sys.path); from roboweave.adapters import amv; '
        'assert sys.path == before; '
        'assert not any(k == "instruction_handoff" or k.startswith("behavior_hl") for k in sys.modules)'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def frozen_inputs(tmp_path):
    root = tmp_path / 'assets'
    directory = root / 'files' / 'fixture'
    directory.mkdir(parents=True)
    instruction = 'Move the cup, then return it to the table.'
    task = {'profile': 'l1-m30-public-v2', 'task_id': 'public-one',
            'runtime_instruction_id': 'native-one', 'instruction_en': instruction,
            'instruction_sha256': hashlib.sha256(instruction.encode()).hexdigest(),
            'scene_model': 'room', 'budget': {'maximum_action_decisions': 12, 'maximum_model_calls': 1000,
                                             'wall_clock_seconds': 3600}, 'source': {'row_index': 0}}
    raw = {'instruction_id': task['task_id'], 'runtime_instruction_id': task['runtime_instruction_id'],
           'instruction_en': instruction, 'budget': task['budget']}
    contents = {'tasks.json': [raw], 'review.json': {'status': 'APPROVED', 'rewrites': [
        {'task_id': 0, 'task_name': task['task_id'], 'instruction': instruction}]},
        'handoff.json': {}, 'index.json': {}}
    rows = []
    for name, value in contents.items():
        path = directory / name
        path.write_text(json.dumps(value))
        rows.append({'key': name, 'source_root': 'fixture', 'source_path': name,
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'task_ids': [task['task_id']]})
    manifest = {'profile': task['profile'], 'inputs_by_task': {task['task_id']: {
        'entrypoint': 'prepared', 'seed': 19506, 'task_json': 'tasks.json',
        'instruction_review_json': 'review.json', 'handoff_json': 'handoff.json',
        'prepared_index_json': 'index.json'}}}
    return task, manifest, rows, root


def test_m_without_source_json_preserves_exact_public_budget_and_review(frozen_inputs):
    task, manifest, rows, root = frozen_inputs
    result = resolve_inputs({}, task, manifest, rows, root)
    assert result['model_task']['task_id'] == 0
    assert result['model_task']['task_name'] == 'native-one'
    assert result['model_task']['maximum_action_decisions'] == 12
    assert result['model_task']['task_seed'] == 19506
    assert 'task_split' not in result['model_task']  # Native worker must resolve the actual identity.
    assert 'source_json' not in result
    assert Path(result['handoff_json']).is_relative_to(root)


@pytest.mark.parametrize('change', ['instruction', 'budget', 'review', 'hash'])
def test_public_binding_changes_fail_closed(frozen_inputs, change):
    task, manifest, rows, root = frozen_inputs
    if change == 'instruction':
        task['instruction_en'] = 'A different task.'
    elif change == 'budget':
        task['budget'] = {**task['budget'], 'maximum_action_decisions': 99}
    elif change == 'hash':
        task['instruction_sha256'] = '0' * 64
    else:
        path = root / 'files/fixture/review.json'
        review = json.loads(path.read_text()); review['status'] = 'DRAFT'
        path.write_text(json.dumps(review))
    with pytest.raises(ValueError):
        resolve_inputs({}, task, manifest, rows, root)


def test_profile_mismatch_rejected_before_native_imports():
    with pytest.raises(ValueError, match='profile mismatch'):
        inspect_inputs('l1-a30', {'profile': 'l1-v30'})


@pytest.mark.parametrize('profile', sorted(PROFILES))
def test_manifest_closure_and_selection_metadata(profile):
    data = Path(__file__).resolve().parents[1] / 'src/roboweave/data/assets' / (profile + '.json')
    manifest = json.loads(data.read_text())
    tasks = {t['task_id']: t for t in list_tasks() if t['profile'] == profile}
    rows = {row['key']: row for row in manifest['files']}
    assert len(manifest['inputs_by_task']) == len(tasks) == 30
    assert manifest['unique_runtime_file_count'] == len(rows)
    assert manifest['unique_runtime_bytes'] == sum(row['bytes'] for row in rows.values())
    for public, inputs in manifest['inputs_by_task'].items():
        for field, key in inputs.items():
            if field.endswith('_json'):
                assert key in rows and public in rows[key]['task_ids']
        assert inputs['seed'] == 19506
        assert inputs['entrypoint'] == ('v_collection' if profile == 'l1-v30' else 'prepared')
    for row in rows.values():
        assert 'selection' in row
        assert not Path(row['source_path']).is_absolute()
        assert '..' not in Path(row['source_path']).parts
        if 'derivation_source_handoff' in row['roles'] or 'donor_v_package' in row['roles']:
            assert row['selection'] is None
        for binding in row.get('hash_bindings', []):
            assert binding['key'] in rows
            for task_id in binding.get('task_ids', row['task_ids']):
                assert task_id in rows[binding['key']]['task_ids']


def test_generator_keeps_derivation_donor_and_rejects_changed_reference(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'tools/build_amv_asset_manifests.py'
    spec = importlib.util.spec_from_file_location('build_amv_asset_manifests', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source = tmp_path / 'source'; source.mkdir()
    donor = source / 'donor.json'
    donor.write_text(json.dumps({'samples': [{'task_spec_id': 'donor-control', 'official_q': 1}]}))
    digest = hashlib.sha256(donor.read_bytes()).hexdigest()
    closure = module.Closure(tmp_path, 'l1-a30')
    _, row = closure.add({'path': str(donor), 'sha256': digest}, 'target-challenge', 'derivation_source_handoff')
    assert row['selection'] is None and row['task_ids'] == ['target-challenge']
    with pytest.raises(ValueError, match='hash changed'):
        closure.add({'path': str(donor), 'sha256': '0' * 64}, 'target-challenge', 'derivation_source_handoff')


def test_v_runtime_binding_requires_the_installed_frozen_source(tmp_path):
    program = '''
import sys
from pathlib import Path
from roboweave._native.bootstrap import activate
from roboweave.profiles import source_revisions
roots = activate('l1-v30')
from v_collection_handoff import verify_runtime
runtime = {'root': str(roots['runtime_root']), 'commit': source_revisions('l1-v30')['runtime']}
module = verify_runtime({'runtime': runtime})
assert Path(module.__file__).resolve().is_relative_to(roots['runtime_root'])
try:
    verify_runtime({'runtime': {**runtime, 'root': sys.argv[1]}})
except (ValueError, FileNotFoundError):
    pass
else:
    raise AssertionError('A task-input path is not an installed frozen Runtime')
'''
    result = subprocess.run([sys.executable, '-B', '-c', program, str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_v_public_scene_name_keeps_native_instruction_separate(frozen_inputs):
    task, manifest, rows, root = frozen_inputs
    task['profile'] = manifest['profile'] = 'l1-v30'
    entry = manifest['inputs_by_task'][task['task_id']]
    entry.update(entrypoint='v_collection', public_config_id=task['task_id'],
                 v_collection_binding_json=entry.pop('handoff_json'))
    entry.pop('prepared_index_json')
    result = resolve_inputs({}, task, manifest, rows, root)
    assert result['model_task']['task_name'] == task['task_id'] == 'public-one'
    assert result['instruction_id'] == 'native-one'


def test_all_v_public_handles_match_frozen_scene_status_contract():
    import ast
    from types import SimpleNamespace
    from roboweave.profiles import source_roots

    source = source_roots('l1-v30')['bench_root'] / 'lab/scene_server.py'
    tree = ast.parse(source.read_text())
    status = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == 'status'
                  and any(isinstance(part, ast.Assign) and any(isinstance(target, ast.Name)
                          and target.id == 'public_name' for target in part.targets) for part in node.body))
    returned = next(node.value for node in status.body if isinstance(node, ast.Return))
    fields = {'task_id', 'task_name', 'task_split', 'task_instance_id'}
    pairs = [(key, value) for key, value in zip(returned.keys, returned.values)
             if isinstance(key, ast.Constant) and key.value in fields]
    # Execute the source's pure identity expressions without importing its simulator service.
    assignments = [node for node in status.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id in {'v_binding', 'public_name'}
                           for target in node.targets)]
    body = assignments + [ast.Assign(targets=[ast.Name(id='result', ctx=ast.Store())],
        value=ast.Dict(keys=[key for key, _ in pairs], values=[value for _, value in pairs]))]
    code = compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(source), 'exec')
    tasks = list_tasks('L1-V')
    assert len(tasks) == 30
    for index, task in enumerate(tasks):
        profile = SimpleNamespace(task_id=40000, task_name=task['runtime_instruction_id'], split='train', instance_id=0)
        scene = SimpleNamespace(profile=profile, _public_task_id=task['source']['row_index'],
                                _public_instruction_id=task['runtime_instruction_id'], _l0_task_spec=None,
                                _v_collection_binding={'collection_config_key': task['task_id']})
        namespace = {'self': scene}
        exec(code, namespace)
        assert namespace['result'] == {'task_id': index, 'task_name': task['task_id'],
                                       'task_split': 'train', 'task_instance_id': '0'}
        assert task['task_id'] != task['runtime_instruction_id']


@pytest.mark.parametrize('profile', sorted({task['profile'] for task in list_tasks()}))
@pytest.mark.parametrize('change', ['instruction', 'budget', 'native_identity'])
def test_public_inspection_rejects_changed_model_binding_before_native_admission(monkeypatch, profile, change):
    from roboweave import adapters

    task = next(task for task in list_tasks() if task['profile'] == profile)
    model_task = {'task_id': task['source']['row_index'],
                  'task_name': task['task_id'] if task['family'] == 'L1-V' else task['runtime_instruction_id'],
                  'instruction': task['instruction_en'], 'budget': dict(task['budget']), **task['budget'],
                  'instruction_provenance': {'profile': profile, 'instruction_sha256': task['instruction_sha256']}}
    inputs = {'profile': profile, 'model_task': model_task, 'instruction_id': task['runtime_instruction_id'],
              'public_config_id': task['task_id']}
    if change == 'native_identity':
        inputs['instruction_id'] = 'different-native-task'
    elif change == 'budget':
        model_task['maximum_action_decisions'] += 1
    else:
        model_task['instruction'] = 'Changed after private input materialization.'

    def unexpected_native(*args):
        raise AssertionError('Changed public binding reached native admission')

    monkeypatch.setattr(adapters, '_adapter', unexpected_native)
    with pytest.raises(ValueError, match='frozen'):
        adapters.inspect_inputs(profile, inputs)


def test_public_inspection_keeps_v_collection_and_native_id_separate(monkeypatch):
    from roboweave import adapters
    from types import SimpleNamespace

    task = list_tasks('L1-V')[0]
    model_task = {'task_id': task['source']['row_index'], 'task_name': task['task_id'],
                  'instruction': task['instruction_en'], 'budget': dict(task['budget']), **task['budget'],
                  'instruction_provenance': {'profile': task['profile'], 'instruction_sha256': task['instruction_sha256']}}
    inputs = {'profile': task['profile'], 'instruction_id': task['runtime_instruction_id'],
              'public_config_id': task['task_id'], 'model_task': model_task}
    monkeypatch.setattr(adapters, '_adapter', lambda profile: SimpleNamespace(inspect_inputs=lambda profile, inputs: inputs))
    assert adapters.inspect_inputs(task['profile'], inputs) == inputs
    inputs['public_config_id'] = task['runtime_instruction_id']
    with pytest.raises(ValueError, match='V collection configuration'):
        adapters.inspect_inputs(task['profile'], inputs)
