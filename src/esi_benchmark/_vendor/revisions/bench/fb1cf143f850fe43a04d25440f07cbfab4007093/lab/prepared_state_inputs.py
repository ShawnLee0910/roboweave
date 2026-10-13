"""Stage exact Oracle prepared inputs without reading credentials or loading scenes."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import re

_evidence_cache = {}

def manifest_projection(raw):
    value = json.loads(raw)
    return json.dumps({**{key: value[key] for key in
        ('seed', 'camera_profile', 'gripper_visual_profile')},
        '_source_sha256': hashlib.sha256(raw).hexdigest()}).encode()

from lab.instruction_handoff import compile_handoff, read_json


def collect_bundle(row, conditions, reader):
    evidence = row['oracle_evidence']
    result_path = Path(evidence['path'])
    run = result_path.parents[3]
    paths = {'prepared': result_path.with_name('prepared_initial_state.json'),
             'camera_structure': run / 'private/predeclared_camera_structure_attestation.json',
             'camera_tuning': run / 'private/gripper_visible_camera_tuning.json',
             'manifest': run / 'manifest.json'}
    values, sources = {}, {}
    for key, path in paths.items():
        raw = reader(evidence, path)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError('prepared input is oversized')
        values[key] = json.loads(raw)
        sources[key] = {'path': str(path), 'sha256': values[key].get('_source_sha256', hashlib.sha256(raw).hexdigest()),
                        'ssh_host': evidence['ssh_host'], 'ssh_port': evidence['ssh_port']}
    if values['prepared'].get('task_name') != row['instruction_id']:
        raise ValueError('prepared Instruction identity differs')
    if values['manifest'].get('seed') != row['seed']:
        raise ValueError('prepared seed differs')
    camera_profile = values['manifest'].get('camera_profile')
    if camera_profile != conditions['observation_profile']:
        raise ValueError('prepared camera profile differs')
    return {'schema_version': 'oracle-prepared-model-state/v1',
            'instruction_id': row['instruction_id'], 'seed': row['seed'],
            'execution_contract': {k: conditions[k] for k in
                ['action_mode', 'physics_steps_per_action', 'scene_load', 'observation_profile', 'overhead_view_mode']},
            'gripper_visual_profile': values['manifest']['gripper_visual_profile'],
            'prepared': values['prepared'], 'camera_structure': values['camera_structure'],
            'camera_tuning': values['camera_tuning'], 'sources': sources}


def read_evidence(evidence, path):
    cache_key = (evidence['ssh_host'], evidence['ssh_port'], str(path))
    if cache_key in _evidence_cache:
        return _evidence_cache[cache_key]
    if evidence['ssh_host'] is None:
        if path.stat().st_size > (128 if path.name == 'manifest.json' else 2) * 1024 * 1024:
            raise ValueError('oversized input')
        raw = path.read_bytes()
        result = manifest_projection(raw) if path.name == 'manifest.json' else raw
        _evidence_cache[cache_key] = result
        return result
    if not isinstance(evidence['ssh_host'], str) or not re.fullmatch(r'[A-Za-z0-9_.-]+@[A-Za-z0-9.-]+', evidence['ssh_host']):
        raise ValueError('invalid prepared-evidence SSH target')
    program = ('from pathlib import Path; import sys; p=Path(sys.argv[1]); '
               'assert p.is_file() and p.stat().st_size<=2097152; sys.stdout.buffer.write(p.read_bytes())')
    if path.name == 'manifest.json':
        program = ('from pathlib import Path; import sys,json,hashlib; p=Path(sys.argv[1]); '
                   'assert p.is_file() and p.stat().st_size<=134217728; raw=p.read_bytes(); v=json.loads(raw); '
                   'print(json.dumps({**{k:v[k] for k in ("seed","camera_profile","gripper_visual_profile")},'
                   '"_source_sha256":hashlib.sha256(raw).hexdigest()}))')
    command = ['python3', '-c', program, str(path)]
    result = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ControlMaster=no',
        '-o', 'ControlPath=none', '-o', 'ProxyCommand=none', '-o', 'ProxyJump=none',
        '-p', str(evidence['ssh_port']), evidence['ssh_host'], shlex.join(command)], timeout=30)
    _evidence_cache[cache_key] = result
    return result


def stage(handoff_path, output, instruction_ids=None):
    if (output / 'index.json').exists():
        raise ValueError('prepared-input index already exists')
    matrix = compile_handoff(handoff_path, {'model': 1})
    if matrix['schema_version'] != 'instruction-model-matrix/v2':
        raise ValueError('prepared replay needs explicit v2 execution conditions and evidence identity')
    selected = set(instruction_ids) if instruction_ids else set(matrix['instructions'])
    if selected - set(matrix['instructions']):
        raise ValueError('unknown selected Instruction')
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    records = {}
    for key, row in matrix['instructions'].items():
        if key not in selected:
            continue
        bundle = collect_bundle(row, matrix['model_conditions'], read_evidence)
        path = output / (key + '.json')
        raw = (json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
        if path.exists() and path.read_bytes() != raw:
            raise ValueError('partial prepared input differs')
        if not path.exists():
            path.write_bytes(raw)
        path.chmod(0o600)
        records[key] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()}
    index = {'schema_version': 'oracle-prepared-model-index/v1',
             'handoff_sha256': matrix['handoff_sha256'], 'instructions': records}
    (output / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    (output / 'index.json').chmod(0o600)
    return index


def load_prepared(index_path, handoff_sha256, instruction_id):
    index = read_json(index_path)
    if index.get('schema_version') != 'oracle-prepared-model-index/v1' or index.get('handoff_sha256') != handoff_sha256:
        raise ValueError('prepared index does not match the handoff')
    ref = index['instructions'][instruction_id]
    path = Path(ref['path']); bundle = read_json(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256'] or bundle.get('instruction_id') != instruction_id:
        raise ValueError('prepared input changed')
    return bundle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--instruction-id', action='append')
    args = parser.parse_args()
    print('staged', len(stage(args.handoff, args.output, args.instruction_id)['instructions']))
