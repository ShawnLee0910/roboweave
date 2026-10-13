#!/usr/bin/env python3
"""Freeze P30/L2 resource references; never copy private assets or run simulations."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path('/liyuanshuai')
REPO = Path(__file__).resolve().parents[1]
BUDGET_KEYS = ('maximum_action_decisions', 'maximum_model_calls', 'wall_clock_seconds')


def read(path):
    path = Path(path)
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError(f'oversized metadata: {path}')
    raw = path.read_bytes()
    return [json.loads(line) for line in raw.splitlines() if line.strip()] if path.suffix == '.jsonl' else json.loads(raw)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ref(path):
    return {'path': str(path), 'sha256': sha(path)}


class Manifest:
    def __init__(self, profile):
        self.profile, self.rows, self.inputs = profile, {}, {}

    def add(self, reference, task, role, selection=None, ids=None):
        path = Path(reference['path'] if isinstance(reference, dict) else reference)
        relative = path.relative_to(BASE)
        root, tail = relative.parts[0], Path(*relative.parts[1:]).as_posix()
        key = root + '/' + tail
        digest = sha(path)
        if isinstance(reference, dict) and reference.get('sha256', digest) != digest:
            raise ValueError(f'frozen reference hash mismatch: {path}')
        row = self.rows.setdefault(key, dict(key=key, source_root=root, source_path=tail,
            sha256=digest, bytes=path.stat().st_size, roles=[], task_ids=[], selection=None))
        if role not in row['roles']:
            row['roles'].append(role)
        if task not in row['task_ids']:
            row['task_ids'].append(task)
        if selection:
            row['selection'] = row['selection'] or {'kind': selection, 'ids': {}}
            row['selection']['ids'][task] = ids or [task]
        return key

    def package(self, package_ref, task, instruction):
        key = self.add(package_ref, task, 'package', 'package', [instruction])
        package = read(package_ref['path'])
        spec_key = self.add(package['task_specs'], task, 'task_specs', 'task_specs', [instruction])
        catref = package['runtime']['registration_catalog']
        catkey = self.add(catref, task, 'registration_catalog', 'registration_catalog', [instruction])
        catalog = read(catref['path'])
        selected = [r for r in catalog['tasks'] if r.get('task_spec_id',r.get('candidate_id')) == instruction]
        if len(selected) != 1:
            raise ValueError('registration identity is not unique')
        for field in ('bddl', 'template', 'tro_state', 'template_contract'):
            self.add(selected[0][field], task, field)
        if catalog.get('prepared_oracle_states'):
            statesref = catalog['prepared_oracle_states']
            statekey = self.add(statesref, task, 'prepared_oracle_states', 'prepared_oracle_states', [instruction])
            states = read(statesref['path'])
            for item_index, item in enumerate(states['items']):
                if item['task_spec_id'] != instruction:
                    continue
                for field in ('donor_success', 'prepared', 'camera_structure', 'camera_tuning', 'donor_initial_construction'):
                    self.add(item[field], task, 'donor_' + field)
                target_key = self.add(selected[0]['template_contract'], task, 'template_contract')
                # Selection reduces items to one record before hash bindings are applied.
                self.rows[statekey].setdefault('hash_bindings', []).append({
                    'pointer': '/items/0/target_template_contract_sha256', 'key': target_key, 'task_ids': [task]})
        return key, spec_key, catkey, selected[0]['profile']

    def handoff(self, handoff_ref, index_ref, task, instruction):
        key = self.add(handoff_ref, task, 'handoff', 'handoff', [instruction])
        handoff = read(handoff_ref['path'])
        samples = [r for r in handoff['samples'] if r['task_spec_id'] == instruction]
        if len(samples) != 1:
            raise ValueError('handoff identity is not unique')
        sample = samples[0]
        package = next(r for r in handoff['packages'] if r['name'] == sample['package_name'])
        pkey, skey, ckey, identity = self.package(package, task, instruction)
        if sample.get('prepared_derivation'):
            proof = sample['prepared_derivation']
            self.add(proof, task, 'prepared_derivation')
            # Native validator reads only donor handoff gate metadata, not donor run artifacts.
            self.add(read(proof['path'])['source_handoff'], task, 'donor_handoff')
        ikey = self.add(index_ref, task, 'prepared_index', 'prepared_index', [instruction])
        index = read(index_ref['path'])
        if index['handoff_sha256'] != sha(handoff_ref['path']):
            raise ValueError('prepared index handoff identity mismatch')
        self.rows[ikey].setdefault('hash_bindings', []).append({
            'pointer': '/handoff_sha256', 'key': key, 'task_ids': [task]})
        bkey = self.add(index['instructions'][instruction], task, 'prepared_bundle')
        bundle = read(index['instructions'][instruction]['path'])
        return dict(handoff_json=key, prepared_index_json=ikey, benchmark_package_json=pkey,
                    task_specs_jsonl=skey, registration_catalog_json=ckey, prepared_bundle_json=bkey,
                    source_task_identity={k: identity[k] for k in ('task_id','task_name','scene_model','split','instance_id')},
                    seed=bundle['seed'], action_mode=bundle['execution_contract']['action_mode'])

    def write(self):
        rows = sorted(self.rows.values(), key=lambda r: r['key'])
        payload = dict(schema_version=2, profile=self.profile,
            source_roots=sorted({r['source_root'] for r in rows}),
            private_assets_not_distributed=True, files=rows, inputs_by_task=self.inputs)
        dest = REPO / 'src/esi_benchmark/data/assets' / (self.profile + '.json')
        dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
        return {'profile': self.profile, 'file_count': len(rows), 'bytes': sum(r['bytes'] for r in rows),
                'manifest_sha256': sha(dest)}


def build():
    summary = []
    for group, profile in [('inside','l1-p-inside5'),('ontop','l1-p-ontop5'),('additions','l1-p-additions20')]:
        source = BASE / 'action20-capability-benchmark/benchmark/L1-P-FINAL/model_batch_20260924' / group
        tasks = read(source / 'tasks.json')
        manifest = Manifest(profile)
        for i, task in enumerate(tasks):
            name = task['runtime_instruction_id']
            entry = manifest.handoff(task['handoff'], task['prepared_index'], name, name)
            entry.update(entrypoint='prepared', public_task_id=i, instruction_id=name,
                instruction_en=task['instruction_en'], budget={k:task['budget'][k] for k in BUDGET_KEYS},
                task_json=manifest.add(source / 'tasks.json', name, 'release_tasks','release_tasks',[name]),
                instruction_review_json=manifest.add(source / 'instructions.json', name, 'instruction_review'),
                history_turns=5)
            manifest.inputs[name] = entry
        summary.append(manifest.write())
    inputs = BASE / 'fb-batchctl-20260825/campaigns/l2_37_gpt6max_pair_19slot_20260925.inputs'
    source_profile = inputs / 'profile.json'
    profile = read(source_profile)
    input_revision = subprocess.check_output(['git', '--no-optional-locks', '-C', str(BASE/'fb-batchctl-20260825'), 'rev-parse', 'HEAD'], text=True).strip()
    ledgers = [lane['initial_ledger'] for lane in profile['dispatcher']['lanes']]
    tasks = read(inputs / 'tasks.json')['stages'][0]['tasks']
    review = read(inputs / 'instructions.json')
    if review['status'] != 'APPROVED':
        raise ValueError('L2 English overlay lacks approval')
    manifest = Manifest('l2-collection37')
    catalog = []
    for task in tasks:
        name, public_id = task['task_name'], task['task_id']
        budget = {k:ledgers[0]['instruction_budgets'][str(public_id)][k] for k in BUDGET_KEYS}
        job = ledgers[0]['instruction_jobs'][str(public_id)]
        for ledger in ledgers[1:]:
            if ({k:ledger['instruction_budgets'][str(public_id)][k] for k in BUDGET_KEYS} != budget
                    or ledger['instruction_jobs'][str(public_id)] != job):
                raise ValueError('L2 lane semantic bindings disagree')
        overlay = [r for r in review['rewrites'] if r['task_id']==public_id and r['task_name']==name]
        if len(overlay) != 1 or overlay[0]['instruction'] != task['instruction']:
            raise ValueError('L2 reviewed instruction differs from frozen public task')
        entry = manifest.handoff({'path':job['handoff_path'],'sha256':job['handoff_sha256']},
                                  ref(job['prepared_state_index']), name, name)
        entry.update(entrypoint='physical_collection' if entry['action_mode']=='Physical' else 'prepared',
                     public_task_id=public_id, instruction_id=name, instruction_en=task['instruction'],
                     budget=budget, history_turns=5,
                     instruction_review_json=manifest.add(inputs/'instructions.json',name,'instruction_review'),
                     source_profile_sha256=sha(source_profile))
        manifest.inputs[name] = entry
        catalog.append(dict(task_id=name,instruction_id=name,runtime_instruction_id=name,family='L2',
            instruction_en=task['instruction'],instruction_sha256=hashlib.sha256(task['instruction'].encode()).hexdigest(),
            scene_model=entry['source_task_identity']['scene_model'],budget=budget,action_mode=entry['action_mode'],
            profile='l2-collection37',source=dict(root='fb-batchctl-20260825',
                path=str((inputs/'tasks.json').relative_to(BASE/'fb-batchctl-20260825')),sha256=sha(inputs/'tasks.json'),
                row_index=public_id,state='untracked',revision=input_revision),split=entry['source_task_identity']['split']))
    summary.append(manifest.write())
    (REPO/'src/esi_benchmark/data/tasks/l2-collection37.json').write_text(json.dumps(
        dict(schema_version=1, tasks=catalog),ensure_ascii=False,indent=2)+'\n')
    sources=[]
    for name in ['action20-capability-benchmark','p30high','l2g25','fb-batchctl-20260825']:
        path=BASE/name
        def git(*args):
            return subprocess.check_output(['git','--no-optional-locks','-C',str(path),*args],text=True).strip()
        sources.append(dict(path=str(path),head=git('rev-parse','HEAD'),
            tracked_status=git('status','--porcelain','--untracked-files=no')))
    payload=dict(schema_version=1, scope='P30 and L2 collection37 explicit metadata/resource references only',
        origin_profile=dict(path=str(source_profile),sha256=sha(source_profile),
            fields_used=['dispatcher.lanes[].initial_ledger.instruction_budgets',
                         'dispatcher.lanes[].initial_ledger.instruction_jobs']),
        sources=sources, manifests=summary, excluded=['credentials','operator profile','raw trajectories','simulator asset binaries'],
        source_file_hashes='Every imported private input has its original complete SHA256 in the corresponding asset manifest.',
        validation='Static manifest construction only; no simulation, external endpoint or environment reproduction.')
    (REPO/'provenance/imports_physical_l2.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
