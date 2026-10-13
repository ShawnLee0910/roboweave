#!/usr/bin/env python3
"""Build metadata-only AMV execution closures from explicit frozen local references.

No directory discovery, simulator import, network, credentials or asset copies.
The authoring source root is supplied explicitly; it is not a runtime default.
"""
import argparse
import hashlib
from functools import lru_cache
import json
from pathlib import Path

PROFILES = {'A': 'l1-a30', 'M': 'l1-m30-public-v2', 'V': 'l1-v30'}
RELEASE = 'action20-capability-benchmark/benchmark/L1-AMV90-RELEASE/20260924_v1/RELEASE.json'
MAX_BYTES = 256 * 1024 * 1024


@lru_cache(maxsize=2048)
def read(path):
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError(f'Missing or oversized explicitly referenced input: {path}')
    return path.read_bytes()


@lru_cache(maxsize=2048)
def decoded(path):
    data = read(path)
    return ([json.loads(line) for line in data.splitlines() if line.strip()]
            if path.suffix == '.jsonl' else json.loads(data))


def references(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and 'sha256' in value:
            yield value
        else:
            for item in value.values():
                yield from references(item)
    elif isinstance(value, list):
        for item in value:
            yield from references(item)


class Closure:
    def __init__(self, root, profile):
        self.root, self.profile = root, profile
        self.files, self.inputs, self.visited = {}, {}, set()

    def add(self, ref, owner, role, *, parent=None, kind=None, identity=None):
        path = Path(ref['path'])
        if not path.is_absolute():
            if parent is None:
                raise ValueError('Relative input without an explicit parent')
            path = parent / path
        path = path.absolute()
        relative = path.relative_to(self.root)
        raw = read(path)
        sha = hashlib.sha256(raw).hexdigest()
        if ref.get('sha256') and ref['sha256'] != sha:
            raise ValueError(f'Frozen input hash changed: {relative}')
        key = self.profile + '-' + hashlib.sha256(str(relative).encode()).hexdigest()[:24]
        row = self.files.setdefault(key, {
            'key': key, 'source_root': relative.parts[0],
            'source_path': str(Path(*relative.parts[1:])), 'sha256': sha,
            'bytes': len(raw), 'roles': [], 'task_ids': [], 'selection': None})
        if role not in row['roles']:
            row['roles'].append(role)
        if owner not in row['task_ids']:
            row['task_ids'].append(owner)
        if kind:
            if row['selection'] is None:
                row['selection'] = {'kind': kind, 'ids': {}}
            if row['selection']['kind'] != kind:
                raise ValueError('Inconsistent asset selection kinds')
            ids = row['selection']['ids'].setdefault(owner, [])
            if identity not in ids:
                ids.append(identity)
        return path, row

    def leaf(self, ref, owner, role, parent):
        return self.add(ref, owner, role, parent=parent)[1]['key']

    def specs(self, ref, owner, native, parent):
        path, row = self.add(ref, owner, 'task_specs', parent=parent,
                             kind='task_specs', identity=native)
        specs = [s for s in decoded(path) if s['task_spec_id'] == native]
        if len(specs) != 1:
            raise ValueError('TaskSpec identity absent or ambiguous')
        # The execution reads native artifacts; historical provenance is not an import dependency.
        for item in references(specs[0].get('private', {}).get('source_artifacts', {})):
            self.leaf(item, owner, 'native_source_artifact', path.parent)
        return row['key']

    def catalog(self, ref, owner, native, parent):
        path, row = self.add(ref, owner, 'registration_catalog', parent=parent,
                             kind='registration_catalog', identity=native)
        values = [s for s in decoded(path)['tasks']
                  if s.get('task_spec_id', s.get('candidate_id')) == native]
        if len(values) != 1:
            raise ValueError('Registered task identity absent or ambiguous')
        contract_key = None
        for name in ('bddl', 'template_contract', 'template', 'tro_state'):
            if name in values[0]:
                dependency_key = self.leaf(values[0][name], owner, 'registered_task_dependency', path.parent)
                if name == 'template_contract':
                    contract_key = dependency_key
        oracle_index = decoded(path).get('prepared_oracle_states')
        if oracle_index:
            ip, index = self.add(oracle_index, owner, 'prepared_oracle_states', parent=path.parent,
                                 kind='prepared_oracle_states', identity=native)
            items = [item for item in decoded(ip)['items'] if item['task_spec_id'] == native]
            if len(items) != 1 or contract_key is None:
                raise ValueError('Prepared Oracle target identity absent or ambiguous')
            for name in ('donor_success', 'prepared', 'camera_structure', 'camera_tuning', 'donor_initial_construction'):
                self.leaf(items[0][name], owner, 'prepared_oracle_dependency', ip.parent)
            index.setdefault('hash_bindings', []).append({
                'pointer': '/items/0/target_template_contract_sha256', 'key': contract_key,
                'task_ids': [owner]})
        return row['key']

    def package(self, ref, owner, native, parent):
        path, row = self.add(ref, owner, 'package', parent=parent,
                             kind='package', identity=native)
        token = (str(path), owner, native)
        if token in self.visited:
            return row['key']
        self.visited.add(token)
        value = decoded(path)
        if native not in value['task_spec_ids']:
            raise ValueError('Package does not contain native task')
        if value.get('task_specs'):
            self.specs(value['task_specs'], owner, native, path.parent)
        runtime = value.get('runtime', {})
        if runtime.get('registration_catalog'):
            self.catalog(runtime['registration_catalog'], owner, native, path.parent)
        for field in ('pool', 'reset_pose_overlay'):
            if runtime.get(field):
                self.leaf(runtime[field], owner, field, path.parent)
        return row['key']

    def prepared(self, raw, owner, native, parent):
        hp, handoff = self.add(raw['handoff'], owner, 'handoff', parent=parent,
                               kind='handoff', identity=native)
        value = decoded(hp)
        samples = [s for s in value['samples'] if s['task_spec_id'] == native]
        if len(samples) != 1:
            raise ValueError('Handoff identity absent or ambiguous')
        package_refs = [r for r in value['packages'] if r['name'] == samples[0]['package_name']]
        if len(package_refs) != 1:
            raise ValueError('Handoff package identity absent or ambiguous')
        package = self.package(package_refs[0], owner, native, hp.parent)
        if samples[0].get('prepared_derivation'):
            dp, _ = self.add(samples[0]['prepared_derivation'], owner, 'prepared_derivation', parent=hp.parent)
            proof = decoded(dp)
            self.leaf(proof['source_handoff'], owner, 'derivation_source_handoff', dp.parent)
        ip, index = self.add(raw['prepared_index'], owner, 'prepared_index', parent=parent,
                             kind='prepared_index', identity=native)
        state = decoded(ip)
        if state['handoff_sha256'] != handoff['sha256']:
            raise ValueError('Prepared index handoff hash differs')
        self.leaf(state['instructions'][native], owner, 'prepared_state', ip.parent)
        index.setdefault('hash_bindings', [])
        binding = {'pointer': '/handoff_sha256', 'key': handoff['key']}
        if binding not in index['hash_bindings']:
            index['hash_bindings'].append(binding)
        return {'entrypoint': 'prepared', 'handoff_json': handoff['key'],
                'prepared_index_json': index['key'], 'benchmark_package_json': package,
                'seed': value['model_conditions']['seed']}

    def v_package(self, ref, owner, native, parent, *, donor=False):
        path, row = self.add(ref, owner, 'donor_v_package' if donor else 'v_package', parent=parent)
        token = (str(path), owner, native)
        if token in self.visited:
            return row['key']
        self.visited.add(token)
        value = decoded(path)
        if value['task_spec_id'] != native:
            raise ValueError('V package native identity differs')
        if value.get('donor_v_package'):
            dp = Path(value['donor_v_package']['path'])
            if not dp.is_absolute():
                dp = path.parent / dp
            donor_native = decoded(dp)['task_spec_id']
            self.v_package(value['donor_v_package'], owner, donor_native, path.parent, donor=True)
        for field in ('task_specs',):
            if value.get(field):
                self.specs(value[field], owner, native, path.parent)
        for field in ('prepared_state', 'reset_pose_overlay'):
            if value.get(field):
                self.leaf(value[field], owner, field, path.parent)
        source = value.get('source', {})
        if source.get('base_package'):
            self.package(source['base_package'], owner, source.get('base_task_spec_id', native), path.parent)
        for name in ('receipt', 'history_attestation'):
            if source.get(name):
                self.leaf(source[name], owner, 'v_source_' + name, path.parent)
        if value.get('native_world'):
            wp, _ = self.add(value['native_world'], owner, 'native_world', parent=path.parent)
            for dependency in references(decoded(wp)):
                self.leaf(dependency, owner, 'native_world_dependency', wp.parent)
        if value.get('historical_full_source'):
            raise ValueError('Historical full R3 source requires an explicit audited closure')
        if value.get('variant_source'):
            vp, _ = self.add(value['variant_source'], owner, 'v_variant_source', parent=path.parent)
            variant = decoded(vp)
            original = Path(variant['original_r4_package']['path'])
            if not original.is_absolute():
                original = path.parent / original
            self.v_package(variant['original_r4_package'], owner, decoded(original)['task_spec_id'], path.parent, donor=True)
            contract = self.leaf(variant['candidate_contract'], owner, 'v_candidate_contract', path.parent)
            for dependency in references(variant.get('target_substitution', {})):
                self.leaf(dependency, owner, 'v_variant_metadata', path.parent)
            specs_key = self.specs(value['task_specs'], owner, native, path.parent)
            spec = next(s for s in decoded(Path(value['task_specs']['path'])) if s['task_spec_id'] == native)
            self.leaf(spec['private']['verifier']['official_goal_membership']['grounder'], owner,
                      'v_variant_grounder_evidence', path.parent)
            self.files[specs_key].setdefault('hash_bindings', []).append({
                'pointer': '/0/candidate_contract_sha256', 'key': contract,
                'encoding': 'canonical_json', 'task_ids': [owner]})
        return row['key']

    def v_collection(self, raw, owner, native, parent):
        bp, binding = self.add(raw['v_binding'], owner, 'v_collection_binding', parent=parent)
        value = decoded(bp)
        key = value['collection_config_key']
        if key != raw['collection_config_key']:
            raise ValueError('V public configuration identity differs')
        ip, index = self.add(value['success_index'], owner, 'v_success_index', parent=bp.parent,
                             kind='v_success_index', identity=key)
        selected = [r for r in decoded(ip)['rows'] if r['collection_config_key'] == key]
        if len(selected) != 1 or selected[0]['task_spec_id'] != native:
            raise ValueError('V success index native identity absent or ambiguous')
        selected = selected[0]
        package = self.v_package(selected['v_package'], owner, native, ip.parent)
        self.specs(selected['source_task_spec'], owner, native, ip.parent)
        if selected.get('prepared_state'):
            self.leaf(selected['prepared_state'], owner, 'prepared_state', ip.parent)
        if value.get('compatibility_report'):
            _, report = self.add(value['compatibility_report'], owner, 'v_compatibility', parent=bp.parent,
                                 kind='v_compatibility', identity=key)
            report.setdefault('hash_bindings', []).extend([
                {'pointer': '/success_index_sha256', 'key': index['key'], 'task_ids': [owner]},
                {'pointer': '/configurations/' + key.replace('~', '~0').replace('/', '~1') + '/v_package_sha256',
                 'key': package, 'task_ids': [owner]}])
        vp = Path(raw['v_package']['path'])
        return {'entrypoint': 'v_collection', 'v_collection_binding_json': binding['key'],
                'public_config_id': key, 'seed': decoded(vp)['seed']}

    def build(self, binding):
        tasks_path = Path(binding['tasks']['path'])
        tasks = decoded(tasks_path)
        for position, raw in enumerate(tasks):
            owner, native = raw['instruction_id'], raw['runtime_instruction_id']
            _, release = self.add({**binding['tasks'], 'path': str(tasks_path.resolve())}, owner,
                                  'release_tasks', kind='release_tasks', identity=owner)
            entry = (self.v_collection(raw, owner, native, tasks_path.parent) if 'v_binding' in raw
                     else self.prepared(raw, owner, native, tasks_path.parent))
            release['selection']['field'] = 'instruction_id'
            review_ref = binding['instruction_review']
            rp, review_row = self.add({**review_ref, 'path': str(Path(review_ref['path']).resolve())},
                                      owner, 'instruction_review')
            review = decoded(rp)
            approved = [r for r in review['rewrites'] if r['task_id'] == position]
            if (review.get('status') != 'APPROVED' or len(approved) != 1
                    or approved[0]['task_name'] != owner or approved[0]['instruction'] != raw['instruction_en']):
                raise ValueError('Public instruction does not match frozen approved review')
            entry['instruction_review_json'] = review_row['key']
            entry['task_json'] = release['key']
            self.inputs[owner] = entry
        rows = sorted(self.files.values(), key=lambda row: (row['source_root'], row['source_path']))
        for row in rows:
            row['task_ids'].sort()
            row['roles'].sort()
        return {'schema_version': 1, 'profile': self.profile,
                'redistribution': 'metadata_only; original content and simulator assets require separate authorization',
                'source_roots': sorted({r['source_root'] for r in rows}),
                'unique_runtime_file_count': len(rows), 'unique_runtime_bytes': sum(r['bytes'] for r in rows),
                'inputs_by_task': self.inputs, 'files': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path, required=True)
    args = parser.parse_args()
    release_path = args.source_root / RELEASE
    release = decoded(release_path)
    record = {'schema_version': 1, 'operation': 'bounded read-only execution dependency indexing',
              'release': {'path': str(release_path), 'sha256': hashlib.sha256(read(release_path)).hexdigest()},
              'profiles': {}, 'asset_content_copied': False}
    for axis, profile in PROFILES.items():
        binding = release['axis_bindings'][axis]
        manifest = Closure(args.source_root, profile).build(binding)
        out = args.candidate_root / 'src/esi_benchmark/data/assets' / (profile + '.json')
        out.write_text(json.dumps(manifest, indent=2) + '\n')
        record['profiles'][profile] = {'sources': binding['sources'], 'tasks': binding['tasks'],
            'instruction_review': binding.get('instruction_review'),
            'resolved_tasks_path': str(Path(binding['tasks']['path']).resolve()),
            'resolved_instruction_review_path': str(Path(binding['instruction_review']['path']).resolve()),
            'files': manifest['unique_runtime_file_count'], 'bytes': manifest['unique_runtime_bytes'],
            'manifest_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}
        print(profile, manifest['unique_runtime_file_count'], manifest['unique_runtime_bytes'])
    (args.candidate_root / 'provenance/imports_amv.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
