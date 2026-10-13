"""Bounded, compatible resident groups; no changes to task or scoring identity."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json


def _template(row):
    # This module is also loaded by the standalone dispatcher, whose `lab`
    # package is not Bench's. Keep the bounded template reader self-contained.
    def read_json(path):
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError('oversized scene input')
        return json.loads(path.read_bytes())
    spec = row['task_spec']
    if row.get('adapter') == 'l0_taskspec':
        ref = spec['private']['source_artifacts']['template']
    else:
        package = read_json(Path(row.get('runtime_package', row['package'])['path']))
        runtime = package['runtime']
        catalog_path = (Path(runtime['registration_catalog']['path']) if runtime['adapter'] == 'registered_l0'
                        else Path(runtime['expansion_root']) / 'benchmark/l0/expansion_v1/official_compatible'
                        / row['scene_model'] / 'runtime_registration/catalog.json')
        record = next(r for r in read_json(catalog_path)['tasks']
                      if r.get('task_spec_id', r.get('candidate_id')) == row['instruction_id'])
        ref = record['template']
    path = Path(ref['path'])
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('oversized scene template')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('scene template changed')
    return json.loads(raw)


def scene_groups(instructions, maximum=4, union_builder=None):
    # None groups all compatible Instructions; explicit limits retain batch planning behavior.
    if maximum is not None and (type(maximum) is not int or maximum < 1):
        raise ValueError('maximum must be None or a positive integer')
    buckets = defaultdict(list)
    for key, row in instructions.items():
        identity = row['source_task_identity']
        package = row.get('runtime_package', row.get('package', {}))
        signature = (row.get('scene_model', identity.get('scene_model')), row['seed'],
                     identity['split'], row.get('adapter', 'l0_taskspec'),
                     package.get('sha256'))
        if not signature[0]:
            raise ValueError('scene identity missing')
        buckets[signature].append(key)
    groups = {}
    for ids in buckets.values():
        width = len(ids) if maximum is None else maximum
        for start in range(0, len(ids), width):
            chunk = ids[start:start + width]
            parts = [chunk]
            if union_builder is not None:
                parts, current, documents = [], [], []
                for key in chunk:
                    document = _template(instructions[key])
                    union_builder([document])  # A malformed singleton must fail.
                    try:
                        union_builder([*documents, document])
                    except (ValueError, RuntimeError) as exc:
                        if not isinstance(exc, ValueError):
                            from behavior_hl.embodied.evaluator_bridge.resident_scene_batch import ResidentSceneBatchError
                            if not isinstance(exc, ResidentSceneBatchError):
                                raise
                        parts.append(current)
                        current, documents = [], []
                    current.append(key)
                    documents.append(document)
                parts.append(current)
            for group in parts:
                for key in group:
                    groups[key] = list(group)
    return groups


def continuation(group, primary):
    if not isinstance(group, list) or not group or len(set(group)) != len(group):
        raise ValueError('invalid resident group')
    if primary not in group:
        raise ValueError('primary Instruction absent from resident group')
    return group[group.index(primary):]
