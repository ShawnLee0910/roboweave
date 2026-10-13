"""Read an Oracle handoff without collapsing Instructions by source Task ID.

The result is private controller input, never model-visible prompt content.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import is_dataclass, replace
from pathlib import Path


def model_profile(profile, resolver):
    """Keep registered profiles; enrich lightweight native L0 source identities."""
    if is_dataclass(profile):
        return profile
    configured = resolver(profile.task_name)
    return replace(configured, **{key: getattr(profile, key) for key in
                                  ('task_id', 'task_name', 'task_language',
                                   'scene_model', 'split', 'instance_id')})


def public_review(handoff_path: Path, instruction_id: str) -> dict:
    matrix = compile_handoff(handoff_path, {"model": 1})
    row = matrix["instructions"][instruction_id]
    package = read_json(Path(row.get('package', matrix.get('package'))['path']))
    specs = Path(package["task_specs"]["path"])
    if specs.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("oversized TaskSpec input")
    raw = specs.read_bytes()
    if hashlib.sha256(raw).hexdigest() != package["task_specs"]["sha256"]:
        raise ValueError("TaskSpec source changed")
    selected = [json.loads(l) for l in raw.splitlines() if l.strip()
                and json.loads(l)["task_spec_id"] == instruction_id]
    if len(selected) != 1 or (selected[0].get('public', {}).get('instruction') or selected[0].get('public_instruction')) != row['instruction']:
        raise ValueError("public Instruction differs from selected TaskSpec")
    identity = row.get('source_task_identity') or selected[0]['private']['source_task_identity']
    return {"schema_version": "behavior2026.certified53-public-instruction-rewrites/v1",
            "status": "DRAFT_FOR_HUMAN_REVIEW",
            "deterministic_public_audit": {"status": "PASS", "basis": "exact TaskSpec instruction match"},
            "source_handoff_sha256": matrix["handoff_sha256"],
            "scope": {"task_ids": [identity["task_id"]]},
            "rewrites": [{"task_id": identity["task_id"], "task_name": identity["task_name"],
                          "instruction_id": instruction_id, "instruction": row["instruction"]}]}


def read_json(path: Path):
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"oversized metadata: {path}")
    return json.loads(path.read_text())


def native_instruction_arguments(handoff_path: Path, instruction_id: str) -> list[str]:
    handoff = read_json(handoff_path)
    if handoff['schema_version'] == 'oracle-model-handoff/v1':
        row = next(r for r in handoff['samples'] if r['task_spec_id'] == instruction_id)
        recipe = read_json(Path(row['campaign_config']))
        return [arg for arg in recipe.get('native_task_args', {}).get(instruction_id, recipe['native_launcher_args'])
                if arg != '--nearby-initial-pose-search']
    if handoff.get('model_conditions', {}).get('action_mode') == 'Physical':
        try:
            from lab.physical_collection_handoff import native_physical_instruction_arguments
        except ModuleNotFoundError:
            from physical_collection_handoff import native_physical_instruction_arguments
        return native_physical_instruction_arguments(handoff_path, instruction_id)
    matrix = compile_handoff(handoff_path, {'model': 1})
    row = matrix['instructions'][instruction_id]
    package_ref = row.get('runtime_package', row['package'])
    package = read_json(Path(package_ref['path']))
    if package['runtime']['adapter'] == 'registered_l0':
        state = package.get('initial_state_compilation', {})
        if state.get('required') and state.get('status') != 'FROZEN':
            raise ValueError('model launch needs frozen initial state; preserve and finalize the measured capture: ' + instruction_id)
        if state.get('status') == 'FROZEN':
            # The model restores an already-compiled state, not an Oracle
            # construction run. Feed its verified catalog to the native model
            # callback without pretending to run construct-and-oracle.
            ref = package['runtime']['registration_catalog']
            catalog = Path(ref['path'])
            registered = read_json(catalog)
            if hashlib.sha256(catalog.read_bytes()).hexdigest() != ref['sha256']:
                raise ValueError('frozen model registration catalog changed')
            if not any(r.get('task_spec_id', r.get('candidate_id')) == instruction_id for r in registered['tasks']):
                raise ValueError('Instruction missing from frozen model catalog')
            return ['--runtime-registration', str(catalog),
                    '--registered-public-action20-marker-se2-navigation']
        return ['--benchmark-package', package_ref['path'],
                '--registered-public-action20-marker-se2-navigation']
    if package['runtime']['adapter'] != 'fixed_l0_pool':
        raise ValueError('unsupported package runtime')
    overlay = package['runtime']['reset_pose_overlay']['path']
    if row['adapter'] == 'l0_taskspec':
        source = ['--l0-taskspecs', package['task_specs']['path']]
    else:
        catalog = Path(package['runtime']['expansion_root']) / 'benchmark/l0/expansion_v1/official_compatible' / row['scene_model'] / 'runtime_registration/catalog.json'
        registered = read_json(catalog)
        if not any(r.get('task_spec_id', r.get('candidate_id')) == instruction_id for r in registered['tasks']):
            raise ValueError('Instruction missing from registered scene catalog')
        source = ['--runtime-registration', str(catalog)]
    return source + ['--reset-pose-overlay', overlay,
                     '--registered-public-action20-marker-se2-navigation']


def compile_handoff(path: Path, models: dict[str, int]) -> dict:
    handoff = read_json(path)
    if handoff.get('schema_version') == 'oracle-model-handoff/v2':
        return compile_multi_package_handoff(path, handoff, models)
    if handoff.get("schema_version") != "oracle-model-handoff/v1":
        raise ValueError("unsupported handoff schema")
    if not models or any(type(n) is not int or n <= 0 for n in models.values()):
        raise ValueError("model capacities must be positive integers")
    package_ref = handoff["package"]
    package_path = Path(package_ref["path"])
    package = read_json(package_path)
    if hashlib.sha256(package_path.read_bytes()).hexdigest() != package_ref["sha256"]:
        raise ValueError("handoff package changed")
    rows = handoff["samples"]
    if len(rows) != handoff["count"]:
        raise ValueError("handoff sample count mismatch")
    instructions = {}
    configs = {}
    for row in rows:
        key = row["task_spec_id"]
        if not isinstance(key, str) or not key or key in instructions:
            raise ValueError("Instruction identities must be unique")
        if key not in package["task_spec_ids"]:
            raise ValueError(f"Instruction absent from package: {key}")
        if row.get("official_q") != 1 or row.get("verification_status") != "COMPLETED":
            raise ValueError(f"Oracle gate missing: {key}")
        if not row.get("prepared_state_available") or not row.get("first_frame_available"):
            raise ValueError(f"prepared initial evidence missing: {key}")
        config_path = row["campaign_config"]
        if config_path not in configs:
            configs[config_path] = read_json(Path(config_path))
        config = configs[config_path]
        identity = config["native_task_identities"][key]
        if identity["scene_model"] != row["scene_model"]:
            raise ValueError(f"scene identity mismatch: {key}")
        if config["source_revision"] != row["runtime_commit"]:
            raise ValueError(f"runtime identity mismatch: {key}")
        instructions[key] = {
            "instruction_id": key,
            "instruction": row["instruction"],
            "source_task_identity": identity,
            "seed": handoff["seed"],
            "max_action_decisions": row["max_action_decisions"],
            "runtime_source_root": row["runtime_source_root"],
            "runtime_commit": row["runtime_commit"],
            "oracle_evidence_directory": row["private_evidence_directory"],
            "prepared_first_frame": row["first_frame"],
            "initial_state_rebuild_required": True,
            "oracle_error_preserved": row.get("oracle_error_preserved"),
        }
    return {
        "schema_version": "instruction-model-matrix/v1",
        "private_operator_manifest": True,
        "handoff_path": str(path.resolve()),
        "handoff_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "package": package_ref,
        "model_capacities": models,
        "instructions": instructions,
        "cells": [
            {"instruction_id": key, "model_key": model, "repeat": 1, "attempt": 1}
            for key in instructions for model in models
        ],
    }


def compile_multi_package_handoff(path: Path, handoff: dict, models: dict[str, int]) -> dict:
    """Normalize v2 metadata without changing TaskSpecs or inventing initial state."""
    if not models or any(type(n) is not int or n <= 0 for n in models.values()):
        raise ValueError('model capacities must be positive integers')
    packages, specs, registered_catalogs = {}, {}, {}
    for ref in handoff['packages']:
        name = ref['name']
        if name in packages:
            raise ValueError('duplicate package name')
        package_path = Path(ref['path'])
        package = read_json(package_path)
        if hashlib.sha256(package_path.read_bytes()).hexdigest() != ref['sha256']:
            raise ValueError('package identity changed: ' + name)
        source = Path(package['task_specs']['path'])
        if source.stat().st_size > 16 * 1024 * 1024:
            raise ValueError('oversized TaskSpec source')
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != package['task_specs']['sha256']:
            raise ValueError('TaskSpec source changed: ' + name)
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        indexed = {row['task_spec_id']: row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError('duplicate Instruction in package')
        packages[name] = dict(ref)
        specs[name] = indexed
    conditions = handoff['model_conditions']
    if not conditions.get('restore_initial_state_before_model'):
        raise ValueError('model initial-state restoration is required')
    instructions = {}
    if len(handoff['samples']) != handoff['count']:
        raise ValueError('handoff count mismatch')
    for row in handoff['samples']:
        key, name = row['task_spec_id'], row['package_name']
        if key in instructions:
            raise ValueError('Instruction identities must be unique')
        spec = specs[name][key]
        if row.get('official_q') != 1 or row.get('verification_status') != 'COMPLETED':
            raise ValueError('Oracle gate missing: ' + key)
        if row.get('seed') != conditions['seed']:
            raise ValueError('seed mismatch: ' + key)
        if not row.get('prepared_state_available') or not row.get('first_frame_available'):
            raise ValueError('prepared evidence missing: ' + key)
        host, port = row.get('ssh_host'), row.get('ssh_port')
        if (host is None) != (port is None) or (port is not None and
                (type(port) is not int or not 1 <= port <= 65535)):
            raise ValueError('invalid evidence host/port')
        evidence = Path(row['oracle_result_path'])
        if not evidence.is_absolute():
            raise ValueError('evidence path must be absolute on its source host')
        frame = Path(row['first_frame_relative_to_run'])
        if frame.is_absolute() or '..' in frame.parts:
            raise ValueError('frame path escapes run')
        legacy = spec.get('task_spec_version') == 'action20-capability-benchmark.l0-task/v1'
        instruction = spec.get('public', {}).get('instruction') or spec.get('public_instruction')
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('public Instruction missing: ' + key)
        instructions[key] = {
            'instruction_id': key, 'instruction': instruction, 'package_name': name,
            'package': packages[name], 'task_spec': spec,
            'source_task_identity': spec.get('private', {}).get('source_task_identity'),
            'scene_model': spec.get('scene_model') or spec.get('private', {}).get('scene_model') or spec.get('private', {}).get('source_task_identity', {}).get('scene_model'),
            'adapter': 'l0_taskspec' if legacy else 'registered_l0',
            'seed': conditions['seed'],
            'max_action_decisions': spec.get('public', {}).get('max_action_decisions', 150),
            'runtime_source_root': row['runtime_source_root'], 'runtime_commit': row['runtime_commit'],
            'oracle_evidence': {'ssh_host': host, 'ssh_port': port, 'path': str(evidence),
                                'sha256': row['oracle_result_sha256']},
            'prepared_first_frame': str(evidence.parents[3] / frame),
            'initial_state_rebuild_required': True,
            'oracle_error_preserved': row.get('execution_error'),
        }
        if not legacy:
            package = read_json(Path(packages[name]['path']))
            runtime = package['runtime']
            catalog_path = (Path(runtime['registration_catalog']['path'])
                if runtime['adapter'] == 'registered_l0' else
                Path(runtime['expansion_root']) / 'benchmark/l0/expansion_v1/official_compatible'
                / instructions[key]['scene_model'] / 'runtime_registration/catalog.json')
            catalog_key = str(catalog_path)
            if catalog_key not in registered_catalogs:
                registered_catalogs[catalog_key] = read_json(catalog_path)
            matches = [r for r in registered_catalogs[catalog_key]['tasks']
                       if r.get('task_spec_id', r.get('candidate_id')) == key]
            if len(matches) != 1:
                raise ValueError('registered Instruction is not unique: ' + key)
            instructions[key]['source_task_identity'] = {
                k: matches[0]['profile'][k] for k in
                ('task_id', 'task_name', 'scene_model', 'split', 'instance_id')}
        frozen_path = path.parent / 'esi_artifacts/instruction88_inputs_20260907/perception_frozen/package.json'
        if name == 'perception114' and frozen_path.is_file():
            frozen = read_json(frozen_path)
            if frozen.get('source_package', {}).get('sha256') != packages[name]['sha256'] or key not in frozen['task_spec_ids']:
                raise ValueError('frozen initial-state package does not match handoff')
            instructions[key]['runtime_package'] = {'path': str(frozen_path),
                'sha256': hashlib.sha256(frozen_path.read_bytes()).hexdigest()}
    return {'schema_version': 'instruction-model-matrix/v2', 'private_operator_manifest': True,
            'handoff_path': str(path.resolve()), 'handoff_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'packages': packages, 'model_conditions': conditions, 'instructions': instructions,
            'model_capacities': models,
            'cells': [{'instruction_id': key, 'model_key': model, 'repeat': 1, 'attempt': 1}
                      for key in instructions for model in models]}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--instruction-id', required=True)
    parser.add_argument('--review-output', type=Path, required=True)
    args = parser.parse_args()
    review = public_review(args.handoff, args.instruction_id)
    with args.review_output.open('x') as stream:
        json.dump(review, stream, ensure_ascii=False, indent=2)
