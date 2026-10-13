"""Bounded read-only task outcomes shared by scheduler and monitoring.

The JSON files remain the source of truth. Missing files are NOT_STARTED only
when the worker's write-before-work protocol explicitly covers the member.
"""
from pathlib import Path
import json
import os

SCHEMA = 'resident-task-progress/v1'
TERMINAL_STATUSES = frozenset({'PASS', 'FAIL', 'TIMEOUT', 'SKIPPED_BASE_BLOCKED', 'CONSTRUCTION_CAPTURED'})
MAX_JSON_BYTES = 4 * 1024 * 1024


def _invalid_constant(token):
    raise ValueError('non-finite JSON value: ' + token)


def read_record(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError('unsafe or oversized task record: ' + str(path))
    value = json.loads(path.read_text(), parse_constant=_invalid_constant)
    if not isinstance(value, dict):
        raise ValueError('task record must be an object')
    return value


def locate_run(artifact_root, worker_id):
    if not artifact_root or not worker_id:
        return None
    if worker_id in {'.', '..'} or Path(worker_id).parts != (worker_id,):
        raise ValueError('invalid worker id')
    parent = Path(artifact_root) / 'workers' / worker_id
    if not parent.resolve().is_relative_to(Path(artifact_root).resolve()):
        raise ValueError('worker directory escapes artifact root')
    if not parent.is_dir():
        return None
    # One declared worker, at most two directory entries; never scan raw runs.
    with os.scandir(parent) as entries:
        found = []
        for entry in entries:
            found.append(Path(entry.path))
            if len(found) == 2:
                break
    if len(found) != 1 or found[0].is_symlink() or not found[0].is_dir():
        raise ValueError('ambiguous worker run directory')
    return found[0]


def task_outcomes(run_dir, task_names, *, worker_finished=True, expected_seed=None,
                  expected_identities=None):
    names = list(task_names)
    if len(names) != len(set(names)) or any(n in {'.', '..'} or Path(n).parts != (n,) for n in names):
        raise ValueError('invalid task membership')
    run = None if run_dir is None else Path(run_dir)
    protocol_ok, protocol_error = False, None
    if run is not None:
        path = run / 'private/resident_task_protocol.json'
        if path.exists():
            try:
                protocol = read_record(path)
                protocol_ok = (protocol.get('schema_version') == SCHEMA
                               and protocol.get('task_names') == names)
                if not protocol_ok:
                    protocol_error = 'task protocol membership/version mismatch'
                if expected_seed is not None and protocol.get('seed') != expected_seed:
                    protocol_error = 'task protocol seed mismatch'
                if expected_identities is not None:
                    identities = protocol.get('task_identities')
                    if not isinstance(identities, list) or len(identities) != len(names):
                        protocol_error = 'task protocol identities missing'
                    else:
                        for name, identity in zip(names, identities):
                            wanted = expected_identities[name]
                            if not isinstance(identity, dict) or any(identity.get(k) != wanted.get(k)
                                    for k in ('task_id', 'scene_model', 'split', 'instance_id')):
                                protocol_error = 'task protocol identity mismatch'
            except (OSError, ValueError) as exc:
                protocol_error = str(exc)
        elif expected_seed is not None or expected_identities is not None:
            protocol_error = 'legacy run lacks identity evidence for automatic continuation'
    result = {}
    for name in names:
        row = dict(task_name=name, state='UNKNOWN', oracle_started=None,
                   result_path=None, progress_path=None, data={}, progress={})
        result[name] = row
        if protocol_error:
            row['warning'] = protocol_error
        if run is None:
            continue
        directory = run / 'private/resident_tasks' / name
        progress_path, terminal_path = directory / 'task_progress.json', directory / 'oracle_result.json'
        malformed = False
        for path, field in ((progress_path, 'progress'), (terminal_path, 'data')):
            if not path.exists():
                continue
            try:
                value = read_record(path)
                identity = value.get('task_name', value.get('resident_item_id', name))
                if identity != name:
                    raise ValueError('task record identity mismatch')
                if field == 'progress' and value.get('schema_version') != SCHEMA:
                    raise ValueError('task progress schema mismatch')
                if field == 'progress' and expected_seed is not None and value.get('seed') != expected_seed:
                    raise ValueError('task progress seed mismatch')
                row[field] = value
                row['progress_path' if field == 'progress' else 'result_path'] = str(path)
            except (OSError, ValueError) as exc:
                row['warning'] = str(exc)
                malformed = True
        progress, data = row['progress'], row['data']
        started = data.get('oracle_started', progress.get('oracle_started'))
        if type(started) is bool:
            row['oracle_started'] = started
        elif data.get('status') == 'TIMEOUT' or data.get('official_q') is not None:
            row['oracle_started'] = True
        elif data.get('construction_error') or data.get('preparation_error'):
            row['oracle_started'] = False
        for field in ('phase', 'failed_phase', 'construction_status', 'oracle_status',
                      'construction_evidence', 'execution_dir', 'executor_actions_path',
                      'first_frame', 'prepared_state', 'timestamps'):
            row[field] = data.get(field, progress.get(field))
        if malformed or protocol_error:
            continue
        if data.get('status') in TERMINAL_STATUSES:
            row['state'] = 'TERMINAL'
        elif (progress.get('phase') == 'TERMINAL'
              and progress.get('terminal_status') == 'CONSTRUCTION_CAPTURED'
              and isinstance(progress.get('prepared_state'), str)
              and Path(progress['prepared_state']).is_file()):
            row['state'] = 'TERMINAL'
            row['data'] = {**progress, 'status': progress['terminal_status']}
        elif progress:
            row['state'] = 'INTERRUPTED' if worker_finished else 'RUNNING'
        elif protocol_ok and not terminal_path.exists():
            row['state'] = 'NOT_STARTED'
    return result
