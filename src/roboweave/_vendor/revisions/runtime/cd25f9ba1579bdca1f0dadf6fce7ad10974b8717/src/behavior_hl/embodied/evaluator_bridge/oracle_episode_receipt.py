"""CPU-only episode evidence: mutable phase journal, immutable final receipt.

The executor owns final outcomes. Controllers and collectors may read them but
must never replace them based on a worker exit code.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from time import monotonic
from datetime import datetime, timezone
from typing import Any, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock

_verification_scope = ContextVar('oracle_verification_scope', default=None)
_finalization_lock = RLock()


class ReceiptConflict(RuntimeError):
    pass


def exception_evidence(exc: Exception) -> dict:
    root = exc
    for _ in range(8):
        if root.__cause__ is None:
            break
        root = root.__cause__
    signature_error = isinstance(root, TypeError) and 'unexpected keyword argument' in str(root)
    return dict(failure_owner='RUNTIME_CODE' if signature_error else 'UNKNOWN',
                failure_code='CALL_SIGNATURE_MISMATCH' if signature_error else type(root).__name__,
                retry_after_fix=signature_error)


def read_document(path: Path, limit: int = 16 * 1024 * 1024) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f'invalid or oversized receipt: {path}')
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f'duplicate receipt field: {key}')
            out[key] = value
        return out
    def reject_constant(value):
        raise ValueError(f'non-finite receipt value: {value}')
    value = json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError('receipt must be a JSON object')
    json.dumps(value, allow_nan=False)
    return value


def task_directory(run_dir: Path, task_id: str) -> Path:
    if not task_id or Path(task_id).parts != (task_id,) or task_id in {'.', '..'}:
        raise ValueError('unsafe episode identity')
    return run_dir / 'private/resident_tasks' / task_id


def _publish_once(target: Path, row: Mapping) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    fd, temporary = tempfile.mkstemp(prefix='.receipt-', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if read_document(target) != row:
                raise ReceiptConflict(f'already finalized: {target}')
        parent = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        os.unlink(temporary)


@contextmanager
def verification_scope(run_dir: Path, task_id: str):
    token = _verification_scope.set((run_dir, task_id))
    try:
        yield
    finally:
        _verification_scope.reset(token)


def publish_final_verification(*, official_q, verifier_ok: bool) -> None:
    """Called at the canonical final snapshot, before rendering/serialization."""
    scope = _verification_scope.get()
    if scope is None:
        return  # Other legacy entrypoints retain their existing evidence path.
    if not isinstance(verifier_ok, bool):
        raise ValueError('verifier availability must be boolean')
    run_dir, task_id = scope
    q = official_q if verifier_ok else None
    if q is not None and (isinstance(q, bool) or not isinstance(q, (int, float))
                          or not math.isfinite(q) or not 0 <= q <= 1):
        raise ValueError('invalid canonical verification Q')
    proof = {
        'schema_version': 'oracle-final-verification/v1', 'task_spec_id': task_id,
        'attempt_id': str(run_dir.resolve()),
        'execution_identity': os.environ.get('ORACLE_EXECUTION_IDENTITY') or str(run_dir.resolve()),
        'verification_status': 'COMPLETED' if verifier_ok and q is not None else 'UNAVAILABLE',
        'official_q': q}
    with _finalization_lock:
        directory = task_directory(run_dir, task_id)
        if (directory/'oracle_result.json').exists():
            raise ReceiptConflict('cannot publish verification after episode closure')
        _publish_once(directory/'final_verification.json', proof)


def record_phase(run_dir: Path, task_id: str, phase: str) -> None:
    from behavior_hl.embodied.evaluator_bridge.resident_scene_batch import write_json_atomic
    # The phase journal is not a final outcome and may advance during execution.
    path = task_directory(run_dir, task_id) / 'episode_phase.json'
    previous = read_document(path) if path.is_file() else {}
    durations = dict(previous.get('phase_durations_seconds', {}))
    now = monotonic()
    if previous.get('phase_started_monotonic') is not None:
        key = previous['last_confirmed_phase']
        durations[key] = durations.get(key, 0) + max(0, now - previous['phase_started_monotonic'])
    write_json_atomic(path, {'task_spec_id': task_id, 'last_confirmed_phase': phase,
        'phase_started_monotonic': now, 'phase_started_at': datetime.now(timezone.utc).isoformat(),
        'phase_durations_seconds': durations})


def commit_episode_result(run_dir: Path, task_id: str, result: Mapping[str, Any], *, lock_timeout=None) -> dict:
    acquired = _finalization_lock.acquire() if lock_timeout is None else _finalization_lock.acquire(timeout=lock_timeout)
    if not acquired:
        raise TimeoutError('episode finalization is still writing evidence')
    try:
        return _commit_episode_result_locked(run_dir, task_id, result)
    finally:
        _finalization_lock.release()


def _commit_episode_result_locked(run_dir: Path, task_id: str, result: Mapping[str, Any]) -> dict:
    directory = task_directory(run_dir, task_id)
    directory.mkdir(parents=True, exist_ok=True)
    phase_file = directory / 'episode_phase.json'
    phase_data = read_document(phase_file) if phase_file.is_file() else {}
    phase = phase_data.get('last_confirmed_phase')
    row = dict(result)
    if row.get('status') not in {'PASS', 'FAIL', 'TIMEOUT', 'ABORTED', 'SKIPPED_BASE_BLOCKED', 'CONSTRUCTION_CAPTURED', 'RENDERED_ONLY'}:
        raise ValueError('episode receipt must have a terminal status')
    proof_path = directory/'final_verification.json'
    proof = read_document(proof_path) if proof_path.is_file() else None
    if proof is not None:
        expected_identity = os.environ.get('ORACLE_EXECUTION_IDENTITY') or str(run_dir.resolve())
        if (proof.get('task_spec_id') != task_id or proof.get('attempt_id') != str(run_dir.resolve())
                or proof.get('execution_identity') != expected_identity
                or proof.get('schema_version') != 'oracle-final-verification/v1'
                or proof.get('verification_status') not in {'COMPLETED', 'UNAVAILABLE'}
                or (proof['verification_status'] == 'COMPLETED') != (proof.get('official_q') is not None)):
            raise ValueError('final verification identity mismatch')
        if row.get('official_q') is not None and row['official_q'] != proof.get('official_q'):
            row['reported_q_conflict'] = repr(row['official_q'])
        row.update(official_q=proof['official_q'], verification_status=proof['verification_status'],
                   final_verification_reference=str(proof_path))
    elif (row.get('verifier_ran') is False or row.get('verification_completed') is False
          or row.get('verification_status') in {'NOT_RUN', 'UNAVAILABLE', 'INCOMPLETE', 'UNKNOWN'}):
        if row.get('official_q') is not None:
            row['reported_q_conflict'] = repr(row['official_q'])
        row['official_q'] = None
        row.setdefault('verification_status', 'NOT_RUN' if row.get('verifier_ran') is False else 'UNKNOWN')
    if row.get('verification_status') == 'COMPLETED' and row.get('official_q') is None:
        row['verification_status'] = 'UNKNOWN'
    q = row.get('official_q')
    if q is not None and (isinstance(q, bool) or not isinstance(q, (int, float))
                          or not math.isfinite(q) or not 0 <= q <= 1):
        raise ValueError('invalid official Q in final receipt')
    entry_path = directory/'oracle_entry.json'
    entry = read_document(entry_path) if entry_path.is_file() else {}
    oracle_ran = entry.get('oracle_invocation_started') is True or row.get('marker_action_gate_attempted') is True or row.get('status') == 'TIMEOUT'
    row.update(receipt_schema_version='oracle-task-receipt/v1', task_spec_id=task_id,
               execution_identity=os.environ.get('ORACLE_EXECUTION_IDENTITY') or str(run_dir.resolve()),
               attempt_id=str(run_dir.resolve()), last_confirmed_phase=phase,
               official_q=q)
    execution = row.get('oracle_execution_status') or (
        'NOT_RUN' if not oracle_ran else 'INCOMPLETE' if row['status'] in {'TIMEOUT', 'ABORTED'}
        or row.get('error_type') else 'COMPLETED')
    row.update(oracle_entered=oracle_ran, oracle_execution_status=execution, oracle_status=execution,
               verification_status=row.get('verification_status') or ('COMPLETED' if q is not None else 'UNKNOWN' if oracle_ran else 'NOT_RUN'),
               verifier_ran=True if proof is not None or q is not None else
                   row.get('verifier_ran') if oracle_ran else False,
               goal_status='UNKNOWN' if q is None else 'PASS' if q == 1 else 'FAIL')
    row['verification_completed'] = row['verification_status'] == 'COMPLETED'
    row['admission_status'] = row.get('admission_status') or 'NOT_RUN'
    row.setdefault('status_scope', 'EPISODE_EXECUTION_COMPATIBILITY')
    row.setdefault('deadline_scope', entry.get('deadline_scope'))
    if row['status'] in {'TIMEOUT', 'ABORTED'} and row['verification_status'] == 'COMPLETED':
        row['failure_stage'] = 'CALLBACK_FINALIZATION'
    row.setdefault('failure_owner', None if row.get('status') in {'PASS', 'CONSTRUCTION_CAPTURED', 'RENDERED_ONLY'} else 'UNKNOWN')
    row.setdefault('failure_stage', phase if row.get('status') not in {'PASS', 'CONSTRUCTION_CAPTURED', 'RENDERED_ONLY'} else None)
    row.setdefault('initial_state_status', 'PASS' if oracle_ran or row.get('status') == 'RENDERED_ONLY' else 'UNKNOWN')
    row['evidence_status'] = row.get('evidence_status') or 'NOT_REVIEWED'
    if 'reported_q_conflict' in row or row['verification_completed'] and execution in {'ERROR', 'INCOMPLETE'}:
        row['evidence_status'] = 'INCOMPLETE'
    row.setdefault('retry_after_fix', False)
    row.setdefault('phase_durations_seconds', phase_data.get('phase_durations_seconds', {}))
    row.setdefault('initial_state_reference', str(directory / 'live_initial_construction.json'))
    target = directory / 'oracle_result.json'
    _publish_once(target, row)
    return row


def close_aborted_episodes(run_dir: Path, task_ids, worker_proof: Path) -> None:
    """Only the execution-host owner may close missing receipts after cleanup."""
    proof = read_document(worker_proof)
    if proof.get('cleanup_confirmed') is not True or proof.get('worker_id') != run_dir.parent.name:
        raise ValueError('matching execution-host cleanup proof required')
    for task_id in task_ids:
        directory = task_directory(run_dir, task_id)
        if (directory/'oracle_result.json').exists():
            continue
        phase_path = directory/'episode_phase.json'
        phase = read_document(phase_path).get('last_confirmed_phase') if phase_path.is_file() else None
        if phase in (None, 'DEPENDENCY_CHECK'):
            continue  # Never fabricate an attempt for a task that did not start.
        status = 'TIMEOUT' if phase == 'RUN_ORACLE' and proof.get('exit_code') == 124 else 'ABORTED'
        commit_episode_result(run_dir, task_id, dict(status=status, official_q=None,
            failure_class='ORACLE_RESOURCE_EXHAUSTED' if status == 'TIMEOUT' else 'UNKNOWN',
            oracle_status='ABORTED' if phase == 'RUN_ORACLE' else 'NOT_RUN',
            failure_stage=None, failure_owner='UNKNOWN', failure_code='WORKER_ABORT',
            resident_state_reusable=False, worker_cleanup_proof=str(worker_proof)))
