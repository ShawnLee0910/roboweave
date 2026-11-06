"""Rebuild a task-level view from a frozen plan and immutable episode receipts.

No Q recomputation, automatic retry, simulator imports, or recursive discovery.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

from .oracle_episode_receipt import read_document


def sole_directory(parent: Path) -> Path | None:
    if not parent.is_dir() or parent.is_symlink():
        return None
    with os.scandir(parent) as entries:
        found = []
        for entry in entries:
            found.append(entry)
            if len(found) > 1:
                raise ValueError(f'ambiguous worker/run directory: {parent}')
    if not found:
        return None
    return Path(found[0].path) if found[0].is_dir(follow_symlinks=False) else None


def collect_task_results(plan: Mapping, state: Mapping) -> list[dict]:
    if plan['plan_sha256'] != state['plan_sha256']:
        raise ValueError('task ledger plan identity mismatch')
    root = Path(plan['artifact_root'])
    rows = []
    for batch in plan['batches']:
        batch_id = batch['batch_id']
        record = state['batches'][batch_id]
        attempts = int(record.get('attempts', 0))
        if not 0 <= attempts <= 128:
            raise ValueError('invalid attempt count')
        evidence = {task: [] for task in batch['task_names']}
        for attempt in range(attempts):
            artifact = root/'batches'/batch['batch_path_id']/f'attempt_{attempt:03d}'
            if not artifact.resolve().is_relative_to(root.resolve()):
                raise ValueError('attempt path escaped campaign root')
            worker_file = Path(str(artifact) + '.worker.json')
            health = read_document(worker_file) if worker_file.is_file() else {}
            worker = sole_directory(artifact/'workers')
            run = sole_directory(worker) if worker else None
            for task in batch['task_names']:
                folder = run/'private/resident_tasks'/task if run else None
                receipt = folder/'oracle_result.json' if folder else None
                phase_path = folder/'episode_phase.json' if folder else None
                phase = read_document(phase_path).get('last_confirmed_phase') if phase_path and phase_path.is_file() else None
                progress_error = None
                progress_path = folder/'task_progress.json' if folder else None
                if phase is None and progress_path and progress_path.is_file():
                    try:
                        from .resident_task_progress import SCHEMA
                        progress = read_document(progress_path)
                        if (progress.get('schema_version') != SCHEMA
                                or progress.get('task_name') != task
                                or ('seed' in plan and progress.get('seed') != plan['seed'])):
                            raise ValueError('task progress identity mismatch')
                        phase = progress.get('phase')
                        if not isinstance(phase, str) or not phase:
                            raise ValueError('task progress phase unavailable')
                    except (ValueError, OSError) as exc:
                        progress_error = str(exc)
                row = dict(attempt_index=attempt, worker_status=health.get('worker_status', 'UNKNOWN'),
                           cleanup_confirmed=health.get('cleanup_confirmed', False),
                           last_confirmed_phase=phase, artifact_root=str(artifact))
                if receipt and receipt.is_file():
                    try:
                        result = read_document(receipt)
                        if result.get('task_spec_id', task) != task:
                            raise ValueError('receipt task mismatch')
                        identity = result.get('execution_identity')
                        if identity is not None and identity != plan['plan_sha256']:
                            raise ValueError('receipt execution identity mismatch')
                        fields = ('status', 'official_q', 'actions_executed', 'oracle_execution_seconds',
                                  'failure_class', 'failure_owner', 'failure_code', 'failure_stage',
                                  'last_confirmed_phase', 'oracle_status', 'initial_state_status',
                                  'verifier_ran', 'error', 'reason', 'blocked_by_base', 'evidence_status',
                                  'retry_after_fix', 'resident_state_reusable', 'phase_name',
                                  'execution_initial_pose', 'initial_state_identity', 'phase_durations_seconds')
                        fields += ('goal_status', 'verification_status', 'oracle_execution_status',
                                   'admission_status', 'route_gate_status', 'final_verification_reference', 'status_scope')
                        fields += ('oracle_started', 'oracle_skip_reason', 'base_task_spec_id',
                                   'base_result_path', 'prepared_state', 'first_frame',
                                   'construction_status', 'preparation_status',
                                   'preparation_failure_type', 'retry_scope')
                        fields += ('canonical_execution_status', 'public_process_report', 'task_success')
                        row.update({key: result[key] for key in fields if key in result})
                        for key in ('error', 'reason'):
                            if isinstance(row.get(key), str) and len(row[key]) > 4096:
                                row[key] = row[key][:4096]
                                row[key + '_preview_truncated'] = True
                        row.update(artifact_path=str(receipt), receipt_valid=True,
                                   receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
                        row['initial_state_receipt_path'] = str(folder/'live_initial_construction.json')
                    except (ValueError, OSError) as exc:
                        row.update(status='INVALID_RECEIPT', official_q=None,
                                   error=str(exc), receipt_valid=False, failure_owner='UNKNOWN')
                elif progress_error is not None:
                    row.update(status='EXECUTION_STATE_UNKNOWN', official_q=None,
                               error=progress_error, failure_owner='UNKNOWN')
                elif record['status'] == 'RUNNING' and attempt == attempts - 1:
                    row.update(status='RUNNING' if phase else 'NOT_STARTED', official_q=None)
                elif health.get('cleanup_confirmed') is True:
                    row.update(status='ABORTED' if phase not in (None, 'DEPENDENCY_CHECK') else 'NOT_RUN_DUE_TO_WORKER_ABORT',
                               official_q=None, failure_owner='UNKNOWN')
                else:
                    row.update(status='EXECUTION_STATE_UNKNOWN', official_q=None, failure_owner='UNKNOWN')
                evidence[task].append(row)
        for task in batch['task_names']:
            history = evidence[task]
            finals = [row for row in history if row.get('receipt_valid')]
            passed = [row for row in finals if row.get('official_q') == 1]
            selected = passed[0] if passed else finals[-1] if finals else history[-1] if history else {'status': 'NOT_RUN', 'official_q': None}
            row = dict(selected, task_spec_id=task, execution_identity=plan['plan_sha256'],
                       batch_id=batch_id, scene_model=batch['scene_model'], attempts=history,
                       family_id=plan.get('native_family_ids', {}).get(task, task),
                       source_revision=plan.get('source_revision'),
                       visual_review_status='NOT_RUN', promotion_status='NOT_REVIEWED')
            if len(passed) > 1:
                row['duplicate_execution_warning'] = True
            row['continuation_eligible'] = row['status'] in {'NOT_RUN', 'NOT_RUN_DUE_TO_WORKER_ABORT'}
            row['task_attempt_count'] = sum(item.get('status') != 'SKIPPED_BASE_BLOCKED' and
                (item.get('last_confirmed_phase') not in (None, 'DEPENDENCY_CHECK') or item.get('receipt_valid', False))
                for item in history)
            row['worker_dispatch_count'] = len(history)
            row.setdefault('failure_owner', 'UNKNOWN' if row.get('status') != 'PASS' else None)
            rows.append(row)
    equivalence = plan.get('oracle_configuration_equivalence') or {}
    by_id = {row['task_spec_id']: row for row in rows}
    for alias, representative in equivalence.get('aliases', {}).items():
        fingerprints = equivalence.get('fingerprints', {})
        if (alias in by_id or representative not in by_id or alias == representative
                or not fingerprints.get(alias) or fingerprints[alias] != fingerprints.get(representative)):
            raise ValueError('invalid Oracle equivalence reference')
        original = by_id[representative]
        available = original.get('receipt_valid') is True
        reuse_status = ('REUSED_ORACLE_EVIDENCE' if available else
                        'WAITING_FOR_REPRESENTATIVE' if original.get('status') in {'RUNNING', 'NOT_STARTED', 'NOT_RUN'}
                        else 'REPRESENTATIVE_EVIDENCE_UNAVAILABLE')
        rows.append(dict(
            task_spec_id=alias, status=reuse_status,
            execution_identity=plan['plan_sha256'], source_revision=plan.get('source_revision'),
            official_q=None, oracle_started=False, oracle_status='NOT_RUN_EQUIVALENT',
            task_attempt_count=0, worker_dispatch_count=0, attempts=[], continuation_eligible=False,
            representative_task_spec_id=representative,
            representative_status=original.get('status'),
            representative_official_q=original.get('official_q') if available else None,
            representative_task_success=original.get('task_success') if available else None,
            representative_public_process_report=original.get('public_process_report') if available else None,
            representative_artifact_path=original.get('artifact_path'),
            representative_receipt_sha256=original.get('receipt_sha256'),
            equivalence_fingerprint=fingerprints[alias], evidence_reused=available,
            independent_initial_state_verified=False,
            evidence_scope=equivalence.get('evidence_scope'),
            visual_review_status='NOT_RUN', promotion_status='NOT_REVIEWED',
        ))
    return rows


def publish_task_ledger(root: Path, rows: list[dict]) -> None:
    target = root/'_campaign/task_results.jsonl'
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.task-ledger-', dir=target.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)  # Derived view, never the immutable receipt.
    finally:
        if os.path.exists(name):
            os.unlink(name)
