"""Private construction progress and an opt-in wall budget, not task scoring."""
from contextlib import contextmanager
from contextvars import ContextVar
import faulthandler
import json
import math
import os
from threading import Lock, Timer
from time import monotonic

from .resident_scene_batch import write_json_atomic
from .oracle_episode_receipt import commit_episode_result

ACTIVE = ContextVar('construction_progress', default=None)


def checkpoint(phase, **fields):
    trace = ACTIVE.get()
    if trace is not None:
        trace(phase, **fields)


@contextmanager
def construction_scope(run_dir, item_id, timeout_seconds=None):
    if timeout_seconds is not None and (
            not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError('construction timeout must be positive and finite')
    root = run_dir / 'private/resident_tasks' / item_id
    root.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    current = {}
    lock = Lock()
    finished = False

    def record(phase, **fields):
        now = monotonic()
        row = dict(phase=phase, elapsed_seconds=now-started, **fields)
        if current:
            row['previous_phase'] = current['phase']
            row['previous_phase_seconds'] = row['elapsed_seconds']-current['elapsed_seconds']
        current.clear()
        current.update(row)
        with (root / 'construction_steps.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        write_json_atomic(root / 'construction_progress.json', row)

    def expire():
        with lock:
            if finished:
                return
            report = dict(status='TIMEOUT', failure_class='CONSTRUCTION_RESOURCE_EXHAUSTED',
                failure_stage='INITIAL_STATE_CONSTRUCTION', initial_state_status='FAIL',
                construction_pass=False, oracle_started=False, oracle_status='NOT_RUN',
                official_q=None, reason='initial_construction_wall_time_budget_exhausted',
                construction_timeout_seconds=timeout_seconds,
                construction_elapsed_seconds=monotonic()-started,
                last_construction_step=dict(current), resident_state_reusable=False,
                deadline_scope='INITIAL_STATE_CONSTRUCTION_ONLY')
            try:
                # No simulator queries from the watchdog: only preserved progress.
                with (root / 'construction_stack.log').open('a') as stack:
                    faulthandler.dump_traceback(file=stack, all_threads=True)
                write_json_atomic(root / 'construction_timeout.json', report)
                commit_episode_result(run_dir, item_id, report, lock_timeout=.5)
            finally:
                os._exit(124)

    token = ACTIVE.set(record)
    timer = None
    stack = None
    previous_threads = None
    try:
        record('CONSTRUCTION_START', timeout_seconds=timeout_seconds)
        if timeout_seconds is not None:
            stack = (root / 'construction_stack.log').open('a')
            # Native watchdog also bounds a C call holding the GIL. The Python
            # timer normally publishes the classified receipt first.
            faulthandler.dump_traceback_later(timeout_seconds+2, file=stack, exit=True)
            timer = Timer(timeout_seconds, expire)
            timer.daemon = True
            timer.start()
        import torch
        previous_threads = torch.get_num_threads()
        # Small collision-boundary reductions regress badly with 96 OpenMP
        # threads per concurrent Kit. This scope never changes geometry or
        # carries the reduced thread count into Oracle / model execution.
        torch.set_num_threads(min(previous_threads, 4))
        record('CONSTRUCTION_CPU_THREADS', previous=previous_threads,
               effective=torch.get_num_threads())
        yield
        record('CONSTRUCTION_PASS')
    except BaseException as exc:
        record('CONSTRUCTION_ERROR', error_type=type(exc).__name__)
        raise
    finally:
        with lock:
            finished = True
            if timer is not None:
                timer.cancel()
                faulthandler.cancel_dump_traceback_later()
            if stack is not None:
                stack.close()
        try:
            if previous_threads is not None:
                torch.set_num_threads(previous_threads)
        finally:
            ACTIVE.reset(token)
