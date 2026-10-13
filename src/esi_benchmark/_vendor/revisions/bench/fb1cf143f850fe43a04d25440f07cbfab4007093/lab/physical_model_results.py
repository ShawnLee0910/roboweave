"""Post-episode private P/process score collection; never model observations."""
import hashlib
import json
import math
from pathlib import Path


def collect_physical_result(summary, *, scene_run_dir, episode_id, output_dir):
    if not episode_id or Path(episode_id).name != episode_id or episode_id in {'.', '..'}:
        raise ValueError('invalid episode identity')
    root = Path(scene_run_dir).resolve()
    source = root / 'episodes' / episode_id / 'private/final_score.json'
    if not source.exists():
        return summary  # Interrupted episodes may have no final score; do not fabricate one.
    if source.is_symlink() or not source.resolve().is_relative_to(root) or source.stat().st_size > 512*1024:
        raise ValueError('unsafe final score')
    raw = source.read_bytes()
    score = json.loads(raw)
    if not isinstance(score, dict):
        raise ValueError('final score must be an object')
    physical = score.get('native_q_alone_is_success') is False
    process = score.get('public_process_report')
    if not physical and process is None:
        return summary  # Existing non-P result semantics are unchanged.
    q = score.get('official_q')
    if (score.get('episode_id') != episode_id
            or (summary.get('current_scene_episode_id') not in (None, episode_id))
            or (q is not None and (type(q) not in (int, float) or not math.isfinite(q) or not 0 <= q <= 1))
            or type(score.get('official_success')) is not bool
            or type(score.get('verifier_ok')) is not bool
            or (physical and score.get('physical_qualification') not in {'PASS', 'FAIL', 'UNKNOWN'})):
        raise ValueError('invalid or mismatched Physical final score')
    if process is not None and (not isinstance(process, dict)
            or type(process.get('operational')) is not bool
            or type(process.get('public_process_satisfied')) is not bool
            or not isinstance(process.get('errors'), list)
            or not isinstance(process.get('violations'), list)):
        raise ValueError('invalid process final score')
    success = (score['official_success'] and score['verifier_ok']
               and score.get('integrity_failure') is None
               and (not physical or score['physical_qualification'] == 'PASS')
               and (process is None or (process['operational']
                    and process['public_process_satisfied']
                    and not process['errors'] and not process['violations']))
               and score.get('task_success') is True)
    target = Path(output_dir) / 'private/scene_final_score.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or json.loads(target.read_bytes()) != score:
            raise ValueError('archived final score differs')
    else:
        with target.open('xb') as stream:
            stream.write(raw)
        target.chmod(0o600)
    result = dict(summary)
    result.update(q_score=q, official_success=success, official_task_success=success,
        verifier_ok=score['verifier_ok'], verifier_pass=success,
        private_score_received_from_scene=True, private_score_read_after_episode=True,
        private_score_exposed_to_model=False, termination_reason=score.get('termination_reason'),
        final_score_source=str(source), final_score_source_sha256=hashlib.sha256(raw).hexdigest())
    if physical:
        result.update(physical_qualification=score['physical_qualification'],
            native_q_alone_is_success=False,
            physics_substeps_observed=score.get('physics_substeps_observed'))
    if process is not None:
        result.update(public_process_report=process,
            public_process_satisfied=process['public_process_satisfied'])
    return result
