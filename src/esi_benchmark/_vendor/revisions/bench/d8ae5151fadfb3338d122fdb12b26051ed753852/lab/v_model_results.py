"""Model-side archive of the shared human/model V episode, never a V executor."""
import json
import math
from pathlib import Path

def apply_v_result(summary, *, scene_run_dir, episode_id, output_dir):
    if scene_run_dir is None:
        return summary
    root = Path(scene_run_dir).resolve()
    if not episode_id or Path(episode_id).name != episode_id or episode_id in {'.', '..'}:
        raise ValueError('invalid V episode identity')
    private = root / 'episodes' / episode_id / 'private'
    identity_path = private / 'v_collection_identity.json'
    if not identity_path.exists() and not (root/'private/v_scene_identity.json').exists():
        return summary  # Existing non-V protocols are unchanged.

    def read(path):
        if not path.resolve().is_relative_to(root) or path.stat().st_size > 2*1024*1024:
            raise ValueError('unsafe V result artifact')
        return json.loads(path.read_text())

    result = dict(summary)
    try:
        identity = read(identity_path)
        scene_identity_path = root/'private/v_scene_identity.json'
        if scene_identity_path.exists() and read(scene_identity_path) != identity:
            raise ValueError('V episode identity differs from resident binding')
        score = read(private / 'final_score.json')
        report = read(private / 'v_protocol_result.json')
        q = score.get('official_q')
        if (identity.get('branch') != 'D1'
                or not identity.get('collection_config_key')
                or score.get('episode_id') != episode_id
                or score.get('collection_config_key') != identity['collection_config_key']
                or score.get('v_protocol') != report
                or type(score.get('official_success')) is not bool
                or type(q) not in (int, float) or not math.isfinite(q) or not 0 <= q <= 1):
            raise ValueError('V identity or final report mismatch')
        result.update(collection_config_key=identity['collection_config_key'],
            v_package_id=identity['package_id'], v_branch=identity['branch'],
            v_protocol=report, canonical_goal_success=score.get('official_success') is True,
            q_score=score.get('official_q'), private_score_received_from_scene=True,
            private_score_exposed_to_model=False)
        complete = (report.get('v_protocol_complete') is True
                    and report.get('operational') is not False
                    and score.get('verifier_ok') is True
                    and score.get('integrity_failure') is None)
        result['v_protocol_complete'] = complete
        result['official_success'] = result['official_task_success'] = complete and result['canonical_goal_success']
        # Raw goal score remains intact. Protocol defects are not model failures.
        if not complete:
            result.update(status='FAIL_V_PROTOCOL', outcome_class='INVALID_FORMAL_RUN',
                formal_score_eligible=False, failure_stage='V_PROTOCOL',
                failure_reason='v_protocol_incomplete',
                formal_ineligibility_reasons=[*result.get('formal_ineligibility_reasons', []), 'v_protocol_incomplete'])
        paths = [identity_path, private/'final_score.json', private/'v_protocol_result.json',
                 private/'v_marker_bindings.jsonl']
        folder = private/'v_intervention'
        if folder.is_dir():
            for p in folder.iterdir():
                if len(paths) >= 64: raise ValueError('V archive file limit exceeded')
                if p.suffix in {'.json', '.jsonl'}: paths.append(p)
        for source in paths:
            if not source.is_file():continue
            if not source.resolve().is_relative_to(root) or source.stat().st_size > 32*1024*1024:
                raise ValueError('unsafe V archive source')
            target = Path(output_dir)/'private/v_scene'/source.relative_to(private)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = source.read_bytes()
            if target.exists():
                if target.read_bytes() != data:raise ValueError('V archive collision')
            else:
                with target.open('xb') as stream:stream.write(data)
                target.chmod(0o600)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result.update(status='FAIL_V_PROTOCOL', outcome_class='INVALID_FORMAL_RUN',
            formal_score_eligible=False, official_success=False, official_task_success=False,
            v_protocol_complete=False, failure_stage='V_PROTOCOL',
            failure_reason='v_result_missing_or_inconsistent', v_archive_error=str(exc))
    return result
