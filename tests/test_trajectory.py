import json
import pytest

from esi_benchmark.models import _archive_scene_result, recorded_score_result
from esi_benchmark.offline import run_replay
from esi_benchmark.trajectory import inspect_run, summarize_runs


def test_failed_initialization_is_preserved_and_not_scored(tmp_path):
    output = tmp_path/'failed'
    output.mkdir()
    summary = {'status': 'FAIL_CLIENT_INITIALIZATION', 'model_calls': 0}
    result = _archive_scene_result(tmp_path/'scene', output, summary, {'seed': 0})
    assert result['status'] == 'unavailable' and result['official_success'] is None
    (output/'summary.json').write_text(json.dumps({**summary, 'candidate_result': result}))
    (output/'conversation_unavailable.json').write_text(json.dumps({'reason': 'post_begin_client_initialization_failed'}))
    (output/'run_record.json').write_text(json.dumps({'mode': 'live'}))
    inspected = inspect_run(output)
    assert inspected['status'] == 'INCOMPLETE' and inspected['client_status'] == summary['status']
    assert inspected['score_available'] is False


def test_summary_keeps_invalid_and_unscored_live_attempts(tmp_path):
    output = tmp_path/'failed'
    output.mkdir()
    (output/'summary.json').write_text(json.dumps({'status': 'INITIALIZATION_FAILED'}))
    (output/'run_record.json').write_text(json.dumps({'mode': 'live'}))
    broken = tmp_path/'broken'
    broken.mkdir()
    (broken/'run_record.json').write_text(json.dumps({'mode': 'live'}))
    report = summarize_runs([output, broken])
    assert report['total_live_attempts'] == 2
    assert report['scored_live_runs'] == 0 and report['unscored_live_runs'] == 2
    assert report['invalid_runs'] == 1 and report['incomplete_runs'] == 1
    assert report['success_rate_all_live_attempts'] == 0
    assert report['success_rate_scored_live'] is None


def test_score_reference_is_verified_without_q_shortcut(tmp_path):
    output = tmp_path/'replay'
    run_replay({}, output)
    scene = tmp_path/'scene/episodes/example/private'
    scene.mkdir(parents=True)
    score = {'episode_id': 'example', 'official_q': 1, 'official_success': False,
             'task_success': True, 'verifier_ok': True, 'integrity_failure': None}
    (scene/'final_score.json').write_text(json.dumps(score))
    summary = json.loads((output/'summary.json').read_text())
    summary['current_scene_episode_id'] = 'example'
    summary['candidate_result'] = _archive_scene_result(tmp_path/'scene', output, summary, {})
    (output/'summary.json').write_text(json.dumps(summary))
    # A synthetic test of the Live record format, never evidence of a Live run.
    (output/'run_record.json').write_text(json.dumps({'mode': 'live'}))
    inspected = inspect_run(output)
    assert inspected['score_available'] is True
    assert inspected['official_success'] is False and inspected['official_q'] == 1
    (output/'private/scene_final_score.json').write_text('{}')
    with pytest.raises(ValueError, match='hash'):
        inspect_run(output)


@pytest.mark.parametrize('field,value', [('official_q', True), ('verifier_ok', 1), ('official_success', 1), ('task_success', 'true')])
def test_malformed_final_score_rejected(field, value):
    score = {'episode_id': 'example', 'official_q': 1, 'official_success': True, 'verifier_ok': True, field: value}
    with pytest.raises(ValueError):
        recorded_score_result(score, 'example')
