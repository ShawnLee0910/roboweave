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


@pytest.fixture
def scored_run(tmp_path):
    output = tmp_path/'scored'
    run_replay({}, output)
    scene = tmp_path/'scene/episodes/example/private'
    scene.mkdir(parents=True)
    score = {'episode_id': 'example', 'official_q': 1, 'official_success': True, 'verifier_ok': True}
    (scene/'final_score.json').write_text(json.dumps(score))
    summary_path = output/'summary.json'
    summary = json.loads(summary_path.read_text())
    summary['current_scene_episode_id'] = 'example'
    summary['candidate_result'] = _archive_scene_result(tmp_path/'scene', output, summary, {})
    summary_path.write_text(json.dumps(summary))
    (output/'run_record.json').write_text(json.dumps({'mode': 'live'}))
    return output


@pytest.mark.parametrize('has_chatml', [True, False])
def test_archive_identity_failure_is_visible_without_rewriting_summary(tmp_path, has_chatml, capsys):
    output = tmp_path/'archive_failure'
    run_replay({}, output)
    summary_path = output/'summary.json'
    summary = json.loads(summary_path.read_text())
    summary['current_scene_episode_id'] = 'example'
    summary_path.write_text(json.dumps(summary))
    before = summary_path.read_bytes()
    scene = tmp_path/'scene/episodes/example/private'
    scene.mkdir(parents=True)
    (scene/'final_score.json').write_text(json.dumps({'episode_id': 'another_episode'}))
    with pytest.raises(ValueError, match='identity mismatch'):
        _archive_scene_result(tmp_path/'scene', output, summary, {})
    # This is the existing CLI failure artifact, not a simulated successful score.
    failure = {'status': 'INITIALIZATION_FAILED', 'stage': 'model_episode', 'error_type': 'ValueError'}
    (output/'candidate_error.json').write_text(json.dumps(failure))
    (output/'run_record.json').write_text(json.dumps({'mode': 'live'}))
    if not has_chatml:
        (output/'conversation_chatml.json').unlink()
    inspected = inspect_run(output)
    assert inspected['status'] == 'INCOMPLETE'
    assert inspected['completeness'] == 'candidate_failure'
    assert inspected['client_status'] == 'DONE'
    assert inspected['candidate_error'] == failure
    assert inspected['score_available'] is False
    assert inspected['official_success'] is None and inspected['official_q'] is None
    assert summary_path.read_bytes() == before
    from esi_benchmark.cli import main
    assert main(['results', 'validate', str(output)]) == 1
    assert json.loads(capsys.readouterr().out)['candidate_error'] == failure


def test_known_failure_does_not_count_even_a_previously_archived_score(scored_run, tmp_path):
    good = tmp_path/'good'
    run_replay({}, good)
    (good/'run_record.json').write_text(json.dumps({'mode': 'live'}))
    summary_path = good/'summary.json'
    summary = json.loads(summary_path.read_text())
    summary['candidate_result'] = {'status': 'unavailable', 'reason': 'Scene did not produce a final score',
                                   'official_success': None, 'official_q': None, 'evidence': {}}
    summary_path.write_text(json.dumps(summary))
    # Missing evidence remains unscored; an explicit execution failure is additionally incomplete.
    assert inspect_run(good)['status'] == 'PASS'
    failure = {'status': 'INITIALIZATION_FAILED', 'stage': 'model_episode', 'error_type': 'ValueError'}
    (scored_run/'candidate_error.json').write_text(json.dumps(failure))
    report = summarize_runs([good, scored_run])
    assert [run['status'] for run in report['runs']] == ['PASS', 'INCOMPLETE']
    assert report['total_live_attempts'] == 2
    assert report['scored_live_runs'] == 0 and report['successful_live_runs'] == 0
    assert report['unscored_live_runs'] == 2 and report['incomplete_runs'] == 1


@pytest.mark.parametrize('artifact,fields,value', [
    ('summary.json', ['candidate_result'], []),
    ('summary.json', ['candidate_result', 'status'], []),
    ('summary.json', ['candidate_result', 'official_success'], 1),
    ('summary.json', ['candidate_result', 'official_q'], True),
    ('summary.json', ['candidate_result', 'evidence'], []),
    ('summary.json', ['candidate_result', 'evidence'], None),
    ('summary.json', ['candidate_result', 'evidence', 'final_score.json'], []),
    ('summary.json', ['candidate_result', 'evidence', 'final_score.json'], None),
    ('client_manifest.json', ['prompt_provenance'], []),
    ('client_manifest.json', ['prompt_provenance'], None),
    ('run_record.json', [], []),
    ('run_record.json', ['mode'], []),
    ('run_record.json', ['evaluation_variant'], {}),
    ('run_record.json', ['source_revisions'], []),
    ('run_record.json', ['task_identity'], []),
    ('run_record.json', ['model'], []),
    ('run_record.json', ['declared_budget'], []),
    ('conversation_chatml.json', ['messages', 0, 'role'], []),
])
def test_malformed_nested_fields_are_invalid_without_losing_other_runs(scored_run, tmp_path, artifact, fields, value):
    good = tmp_path/'good'
    run_replay({}, good)
    path = scored_run/artifact
    payload = json.loads(path.read_text())
    if fields:
        target = payload
        for key in fields[:-1]:
            target = target[key]
        target[fields[-1]] = value
    else:
        payload = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        inspect_run(scored_run)
    report = summarize_runs([good, scored_run])
    assert [run['status'] for run in report['runs']] == ['PASS', 'INVALID']
    assert report['invalid_runs'] == 1 and report['counts']['offline_replay'] == 1


@pytest.mark.parametrize('value', [[], {'status': 'INITIALIZATION_FAILED', 'stage': [], 'error_type': 'ValueError'}])
def test_malformed_candidate_error_is_rejected(scored_run, value):
    (scored_run/'candidate_error.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):
        inspect_run(scored_run)


def test_legacy_missing_optional_record_remains_supported(scored_run):
    (scored_run/'run_record.json').unlink()
    assert inspect_run(scored_run)['mode'] == 'unknown_legacy'


def test_summary_retains_successful_live_run_and_invalid_live_attempt(scored_run, tmp_path):
    broken = tmp_path/'broken_live'
    run_replay({}, broken)
    (broken/'run_record.json').write_text(json.dumps({'mode': 'live'}))
    summary_path = broken/'summary.json'
    summary = json.loads(summary_path.read_text())
    summary['candidate_result'] = {'status': 'recorded', 'evidence': []}
    summary_path.write_text(json.dumps(summary))
    report = summarize_runs([scored_run, broken])
    assert [run['status'] for run in report['runs']] == ['PASS', 'INVALID']
    assert report['total_live_attempts'] == 2 and report['invalid_runs'] == 1
    assert report['scored_live_runs'] == 1 and report['successful_live_runs'] == 1
    assert report['success_rate_scored_live'] == 1
    assert report['success_rate_all_live_attempts'] == report['live_score_coverage'] == 0.5


def test_final_score_extreme_integer_is_a_validation_error():
    score = {'episode_id': 'example', 'official_q': 10**1000, 'official_success': True, 'verifier_ok': True}
    with pytest.raises(ValueError):
        recorded_score_result(score, 'example')


@pytest.mark.parametrize('artifact', ['summary.json', 'run_record.json'])
def test_deeply_nested_json_does_not_interrupt_other_runs(scored_run, tmp_path, artifact):
    broken = tmp_path/'deeply_nested'
    broken.mkdir()
    (broken/'summary.json').write_text(json.dumps({'status': 'INITIALIZATION_FAILED'}))
    (broken/'run_record.json').write_text(json.dumps({'mode': 'live'}))
    (broken/artifact).write_text('{"payload":' + '[' * 10000 + '0' + ']' * 10000 + '}')
    with pytest.raises(ValueError, match='JSON nesting limit'):
        inspect_run(broken)
    report = summarize_runs([scored_run, broken])
    assert [run['status'] for run in report['runs']] == ['PASS', 'INVALID']
    assert report['invalid_runs'] == 1
    assert report['scored_live_runs'] == report['successful_live_runs'] == 1
