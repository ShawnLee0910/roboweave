import json
import pytest

from roboweave.models import _archive_scene_result, recorded_score_result
from roboweave.offline import run_replay
from roboweave.trajectory import inspect_run, summarize_runs


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


TASK_FIELDS = ('task_id', 'task_name', 'task_split', 'task_instance_id', 'task_seed')


def _live_binding(root, profile='l0-english-v2', *, summary=None):
    """Synthetic Live identity for one catalog task; no simulator or model was invoked."""
    from roboweave.assets import asset_manifest
    from roboweave.profiles import source_revisions
    from roboweave.tasks import list_tasks
    task = next(task for task in list_tasks() if task['profile'] == profile)
    frozen = asset_manifest(profile).get('inputs_by_task', {}).get(task['task_id'], {})
    identity = frozen.get('source_task_identity', {'split': 'synthetic', 'instance_id': 0})
    binding = dict(task_id=task['source']['row_index'], task_name=task['task_id'], task_split=identity['split'],
                   task_instance_id=str(identity['instance_id']), task_seed=frozen.get('seed', 0),
                   instruction=task['instruction_en'], budget=task['budget'], **task['budget'],
                   instruction_provenance={'profile': profile, 'instruction_sha256': task['instruction_sha256']})
    limits = {key: task['budget'][key] for key in ('maximum_action_decisions', 'maximum_model_calls')}
    manifest = json.loads((root/'client_manifest.json').read_text())
    manifest.update({field: binding[field] for field in TASK_FIELDS}, instruction_sha256=task['instruction_sha256'])
    manifest.update(limits, physical_history_max=5, experiment_id=root.name)
    manifest.setdefault('model', 'synthetic-model')
    (root/'client_manifest.json').write_text(json.dumps(manifest))
    if summary is None:
        summary = json.loads((root/'summary.json').read_text())
        summary.update(limits, physical_history_max=5)
        (root/'summary.json').write_text(json.dumps(summary))
    else:
        summary.update(limits, physical_history_max=5)
    # A synthetic test of the Live record format, never evidence of a Live run.
    (root/'run_record.json').write_text(json.dumps(dict(
        mode='live', profile=profile, source_revisions=source_revisions(profile), experiment_id=root.name,
        task_identity={field: binding[field] for field in TASK_FIELDS}, model={'model': manifest['model']},
        declared_budget=task['budget'], history_frames=5, evaluation_variant='frozen_history_5', **limits)))
    return binding


def test_score_reference_is_verified_without_q_shortcut(tmp_path):
    output = tmp_path/'replay'
    run_replay({}, output)
    binding = _live_binding(output)
    scene = tmp_path/'scene/episodes/example/private'
    scene.mkdir(parents=True)
    score = {'episode_id': 'example', 'official_q': 1, 'official_success': False,
             'task_success': True, 'verifier_ok': True, 'integrity_failure': None}
    (scene/'final_score.json').write_text(json.dumps(score))
    summary = json.loads((output/'summary.json').read_text())
    summary['current_scene_episode_id'] = 'example'
    summary['candidate_result'] = _archive_scene_result(tmp_path/'scene', output, summary, binding)
    (output/'summary.json').write_text(json.dumps(summary))
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
    binding = _live_binding(output)
    scene = tmp_path/'scene/episodes/example/private'
    scene.mkdir(parents=True)
    score = {'episode_id': 'example', 'official_q': 1, 'official_success': True, 'verifier_ok': True}
    (scene/'final_score.json').write_text(json.dumps(score))
    summary_path = output/'summary.json'
    summary = json.loads(summary_path.read_text())
    summary['current_scene_episode_id'] = 'example'
    summary['candidate_result'] = _archive_scene_result(tmp_path/'scene', output, summary, binding)
    summary_path.write_text(json.dumps(summary))
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
    from roboweave.cli import main
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


def _minimal_trajectory(root):
    """Small artifact-parser fixture; no simulator or model was invoked."""
    import hashlib
    root.mkdir()
    prompt = b'Synthetic trajectory inspection fixture.\n'
    digest = hashlib.sha256(prompt).hexdigest()
    (root/'system_prompt.txt').write_bytes(prompt)
    (root/'conversation_chatml.json').write_text(json.dumps({
        'schema_version': 'behavior-hl-action20-chatml-v3',
        'messages': [{'role': 'user', 'content': 'Synthetic fixture'}]}))
    (root/'client_manifest.json').write_text(json.dumps({
        'prompt_provenance': {'rendered_prompt_sha256': digest},
        'source_sha256': {'synthetic_fixture': digest}, 'model': 'synthetic-model'}))
    (root/'run_record.json').write_text(json.dumps({'mode': 'live'}))
    return {'status': 'DONE', 'prompt_sha256': digest, 'current_scene_episode_id': 'example', 'model': 'synthetic-model'}


def _write(root, name, value):
    (root/name).write_text(json.dumps(value))


def _rehash_evidence(root, name, value=None, raw=None):
    import hashlib
    summary = json.loads((root/'summary.json').read_text())
    entry = summary['candidate_result']['evidence'][name]
    path = root/entry['path']
    if raw is not None:
        path.write_bytes(raw)
    elif value is not None:
        path.write_text(json.dumps(value))
    content = path.read_bytes()
    entry.update(sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content))
    _write(root, 'summary.json', summary)


@pytest.fixture
def rich_evidence_run(tmp_path):
    root = tmp_path/'rich'
    summary = _minimal_trajectory(root)
    scene = tmp_path/'scene/episodes/example/private'
    scene.mkdir(parents=True)
    process = {'operational': True, 'public_process_satisfied': True, 'violations': [], 'errors': [],
               'execution_events': 1, 'existing_frame_references': 1}
    score = {'episode_id': 'example', 'official_q': 1, 'official_success': True, 'verifier_ok': True,
             'task_success': True, 'public_process_report': process}
    for name, value in [('final_score.json', score), ('task_process_result.json', process),
                        ('physical_initial_binding.json', {'synthetic': True})]:
        _write(scene, name, value)
    (scene/'task_process_events.jsonl').write_text('{"event_id":1,"synthetic":true}\n')
    (scene/'task_evidence_index.jsonl').write_text('{"frame_index":1,"synthetic":true}\n')
    summary['candidate_result'] = _archive_scene_result(tmp_path/'scene', root, summary, _live_binding(root, summary=summary))
    _write(root, 'summary.json', summary)
    return root


def test_all_archived_evidence_is_checked_without_replaying_process(rich_evidence_run):
    assert inspect_run(rich_evidence_run)['score_available'] is True
    summary = json.loads((rich_evidence_run/'summary.json').read_text())
    assert summary['candidate_result']['evidence_schema_version'] == 'roboweave/result-evidence-v2'
    assert 'task_binding.json' in summary['candidate_result']['evidence']


@pytest.mark.parametrize('name', ['task_binding.json', 'task_process_result.json',
    'task_process_events.jsonl', 'task_evidence_index.jsonl', 'physical_initial_binding.json'])
@pytest.mark.parametrize('change', ['missing', 'tampered'])
def test_every_auxiliary_evidence_reference_is_required_and_hashed(rich_evidence_run, name, change):
    summary = json.loads((rich_evidence_run/'summary.json').read_text())
    path = rich_evidence_run/summary['candidate_result']['evidence'][name]['path']
    if change == 'missing':
        path.unlink()
    else:
        path.write_bytes(b'{"tampered":true}\n')
    with pytest.raises(ValueError, match='missing|hash'):
        inspect_run(rich_evidence_run)
    assert summarize_runs([rich_evidence_run])['invalid_runs'] == 1


@pytest.mark.parametrize('change', ['size', 'bad_path', 'symlink', 'parent_symlink', 'missing_binding', 'bad_profile'])
def test_evidence_descriptor_and_path_boundaries(rich_evidence_run, tmp_path, change):
    root = rich_evidence_run
    summary = json.loads((root/'summary.json').read_text())
    evidence = summary['candidate_result']['evidence']
    entry = evidence['task_process_events.jsonl']
    if change == 'size':
        entry['size_bytes'] += 1
    elif change == 'bad_path':
        entry['path'] = '../outside.jsonl'
    elif change == 'symlink':
        path = root/entry['path']
        external = tmp_path/'external.jsonl'
        path.rename(external)
        path.symlink_to(external)
    elif change == 'parent_symlink':
        directory = root/'private'
        directory.rename(root/'private-real')
        directory.symlink_to(root/'private-real', target_is_directory=True)
    elif change == 'missing_binding':
        del evidence['task_binding.json']
    else:
        _write(root, 'run_record.json', {'mode': 'live', 'profile': []})
    _write(root, 'summary.json', summary)
    with pytest.raises(ValueError):
        inspect_run(root)


@pytest.mark.parametrize('raw', [b'[]\n', b'{"event_id":NaN}\n', b'{"event_id":1,"event_id":2}\n', b'broken\n'])
def test_jsonl_requires_finite_unambiguous_object_records_even_when_rehashed(rich_evidence_run, raw):
    _rehash_evidence(rich_evidence_run, 'task_process_events.jsonl', raw=raw)
    with pytest.raises(ValueError):
        inspect_run(rich_evidence_run)


def test_auxiliary_limits_and_score_limit_remain_distinct(rich_evidence_run, monkeypatch):
    from roboweave import trajectory
    monkeypatch.setattr(trajectory, 'MAX_JSON_BYTES', 16)
    with pytest.raises(ValueError, match='oversized'):
        trajectory._evidence(rich_evidence_run, json.loads((rich_evidence_run/'summary.json').read_text())['candidate_result'])


@pytest.mark.parametrize('change', ['report', 'missing_report', 'missing_events', 'truncated_events'])
def test_process_report_and_receipt_counts_cannot_be_forged(rich_evidence_run, change):
    root = rich_evidence_run
    if change == 'report':
        _rehash_evidence(root, 'task_process_result.json', value={'operational': True})
    elif change == 'truncated_events':
        _rehash_evidence(root, 'task_process_events.jsonl', raw=b'')
    else:
        summary = json.loads((root/'summary.json').read_text())
        del summary['candidate_result']['evidence']['task_process_result.json' if change == 'missing_report' else 'task_process_events.jsonl']
        _write(root, 'summary.json', summary)
    with pytest.raises(ValueError, match='evidence'):
        inspect_run(root)


def test_unavailable_or_chatml_missing_run_still_checks_its_evidence(tmp_path):
    root = tmp_path/'unavailable'
    root.mkdir()
    summary = {'status': 'INITIALIZATION_FAILED'}
    summary['candidate_result'] = _archive_scene_result(tmp_path/'scene', root, summary, {})
    _write(root, 'summary.json', summary)
    assert inspect_run(root)['status'] == 'INCOMPLETE'
    (root/'private/task_binding.json').unlink()
    with pytest.raises(ValueError, match='missing'):
        inspect_run(root)


def test_legacy_evidence_without_size_metadata_still_gets_full_hash_checks(rich_evidence_run):
    root = rich_evidence_run
    summary = json.loads((root/'summary.json').read_text())
    del summary['candidate_result']['evidence_schema_version']
    for entry in summary['candidate_result']['evidence'].values():
        del entry['size_bytes']
    _write(root, 'summary.json', summary)
    assert inspect_run(root)['score_available'] is True
    (root/'private/task_process_events.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='hash'):
        inspect_run(root)


@pytest.fixture
def v_evidence_run(tmp_path):
    from roboweave.assets import asset_manifest
    from roboweave.profiles import source_revisions
    from roboweave.tasks import list_tasks
    root = tmp_path/'v-run'
    summary = _minimal_trajectory(root)
    task = list_tasks('L1-V')[0]
    key = task['task_id']
    profile = task['profile']
    revisions = source_revisions(profile)
    identity = dict(collection_config_key=key, task_spec_id=task['runtime_instruction_id'],
        package_id='synthetic-v-package', branch='D1', runtime_revision=revisions['runtime'],
        historical_oracle_runtime_revision=revisions['runtime'], source_oracle_success=True,
        runtime_validation_only=False, v_package_sha256='a'*64)
    binding = _live_binding(root, profile, summary=summary)
    binding.update(collection_config_key=key, v_identity=identity)
    report = {'branch': 'D1', 'v_protocol_complete': True, 'operational': True}
    score = {'episode_id': 'example', 'official_q': 1, 'official_success': True, 'verifier_ok': True,
             'collection_config_key': key, 'v_protocol': report}
    scene = tmp_path/'scene/episodes/example/private'
    scene.mkdir(parents=True)
    for name, value in [('final_score.json', score), ('v_collection_identity.json', identity),
                        ('v_protocol_result.json', report)]:
        _write(scene, name, value)
    (scene/'v_marker_bindings.jsonl').write_text('{"sequence":1,"synthetic":true}\n')
    (scene/'v_intervention').mkdir()
    _write(scene/'v_intervention', 'event.json', {'synthetic': True})
    summary.update(model='synthetic-model', official_task_success=True, v_protocol_complete=True)
    summary['candidate_result'] = _archive_scene_result(tmp_path/'scene', root, summary, binding, profile=profile)
    _write(root, 'summary.json', summary)
    return root


def test_v_identity_is_bound_to_frozen_task_and_archived_model(v_evidence_run):
    result = inspect_run(v_evidence_run)
    assert result['score_available'] is True and result['official_success'] is True


@pytest.mark.parametrize('name,field,value', [
    ('v_protocol_result.json', 'v_protocol_complete', False),
    ('v_collection_identity.json', 'branch', 'D0'),
    ('v_collection_identity.json', 'collection_config_key', 'unknown-v-task'),
    ('v_collection_identity.json', 'task_spec_id', 'another-task'),
    ('v_collection_identity.json', 'runtime_revision', 'a'*40),
    ('task_binding.json', 'collection_config_key', 'another-key'),
    ('task_binding.json', 'task_id', 999),
    ('task_binding.json', 'task_name', 'another-task'),
    ('task_binding.json', 'task_seed', 0),
    ('task_binding.json', 'instruction', 'changed instruction'),
])
def test_v_auxiliary_proof_cannot_be_rehashed_to_override_original_score(v_evidence_run, name, field, value):
    root = v_evidence_run
    summary = json.loads((root/'summary.json').read_text())
    payload = json.loads((root/summary['candidate_result']['evidence'][name]['path']).read_text())
    payload[field] = value
    _rehash_evidence(root, name, value=payload)
    with pytest.raises(ValueError):
        inspect_run(root)


@pytest.mark.parametrize('file,field,value', [
    ('client_manifest.json', 'task_name', 'another-task'),
    ('client_manifest.json', 'instruction_sha256', 'b'*64),
    ('client_manifest.json', 'model', 'another-model'),
    ('run_record.json', 'model', {'model': 'another-model'}),
    ('run_record.json', 'task_identity', {}),
    ('run_record.json', 'source_revisions', {}),
    ('run_record.json', 'profile', 'l0-english-v2'),
    ('summary.json', 'current_scene_episode_id', 'another-episode'),
    ('summary.json', 'v_protocol_complete', False),
    ('summary.json', 'official_task_success', False),
    ('summary.json', 'model', 'another-model'),
])
def test_v_archived_model_and_episode_cannot_be_substituted(v_evidence_run, file, field, value):
    payload = json.loads((v_evidence_run/file).read_text())
    payload[field] = value
    _write(v_evidence_run, file, payload)
    with pytest.raises(ValueError):
        inspect_run(v_evidence_run)


@pytest.mark.parametrize('name', ['v_collection_identity.json', 'v_protocol_result.json'])
def test_v_required_identity_and_protocol_cannot_be_removed_from_manifest(v_evidence_run, name):
    summary = json.loads((v_evidence_run/'summary.json').read_text())
    del summary['candidate_result']['evidence'][name]
    _write(v_evidence_run, 'summary.json', summary)
    with pytest.raises(ValueError, match='lacks'):
        inspect_run(v_evidence_run)


def test_l2_combined_physical_success_does_not_require_p30_qualification(tmp_path):
    root = tmp_path/'l2-run'
    summary = _minimal_trajectory(root)
    scene = tmp_path/'scene/episodes/example/private'
    (scene/'l2_v').mkdir(parents=True)
    report = {'branch': 'D1', 'v_protocol_complete': True, 'checkpoint_violations': []}
    score = {'episode_id': 'example', 'official_q': 1, 'official_success': True, 'verifier_ok': True,
             'task_success': True, 'status': 'PASS', 'native_q_alone_is_success': False,
             'physical_template': {'task_success': True}, 'physical_template_complete': True,
             'v_protocol': report, 'v_protocol_complete': True}
    _write(scene, 'final_score.json', score)
    _write(scene/'l2_v', 'final.json', report)
    _write(scene/'l2_v', 'checkpoint_violations.json', [])
    binding = _live_binding(root, 'l2-collection37', summary=summary)
    summary['candidate_result'] = _archive_scene_result(tmp_path/'scene', root, summary, binding, profile='l2-collection37')
    _write(root, 'summary.json', summary)
    assert inspect_run(root)['official_success'] is True
    _rehash_evidence(root, 'l2_v/checkpoint_violations.json', value=[{'synthetic': 'violation'}])
    with pytest.raises(ValueError, match='checkpoint'):
        inspect_run(root)


def test_v_public_scene_name_cannot_be_replaced_by_native_donor_name(v_evidence_run):
    root = v_evidence_run
    binding = json.loads((root/'private/task_binding.json').read_text())
    native_name = binding['v_identity']['task_spec_id']
    binding['task_name'] = native_name
    _rehash_evidence(root, 'task_binding.json', value=binding)
    manifest = json.loads((root/'client_manifest.json').read_text())
    manifest['task_name'] = native_name
    _write(root, 'client_manifest.json', manifest)
    record = json.loads((root/'run_record.json').read_text())
    record['task_identity']['task_name'] = native_name
    _write(root, 'run_record.json', record)
    with pytest.raises(ValueError, match='frozen public identity'):
        inspect_run(root)


FAMILY_SCORES = {
    'l0-english-v2': {},
    'l1-a30': {},
    'l1-m30-public-v2': {},
    'l1-p-inside5': {'task_success': True, 'physical_qualification': 'PASS'},
    'l1-p-ontop5': {'task_success': True, 'physical_qualification': 'PASS'},
    'l1-p-additions20': {'task_success': True, 'physical_qualification': 'PASS'},
    'l2-collection37': {'task_success': True, 'status': 'PASS'},
}


@pytest.fixture(params=sorted(FAMILY_SCORES))
def bound_run(tmp_path, request):
    root = tmp_path/'bound'
    summary = _minimal_trajectory(root)
    binding = _live_binding(root, request.param, summary=summary)
    scene = tmp_path/'scene/episodes/example/private'
    scene.mkdir(parents=True)
    _write(scene, 'final_score.json', {'episode_id': 'example', 'official_q': 1, 'official_success': True,
                                       'verifier_ok': True, **FAMILY_SCORES[request.param]})
    summary['candidate_result'] = _archive_scene_result(tmp_path/'scene', root, summary, binding,
                                                        profile=request.param)
    _write(root, 'summary.json', summary)
    return root


def test_every_family_binds_its_score_to_run_task_and_model(bound_run):
    result = inspect_run(bound_run)
    assert result['status'] == 'PASS' and result['official_success'] is True


@pytest.mark.parametrize('file,field,value', [
    ('summary.json', 'current_scene_episode_id', 'another-attempt'),
    ('summary.json', 'model', 'another-model'),
    ('run_record.json', 'task_identity', {'task_id': 999, 'task_name': 'another-task', 'task_split': 'x',
                                          'task_instance_id': 'y', 'task_seed': 1}),
    ('run_record.json', 'source_revisions', {'bench': '0'*40, 'runtime': '1'*40}),
    ('run_record.json', 'profile', None),
    ('run_record.json', 'model', {'model': 'another-model'}),
    ('client_manifest.json', 'task_seed', 12345),
    ('client_manifest.json', 'instruction_sha256', 'b'*64),
    ('run_record.json', 'declared_budget', {}),
    ('run_record.json', 'maximum_action_decisions', 3000),
    ('summary.json', 'maximum_model_calls', 30000),
    ('client_manifest.json', 'maximum_action_decisions', 3000),
    ('client_manifest.json', 'physical_history_max', 20),
    ('summary.json', 'physical_history_max', 20),
    ('run_record.json', 'history_frames', 20),
    ('client_manifest.json', 'experiment_id', 'another-experiment'),
    ('client_manifest.json', 'source_sha256', {'file': 'not-a-hash'}),
])
def test_non_v_score_cannot_be_rebound_to_another_attempt_task_or_source(bound_run, file, field, value):
    payload = json.loads((bound_run/file).read_text())
    if value is None:
        del payload[field]
    else:
        payload[field] = value
    _write(bound_run, file, payload)
    with pytest.raises(ValueError):
        inspect_run(bound_run)
    assert summarize_runs([bound_run])['invalid_runs'] == 1


@pytest.mark.parametrize('field,value', [('task_name', 'another-task'), ('task_seed', 12345),
                                         ('task_id', 999), ('instruction', 'changed instruction'),
                                         ('budget', {}), ('maximum_model_calls', 30000)])
def test_non_v_task_binding_cannot_be_rehashed_to_another_task(bound_run, field, value):
    from roboweave.tasks import list_tasks
    binding = json.loads((bound_run/'private/task_binding.json').read_text())
    record = json.loads((bound_run/'run_record.json').read_text())
    if field == 'task_name':
        value = next(task['task_id'] for task in list_tasks()
                     if task['profile'] == record['profile'] and task['task_id'] != binding['task_name'])
    binding[field] = value
    _rehash_evidence(bound_run, 'task_binding.json', value=binding)
    with pytest.raises(ValueError):
        inspect_run(bound_run)


def test_candidate_failure_summary_requires_its_candidate_error(tmp_path):
    root = tmp_path/'episode-failed'
    _minimal_trajectory(root)
    _write(root, 'summary.json', {'status': 'EPISODE_FAILED', 'stage': 'model_episode'})
    with pytest.raises(ValueError, match='prompt provenance'):
        inspect_run(root)
    _write(root, 'candidate_error.json', {'status': 'EPISODE_FAILED', 'stage': 'model_episode',
                                          'error_type': 'RuntimeError'})
    result = inspect_run(root)
    assert result['status'] == 'INCOMPLETE' and result['completeness'] == 'candidate_failure'
    # A native summary keeps its own prompt hash requirement even with a candidate error.
    _write(root, 'summary.json', {'status': 'DONE', 'prompt_sha256': '0'*64})
    with pytest.raises(ValueError, match='prompt provenance'):
        inspect_run(root)


def _wall_timeout_run(root, *, signal=True, experiment=None):
    _minimal_trajectory(root)
    _write(root, 'run_record.json', {'mode': 'live', 'profile': 'l2-collection37', 'experiment_id': root.name})
    summary = {'status': 'PAUSED_INTERRUPTED', 'prompt_sha256': json.loads(
        (root/'client_manifest.json').read_text())['prompt_provenance']['rendered_prompt_sha256'],
        'current_scene_episode_id': 'example', 'model': 'synthetic-model',
        'candidate_result': {'status': 'unavailable', 'episode_id': 'example', 'official_success': None,
                             'official_q': None, 'evidence': {}}}
    _write(root, 'summary.json', summary)
    evidence = {'policy': 'model_wall_timeout/v1', 'reason': 'model_wall_timeout', 'observed_phase': 'WAITING_PROVIDER',
                'threshold_elapsed_seconds': 7200, 'observed_elapsed_seconds': 7201.5,
                'experiment_id': experiment or root.name}
    if signal:
        evidence['signal'] = 'SIGINT'
    (root/'private').mkdir(exist_ok=True)
    _write(root, 'private/model_wall_timeout.json', evidence)
    return root


def test_signalled_l2_wall_timeout_is_reported_as_a_historical_failure(rich_evidence_run, tmp_path):
    timed_out = _wall_timeout_run(tmp_path/'timed-out')
    requested = _wall_timeout_run(tmp_path/'requested-only', signal=False)
    result = inspect_run(timed_out)
    assert result['status'] == 'PASS' and result['score_available'] is False
    assert result['wall_timeout']['signal_sent'] is True
    assert result['wall_timeout']['observed_phase'] == 'WAITING_PROVIDER'
    assert inspect_run(requested)['wall_timeout']['signal_sent'] is False
    report = summarize_runs([rich_evidence_run, timed_out, requested])
    assert report['scored_live_runs'] == 1 and report['successful_live_runs'] == 1
    assert report['success_rate_scored_live'] == 1
    assert report['wall_timeout_failed_live_runs'] == 1
    assert report['success_rate_scored_or_wall_timeout_live'] == 0.5
    assert report['success_rate_all_live_attempts'] == pytest.approx(1 / 3)


@pytest.mark.parametrize('change', ['experiment', 'profile', 'signal', 'policy',
                                  'observed_phase', 'threshold_elapsed_seconds', 'observed_elapsed_seconds'])
def test_wall_timeout_evidence_must_belong_to_its_owned_l2_run(tmp_path, change):
    root = _wall_timeout_run(tmp_path/'timed-out', experiment='another-run' if change == 'experiment' else None)
    if change == 'profile':
        record = json.loads((root/'run_record.json').read_text())
        _write(root, 'run_record.json', {**record, 'profile': 'l1-a30'})
    elif change != 'experiment':
        evidence = json.loads((root/'private/model_wall_timeout.json').read_text())
        evidence[change] = {'signal': 'SIGKILL', 'policy': 'model_wall_timeout/v0', 'observed_phase': 'ACTION_RUNNING',
                            'threshold_elapsed_seconds': -1, 'observed_elapsed_seconds': 1}[change]
        _write(root, 'private/model_wall_timeout.json', evidence)
    with pytest.raises(ValueError, match='wall-timeout'):
        inspect_run(root)
    assert summarize_runs([root])['invalid_runs'] == 1


@pytest.mark.parametrize('raw', [
    b'{"episode_id":"example","official_q":1,"official_success":false,"official_success":true,"verifier_ok":true}',
    b'{"episode_id":"example","official_q":1,"official_success":true,"verifier_ok":true,"extra":NaN}',
    b'{"episode_id":"example","official_q":1,"official_success":true,"verifier_ok":true,"extra":1e1000}',
])
def test_archiver_rejects_ambiguous_or_nonfinite_score_before_claiming_success(tmp_path, raw):
    scene = tmp_path/'scene/episodes/example/private'
    scene.mkdir(parents=True)
    (scene/'final_score.json').write_bytes(raw)
    with pytest.raises(ValueError):
        _archive_scene_result(tmp_path/'scene', tmp_path/'run', {'current_scene_episode_id': 'example'}, {})
    assert not (tmp_path/'run/private/scene_final_score.json').exists()


@pytest.mark.parametrize('location', ['private', 'v_scene', 'source_parent'])
def test_archiver_rejects_redirected_private_paths(tmp_path, location):
    root = tmp_path/'run'
    root.mkdir()
    outside = tmp_path/'outside'
    outside.mkdir()
    scene = tmp_path/'scene/episodes/example/private'
    scene.mkdir(parents=True)
    if location == 'private':
        (root/'private').symlink_to(outside, target_is_directory=True)
    elif location == 'v_scene':
        (root/'private').mkdir()
        (root/'private/v_scene').symlink_to(outside, target_is_directory=True)
        _write(scene, 'v_collection_identity.json', {})
    else:
        scene.rmdir()
        scene.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='escaped'):
        _archive_scene_result(tmp_path/'scene', root, {'current_scene_episode_id': 'example'}, {}, profile='l1-v30')
    assert list(outside.iterdir()) == []


def test_archiver_never_replaces_existing_task_binding(tmp_path):
    root = tmp_path/'run'
    result = _archive_scene_result(tmp_path/'scene', root, {}, {'seed': 1})
    binding = root/result['evidence']['task_binding.json']['path']
    original = binding.read_bytes()
    with pytest.raises(ValueError, match='Previously archived'):
        _archive_scene_result(tmp_path/'scene', root, {}, {'seed': 2})
    assert binding.read_bytes() == original
    assert _archive_scene_result(tmp_path/'scene', root, {}, {'seed': 1}) == result


@pytest.mark.parametrize('patch', [
    {'official_success': False}, {'physical_template_complete': True}, {'v_protocol_complete': True},
])
def test_l2_success_cannot_drop_native_completion_evidence(patch):
    score = {'episode_id': 'example', 'official_q': 1, 'official_success': True,
             'verifier_ok': True, 'task_success': True, 'status': 'PASS', **patch}
    with pytest.raises(ValueError, match='L2'):
        recorded_score_result(score, 'example', profile='l2-collection37')


@pytest.mark.parametrize('location', ['root', 'image', 'evidence'])
def test_symlink_loop_is_an_invalid_run_without_aborting_summary(tmp_path, location):
    root = tmp_path/'loop'
    if location == 'root':
        root.symlink_to(root.name, target_is_directory=True)
    else:
        summary = _minimal_trajectory(root)
        if location == 'image':
            (root/'loop.png').symlink_to('loop.png')
            _write(root, 'conversation_chatml.json', {'schema_version': 'behavior-hl-action20-chatml-v3',
                'messages': [{'role': 'user', 'content': [{'type': 'image', 'image': 'loop.png'}]}]})
        else:
            summary['candidate_result'] = _archive_scene_result(tmp_path/'scene', root, {}, {})
            binding = root/'private/task_binding.json'
            binding.unlink()
            binding.symlink_to(binding.name)
        _write(root, 'summary.json', summary)
    with pytest.raises(ValueError):
        inspect_run(root)
    good = tmp_path/'good'
    good.mkdir()
    _write(good, 'summary.json', {'status': 'INITIALIZATION_FAILED'})
    _write(good, 'run_record.json', {'mode': 'live'})
    report = summarize_runs([root, good])
    assert [run['status'] for run in report['runs']] == ['INVALID', 'INCOMPLETE']
    assert report['total_live_attempts'] == (1 if location == 'root' else 2)


@pytest.mark.parametrize('profile,extra', [
    ('l1-p-inside5', {'physical_qualification': 'PASS', 'native_q_alone_is_success': False}),
    ('l1-p-ontop5', {'physical_qualification': 'PASS', 'native_q_alone_is_success': False}),
    ('l1-p-additions20', {'physical_qualification': 'PASS', 'native_q_alone_is_success': False}),
    ('l1-m30-public-v2', {'public_process_report': {
        'operational': True, 'public_process_satisfied': True, 'errors': [], 'violations': []}}),
])
def test_physical_and_process_scores_keep_native_explicit_task_success_gate(tmp_path, profile, extra):
    import runpy
    from roboweave.profiles import source_roots
    native = runpy.run_path(str(source_roots(profile)['bench_root']/'lab/physical_model_results.py'))
    scene = tmp_path/'scene/episodes/example/private'
    scene.mkdir(parents=True)
    for index, task_success in enumerate([None, False, True]):
        score = {'episode_id': 'example', 'official_q': 1, 'official_success': True,
                 'verifier_ok': True, **extra}
        if task_success is not None:
            score['task_success'] = task_success
        _write(scene, 'final_score.json', score)
        original = native['collect_physical_result']({}, scene_run_dir=tmp_path/'scene',
            episode_id='example', output_dir=tmp_path/f'native-{index}')
        result = recorded_score_result(score, 'example', profile=profile)
        assert result['official_success'] is original['official_success'] is (task_success is True)


def test_archived_binding_is_bounded_and_private(tmp_path, monkeypatch):
    from roboweave import trajectory
    root = tmp_path/'run'
    (root/'private').mkdir(parents=True, mode=0o777)
    (root/'private').chmod(0o777)
    _archive_scene_result(tmp_path/'scene', root, {}, {})
    assert (root/'private').stat().st_mode & 0o777 == 0o700
    assert (root/'private/task_binding.json').stat().st_mode & 0o777 == 0o600
    monkeypatch.setattr(trajectory, 'MAX_SCORE_BYTES', 10)
    with pytest.raises(ValueError, match='Oversized'):
        _archive_scene_result(tmp_path/'scene', tmp_path/'oversized', {}, {'value': 'too large'})
    assert not (tmp_path/'oversized/private/task_binding.json').exists()


def test_unknown_legacy_mode_still_prefers_its_verified_score_to_summary(tmp_path):
    root = tmp_path/'legacy'
    summary = _minimal_trajectory(root)
    scene = tmp_path/'scene/episodes/example/private'
    scene.mkdir(parents=True)
    _write(scene, 'final_score.json', {'episode_id': 'example', 'official_q': 1,
                                     'official_success': False, 'verifier_ok': True})
    summary.update(official_task_success=True, q_score=0)
    summary['candidate_result'] = _archive_scene_result(tmp_path/'scene', root, summary, {})
    _write(root, 'summary.json', summary)
    (root/'run_record.json').unlink()
    result = inspect_run(root)
    assert result['mode'] == 'unknown_legacy' and result['score_available'] is True
    assert result['official_success'] is False and result['official_q'] == 1
    assert summarize_runs([root])['total_live_attempts'] == 0


def test_v_native_invalid_classification_is_retained_with_raw_score(v_evidence_run):
    root = v_evidence_run
    score = json.loads((root/'private/scene_final_score.json').read_text())
    score['v_protocol']['v_protocol_complete'] = False
    _rehash_evidence(root, 'final_score.json', value=score)
    _rehash_evidence(root, 'v_protocol_result.json', value=score['v_protocol'])
    summary = json.loads((root/'summary.json').read_text())
    summary.update(status='FAIL_V_PROTOCOL', official_task_success=False, v_protocol_complete=False,
                   formal_score_eligible=False, outcome_class='INVALID_FORMAL_RUN')
    summary['candidate_result'].update(recorded_score_result(score, 'example', profile='l1-v30'))
    _write(root, 'summary.json', summary)
    result = inspect_run(root)
    assert result['status'] == 'PASS' and result['score_available'] is True
    assert result['outcome_class'] == 'INVALID_FORMAL_RUN' and result['formal_score_eligible'] is False
    assert result['official_success'] is False
    report = summarize_runs([root])
    assert report['total_live_attempts'] == report['scored_live_runs'] == 1
    assert report['success_rate_scored_live'] == 0


@pytest.fixture(params=[*FAMILY_SCORES, 'l1-v30'])
def aborted_run(tmp_path, request):
    from roboweave.profiles import source_revisions
    from roboweave.tasks import get_task
    root = tmp_path/'aborted'
    summary = _minimal_trajectory(root)
    binding = _live_binding(root, request.param, summary=summary)
    scene = tmp_path/'scene/episodes/example/private'
    scene.mkdir(parents=True)
    score = {'episode_id': 'example', 'official_q': None, 'official_success': False, 'verifier_ok': False,
             'integrity_failure': 'synthetic_abort', 'termination_reason': 'execution_integrity_failure'}
    if request.param == 'l2-collection37':
        score.update(task_success=False, failure_class='EXECUTION_INTEGRITY_FAILURE')
    if request.param == 'l1-v30':
        task = get_task(binding['task_name'], 'L1-V')
        revision = source_revisions(request.param)['runtime']
        identity = dict(collection_config_key=task['task_id'], task_spec_id=task['runtime_instruction_id'],
            package_id='synthetic-v-package', branch='D1', runtime_revision=revision,
            historical_oracle_runtime_revision=revision, source_oracle_success=True,
            runtime_validation_only=False, v_package_sha256='a'*64)
        binding.update(collection_config_key=task['task_id'], v_identity=identity)
        _write(scene, 'v_collection_identity.json', identity)
        _write(scene, 'v_protocol_abort.json', {'aborted': True, 'v_protocol_complete': False,
            'reason': 'synthetic_environment_error', 'same_scene_recovery_allowed': False})
        summary.update(v_protocol_complete=False, official_task_success=False)
    _write(scene, 'final_score.json', score)
    _write(scene, 'aborted_after_environment_error.json', {'reason': 'synthetic_environment_error',
        'public_actions_executed': 0, 'symbolic_collision_filter_cleanup': {'active': False},
        'same_scene_reset_required_before_retry': True, 'recovery_admitted': False})
    _write(scene, 'task_process_result.json', {'operational': False, 'public_process_satisfied': False,
        'errors': ['synthetic_environment_error'], 'violations': [], 'execution_events': 0, 'existing_frame_references': 0})
    summary['candidate_result'] = _archive_scene_result(tmp_path/'scene', root, summary, binding, profile=request.param)
    _write(root, 'summary.json', summary)
    return root


def test_native_abort_schema_keeps_each_profiles_negative_score(aborted_run):
    result = inspect_run(aborted_run)
    assert result['status'] == 'PASS' and result['score_available'] is True
    assert result['official_success'] is False and result['official_q'] is None
    report = summarize_runs([aborted_run])
    assert report['total_live_attempts'] == report['scored_live_runs'] == 1
    assert report['successful_live_runs'] == report['invalid_runs'] == 0


def test_native_abort_cannot_drop_its_recovery_record(aborted_run):
    summary = json.loads((aborted_run/'summary.json').read_text())
    del summary['candidate_result']['evidence']['aborted_after_environment_error.json']
    _write(aborted_run, 'summary.json', summary)
    with pytest.raises(ValueError, match='recovery evidence'):
        inspect_run(aborted_run)


def test_l2_explicit_environment_exception_can_omit_status():
    score = dict(episode_id='example', official_q=None, official_success=False, verifier_ok=False,
                 task_success=False, failure_class='EXECUTION_EXCEPTION', integrity_failure=None,
                 termination_reason='synthetic_environment_error')
    assert recorded_score_result(score, 'example', profile='l2-collection37')['official_success'] is False
    for changed in ({'task_success': True}, {'official_success': True}, {'verifier_ok': True},
                    {'official_q': 1}, {'failure_class': 'unknown'}, {'status': 'PASS'}):
        with pytest.raises(ValueError):
            recorded_score_result({**score, **changed}, 'example', profile='l2-collection37')
