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
        'source_sha256': {'synthetic_fixture': digest}}))
    (root/'run_record.json').write_text(json.dumps({'mode': 'live'}))
    return {'status': 'DONE', 'prompt_sha256': digest, 'current_scene_episode_id': 'example'}


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
             'public_process_report': process}
    for name, value in [('final_score.json', score), ('task_process_result.json', process),
                        ('physical_initial_binding.json', {'synthetic': True})]:
        _write(scene, name, value)
    (scene/'task_process_events.jsonl').write_text('{"event_id":1,"synthetic":true}\n')
    (scene/'task_evidence_index.jsonl').write_text('{"frame_index":1,"synthetic":true}\n')
    summary['candidate_result'] = _archive_scene_result(tmp_path/'scene', root, summary, {})
    _write(root, 'summary.json', summary)
    return root


def test_all_archived_evidence_is_checked_without_replaying_process(rich_evidence_run):
    assert inspect_run(rich_evidence_run)['score_available'] is True
    summary = json.loads((rich_evidence_run/'summary.json').read_text())
    assert summary['candidate_result']['evidence_schema_version'] == 'esi-benchmark/result-evidence-v2'
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
    from esi_benchmark import trajectory
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
    from esi_benchmark.assets import asset_manifest
    from esi_benchmark.profiles import source_revisions
    from esi_benchmark.tasks import list_tasks
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
    binding = dict(task_id=task['source']['row_index'], task_name=task['task_id'],
        task_split='synthetic', task_instance_id='0',
        task_seed=asset_manifest(profile)['inputs_by_task'][key]['seed'],
        instruction=task['instruction_en'], collection_config_key=key, v_identity=identity)
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
    fields = ('task_id', 'task_name', 'task_split', 'task_instance_id', 'task_seed')
    manifest = json.loads((root/'client_manifest.json').read_text())
    manifest.update({key: binding[key] for key in fields})
    manifest.update(model='synthetic-model', instruction_sha256=task['instruction_sha256'])
    _write(root, 'client_manifest.json', manifest)
    _write(root, 'run_record.json', dict(mode='live', profile=profile, source_revisions=revisions,
        task_identity={key: binding[key] for key in fields}, model={'model': 'synthetic-model'}))
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
    summary['candidate_result'] = _archive_scene_result(tmp_path/'scene', root, summary, {}, profile='l2-collection37')
    _write(root, 'summary.json', summary)
    _write(root, 'run_record.json', {'mode': 'live', 'profile': 'l2-collection37'})
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
