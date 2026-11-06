import json
import pytest

from roboweave.models import validate_model_config, run_episode
from roboweave._native.models import create_provider
from roboweave._native.bootstrap import activate


def test_explicit_key_selection_never_falls_back(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'unrelated-test-key')
    config = {'endpoint': 'https://provider.example/v1', 'model': 'model', 'api_key_env': 'ESI_TEST_KEY'}
    assert validate_model_config(config)['api'] == 'chat_completions'
    with pytest.raises(ValueError, match='unavailable'):
        create_provider(config)
    monkeypatch.setenv('ESI_TEST_KEY', 'selected-test-key')
    provider = create_provider(config)
    assert provider._read_api_key() == 'selected-test-key'
    assert 'selected-test-key' not in json.dumps(validate_model_config(config))


@pytest.mark.parametrize('endpoint', ['http://public.example/v1', 'https://user:secret@example.com', 'https://example.com?token=secret', 'https:///v1'])
def test_unsafe_endpoints_rejected(endpoint):
    with pytest.raises(ValueError):
        validate_model_config({'endpoint': endpoint, 'model': 'model'})


def test_local_anonymous_provider_needs_no_key():
    assert create_provider({'endpoint': 'http://127.0.0.1:8000/v1', 'model': 'local'}).requires_api_key is False


def test_remote_provider_requires_explicit_key_name_without_reading_credentials(monkeypatch):
    monkeypatch.delenv('ESI_TEST_KEY', raising=False)
    config = {'endpoint': 'https://provider.example/v1', 'model': 'model'}
    with pytest.raises(ValueError, match='model.api_key_env'):
        validate_model_config(config)
    assert validate_model_config({**config, 'api_key_env': 'ESI_TEST_KEY'})['api_key_env'] == 'ESI_TEST_KEY'


def test_model_record_matches_frozen_provider_normalization():
    config = {'endpoint': 'http://127.0.0.1:8000/v1', 'model': ' local '}
    assert validate_model_config(config)['model'] == create_provider(config).model == 'local'


def test_native_adapters_validate_frozen_parameters():
    base = {'endpoint': 'http://localhost:8000/v1', 'model': 'model'}
    with pytest.raises(ValueError):
        validate_model_config({**base, 'api': 'responses'})
    assert validate_model_config({**base, 'api': 'responses', 'reasoning_effort': 'medium',
                                  'reasoning_context': 'all_turns', 'reasoning_mode': 'standard',
                                  'text_verbosity': 'low'})['api'] == 'responses'
    assert validate_model_config({**base, 'api': 'anthropic_messages', 'reasoning_effort': 'high'})['api'] == 'anthropic_messages'


def test_scene_registry_containment(tmp_path):
    activate()
    from scene_protocol import SceneClient
    with pytest.raises(ValueError, match='escaped'):
        SceneClient(ready_path=tmp_path / 'elsewhere/ready.json', runtime_root=tmp_path / 'owned')


def test_episode_requires_frozen_budget_before_contacting_scene(tmp_path):
    task = {'task_id': 0, 'task_name': 'test', 'task_split': 'train', 'task_instance_id': '0',
            'task_seed': 0, 'instruction': 'test'}
    with pytest.raises(ValueError, match='maximum_action_decisions'):
        run_episode(model_config={}, task=task, ready_path=tmp_path/'runtime/scene/ready.json',
                    runtime_root=tmp_path/'runtime', output_dir=tmp_path/'output', experiment_id='test')
    assert not (tmp_path/'output').exists()


def test_episode_checks_catalog_binding_before_provider_or_scene(tmp_path, monkeypatch):
    from roboweave._native import models
    from roboweave.tasks import list_tasks
    public = list_tasks('L0')[0]
    task = {'task_id': public['source']['row_index'], 'task_name': public['runtime_instruction_id'],
            'task_split': 'train', 'task_instance_id': '0', 'task_seed': 0,
            'instruction': 'tampered instruction', 'budget': dict(public['budget']), **public['budget'],
            'instruction_provenance': {'profile': public['profile'], 'instruction_sha256': public['instruction_sha256']}}
    monkeypatch.setattr(models, 'create_provider', lambda *args: pytest.fail('contacted provider for changed task'))
    with pytest.raises(ValueError, match='instruction'):
        models.run_episode(model_config={}, task=task, ready_path=tmp_path/'runtime/scene/ready.json',
                           runtime_root=tmp_path/'runtime', output_dir=tmp_path/'output', experiment_id='test')
    assert not (tmp_path/'output').exists()


def test_private_score_archived_without_inventing_success(tmp_path):
    from roboweave.models import _archive_scene_result, recorded_score_result
    scene = tmp_path/'scene'
    output = tmp_path/'run'
    private = scene/'episodes/example/private'
    private.mkdir(parents=True)
    score = {'episode_id': 'example', 'official_q': 1.0, 'official_success': False,
             'verifier_ok': True, 'integrity_failure': None, 'termination_reason': 'done'}
    raw = (json.dumps(score) + '\n').encode()
    (private/'final_score.json').write_bytes(raw)
    result = _archive_scene_result(scene, output, {'current_scene_episode_id': 'example'}, {'seed': 0})
    assert result['official_q'] == 1.0 and result['official_success'] is False
    assert (output/'private/scene_final_score.json').read_bytes() == raw
    assert recorded_score_result({**score, 'official_success': True, 'integrity_failure': 'broken'}, 'example')['official_success'] is False
    assert recorded_score_result({**score, 'official_success': True, 'task_success': False}, 'example')['official_success'] is False
    assert recorded_score_result({**score, 'official_success': True, 'native_q_alone_is_success': False,
                                  'physical_qualification': 'UNKNOWN'}, 'example')['official_success'] is False


def test_reserved_output_rejects_previous_run_artifacts(tmp_path):
    output = tmp_path/'output'
    (output/'private/inputs').mkdir(parents=True)
    (output/'summary.json').write_text('{}')
    task = {'task_id': 0, 'task_name': 'test', 'task_split': 'train', 'task_instance_id': '0',
            'task_seed': 0, 'instruction': 'test', 'maximum_action_decisions': 33, 'maximum_model_calls': 330}
    with pytest.raises(FileExistsError, match='private/inputs'):
        run_episode(model_config={}, task=task, ready_path=tmp_path/'runtime/scene/ready.json',
                    runtime_root=tmp_path/'runtime', output_dir=output, experiment_id='test', output_reserved=True)


def test_profile_specific_scores_preserve_physical_and_v_gates(monkeypatch):
    from roboweave.models import recorded_score_result
    from roboweave import tasks
    score = {'episode_id': 'example', 'official_q': 1.0, 'official_success': True,
             'verifier_ok': True, 'task_success': True}
    monkeypatch.setattr(tasks, 'profile_for', lambda profile: {'family': profile})
    assert recorded_score_result(score, 'example', profile='L1-P')['official_success'] is False
    assert recorded_score_result({**score, 'physical_qualification': 'PASS'}, 'example', profile='L1-P')['official_success'] is True
    with pytest.raises(ValueError, match='protocol'):
        recorded_score_result(score, 'example', profile='L1-V')
    incomplete = {**score, 'v_protocol': {'v_protocol_complete': False, 'operational': True}}
    assert recorded_score_result(incomplete, 'example', profile='L1-V')['official_success'] is False


def test_l2_uses_its_combined_score_without_p30_qualification(monkeypatch):
    from roboweave.models import recorded_score_result
    from roboweave import tasks
    monkeypatch.setattr(tasks, 'profile_for', lambda profile: {'family': 'L2'})
    score = {'episode_id': 'example', 'official_q': 1, 'official_success': True,
             'verifier_ok': True, 'task_success': True, 'status': 'PASS',
             'physical_template': {'task_success': True}, 'physical_template_complete': True,
             'native_q_alone_is_success': False,
             'v_protocol': {'v_protocol_complete': True}, 'v_protocol_complete': True}
    result = recorded_score_result(score, 'example', profile='l2-example')
    assert result['official_success'] is True and result['physical_qualification'] is None
    assert result['physical_template_complete'] is True and result['v_protocol_complete'] is True
    with pytest.raises(ValueError, match='disagree'):
        recorded_score_result({**score, 'physical_template_complete': False}, 'example', profile='l2-example')
    failed = {**score, 'task_success': False, 'status': 'FAIL'}
    assert recorded_score_result(failed, 'example', profile='l2-example')['official_success'] is False


def test_v_archive_preserves_bound_jsonl_and_rejects_summary_disagreement(tmp_path):
    from roboweave.models import _archive_scene_result
    scene = tmp_path/'scene'
    source = scene/'episodes/example/private'
    source.mkdir(parents=True)
    report = {'v_protocol_complete': True, 'operational': True}
    score = {'episode_id': 'example', 'official_q': 1, 'official_success': True,
             'verifier_ok': True, 'collection_config_key': 'v-example', 'v_protocol': report}
    identity = {'branch': 'D1', 'collection_config_key': 'v-example', 'package_id': 'v-package'}
    for name, value in [('final_score.json', score), ('v_collection_identity.json', identity),
                        ('v_protocol_result.json', report)]:
        (source/name).write_text(json.dumps(value))
    marker_raw = b'{"id":"private-marker"}\n'
    (source/'v_marker_bindings.jsonl').write_bytes(marker_raw)
    (source/'v_intervention').mkdir()
    (source/'v_intervention/events.jsonl').write_text('{"event":"recorded"}\n')
    summary = {'current_scene_episode_id': 'example', 'v_protocol_complete': True, 'official_task_success': True}
    result = _archive_scene_result(scene, tmp_path/'run', summary,
                                   {'collection_config_key': 'v-example'}, profile='l1-v30')
    assert result['official_success'] is True
    assert 'v_intervention/events.jsonl' in result['evidence']
    assert (tmp_path/'run/private/v_scene/v_marker_bindings.jsonl').read_bytes() == marker_raw
    with pytest.raises(ValueError, match='disagrees'):
        _archive_scene_result(scene, tmp_path/'failed', {**summary, 'v_protocol_complete': False},
                              {'collection_config_key': 'v-example'}, profile='l1-v30')


def test_l2_archive_keeps_native_process_and_intervention_records(tmp_path, monkeypatch):
    from roboweave.models import _archive_scene_result
    from roboweave import tasks
    monkeypatch.setattr(tasks, 'profile_for', lambda profile: {'family': 'L2'})
    private = tmp_path/'scene/episodes/example/private'
    (private/'l2_v').mkdir(parents=True)
    report = {'v_protocol_complete': True}
    score = {'episode_id': 'example', 'official_q': 1, 'official_success': True,
             'verifier_ok': True, 'task_success': True, 'status': 'PASS',
             'v_protocol': report, 'v_protocol_complete': True}
    (private/'final_score.json').write_text(json.dumps(score))
    (private/'l2_v/final.json').write_text(json.dumps(report))
    (private/'l2_v/checkpoint_violations.json').write_text('[]\n')
    process = b'{"actor":"AGENT","event_id":1}\n'
    (private/'task_process_events.jsonl').write_bytes(process)
    result = _archive_scene_result(tmp_path/'scene', tmp_path/'run',
        {'current_scene_episode_id': 'example'}, {}, profile='l2-example')
    assert result['official_success'] is True
    assert set(result['evidence']) == {'final_score.json', 'task_binding.json', 'task_process_events.jsonl',
                                      'l2_v/final.json', 'l2_v/checkpoint_violations.json'}
    assert (tmp_path/'run/private/task_process_events.jsonl').read_bytes() == process
    assert (tmp_path/'run/private/l2_v/checkpoint_violations.json').read_text() == '[]\n'


@pytest.mark.parametrize('profile,watchdog', [('l0-english-v2', False), ('l1-a30', False), ('l1-m30-public-v2', False),
                                              ('l1-v30', False), ('l1-p-inside5', False), ('l2-collection37', True)])
def test_every_profile_keeps_frozen_bench_pause_semantics(tmp_path, monkeypatch, profile, watchdog):
    from types import SimpleNamespace
    from roboweave._native import models
    from roboweave import tasks
    import model_client
    import scene_protocol
    captured = {}
    monkeypatch.setattr(models, 'activate', lambda profile: None)
    monkeypatch.setattr(tasks, 'validate_model_task', lambda profile, task: task)
    monkeypatch.setattr(models, 'create_provider', lambda config: SimpleNamespace())
    monkeypatch.setattr(models, 'validate_model_config', lambda config: dict(config))
    monkeypatch.setattr(models, '_archive_scene_result', lambda *args, **kwargs: {'status': 'unavailable'})
    monkeypatch.setattr(scene_protocol, 'SceneClient', lambda **kwargs: SimpleNamespace(scene_run_dir=tmp_path/'scene'))
    def episode(**kwargs):
        captured.update(kwargs)
        return {'status': 'PAUSED_INTERRUPTED'}
    monkeypatch.setattr(model_client, 'run_model_episode', episode)
    task = {'task_id': 0, 'task_name': 'test', 'task_split': 'train', 'task_instance_id': '0',
            'task_seed': 0, 'instruction': 'test', 'maximum_action_decisions': 300, 'maximum_model_calls': 3810,
            'budget': {'wall_clock_seconds': 7200}}
    result = models.run_episode(model_config={'model': 'fixture'}, task=task,
        ready_path=tmp_path/'scene/ready.json', runtime_root=tmp_path/'scene', output_dir=tmp_path/'run',
        experiment_id='test', profile=profile)
    assert captured['pause_on_client_failure'] is True
    assert result['summary']['status'] == 'PAUSED_INTERRUPTED'
    record = json.loads((tmp_path/'run/run_record.json').read_text())
    assert ('safe_point_watchdog' in record['wall_clock_enforcement']) is watchdog


@pytest.mark.parametrize('profile', ['l0-english-v2', 'l1-a30', 'l1-m30-public-v2',
                                   'l1-v30', 'l1-p-inside5', 'l2-collection37'])
def test_frozen_provider_failures_redact_credentials_without_changing_raw_evidence(tmp_path, profile):
    from pathlib import Path
    import subprocess
    import sys
    probe = Path(__file__).with_name('provider_redaction_probe.py')
    result = subprocess.run([sys.executable, '-I', '-B', str(probe), profile, str(tmp_path),
                             str(probe.parents[1] / 'src')], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report['status'] == 'PASS' and report['network_used'] is False
