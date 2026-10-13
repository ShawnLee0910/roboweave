import json
import pytest

from esi_benchmark.models import create_provider, validate_model_config, run_episode
from esi_benchmark.bootstrap import activate


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


def test_private_score_archived_without_inventing_success(tmp_path):
    from esi_benchmark.models import _archive_scene_result, recorded_score_result
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
