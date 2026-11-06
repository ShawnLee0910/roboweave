"""Output budgets survive registration, launch construction and API serialization."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src/roboweave/_vendor/bench")))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from roboweave.models import validate_model_config
from model_client import OpenAICompatibleProvider
from native_tool_preflight import run_native_tool_preflight, PROBE_STEPS
from test_native_tool_preflight import _completion, _prompt_revision


@pytest.mark.parametrize('budget', [1, 4096, 32768, 131072])
def test_budget_reaches_request(tmp_path, monkeypatch, budget):
    model = validate_model_config({'endpoint': 'http://127.0.0.1:1/v1', 'model': 'test-model',
                                   'maximum_output_tokens': budget})
    assert model['maximum_output_tokens'] == budget
    provider = OpenAICompatibleProvider(model='test-model', base_url='http://127.0.0.1:1/v1',
                                       timeout_seconds=30, api_key_file=None,
                                       maximum_output_tokens=budget)
    sent = []
    monkeypatch.setattr(provider, '_post', lambda request: (sent.append(request) or _completion('wait', None, 1)))
    provider.complete({'model': 'test-model', 'messages': [], 'tools': []},
                      call_dir=tmp_path / 'call', seed=1)
    assert sent[0]['max_tokens'] == budget
    probes = []
    def post(request):
        step = PROBE_STEPS[len(probes)]
        probes.append(request)
        return _completion(step.expected_action, step.expected_target, len(probes),
                           step.expected_target_description)
    assert run_native_tool_preflight(post=post, model='test-model',
        prompt_revision=_prompt_revision(), max_tokens=budget)['status'] == 'PASS'
    assert all(x['max_tokens'] == budget for x in probes)


@pytest.mark.parametrize('budget', [True, False, 0, -1, 1.5, '32768'])
def test_invalid_budget_rejected(tmp_path, budget):
    with pytest.raises(ValueError):
        validate_model_config({'endpoint': 'http://127.0.0.1:1/v1', 'model': 'test-model',
                               'maximum_output_tokens': budget})
    with pytest.raises(ValueError):
        OpenAICompatibleProvider(model='test-model', base_url='http://127.0.0.1:1/v1',
                                 timeout_seconds=30, api_key_file=None,
                                 maximum_output_tokens=budget)
    with pytest.raises(ValueError):
        run_native_tool_preflight(post=lambda _: {}, model='test-model',
                                 prompt_revision=_prompt_revision(), max_tokens=budget)
