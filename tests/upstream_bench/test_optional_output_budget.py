import json
import sys
from pathlib import Path
import pytest

sys.path[:0]=[str(Path(__file__).parents[1]),str(Path(__file__).parent)]
from esi_benchmark.models import validate_model_config
import model_client as m
from test_responses_adapter import _request, _opaque_response_bundle


@pytest.mark.parametrize('api',['chat_completions','responses'])
@pytest.mark.parametrize('budget',[None,1,16384])
def test_wire_budget_omitted_or_preserved(api,budget,tmp_path,monkeypatch):
    generation=dict(reasoning_effort='medium',reasoning_context='all_turns',
                    reasoning_mode='standard',text_verbosity='low') if api=='responses' else {}
    p=m.OpenAICompatibleProvider(model='test-model',base_url='http://127.0.0.1:57120/v1',
        timeout_seconds=30,api_key_file=None,provider_api=api,maximum_output_tokens=budget,**generation)
    sent=[]
    raw=_opaque_response_bundle() if api=='responses' else {'choices':[{'message':{'content':'OK'}}]}
    monkeypatch.setattr(p,'_post',lambda r:(sent.append(r) or raw))
    p.complete(_request(),call_dir=tmp_path/'call',seed=1)
    field='max_tokens' if api=='chat_completions' else 'max_output_tokens'
    assert (field not in sent[0]) if budget is None else sent[0][field]==budget
    manifest=json.loads((tmp_path/'call/request_manifest.json').read_text())
    assert (field not in manifest) if budget is None else manifest[field]==budget
    assert p.maximum_output_tokens==budget


def test_public_config_preserves_optional_budget():
    config = validate_model_config({'endpoint': 'http://127.0.0.1:1/v1', 'model': 'test-model',
                                    'maximum_output_tokens': None})
    assert config['maximum_output_tokens'] is None


def test_episode_summary_preserves_unset_budget(tmp_path,monkeypatch):
    from test_model_protocol_retry import _Provider, _run, _done
    monkeypatch.setattr(_Provider,'maximum_output_tokens',None)
    summary,provider,scene,output=_run(tmp_path,[_done()])
    assert summary['maximum_output_tokens'] is None and summary['status']=='DONE'


@pytest.mark.parametrize('budget',[True,False,0,-1,1.5,'unlimited'])
def test_bad_typed_budgets_still_rejected(budget):
    with pytest.raises(ValueError):
        m.OpenAICompatibleProvider(model='test',base_url='http://127.0.0.1:57120/v1',
            timeout_seconds=30,api_key_file=None,maximum_output_tokens=budget)
