from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import pytest

from behavior_hl.embodied.evaluator_bridge.l0_taskspec_live_adapter import (
    L0TaskSpecEpisodeVerifier,
)


def test_projected_report_becomes_private_predicate_snapshot() -> None:
    snapshot = L0TaskSpecEpisodeVerifier._snapshot({
        "operational": True,
        "success": False,
        "q_score": 0.5,
        "clause_reports": [
            {"index": 0, "satisfied": True},
            {"index": 1, "satisfied": False},
        ],
    })
    assert snapshot.ok is True
    assert snapshot.official_q == 0.5
    assert dict(snapshot.leaf_truth) == {
        "l0_clause/0": True,
        "l0_clause/1": False,
    }


def test_vendored_projected_verifier_keeps_frozen_sha256() -> None:
    source = __import__("behavior_hl.project_identity", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT / (
        "src/behavior_hl/embodied/evaluator_bridge/l0_verifier_adapter.py"
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "198a14cb47b16a505532b57a2e1feed968ca2210771713416459d54a7bb72cff"
    )


@pytest.mark.parametrize("mode", ["official", "action20_symbolic"])
def test_episode_constructor_forwards_runtime_predicate_mode(monkeypatch, mode):
    from behavior_hl.embodied.evaluator_bridge import l0_taskspec_live_adapter as module
    captured = {}
    def adapter(spec, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(bind=lambda *_a, **_k: {"binding_pass": True},
                               initialize=lambda *_: {"admission_pass": True})
    monkeypatch.setattr(module, 'L0TaskSpecLiveAdapter', adapter)
    monkeypatch.setattr(module, 'canonical_task_spec_sha256', lambda _: 'frozen')
    verifier = L0TaskSpecEpisodeVerifier(
        {"private": {"verifier": {"route": module.OFFICIAL_FULL}}},
        evaluator_state=object(), runtime_identity={}, marker_session_id='session', predicate_mode=mode,
    )
    assert captured['predicate_mode'] == mode
    assert captured['stability_window'] == 1
    assert verifier._initial_report['admission_pass'] is True


@pytest.mark.parametrize('route', ['official_projection', 'canonical_fallback'])
@pytest.mark.parametrize('mode', ['official', 'action20_symbolic'])
@pytest.mark.parametrize('predicate', ['on_top', 'next_to'])
def test_episode_real_reader_and_projected_q_follow_mode(monkeypatch, route, mode, predicate):
    from behavior_hl.embodied.evaluator_bridge import l0_taskspec_live_adapter as module
    from behavior_hl.embodied.evaluator_bridge.l0_verifier_adapter import ProjectedGoalVerifier
    from behavior_hl.embodied.evaluator_bridge.r1pro_b1_component_builders import _read_marker_symbolic_canonical_state
    from test_official_predicate_verifier import _Box, _mock_box_surface
    item = _Box((5, 0, .05), (.1, .1, .1))
    reference = _Box((0, 0, .25), (1, 1, .5))
    _mock_box_surface(monkeypatch, item, reference)
    state_name = {'on_top': 'OnTop', 'next_to': 'NextTo'}[predicate]
    native = {'value': False}
    item.states[type(state_name, (), {})] = SimpleNamespace(get_value=lambda _: native['value'])
    task = object()
    evaluator = SimpleNamespace(env=SimpleNamespace(task=task))
    # Only file provenance / simulator scope binding are substituted. The
    # constructor, truth reader, initial credit and final Q are production code.
    def validate(adapter):
        adapter._projected_verifier_class = ProjectedGoalVerifier
    def bind(adapter, *_args, **_kwargs):
        adapter._binding = SimpleNamespace(task=task, generation=1,
            scopes={'item': item, 'reference': reference})
        return {'binding_pass': True}
    monkeypatch.setattr(module.L0TaskSpecLiveAdapter, '_validate_task_spec', validate)
    monkeypatch.setattr(module.L0TaskSpecLiveAdapter, 'bind', bind)
    spec = {'task_spec_id': 'mode_test', 'private': {'verifier': {'route': route, 'clauses': [{
        'canonical_predicate': predicate.replace('_', ''), 'quantifier': 'ALL_LISTED',
        'target_scopes': ['item'], 'destination_scopes': ['reference'],
    }]}}}
    verifier = L0TaskSpecEpisodeVerifier(spec, evaluator_state=evaluator,
        runtime_identity={}, marker_session_id='session', predicate_mode=mode)
    assert verifier.evaluate(None, evaluator).official_q == 0
    item.center = (0, 0, .553) if predicate == 'on_top' else (.56, 0, .05)
    action_truth = _read_marker_symbolic_canonical_state(item, state_name, reference,
        symbolic_on_top=mode == 'action20_symbolic')
    result = verifier.evaluate(None, evaluator)
    assert result.ok and result.official_success is action_truth
    assert result.official_q == float(action_truth) == float(mode == 'action20_symbolic')
    if mode == 'official':
        native['value'] = True
        assert verifier.evaluate(None, evaluator).official_q == 1
    else:
        item.extent = (float('nan'), .1, .1)
        failed = verifier.evaluate(None, evaluator)
        assert not failed.ok and failed.official_q is None
