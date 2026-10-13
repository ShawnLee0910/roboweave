import json
import subprocess
import sys

import pytest

from behavior_hl.embodied.evaluator_bridge import resident_scene_live as live


def test_oracle_deadline_scope_cancel_and_timeout(monkeypatch, tmp_path):
    timers = []

    class Timer:
        def __init__(self, seconds, callback):
            assert seconds == 1800
            self.callback = callback
            self.started = self.cancelled = False
            timers.append(self)

        def start(self):
            self.started = True

        def cancel(self):
            self.cancelled = True

    class Exit(BaseException):
        pass

    def exit_process(code):
        assert code == 124
        raise Exit()

    monkeypatch.setattr(live, "Timer", Timer)
    monkeypatch.setattr(live.os, "_exit", exit_process)
    kwargs = dict(timeout_seconds=1800, item_id="task_a", run_dir=tmp_path,
                  stages=tmp_path / "stages.jsonl")

    def success(**kwargs):
        assert timers[-1].started
        return "PASS"

    assert not timers  # scene loading has not armed anything
    assert live._run_oracle_with_deadline(success, **kwargs) == "PASS"
    assert timers[-1].cancelled
    timers[-1].callback()  # racing timer after completion must not exit
    assert not (tmp_path / "stages.jsonl").exists()

    def failure(**kwargs):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        live._run_oracle_with_deadline(failure, **kwargs)
    assert timers[-1].cancelled
    timers[-1].callback()

    def timeout(**kwargs):
        timers[-1].callback()
        pytest.fail("resident execution continued after timeout")

    with pytest.raises(Exit):
        live._run_oracle_with_deadline(timeout, **kwargs)
    report = json.loads((tmp_path / "private/resident_tasks/task_a/oracle_result.json").read_text())
    assert report["failure_class"] == "ORACLE_RESOURCE_EXHAUSTED"
    assert report["resident_item_id"] == "task_a"
    assert report["resident_state_reusable"] is False
    assert report["official_q"] is None
    assert report["oracle_execution_seconds"] >= 0
    assert json.loads((tmp_path / "stages.jsonl").read_text())["stage"] == "ORACLE_RESOURCE_EXHAUSTED"


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_deadline_rejected_before_callback(seconds, tmp_path):
    with pytest.raises(ValueError, match="positive and finite"):
        live._run_oracle_with_deadline(
            lambda **kwargs: pytest.fail("callback ran"), timeout_seconds=seconds,
            item_id="task", run_dir=tmp_path, stages=tmp_path / "stages.jsonl",
        )


def test_disabled_deadline_does_not_arm_timer(monkeypatch, tmp_path):
    monkeypatch.setattr(live, "Timer", lambda *args: pytest.fail("timer armed"))
    assert live._run_oracle_with_deadline(
        lambda **kwargs: 42, timeout_seconds=None, item_id="task",
        run_dir=tmp_path, stages=tmp_path / "stages.jsonl",
    ) == 42


@pytest.mark.parametrize('verified', [False, True])
def test_real_timer_exits_process_and_preserves_evidence(tmp_path, verified):
    result = subprocess.run([
        sys.executable, "-c",
        "from pathlib import Path; import sys, time; "
        "from esi_benchmark.bootstrap import activate; activate(); "
        "from behavior_hl.embodied.evaluator_bridge.resident_scene_live import _run_oracle_with_deadline; "
        "from behavior_hl.embodied.evaluator_bridge.oracle_episode_receipt import publish_final_verification; "
        "root=Path(sys.argv[1]); "
        "_run_oracle_with_deadline(lambda **kw: (publish_final_verification(official_q=1, verifier_ok=True) if sys.argv[2]=='True' else None, time.sleep(10)), "
        "timeout_seconds=0.05, item_id='task', run_dir=root, stages=root/'stages.jsonl')",
        str(tmp_path),
        str(verified),
    ], timeout=15, capture_output=True, text=True)
    assert result.returncode == 124, result.stderr
    report = json.loads((tmp_path / "private/resident_tasks/task/oracle_result.json").read_text())
    assert report["failure_class"] == "ORACLE_RESOURCE_EXHAUSTED"
    assert report['official_q'] == (1 if verified else None)
    assert report['oracle_execution_status'] == 'INCOMPLETE'
    assert report['verification_completed'] is verified
    if verified:
        assert report['goal_status'] == 'PASS' and report['failure_stage'] == 'CALLBACK_FINALIZATION'




@pytest.mark.parametrize('deadline', [None, 1800])
def test_real_public_oracle_callback_through_resident(monkeypatch, tmp_path, deadline):
    from types import SimpleNamespace as NS
    from behavior_hl.embodied.evaluator_bridge import run_b1_marker_symbolic_rollout as oracle
    from behavior_hl.embodied.evaluator_bridge.resident_scene_batch import ResidentTaskSpec
    calls = []
    monkeypatch.setattr(live, '_capture_prepared_initial_state', lambda **kw: {
        'state_path': str(tmp_path/'prepared.json'), 'frame_path': str(tmp_path/'prepared.png')})
    monkeypatch.setattr(oracle, '_require_loaded_profile', lambda *a: None)
    monkeypatch.setattr(oracle, '_prepare_action20_private_segmentation_channel', lambda **kw: None)
    monkeypatch.setattr(oracle, '_build_full_motion_safety_starter', lambda *a: None)
    def phase(**kw):
        calls.append(kw)
        return {'status': 'PASS', 'official_q': 1}
    monkeypatch.setattr(oracle, '_full_oracle_phase', phase)
    profile = NS(task_name='activity')
    actual = oracle.make_public_action20_marker_oracle_callback(seed=1, profile=profile,
                  phase_name='public_action20_marker_full_oracle')
    callback = live.make_resident_scene_batch_callback(
        tasks=[ResidentTaskSpec('activity', 1, 'scene', 'train', 0)], profiles=[profile],
        templates=[{}], union_template={}, seed=1, filter_embedded_scene_agents=True,
        oracle_callback=actual, oracle_timeout_seconds=deadline)
    evaluator = NS(robot=NS(get_position_orientation=lambda: ([0, 0, 0], [0, 0, 0, 1])), env=NS(task=NS(activity_name='activity'), scene=NS(save=lambda: {})))
    callback(evaluator=evaluator, control=None, bindings=None, run_dir=tmp_path, stages=tmp_path/'stages.jsonl')
    assert len(calls) == 1
    assert calls[0]['phase_name'] == 'public_action20_marker_full_oracle'
    progress = json.loads((tmp_path/'private/resident_tasks/activity/task_progress.json').read_text())
    assert progress['oracle_started'] is True


def test_signature_failure_does_not_record_start_or_arm_timer(monkeypatch, tmp_path):
    monkeypatch.setattr(live, 'Timer', lambda *a: pytest.fail('must validate signature before timer'))
    updates = []
    def callback(*, run_dir, stages):
        pytest.fail('invalid invocation must not enter callback')
    with pytest.raises(TypeError):
        live._run_oracle_with_deadline(callback, timeout_seconds=1, item_id='task',
            run_dir=tmp_path, stages=tmp_path/'stages.jsonl', progress_update=lambda *a, **kw: updates.append(kw),
            unexpected=True)
    assert updates == []


def test_incompatible_callback_rejected_during_resident_setup():
    from types import SimpleNamespace as NS
    from behavior_hl.embodied.evaluator_bridge.resident_scene_batch import ResidentTaskSpec
    def incompatible(*, evaluator, control, bindings, run_dir, stages, required_extra):
        pytest.fail('setup cannot execute Oracle')
    with pytest.raises(TypeError, match='required_extra'):
        live.make_resident_scene_batch_callback(
            tasks=[ResidentTaskSpec('activity', 1, 'scene', 'train', 0)],
            profiles=[NS(task_name='activity')], templates=[{}], union_template={},
            seed=1, filter_embedded_scene_agents=True, oracle_callback=incompatible)
