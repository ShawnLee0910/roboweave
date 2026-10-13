import json
from types import SimpleNamespace as NS

import pytest

from behavior_hl.embodied.evaluator_bridge import resident_scene_live as live
from behavior_hl.embodied.evaluator_bridge.registered_l0_task_runtime import (
    RegisteredL0ConstructionError,
)
from behavior_hl.embodied.evaluator_bridge.resident_scene_batch import ResidentTaskSpec
from behavior_hl.embodied.evaluator_bridge.resident_task_progress import task_outcomes


class _Robot:
    def __init__(self):
        self.position = [0.0, 0.0, 0.005]
        self.orientation = [0.0, 0.0, 0.0, 1.0]
        self.joints = [0.0]

    def get_position_orientation(self):
        return self.position, self.orientation

    def set_position_orientation(self, position, orientation):
        self.position, self.orientation = list(position), list(orientation)

    def get_joint_positions(self):
        return self.joints

    def set_joint_positions(self, joints):
        self.joints = list(joints)


def _evaluator(name="first"):
    return NS(
        robot=_Robot(),
        env=NS(
            task=NS(activity_name=name),
            scene=NS(save=lambda: "{}"),
        )
    )


@pytest.mark.parametrize("q,success,gated,expected_calls", [
    (1., True, True, 2), (0., False, True, 1),
    (None, False, True, 1), (1., False, True, 1),
    (True, True, True, 1), (None, False, False, 2),
])
def test_l0_visual_variant_renders_before_conditional_oracle(tmp_path, monkeypatch, q, success, gated, expected_calls):
    tasks = [ResidentTaskSpec(name, i, "scene", "train", 0) for i, name in enumerate(("first", "second"))]
    evaluator = _evaluator()
    captures, calls = [], []
    def capture(**kwargs):
        captures.append(str(kwargs["path"]))
        return {"status": "CAPTURED", "path": str(kwargs["path"]), "head_camera_readback": {}}
    monkeypatch.setattr(live, "capture_construction_rgb", capture)
    def restore(**kwargs):
        evaluator.env.task.activity_name = kwargs["task"].task_name
        return {"status": "PASS"}
    monkeypatch.setattr(live, "switch_resident_task", restore)
    def oracle(**kwargs):
        calls.append(True)
        return {"strategies": [{"status": "PASS" if success else "FAIL", "official_success": success, "official_q": q}]}
    callback = live.make_resident_scene_batch_callback(
        tasks=tasks, profiles=[NS(task_name=t.task_name) for t in tasks], templates=[{},{}], union_template={},
        seed=123, filter_embedded_scene_agents=False, oracle_callback=oracle,
        l0_visual_variant_dependencies={"second":"first"} if gated else {})
    result=callback(evaluator=evaluator, control=None, bindings=None, run_dir=tmp_path, stages=tmp_path/'stages.jsonl')
    assert len(calls) == expected_calls
    assert len(captures) == 2
    row=json.loads((tmp_path/'private/resident_tasks/second/oracle_result.json').read_text())
    if expected_calls == 1:
        assert row['status'] == 'RENDERED_ONLY' and row['official_q'] is None
        assert row['oracle_status'] == 'NOT_RUN' and row['oracle_started'] is False
        assert row['first_frame'] and row['prepared_state']
        assert row['failure_class'] is None
        assert result['rendered_only_count'] == 1
    else:
        assert row['oracle_started'] is True


def _oracle(events):
    def callback(**kwargs):
        events.append(kwargs["evaluator"].env.task.activity_name)
        return {"strategies": [{"status": "PASS", "official_q": 1}]}

    return callback


def test_restore_failure_preserves_task_terminal_before_worker_exit(tmp_path):
    task = ResidentTaskSpec('first', 1, 'scene', 'train', 0)
    def broken_reset(_):
        raise RuntimeError('restore failed')
    cb = live.make_resident_scene_batch_callback(tasks=[task], profiles=[NS(task_name='first')],
        templates=[{}], union_template={}, seed=1, filter_embedded_scene_agents=False,
        before_task_reset=broken_reset, oracle_callback=_oracle([]))
    with pytest.raises(RuntimeError,match='restore failed'):
        cb(evaluator=_evaluator(),control=None,bindings=None,run_dir=tmp_path,stages=tmp_path/'stages.jsonl')
    r=json.loads((tmp_path/'private/resident_tasks/first/oracle_result.json').read_text())
    assert r['oracle_started'] is False and r['failure_class']=='RESTORE_ERROR'
    assert r.get('failed_phase') != 'CONSTRUCTION'


def test_protocol_uses_subset_item_ids_and_records_complete_phase_sequence(tmp_path, monkeypatch):
    tasks = [ResidentTaskSpec(name, index, "scene", "train", 0) for index, name in enumerate(("first", "second"))]
    evaluator = _evaluator()
    monkeypatch.setattr(
        live,
        "capture_construction_rgb",
        lambda **kwargs: {
            "status": "CAPTURED",
            "path": str(kwargs["path"]),
            "head_camera_readback": {},
        },
    )

    def restore(**kwargs):
        evaluator.env.task.activity_name = kwargs["task"].task_name
        return {"status": "PASS"}

    monkeypatch.setattr(live, "switch_resident_task", restore)
    callback = live.make_resident_scene_batch_callback(
        tasks=tasks,
        profiles=[NS(task_name=task.task_name) for task in tasks],
        templates=[{}, {}],
        union_template={},
        seed=123,
        filter_embedded_scene_agents=False,
        item_ids=["subset_a", "subset_b"],
        oracle_callback=_oracle([]),
    )
    result = callback(
        evaluator=evaluator,
        control=None,
        bindings=None,
        run_dir=tmp_path,
        stages=tmp_path / "stages.jsonl",
    )
    protocol = json.loads((tmp_path / "private/resident_task_protocol.json").read_text())
    assert protocol["schema_version"] == "resident-task-progress/v1"
    assert protocol["seed"] == 123
    assert protocol["task_names"] == ["subset_a", "subset_b"]
    assert [row["task_name"] for row in protocol["task_identities"]] == ["first", "second"]
    for item_id in protocol["task_names"]:
        progress = json.loads((tmp_path / f"private/resident_tasks/{item_id}/task_progress.json").read_text())
        assert progress["phase"] == "TERMINAL"
        assert progress["seed"] == 123
        assert progress["oracle_started"] is True
        assert progress["timestamps"]["TASK_STARTED"] <= progress["timestamps"]["ORACLE_STARTED"]
        assert progress["execution_dir"]
        assert progress["executor_actions_path"]
    assert result["oracle_status"] == "COMPLETED"
    assert result["oracle_started_count"] == 2


@pytest.mark.parametrize("failure_kind", ["reported_exception", "bare_exception", "failed_report"])
def test_construction_failure_writes_terminal_without_oracle_started(tmp_path, failure_kind):
    task = ResidentTaskSpec("first", 0, "scene", "train", 0)

    def fail(**_kwargs):
        if failure_kind == "bare_exception":
            raise ValueError("unreported construction failure")
        if failure_kind == "failed_report":
            return {"status": "FAIL", "construction_pass": False, "error": "occupied support"}
        raise RegisteredL0ConstructionError(
            "construction failed",
            {"status": "FAIL", "construction_pass": False, "failure_class": "RESTORE_ERROR"},
        )

    callback = live.make_resident_scene_batch_callback(
        tasks=[task],
        profiles=[NS(task_name="first")],
        templates=[{}],
        union_template={},
        seed=1,
        filter_embedded_scene_agents=False,
        construction_callback=fail,
        oracle_callback=lambda **_: pytest.fail("Oracle must not run"),
    )
    result = callback(
        evaluator=_evaluator(),
        control=None,
        bindings=None,
        run_dir=tmp_path,
        stages=tmp_path / "stages.jsonl",
    )
    progress = json.loads((tmp_path / "private/resident_tasks/first/task_progress.json").read_text())
    row = json.loads((tmp_path / "private/resident_tasks/first/oracle_result.json").read_text())
    assert progress["phase"] == "TERMINAL" and progress["oracle_started"] is False
    assert progress["terminal_status"] == "FAIL"
    assert row["oracle_started"] is False
    assert row["failed_phase"] == "CONSTRUCTION"
    assert row["construction_status"] == "FAIL"
    assert row["oracle_status"] == "NOT_RUN"
    assert row["route_gate_attempted"] is False
    assert row["marker_action_gate_attempted"] is False
    evidence = tmp_path / "private/resident_tasks/first/live_initial_construction.json"
    assert row["construction_evidence"] == str(evidence.resolve())
    assert json.loads(evidence.read_text())["construction_pass"] is False
    assert progress["failed_phase"] == "CONSTRUCTION"
    observed = task_outcomes(tmp_path, ["first"], expected_seed=1)["first"]
    assert observed["state"] == "TERMINAL"
    assert observed["oracle_started"] is False
    assert observed["oracle_status"] == "NOT_RUN"
    assert observed["failed_phase"] == "CONSTRUCTION"
    assert observed["construction_evidence"] == row["construction_evidence"]
    assert result["oracle_status"] == "NOT_RUN"
    assert result["oracle_started_count"] == 0


@pytest.mark.parametrize('reuse_failure', [False, True])
def test_explicit_pair_reuses_preparation_without_second_preflight(tmp_path, monkeypatch, reuse_failure):
    first = NS(task_name="first", task_id=0, scene_model="scene", split="train", instance_id=0,
               pair_base_task_spec_id="pair-base")
    second = NS(task_name="second", task_id=1, scene_model="scene", split="train", instance_id=0,
                pair_base_task_spec_id="pair-base")
    evaluator = _evaluator()
    if reuse_failure:
        def fail_pose(*args):
            raise RuntimeError('native pose restore failed')
        evaluator.robot.set_position_orientation = fail_pose
    monkeypatch.setattr(
        live,
        "capture_construction_rgb",
        lambda **kwargs: {
            "status": "CAPTURED",
            "path": str(kwargs["path"]),
            "head_camera_readback": {},
        },
    )
    events = []

    def restore(**kwargs):
        evaluator.env.task.activity_name = kwargs["task"].task_name
        return {"status": "PASS"}

    monkeypatch.setattr(live, "switch_resident_task", restore)

    def preflight(**kwargs):
        events.append(kwargs["task"].task_name)
        return {"status": "PASS", "preparation": {"selected_rank": 2}}

    callback = live.make_resident_scene_batch_callback(
        tasks=[first, second],
        profiles=[NS(task_name="first"), NS(task_name="second")],
        templates=[{"physical_contract_sha256": "same"}, {"physical_contract_sha256": "same"}],
        union_template={},
        seed=1,
        filter_embedded_scene_agents=False,
        present_inward_grippers=True,
        candidate_preflight_callback=preflight,
        oracle_callback=_oracle([]),
    )
    from contextlib import nullcontext
    with pytest.raises(RuntimeError, match='native pose restore failed') if reuse_failure else nullcontext():
        callback(evaluator=evaluator, control=None, bindings=None, run_dir=tmp_path,
                 stages=tmp_path / 'stages.jsonl')
    assert events == ["first"]
    second_presentation = json.loads(
        (tmp_path / "private/resident_tasks/second/gripper_presentation.json").read_text()
    )
    if reuse_failure:
        outcomes = task_outcomes(tmp_path, ['first', 'second'], expected_seed=1)
        assert outcomes['first']['data']['status'] == 'PASS'
        assert outcomes['second']['state'] == 'TERMINAL'
        assert outcomes['second']['oracle_started'] is False
        assert outcomes['second']['oracle_status'] == 'NOT_RUN'
        assert outcomes['second']['failed_phase'] == 'PREPARATION'
        result = json.loads((tmp_path/'private/resident_tasks/second/oracle_result.json').read_text())
        assert result['resident_session_invalidated'] is True
        assert second_presentation['status'] == 'FAIL'
    else:
        assert second_presentation["pair_preparation_reused"] is True


def test_timeout_writes_started_terminal_progress(tmp_path, monkeypatch):
    timers = []

    class Timer:
        def __init__(self, _seconds, callback):
            self.callback = callback
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            pass

    class Exit(BaseException):
        pass

    monkeypatch.setattr(live, "Timer", Timer)
    monkeypatch.setattr(live.os, "_exit", lambda code: (_ for _ in ()).throw(Exit(code)))
    updates = []
    with pytest.raises(Exit):
        live._run_oracle_with_deadline(
            lambda **_: timers[-1].callback(),
            timeout_seconds=1,
            item_id="item",
            run_dir=tmp_path,
            stages=tmp_path / "stages.jsonl",
            execution_phase_name="resident_tasks/00_item/oracle",
            progress_update=lambda *args, **kwargs: updates.append((args, kwargs)),
        )
    assert updates[-1][0] == ("TERMINAL",)
    assert updates[-1][1]["oracle_started"] is True
    report = json.loads((tmp_path / "private/resident_tasks/item/oracle_result.json").read_text())
    assert report["oracle_started"] is True and report["phase_name"].endswith("/oracle")


def test_prepared_boundary_rejects_missing_robot_bad_quaternion_and_setter(monkeypatch, tmp_path):
    monkeypatch.setattr(
        live,
        "capture_construction_rgb",
        lambda **kwargs: {
            "status": "CAPTURED",
            "path": str(kwargs["path"]),
            "head_camera_readback": {},
        },
    )
    with pytest.raises(RuntimeError, match="robot readback"):
        live._capture_prepared_initial_state(
            evaluator=NS(), control=None, profile=None, run_dir=tmp_path,
            item_id="item", initial_pose_validation_callback=None, task=NS(),
        )
    bad_robot = _Robot()
    bad_robot.orientation = [0.0, 0.0, 0.0, 0.0]
    with pytest.raises(RuntimeError, match="zero quaternion"):
        live._capture_prepared_initial_state(
            evaluator=NS(robot=bad_robot), control=None, profile=None, run_dir=tmp_path,
            item_id="item", initial_pose_validation_callback=None, task=NS(),
        )
    no_setter = _Robot()
    no_setter.set_joint_positions = None
    with pytest.raises(RuntimeError, match="setter"):
        live._reuse_prepared_robot_pose(
            evaluator=NS(robot=no_setter),
            prepared={
                "robot_position": [0.0, 0.0, 0.005],
                "robot_orientation": [0.0, 0.0, 0.0, 1.0],
                "robot_joint_positions": [0.0],
            },
        )
