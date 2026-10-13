from __future__ import annotations

import ast
from contextvars import ContextVar
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from lab.l1_episode import audited_step


SCENE_SERVER_PATH = (Path(__file__).resolve().parents[2] / "src/esi_benchmark/_vendor/bench/lab") / "scene_server.py"


def _scene_class() -> ast.ClassDef:
    tree = ast.parse(SCENE_SERVER_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ResidentInsideTaskScene"
    )


def _scene_method(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in _scene_class().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


@pytest.mark.parametrize("substeps,stop,expected_steps", [
    (1, None, 1), (3, None, 3), (3, "blocked", 2), (3, "success", 2),
])
def test_navigation_capture_is_owned_by_outer_boundary(tmp_path, substeps, stop, expected_steps):
    branch = next(node for node in ast.walk(_scene_method("_step_impl"))
                  if isinstance(node, ast.If) and ast.unparse(node.test) == "navigation_substeps == 1")
    calls, captures, saved = [], [], []
    def execute(*, runtime, action, action_index, phase_dir, terminal_success_check,
                capture_before_frame=True, capture_after_frame=True):
        # Exact supported keyword names: the obsolete name fails this test.
        calls.append((capture_before_frame, capture_after_frame))
        blocked = stop == "blocked" and len(calls) == 2
        result = SimpleNamespace(status="blocked" if blocked else "completed",
                                 public_feedback={"body_odometry": {"forward_m": .15, "left_m": 0, "yaw_deg": 0}})
        return result, ("inner_before", "inner_after") if capture_before_frame else (), 0
    def capture(runtime):
        captures.append(len(calls))
        return f"frame_at_step_{len(calls)}"
    checkpoints = []
    runtime = SimpleNamespace(environment=SimpleNamespace(checkpoint_once=lambda: checkpoints.append(True) or 0),
                              symbolic_primitives=SimpleNamespace())
    owner = SimpleNamespace(_episode=object(), evaluator=object(), _verifier=SimpleNamespace(
        evaluate=lambda *_: SimpleNamespace(official_success=stop == "success" and len(calls) == 2)))
    namespace = dict(navigation_substeps=substeps, runtime=runtime, parsed=object(), sequence=7,
        action_artifact_dir=tmp_path, action="move_forward", self=owner, Mapping=dict,
        SceneIntegrationError=RuntimeError, _execute=execute,
        _capture_action_boundary_observation=capture,
        _save_frame=lambda **kwargs: saved.append((kwargs["path"].name, kwargs["observation"])),
        model_visible_action20_last_result=lambda value, **_: value)
    exec(compile(ast.Module(body=[branch], type_ignores=[]), str(SCENE_SERVER_PATH), "exec"), namespace)
    assert len(calls) == expected_steps
    if substeps == 1:
        assert calls == [(True, True)]
        assert captures == saved == []
        assert namespace["action_frames"] == ("inner_before", "inner_after")
    else:
        assert calls == [(False, False)] * expected_steps
        assert captures == [0, expected_steps]
        assert saved == [("action_007_before.png", "frame_at_step_0"),
                         ("action_007_after.png", f"frame_at_step_{expected_steps}")]
        assert len(namespace["navigation_substep_records"]) == expected_steps
        assert bool(checkpoints) is (stop != "success")


def _load_step_only_scene() -> type[Any]:
    method = _scene_method("step")
    shell = ast.ClassDef(
        name="StepOnlyScene",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                shell,
            ],
            type_ignores=[],
        )
    )

    class SceneProtocolError(ValueError):
        pass

    namespace: dict[str, Any] = {
        "Any": Any,
        "Mapping": dict,
        "SceneProtocolError": SceneProtocolError,
        "audited_step": audited_step,
    }
    exec(compile(module, str(SCENE_SERVER_PATH), "exec"), namespace)
    return namespace["StepOnlyScene"]


def test_step_impl_binds_live_official_verifier_to_execute() -> None:
    method = _scene_method("_step_impl")
    execute_call = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_execute"
    )
    terminal_check = next(
        keyword.value
        for keyword in execute_call.keywords
        if keyword.arg == "terminal_success_check"
    )
    assert isinstance(terminal_check, ast.Lambda)
    assert terminal_check.args.args == []
    expected_callback_body = ast.parse(
        "self._verifier.evaluate(self._episode, self.evaluator)", mode="eval"
    ).body
    assert ast.dump(terminal_check.body, include_attributes=False) == ast.dump(
        expected_callback_body,
        include_attributes=False,
    )

    source = ast.unparse(method)
    assert (
        "None if snapshot.official_success else "
        "self._maybe_capture_rolling_checkpoint()"
    ) in source


@pytest.fixture
def runtime_audit_context(monkeypatch):
    # Step-only tests do not import the simulator; preserve the audit boundary.
    active = ContextVar("test_active_audit", default=None)
    monkeypatch.setitem(sys.modules,
        "behavior_hl.embodied.evaluator_bridge.l1_process_audit",
        SimpleNamespace(ACTIVE=active))
    yield active
    assert active.get() is None


@pytest.mark.parametrize("native_q_alone,qualification,should_end", [
    (True, None, True),
    (False, "PASS", True),
    (False, "FAIL", False),
    (False, "UNKNOWN", False),
    (False, None, False),
])
def test_verified_success_auto_ends_episode_without_another_step(
    runtime_audit_context, native_q_alone, qualification, should_end,
) -> None:
    scene_type = _load_step_only_scene()
    scene = scene_type()
    scene._session_valid = True
    scene._native_q_alone_is_success = native_q_alone
    scene._last_verifier = SimpleNamespace(official_success=True,
        research_diagnostics={"physical_template_review": {
            "qualification_status": qualification}})
    calls: list[tuple[Any, ...]] = []
    scene._step_impl = lambda **kwargs: (
        calls.append(("step_impl", kwargs))
        or {"status": "PASS", "simulator_steps": 1}
    )
    scene.end_episode = lambda *, reason: (
        calls.append(("end_episode", reason))
        or {"session_valid": True}
    )
    scene._recover_failed_episode_runtime = lambda **_kwargs: (
        calls.append(("recover",)) or False
    )

    result = scene.step(sequence=0, action="toggle_on", target=1)

    if not should_end:
        assert result == {"status": "PASS", "simulator_steps": 1}
        assert len(calls) == 1 and calls[0][0] == "step_impl"
        return

    assert result == {
        "status": "PASS",
        "simulator_steps": 1,
        "episode_terminal": True,
        "session_valid": True,
    }
    assert calls == [
        (
            "step_impl",
            {
                "sequence": 0,
                "action": "toggle_on",
                "target": 1,
                "navigation_substeps": 1,
            },
        ),
        ("end_episode", "done"),
    ]


def test_unverified_terminal_exception_keeps_environment_error_path(runtime_audit_context) -> None:
    scene_type = _load_step_only_scene()
    scene = scene_type()
    scene._session_valid = True
    scene._last_verifier = SimpleNamespace(official_success=False)
    terminal = RuntimeError("resident evaluator terminated during bounded step")
    recover_reasons: list[str] = []

    def fail(**_kwargs: Any) -> Any:
        raise terminal

    scene._step_impl = fail
    scene.end_episode = lambda **_kwargs: pytest.fail(
        "unverified terminal must not end as success"
    )
    scene._recover_failed_episode_runtime = lambda *, reason: (
        recover_reasons.append(reason) or False
    )

    with pytest.raises(RuntimeError) as caught:
        scene.step(sequence=0, action="toggle_on", target=1)

    assert caught.value is terminal
    assert recover_reasons == ["step_environment_error"]
    assert scene._session_valid is False
