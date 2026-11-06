import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from roboweave import engine
from roboweave.profiles import DEFAULT_PROFILE


@pytest.mark.parametrize("change", [
    {"operation": "arbitrary_import"}, {"profile": "not-a-profile"},
    {"payload": {"unexpected": True}}, {"extra": True},
    {"operation": []}, {"profile": []},
])
def test_fixed_protocol_rejects_unknown_fields_before_spawning(monkeypatch, change):
    request = {"schema_version": engine.SCHEMA, "operation": "inspect",
               "profile": DEFAULT_PROFILE, "payload": {}}
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("spawned invalid request"))
    with pytest.raises(ValueError):
        engine.validate_request({**request, **change})


def test_nested_profile_must_match():
    with pytest.raises(ValueError, match="differs"):
        engine.call("replay", {"config": {"profile": "other"}, "output_dir": "unused"})
    with pytest.raises(ValueError, match="object"):
        engine.call("inspect", [])


def test_public_calls_do_not_load_or_replace_legacy_modules(tmp_path):
    package = Path(__file__).resolve().parents[1] / "src"
    script = """
import json, sys, types
from pathlib import Path
from roboweave.bootstrap import resolve, activate
from roboweave.engine import call
from roboweave.models import validate_model_config
from roboweave.offline import run_replay
legacy = types.ModuleType('agent_protocol')
legacy.foreign = True
sys.modules['agent_protocol'] = legacy
before_path = list(sys.path)
before_owned = {name for name in sys.modules if name in {'Agent', 'lab', 'behavior_hl', 'model_client'}}
resolve(); activate()
assert validate_model_config({'endpoint': 'http://localhost:8000/v1', 'model': 'unused'})['api'] == 'chat_completions'
assert call('inspect', {})['ok']
assert run_replay({}, Path(sys.argv[1]))['mode'] == 'offline_replay'
assert sys.path == before_path
assert sys.modules['agent_protocol'] is legacy
assert {name for name in sys.modules if name in {'Agent', 'lab', 'behavior_hl', 'model_client'}} == before_owned
"""
    result = subprocess.run([sys.executable, "-B", "-c", script, str(tmp_path / "run")],
                            env={**os.environ, "PYTHONPATH": str(package)},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_credentials_are_not_serialized_and_episode_has_no_new_deadline(monkeypatch):
    observed = {}

    class Process:
        pid = 432123
        returncode = 0
        stdin = io.BytesIO()
        stdout = io.BytesIO()

        def communicate(self, raw, timeout):
            observed.update(request=json.loads(raw), timeout=timeout)
            return json.dumps({"schema_version": engine.SCHEMA, "ok": True,
                               "result": {"synthetic": True}, "error": None}).encode(), None

    def spawn(command, **kwargs):
        observed.update(command=command, options=kwargs)
        return Process()

    monkeypatch.setenv("ESI_TEST_ENGINE_KEY", "test-secret-not-in-json")
    monkeypatch.setattr(subprocess, "Popen", spawn)
    result = engine.call("episode", {
        "model_config": {"api_key_env": "ESI_TEST_ENGINE_KEY"}, "task": {},
        "ready_path": "scene/ready.json", "runtime_root": "scene", "output_dir": "run",
        "experiment_id": "test", "history_frames": 5, "output_reserved": False,
    })
    assert result == {"synthetic": True}
    assert observed["timeout"] is None
    assert observed["options"]["start_new_session"] is True
    assert observed["options"]["env"]["ESI_TEST_ENGINE_KEY"] == "test-secret-not-in-json"
    assert "test-secret-not-in-json" not in json.dumps(observed["request"])
    assert "-I" in observed["command"]


def test_cancellation_only_stops_the_owned_worker_session(monkeypatch):
    calls = []

    class Process:
        pid = 432124
        stdin = io.BytesIO()
        stdout = io.BytesIO()

        def communicate(self, *args, **kwargs):
            raise KeyboardInterrupt()

        def wait(self, **kwargs):
            calls.append(("wait", kwargs))

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    with pytest.raises(KeyboardInterrupt):
        engine.call("inspect", {})
    assert [item for item in calls if item[0] == Process.pid] == [
        (Process.pid, signal.SIGTERM), (Process.pid, signal.SIGKILL)]


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"value":NaN}'])
def test_protocol_rejects_duplicate_or_nonfinite_json(raw):
    with pytest.raises(ValueError):
        engine._read_json(raw)


@pytest.fixture
def watchdog_payload(tmp_path):
    run = tmp_path / "owned-run"
    run.mkdir()
    task = {"task_id": 3, "task_name": "frozen-task", "task_split": "train",
            "task_instance_id": "0", "task_seed": 19506,
            "maximum_model_calls": 3810, "budget": {"wall_clock_seconds": 7200}}
    identity = {key: task[key] for key in ("task_id", "task_name", "task_split", "task_instance_id", "task_seed")}
    (run / "run_record.json").write_text(json.dumps({
        "profile": "l2-collection37", "experiment_id": "owned", "task_identity": identity}))
    (run / "client_manifest.json").write_text(json.dumps({
        **identity, "experiment_id": "owned", "model": "fixture", "pause_on_client_failure": True}))
    (run / "provider_preflight.json").write_text('{"status":"PASS"}')
    (run / "provider_calls/call_0001").mkdir(parents=True)
    return {"output_dir": str(run), "experiment_id": "owned", "task": task,
            "model_config": {"model": "fixture"}}


@pytest.mark.parametrize("artifacts,phase", [
    ({}, "STARTING_PROVIDER_CALL"),
    ({"request_manifest.json": {}}, "WAITING_PROVIDER"),
    ({"response.json": {}, "request_manifest.json": {}}, None),
    ({"protocol_attempt.json": {"status": "PASS"}}, None),
    ({"protocol_attempt.json": {"status": "INVALID_MODEL_RESPONSE", "retry_allowed": True}}, None),
    ({"provider_error.json": {"retry_allowed": False}}, None),
    ({"provider_error.json": {"retry_allowed": True}, "protocol_attempt.json": {}}, "RETRYING_PROVIDER"),
])
def test_watchdog_preserves_original_common_phase_precedence(watchdog_payload, artifacts, phase):
    run = Path(watchdog_payload["output_dir"])
    for name, value in artifacts.items():
        (run / "provider_calls/call_0001" / name).write_text(json.dumps(value))
    assert engine._safe_model_phase(watchdog_payload, run) == phase


def test_watchdog_requires_exact_run_and_task_identity(watchdog_payload, tmp_path):
    run = Path(watchdog_payload["output_dir"])
    assert engine._safe_model_phase(watchdog_payload, tmp_path) is None
    path = run / "client_manifest.json"
    manifest = json.loads(path.read_text())
    path.write_text(json.dumps({**manifest, "task_id": 4}))
    assert engine._safe_model_phase(watchdog_payload, run) is None


@pytest.mark.parametrize("configured,expected", [(" fixture ", "STARTING_PROVIDER_CALL"), ("other", None)])
def test_watchdog_matches_the_frozen_provider_normalized_model(watchdog_payload, configured, expected):
    watchdog_payload["model_config"]["model"] = configured
    assert engine._safe_model_phase(watchdog_payload, Path(watchdog_payload["output_dir"])) == expected


def test_watchdog_observation_capacity_covers_the_declared_l2_budget(watchdog_payload):
    run = Path(watchdog_payload["output_dir"])
    for index in range(2, 514):
        (run / "provider_calls" / f"call_{index:04d}").mkdir()
    (run / "provider_calls/call_0513/request_manifest.json").write_text('{}')
    assert engine._safe_model_phase(watchdog_payload, run) == "WAITING_PROVIDER"
    watchdog_payload["task"]["maximum_model_calls"] = 512
    assert engine._safe_model_phase(watchdog_payload, run) is None


def test_l2_wall_budget_never_applies_to_other_profiles_or_cpu_operations():
    request = {"operation": "episode", "profile": "l2-collection37",
               "payload": {"task": {"budget": {"wall_clock_seconds": 7200}}}}
    assert engine._task_wall_seconds(request) == 7200
    assert engine._task_wall_seconds({**request, "profile": DEFAULT_PROFILE}) is None
    assert engine._task_wall_seconds({**request, "operation": "replay"}) is None
    with pytest.raises(ValueError, match="wall budget"):
        engine._task_wall_seconds({**request, "payload": {"task": {}}})


def test_watchdog_waits_for_threshold_and_sigint_acceptance(watchdog_payload, monkeypatch):
    class Process:
        pid = 432125
    process = Process()
    run = Path(watchdog_payload["output_dir"])
    monkeypatch.setattr(engine, "_process_elapsed_seconds", lambda process: 7199.9)
    monkeypatch.setattr(engine, "_owned_sigint_ready", lambda process: True)
    assert engine._wall_timeout_evidence(process, watchdog_payload, run, 7200) is None
    monkeypatch.setattr(engine, "_process_elapsed_seconds", lambda process: 7200.0)
    evidence = engine._wall_timeout_evidence(process, watchdog_payload, run, 7200)
    assert evidence["observed_elapsed_seconds"] == 7200.0
    assert evidence["scene_load_included"] is False
    monkeypatch.setattr(engine, "_owned_sigint_ready", lambda process: False)
    assert engine._wall_timeout_evidence(process, watchdog_payload, run, 7200) is None


def test_watchdog_clock_starts_at_kernel_model_process_birth(monkeypatch):
    class Process:
        pid = 432129
    fields = ["0"] * 20
    fields[0], fields[19] = "S", "1000"
    stat = "432129 (owned model worker) " + " ".join(fields)
    def read(path, *args, **kwargs):
        return io.StringIO(stat if path.name == "stat" else "7210.0 0.0\n")
    monkeypatch.setattr(Path, "open", read)
    monkeypatch.setattr(os, "sysconf", lambda name: 100)
    assert engine._process_elapsed_seconds(Process()) == 7200.0


@pytest.mark.parametrize("ignored,pgid,expected", [("0", 432126, True), ("2", 432126, False), ("0", 432999, False)])
def test_watchdog_checks_the_owned_process_group_and_sigint_mask(monkeypatch, ignored, pgid, expected):
    class Process:
        pid = 432126
        def poll(self):
            return None
    monkeypatch.setattr(os, "getpgid", lambda pid: pgid)
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: io.StringIO("SigIgn:\t" + ignored + "\n"))
    assert engine._owned_sigint_ready(Process()) is expected


def test_watchdog_interrupts_once_and_keeps_original_worker_result(watchdog_payload, monkeypatch):
    signals = []
    inputs = []
    class Process:
        pid = 432127
        attempts = 0
        def communicate(self, raw, timeout):
            inputs.append((raw, timeout))
            self.attempts += 1
            if self.attempts < 4:
                raise subprocess.TimeoutExpired("fake-worker", timeout)
            return b'{"status":"PAUSED_INTERRUPTED"}', None
    ages = iter([7199, 7201, 7201])
    monkeypatch.setattr(engine, "_process_elapsed_seconds", lambda process: next(ages))
    monkeypatch.setattr(engine, "_owned_sigint_ready", lambda process: True)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    result = engine._communicate_l2_episode(Process(), b"request", watchdog_payload, 7200)
    assert result == (b'{"status":"PAUSED_INTERRUPTED"}', None)
    assert signals == [(Process.pid, signal.SIGINT)]
    assert inputs == [(b"request", 1), (None, 1), (None, 1), (None, 1)]
    evidence = json.loads((Path(watchdog_payload["output_dir"]) / "private/model_wall_timeout.json").read_text())
    assert evidence["policy"] == "model_wall_timeout/v1"
    assert evidence["observed_phase"] == "STARTING_PROVIDER_CALL"
    assert evidence["signal"] == "SIGINT" and evidence["budget_scope"] == "task"


def test_watchdog_rechecks_phase_and_never_escalates_to_a_hard_kill(watchdog_payload, monkeypatch):
    class Process:
        pid = 432128
        attempts = 0
        def communicate(self, raw, timeout):
            self.attempts += 1
            if self.attempts == 1:
                raise subprocess.TimeoutExpired("fake-worker", timeout)
            return b"finished", None
    phases = iter(["WAITING_PROVIDER", None])
    monkeypatch.setattr(engine, "_process_elapsed_seconds", lambda process: 8000)
    monkeypatch.setattr(engine, "_safe_model_phase", lambda *args: next(phases))
    monkeypatch.setattr(engine, "_owned_sigint_ready", lambda process: True)
    monkeypatch.setattr(os, "killpg", lambda *args: pytest.fail("unsafe signal"))
    monkeypatch.setattr(engine, "_stop_owned", lambda *args: pytest.fail("forced shutdown"))
    assert engine._communicate_l2_episode(Process(), b"request", watchdog_payload, 7200) == (b"finished", None)
    assert not (Path(watchdog_payload["output_dir"]) / "private/model_wall_timeout.json").exists()
