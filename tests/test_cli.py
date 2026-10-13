import copy
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from esi_benchmark import cli


def _config(tmp_path):
    return {"schema_version": "esi-benchmark/config-v1", "profile": "l0-english-v2", "mode": "replay",
            "run": {"task_id": "am200v1_a001_b2_control", "family": "L0",
                    "output_root": str(tmp_path / "outputs"), "history_frames": 5}}


def test_help_and_config_cli(tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_config(tmp_path)))
    assert cli.main(["config", "validate", "--config", str(config)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    with pytest.raises(SystemExit) as result:
        cli.main(["--help"])
    assert result.value.code == 0


def test_dry_replay_creates_nothing(tmp_path):
    target = tmp_path / "not-created"
    result = cli._run(_config(tmp_path), target, True)
    assert result["ok"] and not target.exists()


def test_existing_output_rejected(tmp_path):
    with pytest.raises(ValueError, match="already exists"):
        cli._run(_config(tmp_path), tmp_path, False)


@pytest.mark.parametrize("model_fails", [False, True])
def test_owned_live_wiring_and_cleanup_without_live(tmp_path, monkeypatch, model_fails):
    from esi_benchmark import assets, models, runtime
    config = _config(tmp_path)
    config.update(mode="live", runtime={"scene_load_timeout": 1}, model={})
    ready = tmp_path / "owned" / "scene" / "ready.json"
    plan = {"runtime_root": ready.parent.parent, "ready_path": ready}
    task_inputs = {"model_task": {"task_id": 3, "instruction": "public"}}
    calls = []
    monkeypatch.setattr(assets, "resolve_task_inputs", lambda *a: task_inputs)
    def materialize(_config, task, output):
        output.mkdir(parents=True)
        return task_inputs
    monkeypatch.setattr(assets, "materialize_task_inputs", materialize)
    monkeypatch.setattr(runtime, "runtime_plan", lambda *a: plan)
    def launch(*a):
        ready.parent.mkdir(parents=True)
        ready.write_text("{}")
        calls.append("launch")
        return SimpleNamespace(pid=424242, poll=lambda: None)
    monkeypatch.setattr(runtime, "launch_runtime", launch)
    monkeypatch.setattr(runtime, "validate_owned_ready", lambda *a: calls.append("ownership"))
    def episode(**kwargs):
        assert kwargs["task"] == task_inputs["model_task"]
        assert kwargs["runtime_root"] == ready.parent.parent
        assert kwargs["output_reserved"] is True
        assert (kwargs["output_dir"] / "private/inputs").is_dir()
        calls.append("model")
        if model_fails:
            raise ValueError("deliberate fake provider error")
        return {"mode": "test_only"}
    monkeypatch.setattr(models, "run_episode", episode)
    monkeypatch.setattr(cli, "_stop_owned", lambda p: calls.append("stop"))
    if model_fails:
        with pytest.raises(ValueError, match="deliberate"):
            cli._run(config, tmp_path / "new-run", False)
    else:
        assert cli._run(config, tmp_path / "new-run", False)["mode"] == "test_only"
    assert calls == ["launch", "ownership", "model", "stop"]
    if model_fails:
        summary = json.loads((tmp_path / "new-run/summary.json").read_text())
        assert summary["status"] == "INITIALIZATION_FAILED"
        assert summary["official_success"] is None


@pytest.mark.skipif(os.name != "posix", reason="Live runtime requires POSIX sessions")
def test_cleanup_reaps_owned_child_after_leader_exits():
    child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready',flush=True); time.sleep(60)"
    leader = ("import subprocess,sys; "
              f"p=subprocess.Popen([sys.executable,'-c',{child!r}],stdout=subprocess.PIPE,text=True); "
              "p.stdout.readline(); print(p.pid,flush=True)")
    process = subprocess.Popen([sys.executable, "-c", leader], stdout=subprocess.PIPE,
                               text=True, start_new_session=True)
    child_pid = int(process.stdout.readline())
    try:
        process.wait(timeout=5)
        assert os.getpgid(child_pid) == process.pid
        cli._stop_owned(process)
        stat = Path(f"/proc/{child_pid}/stat")
        deadline = time.monotonic() + 2
        while stat.exists() and stat.read_text().split()[2] != "Z" and time.monotonic() < deadline:
            time.sleep(0.02)
        if stat.exists():
            assert stat.read_text().split()[2] == "Z"
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_score_cli_forwards_process_output(tmp_path, monkeypatch, capsys):
    from esi_benchmark import scoring
    path = tmp_path / "record.json"
    path.write_text("{}")
    destination = tmp_path / "process"
    def score(record, *, process_directory):
        assert process_directory == destination
        return {"mode": "cpu-record-replay"}
    monkeypatch.setattr(scoring, "score_record", score)
    assert cli.main(["score", str(path), "--process-output", str(destination)]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "cpu-record-replay"


def test_incomplete_validation_returns_nonzero(tmp_path, monkeypatch, capsys):
    from esi_benchmark import trajectory
    monkeypatch.setattr(trajectory, "validate_run", lambda _: {"status": "INCOMPLETE"})
    assert cli.main(["results", "validate", str(tmp_path)]) == 1
