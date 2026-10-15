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


def test_replay_override_validates_selected_mode_without_live_dependencies(tmp_path, capsys):
    config = _config(tmp_path)
    config["mode"] = "live"
    path = tmp_path / "live.json"
    path.write_text(json.dumps(config))
    assert cli.main(["run", "--config", str(path), "--mode", "replay", "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "replay" and result["ok"]
    assert not Path(result["output_dir"]).exists()


@pytest.mark.parametrize("relative", ["workspace", "workspace/episode", "."])
def test_output_workspace_overlap_fails_before_resolving_or_loading_scene(tmp_path, monkeypatch, relative):
    from esi_benchmark import assets
    config = _config(tmp_path)
    config.update(mode="live", runtime={"workspace": str(tmp_path / "outputs/workspace")})
    def unexpected(*args):
        pytest.fail("Overlapping output must be rejected before loading private task inputs")
    monkeypatch.setattr(assets, "resolve_task_inputs", unexpected)
    with pytest.raises(ValueError, match="non-overlapping"):
        cli._run(config, tmp_path / "outputs" / relative, False)
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("name", ["bad name", "x" * 129])
def test_output_id_is_checked_before_scene_load(tmp_path, name):
    config = _config(tmp_path)
    config.update(mode="live", runtime={"workspace": str(tmp_path / "runtime")})
    with pytest.raises(ValueError, match="experiment ID"):
        cli._run(config, tmp_path / name, False)
    assert not (tmp_path / name).exists()


@pytest.mark.parametrize("model_fails", [False, "before_views", "after_views"])
def test_owned_live_wiring_and_cleanup_without_live(tmp_path, monkeypatch, model_fails):
    from esi_benchmark import assets, engine, models, runtime
    config = _config(tmp_path)
    config.update(mode="live", runtime={"scene_load_timeout": 1, "workspace": str(tmp_path / "owned")}, model={})
    ready = tmp_path / "owned" / "scene" / "ready.json"
    plan = {"runtime_root": ready.parent.parent, "ready_path": ready}
    task_inputs = {"model_task": {"task_id": 3, "instruction": "public"}}
    calls = []
    monkeypatch.setattr(assets, "resolve_task_inputs", lambda *a: task_inputs)
    def materialize(_config, task, output):
        output.mkdir(parents=True)
        return task_inputs
    monkeypatch.setattr(assets, "materialize_task_inputs", materialize)
    def inspect(operation, payload, *, profile):
        assert operation == "task_info" and payload == {"inputs": task_inputs}
        assert profile == config["profile"]
        calls.append("admission")
        return task_inputs
    monkeypatch.setattr(engine, "call", inspect)
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
        assert kwargs["profile"] == config["profile"]
        assert (kwargs["output_dir"] / "private/inputs").is_dir()
        calls.append("model")
        if model_fails == "after_views":
            # The native loop writes these before re-raising an episode exception.
            import hashlib
            output = kwargs["output_dir"]
            prompt = b"Synthetic CLI failure fixture.\n"
            (output / "system_prompt.txt").write_bytes(prompt)
            (output / "client_manifest.json").write_text(json.dumps({
                "prompt_provenance": {"rendered_prompt_sha256": hashlib.sha256(prompt).hexdigest()},
                "source_sha256": {"fixture": "0" * 64}}))
            (output / "conversation_chatml.json").write_text(json.dumps({
                "schema_version": "behavior-hl-action20-chatml-v3", "messages": [{"role": "user", "content": "x"}]}))
            (output / "run_record.json").write_text(json.dumps({"mode": "live", "profile": config["profile"]}))
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
    assert calls == ["admission", "launch", "ownership", "model", "stop"]
    if model_fails:
        from esi_benchmark.trajectory import inspect_run, summarize_runs
        output = tmp_path / "new-run"
        summary = json.loads((output / "summary.json").read_text())
        assert summary["status"] == "EPISODE_FAILED" and summary["stage"] == "model_episode"
        assert summary["official_success"] is None
        assert json.loads((output / "candidate_error.json").read_text())["status"] == "EPISODE_FAILED"
        assert (output / "conversation_unavailable.json").exists() is (model_fails == "before_views")
        inspected = inspect_run(output)
        assert inspected["status"] == "INCOMPLETE" and inspected["completeness"] == "candidate_failure"
        assert inspected["score_available"] is False
        report = summarize_runs([output])
        assert report["incomplete_runs"] == 1 and report["invalid_runs"] == 0


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_model_key_fails_before_assets_or_scene(tmp_path, monkeypatch, value):
    from esi_benchmark import assets
    config = _config(tmp_path)
    config.update(mode="live", runtime={"workspace": str(tmp_path / "runtime")},
                  model={"api_key_env": "ESI_TEST_KEY"})
    if value is None:
        monkeypatch.delenv("ESI_TEST_KEY", raising=False)
    else:
        monkeypatch.setenv("ESI_TEST_KEY", value)
    monkeypatch.setattr(assets, "resolve_task_inputs", lambda *a: pytest.fail("Missing key must fail before assets"))
    with pytest.raises(ValueError, match="environment variable is unavailable"):
        cli._run(config, tmp_path / "new-run", False)
    assert not (tmp_path / "new-run").exists()
    assert not (tmp_path / "runtime").exists()


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
    def score(record, *, process_directory, profile):
        assert process_directory == destination
        assert profile == "l1-a30"
        return {"mode": "cpu-record-replay"}
    monkeypatch.setattr(scoring, "score_record", score)
    assert cli.main(["score", str(path), "--process-output", str(destination), "--profile", "l1-a30"]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "cpu-record-replay"


def test_incomplete_validation_returns_nonzero(tmp_path, monkeypatch, capsys):
    from esi_benchmark import trajectory
    monkeypatch.setattr(trajectory, "validate_run", lambda _: {"status": "INCOMPLETE"})
    assert cli.main(["results", "validate", str(tmp_path)]) == 1


@pytest.mark.parametrize("raw", ['{"official":{},"official":{"snapshots":[]}}',
                                  '{"truth":NaN}', None],
                         ids=["duplicate", "nonfinite", "nesting"])
def test_malformed_score_json_is_rejected_before_evaluation(tmp_path, monkeypatch, capsys, raw):
    from esi_benchmark import scoring
    if raw is None:
        depth = 20000  # CPython 3.12's C JSON decoder has a separate recursion limit.
        raw = '[' * depth + '0' + ']' * depth
    path = tmp_path / "record.json"
    path.write_text(raw)
    monkeypatch.setattr(scoring, "score_record", lambda *args, **kwargs: pytest.fail("Malformed JSON must not reach scoring"))
    assert cli.main(["score", str(path)]) == 2
    assert "esi-benchmark:" in capsys.readouterr().err


def test_score_rejects_nonregular_input_without_opening_it(tmp_path, capsys):
    path = tmp_path / "record-pipe"
    os.mkfifo(path)
    assert cli.main(["score", str(path)]) == 2
    assert "JSON file" in capsys.readouterr().err
