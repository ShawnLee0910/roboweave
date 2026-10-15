import hashlib
import json
from pathlib import Path
import socket
import sys

import pytest

from esi_benchmark import runtime
from esi_benchmark.bootstrap import REVISIONS, activate


@pytest.fixture
def frozen_task():
    from esi_benchmark.tasks import list_tasks
    public = list_tasks("L0")[0]
    model = dict(task_id=public["source"]["row_index"], task_name=public["runtime_instruction_id"],
                 task_split="train", task_instance_id="0", task_seed=19506,
                 instruction=public["instruction_en"], budget=dict(public["budget"]), **public["budget"],
                 instruction_provenance={"profile": public["profile"], "instruction_sha256": public["instruction_sha256"]})
    return public, model


def test_source_identity_is_upstream_plus_candidate_hashes():
    report = runtime.verify_runtime_sources()
    assert report["upstream_revision"] == REVISIONS["runtime"]
    assert report["files_verified"] >= 291
    assert report["candidate_adaptations"]
    from behavior_hl.embodied.evaluator_bridge.benchmark_execution_contract import runner_identity
    identity = runner_identity(activate()["runtime_root"])
    assert identity["commit"] == REVISIONS["runtime"]
    assert identity["candidate_source_manifest_sha256"]


def test_materialized_input_identity_and_no_path_escape(tmp_path, frozen_task):
    public, model = frozen_task
    inputs = dict(instruction_id=public["runtime_instruction_id"], handoff_json=str(tmp_path/"handoff.json"),
                  prepared_index_json=str(tmp_path/"prepared.json"), model_task=model)
    with pytest.raises(ValueError, match="materialized"):
        runtime._verify_materialized_inputs(inputs, profile=public["profile"])
    files = []
    for filename in ("handoff.json", "prepared.json"):
        path = tmp_path / filename
        path.write_text("{}")
        files.append(dict(derived_path=filename, derived_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    receipt = tmp_path / "materialization_receipt.json"
    report = dict(instruction_id=public["runtime_instruction_id"], files=files,
                  profile=public["profile"], task_id=public["task_id"])
    receipt.write_text(json.dumps(report))
    inputs["materialization_receipt"] = str(receipt)
    runtime._verify_materialized_inputs(inputs, profile=public["profile"])
    for change in ({"profile": "l1-a30"}, {"task_id": "another-task"}):
        receipt.write_text(json.dumps({**report, **change}))
        with pytest.raises(ValueError, match="receipt"):
            runtime._verify_materialized_inputs(inputs, profile=public["profile"])
    receipt.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="receipt"):
        runtime._verify_materialized_inputs(inputs, profile=public["profile"], task_id="another-task")
    (tmp_path / "prepared.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed"):
        runtime._verify_materialized_inputs(inputs, profile=public["profile"])


def test_ready_owner(tmp_path):
    scene = tmp_path / "scene"
    scene.mkdir()
    run = scene / "simulator"
    run.mkdir()
    plan = dict(workspace=str(tmp_path), ready_path=str(scene / "ready.json"))
    with socket.socket(socket.AF_UNIX) as sock:
        sock.bind(str(scene / "s"))
        ready = dict(socket_path=str(scene / "s"), run_dir=str(run), pid=42)
        (scene / "ready.json").write_text(json.dumps(ready))
        assert runtime.validate_owned_ready(plan, 42) == ready
        with pytest.raises(ValueError, match="process identity"):
            runtime.validate_owned_ready(plan, 43)
        ready["run_dir"] = str(tmp_path.parent)
        (scene / "ready.json").write_text(json.dumps(ready))
        with pytest.raises(ValueError, match="escaped"):
            runtime.validate_owned_ready(plan, 42)


def test_explicit_host_shim_absence_keeps_validation_safe(monkeypatch):
    from behavior_hl.embodied.evaluator_bridge import run_g1_canary as module
    monkeypatch.setattr(module, "HOST_DLSYM_SHIM", None)
    monkeypatch.setattr(module, "OVERLAY_ROOT", Path(sys.prefix))
    monkeypatch.setenv("BEHAVIOR_HL_PROJECT_ROOT", str(activate()["runtime_root"]))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("OMNIGIBSON_GPU_ID", "0")
    monkeypatch.setenv("TORCHINDUCTOR_DISABLE", "1")
    monkeypatch.setenv("CUDA_HOME", str(module.TOOLCHAIN_ROOT))
    report = module._validate_launch_contract(physical_gpu=0)
    assert report["passed"]
    assert report["host_dlsym_shim"] is None
    assert report["host_dlsym_shim_sha256"] is None


def test_single_load_resource_contract_is_complete():
    from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import load_single_load_contract
    contract = load_single_load_contract()
    assert contract["scene_load_budget"] == 1
    assert contract["automatic_retry"] is False
    assert contract["evaluator_instances"] == 1


def test_official_checkout_rejects_untracked_files_before_scene_launch(tmp_path, monkeypatch):
    from types import SimpleNamespace
    def git(command, **kwargs):
        if command[-1] == "--show-toplevel":
            return SimpleNamespace(stdout=str(tmp_path))
        if command[-1] == "HEAD":
            return SimpleNamespace(stdout=runtime.OFFICIAL_REVISION)
        return SimpleNamespace(stdout="?? unexpected.py\n" if "--untracked-files=normal" in command else "")
    monkeypatch.setattr(runtime.subprocess, "run", git)
    with pytest.raises(ValueError, match="untracked"):
        runtime._official_identity(tmp_path)


def test_plan_is_read_only_and_launch_keeps_virtualenv_path(tmp_path, monkeypatch, frozen_task):
    from esi_benchmark.bootstrap import PROFILE
    public, model = frozen_task
    behavior = tmp_path / "official"
    for child in ("OmniGibson/omnigibson", "bddl3/bddl", "joylo"):
        (behavior / child).mkdir(parents=True)
    data = tmp_path / "assets"
    for child, value in (("behavior-1k-assets/VERSION", "3.9.0"), ("omnigibson-robot-assets/VERSION", "3.8.2")):
        path = data / child
        path.parent.mkdir(parents=True)
        path.write_text(value)
    (data / "2026-challenge-task-instances").mkdir()
    environment = tmp_path / "env"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").symlink_to(sys.executable)
    (environment / "bin/nvcc").write_text("synthetic compiler marker; not executed")
    mdl = tmp_path / "mdl"
    mdl.mkdir()
    (mdl / "synthetic.mdl").write_text("CPU fixture, not an NVIDIA module")
    monkeypatch.setattr(runtime, "MDL_SHA256", {"synthetic.mdl": hashlib.sha256((mdl / "synthetic.mdl").read_bytes()).hexdigest()})
    icd = tmp_path / "icd.json"
    icd.write_text("{}")
    output = tmp_path.parent / "plan-outputs"
    config = dict(schema_version="esi-benchmark/config-v1", mode="live", profile=PROFILE,
                  assets=dict(root=str(data)), run=dict(task_id=public["task_id"], family="L0", output_root=str(output), history_frames=20),
                  model=dict(endpoint="https://example.invalid/v1", model="example", api_key_env="EXAMPLE_TEST_KEY"),
                  runtime=dict(workspace=str(output / "w"), python=str(environment / "bin/python"),
                               behavior_root=str(behavior), data_root=str(data), toolchain_root=str(environment),
                               overlay_root=str(environment), gpu=0, expected_gpu_uuid="GPU-synthetic", server_port=19500,
                               accept_eula=True, scene_load_timeout=1800, max_scene_loads=1,
                               vulkan_icd=str(icd), mdl_root=str(mdl)))
    for name in ("handoff.json", "prepared.json"):
        (tmp_path / name).write_text("{}")
    inputs = dict(instruction_id=public["runtime_instruction_id"], handoff_json=str(tmp_path / "handoff.json"),
                  prepared_index_json=str(tmp_path / "prepared.json"), model_task=model)
    monkeypatch.setattr(runtime, "_official_identity", lambda path: runtime.OFFICIAL_REVISION)
    plan = runtime.runtime_plan(config, inputs)
    assert not output.exists()
    assert plan["command"][0] == str(environment / "bin/python")
    assert plan["runtime_root"] == str(output / "w")
    assert plan["scene_root"] == str(output / "w/scene")
    assert "EXAMPLE_TEST_KEY" not in plan["env"]
    with pytest.raises(ValueError, match="materialized"):
        runtime.launch_runtime(config, inputs)
    assert not output.exists()
    monkeypatch.setattr(runtime, "verify_runtime_sources", lambda *args: pytest.fail("read runtime before checking task"))
    with pytest.raises(ValueError, match="instruction"):
        runtime.runtime_plan(config, {**inputs, "model_task": {**model, "instruction": "changed"}})
    with pytest.raises(ValueError, match="selection"):
        runtime.runtime_plan({**config, "run": {**config["run"], "task_id": "other"}}, inputs)


def test_scene_launch_strips_all_supported_provider_credentials(tmp_path, monkeypatch):
    from esi_benchmark.profiles import DEFAULT_PROFILE
    workspace = tmp_path / "owned"
    plan = {"workspace": str(workspace), "command": ["synthetic-not-executed"], "env": {"PYTHONDONTWRITEBYTECODE": "1"}}
    observed = {}
    credentials = ("OPENAI_API_KEY", "DASHSCOPE_API_KEY", "TOKEN_PLAN_API_KEY",
                   "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CUSTOM_SELECTED_KEY")
    for key in credentials:
        monkeypatch.setenv(key, "synthetic-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://unused.invalid")
    monkeypatch.setenv("ESI_TEST_PRIVATE_OVERRIDE", "private-state")
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setattr(runtime, "runtime_plan", lambda *args: plan)
    monkeypatch.setattr(runtime, "_verify_materialized_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "MDL_SHA256", {})
    def spawn(command, **kwargs):
        observed.update(command=command, **kwargs)
        return object()
    monkeypatch.setattr(runtime.subprocess, "Popen", spawn)
    runtime.launch_runtime({"profile": DEFAULT_PROFILE, "runtime": {}, "run": {"task_id": "task"},
                            "model": {"api_key_env": "CUSTOM_SELECTED_KEY"}},
        {"handoff_json": "unused", "prepared_index_json": "unused", "instruction_id": "task", "model_task": {}})
    assert not set(credentials) & observed["env"].keys()
    assert "ANTHROPIC_BASE_URL" not in observed["env"]
    assert "ESI_TEST_PRIVATE_OVERRIDE" not in observed["env"]
    assert observed["env"]["LANG"] == "C.UTF-8"
    assert observed["start_new_session"] is True
    assert "synthetic-secret" not in (workspace / "launch_plan.json").read_text()
