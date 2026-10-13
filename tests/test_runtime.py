import hashlib
import json
from pathlib import Path
import socket
import sys

import pytest

from esi_benchmark import runtime
from esi_benchmark.bootstrap import REVISIONS, activate


def test_source_identity_is_upstream_plus_candidate_hashes():
    report = runtime.verify_runtime_sources()
    assert report["upstream_revision"] == REVISIONS["runtime"]
    assert report["files_verified"] >= 291
    assert report["candidate_adaptations"]
    from behavior_hl.embodied.evaluator_bridge.benchmark_execution_contract import runner_identity
    identity = runner_identity(activate()["runtime_root"])
    assert identity["commit"] == REVISIONS["runtime"]
    assert identity["candidate_source_manifest_sha256"]


def test_materialized_input_identity_and_no_path_escape(tmp_path):
    inputs = dict(instruction_id="task", handoff_json=str(tmp_path/"handoff.json"),
                  prepared_index_json=str(tmp_path/"prepared.json"))
    with pytest.raises(ValueError, match="materialized"):
        runtime._verify_materialized_inputs(inputs)
    files = []
    for filename in ("handoff.json", "prepared.json"):
        path = tmp_path / filename
        path.write_text("{}")
        files.append(dict(derived_path=filename, derived_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    receipt = tmp_path / "materialization_receipt.json"
    receipt.write_text(json.dumps(dict(instruction_id="task", files=files)))
    inputs["materialization_receipt"] = str(receipt)
    runtime._verify_materialized_inputs(inputs)
    (tmp_path / "prepared.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed"):
        runtime._verify_materialized_inputs(inputs)


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


def test_plan_is_read_only_and_launch_keeps_virtualenv_path(tmp_path, monkeypatch):
    from esi_benchmark.bootstrap import PROFILE
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
                  assets=dict(root=str(data)), run=dict(task_id="example", family="L0", output_root=str(output), history_frames=20),
                  model=dict(endpoint="https://example.invalid/v1", model="example", api_key_env="EXAMPLE_TEST_KEY"),
                  runtime=dict(workspace=str(output / "w"), python=str(environment / "bin/python"),
                               behavior_root=str(behavior), data_root=str(data), toolchain_root=str(environment),
                               overlay_root=str(environment), gpu=0, expected_gpu_uuid="GPU-synthetic", server_port=19500,
                               accept_eula=True, scene_load_timeout=1800, max_scene_loads=1,
                               vulkan_icd=str(icd), mdl_root=str(mdl)))
    for name in ("handoff.json", "prepared.json"):
        (tmp_path / name).write_text("{}")
    inputs = dict(instruction_id="example", handoff_json=str(tmp_path / "handoff.json"),
                  prepared_index_json=str(tmp_path / "prepared.json"), model_task=dict(task_id=0, task_seed=19506))
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
