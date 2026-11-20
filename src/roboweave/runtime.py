"""Owned, one-scene Live launch using the original prepared-state entry point.

Planning is CPU-only. Only launch_runtime (or the private child command) can
start a process, and no function attaches to an existing resident worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import shutil
import stat
import subprocess
import sys
from typing import Any, Mapping

from .profiles import DEFAULT_PROFILE, source_roots, source_revisions, verify_sources

OFFICIAL_REVISION = "802356a83d210a42bb6bfde93544bda4e33124e1"
MDL_SHA256 = {
    "vray_maps.mdl": "279d9e8ded03c2db928033c843b40a04b104f3fd6fc7ccd284aadc6459a2aa1c",
    "vray_materials.mdl": "eac4af88c7e39cdba091bf2e76c20adc3f9c9cd65d6d044479065537f70de696",
}


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def verify_runtime_sources(profile=DEFAULT_PROFILE) -> dict[str, Any]:
    report = verify_sources(profile)
    root = source_roots(profile)["runtime_root"]
    identity = json.loads((root / "SOURCE_IDENTITY.json").read_text())
    return {"upstream_revision": identity["upstream_revision"], "files_verified": len(identity["files"]),
            "candidate_adaptations": identity["candidate_adaptations"], "profile": profile,
            "source_verification": report}


def _file(path, label):
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise ValueError(f"{label} is not a file: {result}")
    return result


def _directory(path, label):
    result = Path(path).expanduser().resolve()
    if not result.is_dir():
        raise ValueError(f"{label} is not a directory: {result}")
    return result


def _official_identity(root: Path):
    git = ["git", "--no-optional-locks", "-C", str(root)]
    completed = subprocess.run(git + ["rev-parse", "--show-toplevel"], check=True,
                               capture_output=True, text=True, timeout=15)
    if Path(completed.stdout.strip()).resolve() != root:
        raise ValueError("external BEHAVIOR directory must be its own frozen checkout")
    revision = subprocess.run(git + ["rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True, timeout=15).stdout.strip()
    if revision != OFFICIAL_REVISION:
        raise ValueError("external BEHAVIOR checkout does not match the frozen dependency revision")
    dirty = subprocess.run(git + ["status", "--porcelain=v1", "--untracked-files=normal"], check=True,
                           capture_output=True, text=True, timeout=15).stdout
    if dirty.strip():
        raise ValueError("external BEHAVIOR checkout has tracked or untracked changes")
    return revision


def _launch_inputs(task_inputs):
    entrypoint = task_inputs.get("entrypoint", "prepared")
    entry_fields = _entry_files(entrypoint)
    required = (*entry_fields, "model_task", "instruction_id")
    if any(key not in task_inputs for key in required):
        raise ValueError("task inputs lack the selected entrypoint files or model identity")
    fields = {key for key in task_inputs if key.endswith(("_json", "_jsonl"))}
    fields.update({"profile", "entrypoint", "model_task", "instruction_id", "materialization_receipt",
                   "source_task_identity", "scene_model", "action_mode", "public_config_id", "native_args"})
    return {**{key: _jsonable(task_inputs[key]) for key in fields if key in task_inputs},
            "entrypoint": entrypoint, "materialization_receipt": task_inputs.get("materialization_receipt")}


def _entry_files(entrypoint):
    if entrypoint in {"prepared", "physical_collection"}:
        return ("handoff_json", "prepared_index_json")
    if entrypoint == "v_collection":
        return ("v_collection_binding_json",)
    raise ValueError(f"Unsupported frozen scene entrypoint: {entrypoint}")


def runtime_plan(config: Mapping[str, Any], task_inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Validate local files and return an explicit launch plan without writes."""
    from .config import validate_config
    validate_config(dict(config))
    if config["mode"] != "live":
        raise ValueError("runtime launch requires mode=live")
    from .tasks import validate_model_task
    selected = validate_model_task(config["profile"], task_inputs["model_task"])
    if (config["run"]["task_id"], config["run"]["family"]) != (selected["task_id"], selected["family"]):
        raise ValueError("Runtime task selection differs from its frozen model binding")
    roots = source_roots(config["profile"])
    source = verify_runtime_sources(config["profile"])
    runtime = config["runtime"]
    workspace = Path(runtime["workspace"]).resolve()
    output_root = Path(config["run"]["output_root"]).resolve()
    if workspace.exists() or workspace.is_symlink():
        raise ValueError("runtime.workspace must be a fresh, independently owned directory")
    if not workspace.is_relative_to(output_root) or workspace == output_root:
        raise ValueError("runtime.workspace must be a fresh child of run.output_root")
    _file(runtime["python"], "runtime.python")
    python = Path(runtime["python"]).expanduser().absolute()
    if not os.access(python, os.X_OK):
        raise ValueError("runtime.python is not executable")
    overlay = _directory(runtime["overlay_root"], "runtime.overlay_root")
    if python.resolve() != (overlay / "bin/python").resolve():
        raise ValueError("runtime.python must resolve to overlay_root/bin/python")
    behavior = _directory(runtime["behavior_root"], "runtime.behavior_root")
    data = _directory(runtime["data_root"], "runtime.data_root")
    toolchain = _directory(runtime["toolchain_root"], "runtime.toolchain_root")
    _file(toolchain / "bin/nvcc", "CUDA compiler")
    assets = _directory(config["assets"]["root"], "assets.root")
    for root in (*roots.values(), behavior, data, toolchain, overlay, assets):
        if workspace.is_relative_to(root) or root.is_relative_to(workspace):
            raise ValueError("runtime.workspace overlaps a read-only code, environment or asset root")
    official_revision = _official_identity(behavior)
    for component in ("OmniGibson/omnigibson", "bddl3/bddl", "joylo"):
        _directory(behavior / component, component)
    for relative, expected in (("behavior-1k-assets/VERSION", "3.9.0"),
                               ("omnigibson-robot-assets/VERSION", "3.8.2")):
        if _file(data / relative, relative).read_text().strip() != expected:
            raise ValueError(f"external asset version differs: {relative}")
    _directory(data / "2026-challenge-task-instances", "task instances")
    mdl_root = _directory(runtime["mdl_root"], "runtime.mdl_root")
    for filename, expected in MDL_SHA256.items():
        path = _file(mdl_root / filename, filename)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"licensed external material module changed: {filename}")
    icd = runtime.get("vulkan_icd")
    if icd is None:
        icd = next((p for p in ("/usr/share/vulkan/icd.d/nvidia_icd.json",
                               "/etc/vulkan/icd.d/nvidia_icd.json") if Path(p).is_file()), None)
    if icd is None:
        raise ValueError("NVIDIA Vulkan driver ICD is missing; set runtime.vulkan_icd")
    icd = _file(icd, "runtime.vulkan_icd")
    launch_inputs = _launch_inputs(task_inputs)
    for key in _entry_files(launch_inputs["entrypoint"]):
        _file(launch_inputs[key], key)
    task = launch_inputs["model_task"]
    if type(task.get("task_id")) is not int or type(task.get("task_seed")) is not int:
        raise ValueError("frozen public task handle and seed must be integers")
    scene_root = workspace / "scene"
    if len(os.fsencode(scene_root / "s")) > 107:
        raise ValueError("Owned scene socket path exceeds the Unix socket address limit")
    paths = [Path(__file__).resolve().parents[1], roots["bench_root"], roots["bench_root"] / "lab",
             roots["runtime_root"] / "src", roots["runtime_root"] / "scripts",
             behavior / "OmniGibson", behavior / "bddl3", behavior / "joylo"]
    env = {"PYTHONPATH": os.pathsep.join(map(str, paths)), "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONHASHSEED": str(task["task_seed"]), "BEHAVIOR_HL_PROJECT_ROOT": str(roots["runtime_root"]),
           "BEHAVIOR_1K_PROJECT_ROOT": str(behavior), "ESI_BEHAVIOR_ROOT": str(behavior),
           "ESI_DATA_ROOT": str(data), "ESI_OVERLAY_ROOT": str(overlay),
           "ESI_TOOLCHAIN_ROOT": str(toolchain), "ESI_MODEL_PYTHON": str(python),
           "ESI_VULKAN_ICD_PATH": str(icd), "CUDA_HOME": str(toolchain),
           "ESI_RUNTIME_MDL_PATH": str(workspace / "materials/behavior_hl_vray_roughness.mdl"),
           "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "OMNIGIBSON_GPU_ID": str(runtime["gpu"]),
           "OMNIGIBSON_DATA_PATH": str(data), "BEHAVIOR2026_LOCAL_DATA_ROOT": str(data),
           "OMNIGIBSON_APPDATA_PATH": str(workspace / "appdata"),
           "BEHAVIOR2026_LOCAL_APPDATA_PATH": str(workspace / "appdata"),
           "OMNIGIBSON_HEADLESS": "1", "OMNI_KIT_ACCEPT_EULA": "YES",
           "TORCHDYNAMO_DISABLE": "1", "TORCHINDUCTOR_DISABLE": "1",
           "TMPDIR": str(workspace / "tmp"), "XDG_CACHE_HOME": str(workspace / "cache"),
           "XDG_CONFIG_HOME": str(workspace / "config"), "XDG_DATA_HOME": str(workspace / "data"),
           "XDG_RUNTIME_DIR": str(workspace / "xdg"),
           "CUDA_CACHE_PATH": str(workspace / "cache/cuda"), "TORCH_HOME": str(workspace / "cache/torch"),
           "HF_HOME": str(workspace / "cache/huggingface"), "NUMBA_CACHE_DIR": str(workspace / "cache/numba"),
           "MPLCONFIGDIR": str(workspace / "config/matplotlib"), "NO_PROXY": "127.0.0.1,localhost",
           "no_proxy": "127.0.0.1,localhost", "VK_ICD_FILENAMES": str(icd), "VK_DRIVER_FILES": str(icd)}
    if config["profile"] == DEFAULT_PROFILE:
        for flag in ("ROBOWEAVE_EXPERIMENTAL_WIPE", "ROBOWEAVE_EXPERIMENTAL_HOUSEHOLD_ACTIONS"):
            if os.environ.get(flag) == "1":
                env[flag] = "1"
    if runtime.get("host_dlsym_shim"):
        shim = str(_file(runtime["host_dlsym_shim"], "runtime.host_dlsym_shim"))
        env.update(ESI_HOST_DLSYM_SHIM=shim, LD_PRELOAD=shim)
    payload = {"profile": config["profile"], "runtime": _jsonable(runtime), "task_inputs": launch_inputs}
    command = [str(python), "-m", "roboweave.runtime", "--launch-json",
               json.dumps(payload, separators=(",", ":"), allow_nan=False)]
    return {"command": command, "env": env, "workspace": str(workspace),
            "runtime_root": str(workspace), "scene_root": str(scene_root), "ready_path": str(scene_root / "ready.json"),
            "preflight": {"runtime": source, "official_revision": official_revision,
                          "gpu_uuid_check": "deferred-to-explicit-launch", "scene_load_budget": 1,
                          "live_verified": False}}


def build_runtime_command(config, task_inputs) -> list[str]:
    return runtime_plan(config, task_inputs)["command"]


def launch_runtime(config, task_inputs) -> subprocess.Popen:
    """Create one owned workspace and process; never retry a scene load."""
    plan = runtime_plan(config, task_inputs)
    _verify_materialized_inputs(_launch_inputs(task_inputs), profile=config["profile"], task_id=config["run"]["task_id"])
    workspace = Path(plan["workspace"])
    workspace.mkdir(parents=True, mode=0o700, exist_ok=False)
    for name in ("appdata", "tmp", "cache", "config", "data", "xdg", "materials"):
        (workspace / name).mkdir(mode=0o700)
    for filename in MDL_SHA256:
        shutil.copyfile(Path(config["runtime"]["mdl_root"]) / filename, workspace / "materials" / filename)
    shutil.copyfile(source_roots(config["profile"])["runtime_root"] / "configs/materials/behavior_hl_vray_roughness.mdl",
                    workspace / "materials/behavior_hl_vray_roughness.mdl")
    stripped = {"CUDA_VISIBLE_DEVICES", "VCUDA_PORT", "LOCAL_HOST_IP", "PYTHONPATH", "LD_PRELOAD",
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
                "TOKEN_PLAN_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                "TOKEN_PLAN_BASE_URL", "DASHSCOPE_BASE_URL", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"}
    credential = config.get("model", {}).get("api_key_env")
    env = {key: value for key, value in os.environ.items()
           if key not in stripped and key != credential and not key.startswith(("BEHAVIOR_HL_", "ESI_", "ROBOWEAVE_", "OMNIGIBSON_", "BEHAVIOR2026_"))}
    env.update(plan["env"])
    (workspace / "launch_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    with (workspace / "launcher.log").open("xb") as log:
        return subprocess.Popen(plan["command"], env=env, cwd=workspace, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)



def validate_owned_ready(plan, process_pid: int) -> dict[str, Any]:
    """Validate readiness on disk before any connection to the scene socket."""
    workspace = Path(plan["workspace"]).resolve()
    scene = workspace / "scene"
    ready_path = Path(plan["ready_path"])
    if ready_path != scene / "ready.json" or scene.is_symlink() or ready_path.is_symlink():
        raise ValueError("ready path does not belong to the owned scene")
    if not ready_path.is_file() or ready_path.stat().st_size > 64 * 1024:
        raise ValueError("owned ready record is missing or oversized")
    ready = json.loads(ready_path.read_text())
    expected_socket = scene / "s"
    if (ready.get("socket_path") != str(expected_socket) or expected_socket.is_symlink()
            or not stat.S_ISSOCK(expected_socket.stat().st_mode)):
        raise ValueError("ready socket does not belong to the owned scene")
    run_dir = Path(ready["run_dir"])
    if not run_dir.resolve().is_relative_to(workspace) or not run_dir.is_dir():
        raise ValueError("ready run directory escaped the owned workspace")
    if ready.get("pid") != process_pid:
        raise ValueError("ready process identity differs from the launched worker")
    return ready



def _verify_materialized_inputs(inputs, *, profile, task_id=None):
    receipt = inputs.get("materialization_receipt")
    if not receipt:
        raise ValueError("Live requires independently materialized and hashed task inputs")
    path = _file(receipt, "materialization receipt")
    root = path.parent
    report = json.loads(path.read_text())
    from .tasks import validate_model_task
    selected = validate_model_task(profile, inputs["model_task"])
    if (report.get("profile") != profile or inputs.get("profile", profile) != profile
            or report.get("task_id") != selected["task_id"]
            or (task_id is not None and task_id != selected["task_id"])
            or inputs["instruction_id"] != selected["runtime_instruction_id"]):
        raise ValueError("Materialized receipt differs from its frozen task or profile")
    if report.get("instruction_id") != inputs["instruction_id"]:
        raise ValueError("materialized input selection changed")
    members = set()
    for row in report["files"]:
        member = (root / row["derived_path"]).resolve()
        if not member.is_relative_to(root) or not member.is_file():
            raise ValueError("materialized input escaped the owned input root")
        if hashlib.sha256(member.read_bytes()).hexdigest() != row["derived_sha256"]:
            raise ValueError("materialized input changed after preparation")
        members.add(member)
    if any(Path(inputs[key]).resolve() not in members for key in _entry_files(inputs.get("entrypoint", "prepared"))):
        raise ValueError("launcher input is absent from its materialization receipt")


def _verify_launch_device(runtime):
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=15)
    devices = {}
    for line in result.stdout.splitlines():
        index, uuid = (part.strip() for part in line.split(",", 1))
        devices[int(index)] = uuid
    if devices.get(runtime["gpu"]) != runtime["expected_gpu_uuid"]:
        raise ValueError("physical GPU index no longer matches the explicitly owned UUID")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", runtime["server_port"]))


def _main(argv=None):
    parser = argparse.ArgumentParser(description="Private owned scene launcher; use roboweave run")
    parser.add_argument("--launch-json", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.launch_json)
    runtime, inputs = payload["runtime"], payload["task_inputs"]
    from ._native.bootstrap import activate
    activate(payload["profile"])
    verify_runtime_sources(payload["profile"])
    _verify_materialized_inputs(inputs, profile=payload["profile"])
    if sys.version_info[:2] != (3, 11):
        raise ValueError("Live requires the frozen Python 3.11 environment")
    if Path(sys.prefix).resolve() != Path(runtime["overlay_root"]).resolve():
        raise ValueError("Live interpreter did not activate the configured isolated environment")
    from importlib import metadata, util
    # The official 5.1.0 release reports 5.1.0.0 in its package metadata.
    if metadata.version("isaacsim") not in ("5.1.0", "5.1.0.0"):
        raise ValueError("Live requires the frozen Isaac Sim 5.1.0 dependency")
    for module, expected in (("omnigibson", "OmniGibson/omnigibson"), ("bddl", "bddl3/bddl")):
        spec = util.find_spec(module)
        if spec is None or not spec.origin or not Path(spec.origin).resolve().is_relative_to(Path(runtime["behavior_root"]) / expected):
            raise ValueError(f"{module} resolves outside the configured frozen checkout")
    for module in ("curobo", "torch", "warp"):
        if util.find_spec(module) is None:
            raise ValueError(f"Live dependency is missing: {module}")
    _verify_launch_device(runtime)
    from .adapters import inspect_inputs
    from package_scene_server import main as package_main
    inputs = inspect_inputs(payload["profile"], inputs)
    task = inputs["model_task"]
    arguments = ["--instruction-id", inputs["instruction_id"], "--public-task-id", str(task["task_id"]),
        "--runtime-root", str(Path(runtime["workspace"]) / "scene"),
        "--worker-id", "public-single-scene", "--physical-gpu", str(runtime["gpu"]),
        "--server-port", str(runtime["server_port"]), "--seed", str(task["task_seed"]),
        "--artifact-root", str(Path(runtime["workspace"]) / "scene/simulator")]
    if inputs["entrypoint"] == "v_collection":
        arguments += ["--v-collection-binding", inputs["v_collection_binding_json"],
                      "--public-config-id", inputs["public_config_id"],
                      "--registered-public-action20-marker-se2-navigation"]
    else:
        arguments += ["--handoff", inputs["handoff_json"], "--prepared-state-index", inputs["prepared_index_json"],
                      "--task-names", inputs["instruction_id"], *inputs["native_args"]]
    return package_main(arguments)


if __name__ == "__main__":
    raise SystemExit(_main())
