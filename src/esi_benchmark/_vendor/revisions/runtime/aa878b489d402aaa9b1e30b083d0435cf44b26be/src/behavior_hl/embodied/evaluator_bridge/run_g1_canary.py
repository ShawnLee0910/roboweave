"""Run the one-load GPU-0 G1 RGB-D hold-action canary."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import multiprocessing
import os
import sys
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from behavior_hl.embodied.policy.hold_server import server_process_main
from behavior_hl.embodied.policy.observations import schema_sha256
from behavior_hl.embodied.policy.profiles import (
    build_picking_up_toys_registry,
)
from behavior_hl.project_identity import (
    BUNDLE_ROOT,
    PROJECT_ROOT,
    PROJECT_ROOT_ENV,
    bind_project_root_environment,
    capture_project_identity,
)


BEHAVIOR_ROOT = Path(os.environ.get("ESI_BEHAVIOR_ROOT", str(BUNDLE_ROOT / "BEHAVIOR-1K-2026")))
DATA_ROOT = Path(os.environ.get("ESI_DATA_ROOT", str(BUNDLE_ROOT / "BEHAVIOR-1K-2026-data")))
CONFIG_PATH = PROJECT_ROOT / "configs" / "picking_up_toys_r1pro_hla.yaml"
OVERLAY_ROOT = Path(os.environ.get("ESI_OVERLAY_ROOT", sys.prefix))
TOOLCHAIN_ROOT = Path(os.environ.get("ESI_TOOLCHAIN_ROOT", str(BUNDLE_ROOT / "cuda-toolchain")))
HOST_DLSYM_SHIM = (Path(os.environ["ESI_HOST_DLSYM_SHIM"])
                   if os.environ.get("ESI_HOST_DLSYM_SHIM") else None)
LOCAL_DATA_ROOT_ENV = "BEHAVIOR2026_LOCAL_DATA_ROOT"
LOCAL_APPDATA_PATH_ENV = "BEHAVIOR2026_LOCAL_APPDATA_PATH"
OFFICIAL_WRAPPER = "omnigibson.eval.wrappers.RGBDFullResWrapper"
EFFECTIVE_WRAPPER = (
    "behavior_hl.embodied.evaluator_bridge.camera_only_rgbd_wrapper."
    "CameraOnlyRGBDFullResWrapper"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_aggregate(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=float)
        stream.write("\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, default=float) + "\n")


def _record_stage(path: Path, stage: str, **details: Any) -> None:
    _append_jsonl(
        path,
        {
            "stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **details,
        },
    )


def _wait_for_port(host: str, port: int, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        connection = http.client.HTTPConnection(host, port, timeout=0.5)
        try:
            connection.request("GET", "/healthz")
            response = connection.getresponse()
            if response.status == 200:
                return
        except (OSError, http.client.HTTPException):
            time.sleep(0.1)
        finally:
            connection.close()
    raise TimeoutError(f"policy server did not listen on {host}:{port}")


def _to_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def _norm_delta(first: list[float], second: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _configure_process_environment(
    run_dir: Path,
    *,
    physical_gpu: int = 0,
) -> None:
    # The BEHAVIOR 2026 launch contract addresses the physical GPU directly.
    # Masking it changes Isaac/Kit's Vulkan-CUDA device mapping.
    if (
        isinstance(physical_gpu, bool)
        or not isinstance(physical_gpu, int)
        or physical_gpu < 0
    ):
        raise ValueError("physical_gpu must be a non-negative integer")
    bind_project_root_environment()
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.pop("VCUDA_PORT", None)
    os.environ.pop("LOCAL_HOST_IP", None)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["OMNIGIBSON_GPU_ID"] = str(physical_gpu)
    local_data_root = os.environ.get(LOCAL_DATA_ROOT_ENV)
    if local_data_root:
        resolved_data_root = Path(local_data_root).expanduser().resolve()
        if not resolved_data_root.is_dir():
            raise FileNotFoundError(
                f"{LOCAL_DATA_ROOT_ENV} is not a directory: "
                f"{resolved_data_root}"
            )
        os.environ["OMNIGIBSON_DATA_PATH"] = str(resolved_data_root)
    else:
        os.environ["OMNIGIBSON_DATA_PATH"] = str(DATA_ROOT)

    local_appdata_path = os.environ.get(LOCAL_APPDATA_PATH_ENV)
    if local_appdata_path:
        resolved_appdata_path = Path(local_appdata_path).expanduser().resolve()
        resolved_appdata_path.mkdir(parents=True, exist_ok=True)
    else:
        resolved_appdata_path = (
            run_dir.parent / f"_shared_kit_appdata_gpu{physical_gpu}"
        ).resolve()
    os.environ["OMNIGIBSON_APPDATA_PATH"] = str(resolved_appdata_path)
    os.environ["OMNIGIBSON_HEADLESS"] = "1"
    if os.environ.get("OMNI_KIT_ACCEPT_EULA") != "YES":
        raise ValueError("explicit Isaac Sim EULA acceptance is required before launch")
    os.environ["OG_DISABLE_EMITTER_APIS"] = "1"
    os.environ["CUDA_HOME"] = str(TOOLCHAIN_ROOT)
    os.environ["VK_ICD_FILENAMES"] = os.environ.get("ESI_VULKAN_ICD_PATH", "/etc/vulkan/icd.d/nvidia_icd.json")
    os.environ["VK_DRIVER_FILES"] = os.environ["VK_ICD_FILENAMES"]
    os.environ["NVIDIA_DRIVER_CAPABILITIES"] = os.environ.get(
        "NVIDIA_DRIVER_CAPABILITIES",
        "all",
    )
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["TORCHINDUCTOR_DISABLE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    isolation = run_dir / "kit_isolation"
    runtime = isolation / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    runtime.chmod(0o700)
    for variable, leaf in (
        ("XDG_CACHE_HOME", "cache"),
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"),
    ):
        path = isolation / leaf
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["XDG_RUNTIME_DIR"] = str(runtime)

    component_roots = (
        BEHAVIOR_ROOT / "OmniGibson",
        BEHAVIOR_ROOT / "bddl3",
        BEHAVIOR_ROOT / "joylo",
    )
    for component_root in reversed(component_roots):
        sys.path.insert(0, str(component_root))


def _validate_launch_contract(
    *,
    physical_gpu: int = 0,
) -> dict[str, Any]:
    errors: list[str] = []
    project_identity = capture_project_identity(
        expected_root=PROJECT_ROOT,
        module_files={"run_g1_canary": Path(__file__)},
        require_environment=True,
    )
    expected_python = OVERLAY_ROOT / "bin" / "python"
    if Path(sys.executable).resolve() != expected_python.resolve():
        errors.append(
            f"expected isolated overlay Python {expected_python}, "
            f"got {sys.executable}"
        )
    if HOST_DLSYM_SHIM is not None and not HOST_DLSYM_SHIM.is_file():
        errors.append(f"missing host compatibility shim: {HOST_DLSYM_SHIM}")
    preload_entries = {
        Path(entry).resolve()
        for entry in os.environ.get("LD_PRELOAD", "").split(os.pathsep)
        if entry
    }
    if HOST_DLSYM_SHIM is not None and HOST_DLSYM_SHIM.resolve() not in preload_entries:
        errors.append(
            "host compatibility shim was not preloaded before Python startup"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        errors.append(
            "CUDA_VISIBLE_DEVICES must be unset for physical GPU "
            "Vulkan/CUDA alignment"
        )
    if os.environ.get("OMNIGIBSON_GPU_ID") != str(physical_gpu):
        errors.append(
            f"OMNIGIBSON_GPU_ID must be {physical_gpu}"
        )
    if os.environ.get("TORCHINDUCTOR_DISABLE") != "1":
        errors.append("TORCHINDUCTOR_DISABLE must be 1")
    if os.environ.get("CUDA_HOME") != str(TOOLCHAIN_ROOT):
        errors.append(f"CUDA_HOME must be {TOOLCHAIN_ROOT}")
    report = {
        "passed": not errors,
        "errors": errors,
        "python": sys.executable,
        "overlay": str(OVERLAY_ROOT),
        "toolchain": str(TOOLCHAIN_ROOT),
        "host_dlsym_shim": None if HOST_DLSYM_SHIM is None else str(HOST_DLSYM_SHIM),
        "host_dlsym_shim_sha256": (
            _sha256(HOST_DLSYM_SHIM)
            if HOST_DLSYM_SHIM is not None and HOST_DLSYM_SHIM.is_file()
            else None
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omnigibson_gpu_id": os.environ.get("OMNIGIBSON_GPU_ID"),
        "omnigibson_data_path": os.environ.get("OMNIGIBSON_DATA_PATH"),
        "omnigibson_appdata_path": os.environ.get("OMNIGIBSON_APPDATA_PATH"),
        "local_data_root_override": os.environ.get(LOCAL_DATA_ROOT_ENV),
        "local_appdata_path_override": os.environ.get(LOCAL_APPDATA_PATH_ENV),
        "torchinductor_disable": os.environ.get("TORCHINDUCTOR_DISABLE"),
        "project_root": str(PROJECT_ROOT.resolve()),
        "project_root_environment": os.environ.get(PROJECT_ROOT_ENV),
        "project_identity": project_identity,
    }
    if errors:
        raise RuntimeError(f"invalid G1 launch contract: {errors}")
    return report


def _load_and_validate_config() -> dict[str, Any]:
    import yaml

    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    registry = build_picking_up_toys_registry()
    checks = {
        "schema_version": (
            config["schema_version"],
            registry.schema_version,
        ),
        "task.name": (config["task"]["name"], "picking_up_toys"),
        "task.b100_id": (config["task"]["b100_id"], 7),
        "runtime.gpu": (config["runtime"]["gpu"], 0),
        "runtime.wrapper": (
            config["runtime"]["wrapper"],
            "omnigibson.eval.wrappers.RGBDFullResWrapper",
        ),
        "runtime.max_steps_override": (
            config["runtime"]["max_steps_override"],
            None,
        ),
        "observation.proprio_dim": (
            config["observation"]["proprio_dim"],
            61,
        ),
        "action.dimension": (config["action"]["dimension"], 23),
        "action.enabled": (
            set(config["action"]["enabled"]),
            set(registry.names),
        ),
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in checks.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(f"G1 config mismatch: {mismatches}")
    return config


def _validate_video(path: Path) -> dict[str, Any]:
    import av

    frames = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        width = stream.codec_context.width
        height = stream.codec_context.height
        for _ in container.decode(video=0):
            frames += 1
    return {
        "exists": path.is_file(),
        "bytes": path.stat().st_size,
        "frames": frames,
        "width": width,
        "height": height,
        "sha256": _sha256(path),
    }


def run_canary(
    *,
    steps: int | None,
    host: str | None,
    port: int | None,
) -> Path:
    run_config = _load_and_validate_config()
    steps = (
        int(run_config["g1"]["simulator_steps"])
        if steps is None
        else steps
    )
    host = (
        str(run_config["g1"]["server_host"])
        if host is None
        else host
    )
    port = (
        int(run_config["g1"]["server_port"])
        if port is None
        else port
    )
    if steps <= 0:
        raise ValueError("G1 simulator steps must be positive")
    artifact_root = Path(run_config["runtime"]["artifact_root"])
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"_g1_{os.getpid()}"
    )
    run_dir = artifact_root / run_id
    for relative in ("private", "policy", "video", "logs", "canary"):
        (run_dir / relative).mkdir(parents=True, exist_ok=False)

    _configure_process_environment(run_dir)
    stage_path = run_dir / "canary" / "stages.jsonl"
    _record_stage(stage_path, "RUN_DIRECTORY_READY")
    launch_contract = _validate_launch_contract()
    _write_json(run_dir / "private" / "launch_contract.json", launch_contract)
    _record_stage(stage_path, "ZERO_SCENE_LAUNCH_CONTRACT_PASS")
    registry = build_picking_up_toys_registry()
    initial_manifest = {
        "run_id": run_id,
        "gate": "G1",
        "status": "RUNNING",
        "task": "picking_up_toys",
        "mode": "train",
        "instance_id": 0,
        "simulator_seed": 0,
        "gpu": 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omnigibson_gpu_id": os.environ["OMNIGIBSON_GPU_ID"],
        "gpu_binding_mode": "physical_unmasked_single_icd",
        "python": sys.executable,
        "scene_load_budget": 1,
        "scene_loads_attempted": 0,
        "max_steps_override": None,
        "wrapper": run_config["runtime"]["wrapper"],
        "effective_wrapper": EFFECTIVE_WRAPPER,
        "write_video": True,
        "config_sha256": _sha256(CONFIG_PATH),
        "registry_sha256": registry.sha256(),
        "observation_schema_sha256": schema_sha256(),
        "embodied_source_sha256": _source_aggregate(
            PROJECT_ROOT / "src" / "behavior_hl" / "embodied"
        ),
        "viewer_camera_control": False,
        "teleport_enabled": False,
        "privileged_policy_inputs": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_dir / "manifest.json", initial_manifest)

    _record_stage(stage_path, "POLICY_SERVER_STARTING")
    context = multiprocessing.get_context("spawn")
    ready_queue = context.Queue()
    server_process = context.Process(
        target=server_process_main,
        args=(
            host,
            port,
            str(run_dir / "policy"),
            float(run_config["observation"]["minimum_depth_finite_ratio"]),
            ready_queue,
        ),
        daemon=False,
    )
    server_process.start()
    scene_loads_attempted = 0
    evaluator_context = None
    try:
        actual_port = int(ready_queue.get(timeout=20))
        _wait_for_port(host, actual_port)
        _record_stage(
            stage_path,
            "POLICY_SERVER_READY",
            host=host,
            port=actual_port,
        )

        _record_stage(stage_path, "OFFICIAL_RUNTIME_IMPORT_START")
        from omegaconf import OmegaConf
        from omnigibson.eval.evaluator import Evaluator
        from omnigibson.eval.utils.eval_utils import seed_everything
        from omnigibson.macros import gm
        expected_appdata = Path(
            os.environ["OMNIGIBSON_APPDATA_PATH"]
        ).resolve()
        if Path(gm.APPDATA_PATH).resolve() != expected_appdata:
            raise RuntimeError(
                "OmniGibson appdata escaped the isolated artifact directory: "
                f"{gm.APPDATA_PATH} != {expected_appdata}"
            )
        _record_stage(stage_path, "OFFICIAL_RUNTIME_IMPORT_PASS")

        gm.HEADLESS = True
        seed_everything(0)
        cfg = OmegaConf.create(
            {
                "env_wrapper": {
                    "_target_": EFFECTIVE_WRAPPER
                },
                "policy_name": "behavior_hl_g1_whitelisted_websocket",
                "model": {
                    "_target_": (
                        "behavior_hl.embodied.evaluator_bridge."
                        "whitelist_policy.WhitelistedWebsocketPolicy"
                    ),
                    "host": host,
                    "port": actual_port,
                    "allow_reconnect": False,
                },
                "headless": True,
                "partial_scene_load": run_config["runtime"][
                    "partial_scene_load"
                ],
                "max_steps": run_config["runtime"]["max_steps_override"],
                "write_video": run_config["runtime"]["write_video"],
                "mode": "train",
                "seed": 0,
                "task": {"name": "picking_up_toys"},
                "robot": None,
            }
        )

        video_path = run_dir / "video" / "official_rollout.mp4"
        terminated = False
        truncated = False
        scene_loads_attempted = 1
        _record_stage(
            stage_path,
            "SCENE_CONSTRUCTION_START",
            scene_load_number=1,
        )
        # Evaluator.__exit__ calls og.shutdown(), which can terminate this
        # process before statements after a normal with-block are executed.
        # Enter it explicitly and defer __exit__ until the outer finally so
        # the gate result and final manifest are durable first.
        evaluator_context = Evaluator(cfg)
        with nullcontext(evaluator_context.__enter__()) as evaluator:
            from omnigibson.eval.wrappers import RGBDFullResWrapper

            uses_official_rgbd_wrapper = isinstance(
                evaluator.env,
                RGBDFullResWrapper,
            )
            _record_stage(
                stage_path,
                "SCENE_CONSTRUCTION_PASS",
                scene_load_number=1,
            )
            wrapper_class = (
                f"{type(evaluator.env).__module__}."
                f"{type(evaluator.env).__name__}"
            )
            timeout_steps = int(
                evaluator.env.task._termination_conditions["timeout"]._max_steps
            )
            action_dim = int(evaluator.robot.action_dim)

            _record_stage(stage_path, "INITIAL_RESET_START")
            evaluator.reset()
            _record_stage(stage_path, "INITIAL_RESET_PASS")
            _record_stage(stage_path, "TASK_INSTANCE_STATE_LOAD_START")
            evaluator.load_task_instance(0)
            _record_stage(stage_path, "TASK_INSTANCE_STATE_LOAD_PASS")
            _record_stage(stage_path, "POST_LOAD_RESET_START")
            evaluator.reset()
            _record_stage(stage_path, "POST_LOAD_RESET_PASS")

            start_position, start_orientation = (
                evaluator.robot.get_position_orientation()
            )
            start_joints = evaluator.robot.get_joint_positions()

            evaluator.start_recording(
                str(video_path),
                rate=int(run_config["runtime"]["video_fps"]),
            )
            _record_stage(stage_path, "HOLD_STEPS_START", steps=steps)
            try:
                executed_steps = 0
                for _ in range(steps):
                    terminated, truncated = evaluator.step()
                    executed_steps += 1
                    if terminated or truncated:
                        break
            finally:
                evaluator.stop_recording()
            _record_stage(
                stage_path,
                "HOLD_STEPS_PASS",
                executed_steps=executed_steps,
            )

            end_position, end_orientation = (
                evaluator.robot.get_position_orientation()
            )
            end_joints = evaluator.robot.get_joint_positions()

        start_position_list = _to_list(start_position)
        end_position_list = _to_list(end_position)
        start_orientation_list = _to_list(start_orientation)
        end_orientation_list = _to_list(end_orientation)
        start_joint_list = _to_list(start_joints)
        end_joint_list = _to_list(end_joints)
        max_joint_drift = max(
            abs(a - b)
            for a, b in zip(start_joint_list, end_joint_list)
        )
        video = _validate_video(video_path)

        observation_summaries_path = (
            run_dir / "policy" / "observation_summaries.jsonl"
        )
        summaries = [
            json.loads(line)
            for line in observation_summaries_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        depth_ratios = [
            array_summary["finite_ratio"]
            for summary in summaries
            for field, array_summary in summary["arrays"].items()
            if field.endswith(".depth_linear")
        ]

        result = {
            "gate": "G1",
            "wrapper_class": wrapper_class,
            "official_rgbd_wrapper_base": uses_official_rgbd_wrapper,
            "action_dim": action_dim,
            "proprio_dim": 61,
            "resolved_timeout_steps": timeout_steps,
            "requested_canary_steps": steps,
            "executed_steps": executed_steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "observation_messages": len(summaries),
            "minimum_depth_finite_ratio": min(depth_ratios),
            "base_position_drift_m": _norm_delta(
                start_position_list,
                end_position_list,
            ),
            "base_orientation_l2_drift": _norm_delta(
                start_orientation_list,
                end_orientation_list,
            ),
            "max_joint_drift": max_joint_drift,
            "video": video,
        }
        passed = (
            uses_official_rgbd_wrapper
            and action_dim == 23
            and timeout_steps == 28_335
            and executed_steps == steps
            and len(summaries) == steps
            and min(depth_ratios)
            >= float(
                run_config["observation"]["minimum_depth_finite_ratio"]
            )
            and video["bytes"] > 0
            and video["frames"] >= 1
            and video["width"] == 672
            and video["height"] == 448
        )
        result["status"] = "PASS" if passed else "FAIL"
        _write_json(run_dir / "private" / "g1_result.json", result)
        _record_stage(stage_path, f"G1_{result['status']}")
        _append_jsonl(
            run_dir / "canary" / "gate_checkpoints.jsonl",
            {
                "gate": "G1",
                "status": result["status"],
                "reset_epochs_expected": 2,
                "source_hash": initial_manifest["embodied_source_sha256"],
                "registry_hash": registry.sha256(),
                "observation_schema_hash": schema_sha256(),
                "result_path": "private/g1_result.json",
            },
        )

        final_manifest = {
            **initial_manifest,
            "status": result["status"],
            "resolved_timeout_steps": timeout_steps,
            "executed_steps": executed_steps,
            "scene_loads_attempted": scene_loads_attempted,
            "video_sha256": video["sha256"],
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "source_files": {
                "behavior_head": _git_head(BEHAVIOR_ROOT),
                "r1pro_yaml_sha256": _sha256(
                    BEHAVIOR_ROOT
                    / "OmniGibson"
                    / "omnigibson"
                    / "eval"
                    / "r1pro.yaml"
                ),
                "wrapper_sha256": _sha256(
                    BEHAVIOR_ROOT
                    / "OmniGibson"
                    / "omnigibson"
                    / "eval"
                    / "wrappers"
                    / "rgbd_full_res_wrapper.py"
                ),
                "plan_sha256": _sha256(
                    PROJECT_ROOT
                    / "docs"
                    / "picking_up_toys_r1pro_esi_action_plan_20260730.md"
                ),
                "config_sha256": _sha256(CONFIG_PATH),
            },
        }
        _write_json(run_dir / "manifest.json", final_manifest)
        _record_stage(
            stage_path,
            "RESULTS_PERSISTED_BEFORE_EVALUATOR_EXIT",
            status=result["status"],
        )
        if not passed:
            raise RuntimeError(f"G1 did not pass; see {run_dir}")
        return run_dir
    except Exception as exc:
        _record_stage(
            stage_path,
            "PYTHON_EXCEPTION",
            error_type=type(exc).__name__,
            message=str(exc),
        )
        _write_json(
            run_dir / "failure.json",
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        _write_json(
            run_dir / "manifest.json",
            {
                **initial_manifest,
                "status": "FAIL",
                "scene_loads_attempted": scene_loads_attempted,
                "failure_type": type(exc).__name__,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
    finally:
        if server_process.is_alive():
            server_process.terminate()
        server_process.join(timeout=10)
        if evaluator_context is not None:
            evaluator_context.__exit__(*sys.exc_info())


def _git_head(repo: Path) -> str:
    head = repo / ".git" / "HEAD"
    content = head.read_text(encoding="utf-8").strip()
    if content.startswith("ref: "):
        ref = repo / ".git" / content[5:]
        return ref.read_text(encoding="utf-8").strip()
    return content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Local websocket port; config default 0 asks the OS for a free port.",
    )
    args = parser.parse_args()
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    run_dir = run_canary(steps=args.steps, host=args.host, port=args.port)
    print(run_dir)


if __name__ == "__main__":
    main()
