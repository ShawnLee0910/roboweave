"""Run the observation-grounded G4a tennis-ball grasp calibration."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Sequence

import numpy as np

from behavior_hl.embodied.evaluator_bridge.pixel_manipulation import (
    load_public_snapshot,
    public_snapshot_metadata,
    robot_base_point_to_world,
    validate_local_manipulation_target,
)
from behavior_hl.embodied.evaluator_bridge.run_g1_canary import (
    BEHAVIOR_ROOT,
    CONFIG_PATH,
    EFFECTIVE_WRAPPER,
    PROJECT_ROOT,
    _append_jsonl,
    _configure_process_environment,
    _git_head,
    _load_and_validate_config,
    _record_stage,
    _sha256,
    _source_aggregate,
    _validate_launch_contract,
    _validate_video,
    _wait_for_port,
    _write_json,
)
from behavior_hl.embodied.evaluator_bridge.run_g2_g3a_suite import (
    _array,
    _base_contact,
    _robot_pose,
    _trunk_arm_contact,
)
from behavior_hl.embodied.policy.canary_server import (
    SharedCanaryControl,
    server_process_main,
)
from behavior_hl.embodied.policy.approach import (
    APPROACH_BASE_ACTIONS,
    APPROACH_LOOK_ACTIONS,
    ApproachLedger,
    ApproachLimits,
    ApproachStopped,
    build_approach_prompt,
    prompt_sha256,
)
from behavior_hl.embodied.policy.approach_execution import (
    BaseMacroConfig,
    BaseMacroProgress,
    classify_macro_displacement,
    look_translate_for_action,
)
from behavior_hl.embodied.policy.feedback import (
    PublicActionResult,
    PublicStatus,
    movement_result,
)
from behavior_hl.embodied.policy.observations import schema_sha256
from behavior_hl.embodied.policy.profiles import (
    PICKING_UP_TOYS_PROFILE,
    build_picking_up_toys_registry,
)
from behavior_hl.embodied.policy.r1pro.command_policy import (
    torso_profile_from_translate,
)
from behavior_hl.embodied.policy.rgbd_grounding import ground_image_point
from behavior_hl.embodied.policy.tokenplan_adapter import (
    TOKENPLAN_MODEL,
    TOKENPLAN_WALL_TIMEOUT_SECONDS,
    TokenPlanApproachRequest,
    parse_strict_approach_response,
)


TOKENPLAN_PYTHON = Path(os.environ.get("ESI_MODEL_PYTHON", sys.executable))
TOKENPLAN_SCRIPT = (
    PROJECT_ROOT / "scripts" / "tokenplan_approach_decision.py"
)
# A provider timeout is an infrastructure event, not a policy decision.  Keep
# one logical model call / ledger budget entry, but allow a bounded number of
# fresh child-process attempts before failing closed.  Every attempt gets its
# own process artifact and the canonical response remains the first successful
# response, preserving a complete audit trail without silently counting retries
# as extra policy actions.
TOKENPLAN_APPROACH_TIMEOUT_RETRIES = 2


def _run_tokenplan_approach_call(
    *,
    run_dir: Path,
    snapshot_path: Path,
    registry,
    ledger: ApproachLedger,
    wall_timeout_seconds: int,
    history_snapshot_paths: Sequence[Path] = (),
) -> tuple[Any | None, dict[str, Any], str]:
    """Return one strict action or None for a budgeted invalid response."""

    snapshot = load_public_snapshot(snapshot_path)
    head_rgb = np.ascontiguousarray(
        snapshot["cameras"]["head"]["rgb"],
        dtype=np.uint8,
    )
    frame_sha256 = hashlib.sha256(head_rgb.tobytes()).hexdigest()
    history_paths = tuple(Path(path) for path in history_snapshot_paths)
    if len(history_paths) > 5:
        raise ValueError("TokenPlan approach history exceeds five frames")
    history_frame_sha256s: list[str] = []
    for history_path in history_paths:
        history_snapshot = load_public_snapshot(history_path)
        history_head = np.ascontiguousarray(
            history_snapshot["cameras"]["head"]["rgb"],
            dtype=np.uint8,
        )
        history_frame_sha256s.append(
            hashlib.sha256(history_head.tobytes()).hexdigest()
        )
    call_index = ledger.note_model_call()
    prompt = build_approach_prompt(
        task_language=PICKING_UP_TOYS_PROFILE.task_language,
        registry=registry,
        ledger=ledger,
        frame_sha256=frame_sha256,
    )
    request = TokenPlanApproachRequest(
        call_index=call_index,
        prompt=prompt,
        prompt_sha256=prompt_sha256(prompt),
        frame_sha256=frame_sha256,
        history_frame_sha256s=tuple(history_frame_sha256s),
        registry_schema_version=registry.schema_version,
    )
    qwen_dir = run_dir / "policy" / "qwen"
    request_path = qwen_dir / f"request_{call_index:03d}.json"
    response_path = qwen_dir / f"response_{call_index:03d}.json"
    process_path = qwen_dir / f"process_{call_index:03d}.json"
    _write_json(request_path, request.to_dict())

    child_environment = os.environ.copy()
    child_environment.pop("LD_PRELOAD", None)
    child_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    child_environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    child_environment["BEHAVIOR_HL_PROJECT_ROOT"] = str(
        PROJECT_ROOT.resolve()
    )
    command = [
        str(TOKENPLAN_PYTHON),
        str(TOKENPLAN_SCRIPT),
    ]
    for history_path in history_paths:
        command.extend(("--history-snapshot", str(history_path)))
    command.extend(
        (
            "--snapshot",
            str(snapshot_path),
            "--request",
            str(request_path),
            "--output",
            str(response_path),
        )
    )
    attempt_records: list[dict[str, Any]] = []
    completed = None
    response = None
    for attempt_index in range(1, TOKENPLAN_APPROACH_TIMEOUT_RETRIES + 2):
        attempt_response_path = qwen_dir / (
            f"response_{call_index:03d}_attempt{attempt_index:02d}.json"
        )
        attempt_process_path = qwen_dir / (
            f"process_{call_index:03d}_attempt{attempt_index:02d}.json"
        )
        attempt_command = [
            value if value != str(response_path) else str(attempt_response_path)
            for value in command
        ]
        try:
            completed = subprocess.run(
                attempt_command,
                cwd=PROJECT_ROOT,
                env=child_environment,
                capture_output=True,
                text=True,
                timeout=wall_timeout_seconds + 15,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            record = {
                "attempt": attempt_index,
                "status": "timeout",
                "timeout_seconds": wall_timeout_seconds + 15,
                "error_type": type(exc).__name__,
            }
            attempt_records.append(record)
            _write_json(attempt_process_path, record)
            if attempt_index <= TOKENPLAN_APPROACH_TIMEOUT_RETRIES:
                continue
            _write_json(
                process_path,
                {
                    "call_index": call_index,
                    "status": "timeout",
                    "attempts": attempt_records,
                    "timeout_retries": max(attempt_index - 1, 0),
                },
            )
            raise RuntimeError(
                "TokenPlan approach call timed out after bounded retries"
            ) from exc

        record = {
            "attempt": attempt_index,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
        if attempt_response_path.is_file():
            try:
                attempt_payload = json.loads(
                    attempt_response_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                record.update(
                    {
                        "status": "invalid_attempt_artifact",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            else:
                record["status"] = str(attempt_payload.get("status"))
                record["error_type"] = attempt_payload.get("error_type")
                record["error"] = attempt_payload.get("error")
        else:
            record["status"] = "missing_response"
        attempt_records.append(record)
        _write_json(attempt_process_path, record)
        if (
            completed.returncode == 0
            and attempt_response_path.is_file()
        ):
            response = json.loads(
                attempt_response_path.read_text(encoding="utf-8")
            )
            if response.get("status") != "infrastructure_error":
                os.replace(attempt_response_path, response_path)
                break
        if attempt_index <= TOKENPLAN_APPROACH_TIMEOUT_RETRIES:
            continue
        _write_json(
            process_path,
            {
                "call_index": call_index,
                "status": "infrastructure_error",
                "attempts": attempt_records,
                "timeout_retries": max(attempt_index - 1, 0),
            },
        )
        failure = (
            response.get("error_type", "unknown")
            if isinstance(response, dict)
            else "unknown"
        )
        raise RuntimeError(
            "TokenPlan approach infrastructure failure after bounded retries: "
            f"{failure}"
        )

    if response is None or completed is None or not response_path.is_file():
        _write_json(
            process_path,
            {
                "call_index": call_index,
                "status": "missing_response",
                "attempts": attempt_records,
                # Record what actually happened, rather than the configured
                # retry budget.  A successful first attempt must not be
                # mislabeled as having consumed two timeout retries.
                "timeout_retries": max(len(attempt_records) - 1, 0),
            },
        )
        raise RuntimeError("TokenPlan approach call produced no response file")
    _write_json(
        process_path,
        {
            "call_index": call_index,
            "status": "success",
            "attempts": attempt_records,
            "timeout_retries": max(attempt_index - 1, 0),
        },
    )
    status = response.get("status")
    response_contract = {
        "call_index": call_index,
        "prompt_sha256": request.prompt_sha256,
        "head_rgb_sha256": frame_sha256,
        "history_head_rgb_sha256s": history_frame_sha256s,
        "project_root": str(PROJECT_ROOT.resolve()),
        "registry_schema_version": registry.schema_version,
    }
    for field, expected in response_contract.items():
        if response.get(field) != expected:
            raise RuntimeError(
                f"TokenPlan approach response {field} changed"
            )
    if completed.returncode != 0 or status == "infrastructure_error":
        raise RuntimeError(
            "TokenPlan approach infrastructure failure: "
            f"{response.get('error_type', 'unknown')}"
        )
    if status == "invalid_response":
        ledger.note_invalid_response()
        return None, snapshot, frame_sha256
    if status != "valid":
        raise RuntimeError("TokenPlan approach response status is invalid")
    if response.get("model") != TOKENPLAN_MODEL:
        raise RuntimeError("TokenPlan approach model identity changed")
    envelope = parse_strict_approach_response(
        response["raw_response"],
        registry=registry,
    )
    if envelope.to_dict() != response.get("action"):
        raise RuntimeError(
            "TokenPlan parsed action does not match raw response"
        )
    return envelope, snapshot, frame_sha256


def run_suite() -> Path:
    run_config = _load_and_validate_config()
    g4a = run_config["g4a"]
    approach_config = run_config["g4a_approach"]
    if not bool(g4a["calibration_segment"]):
        raise ValueError("G4a grasp canary must run as a calibration segment")
    if bool(g4a["scored_main_protocol"]):
        raise ValueError("G4a calibration cannot be marked as scored")
    if not bool(approach_config["enabled"]) or not bool(
        approach_config["calibration_segment"]
    ):
        raise ValueError("G4a approach calibration must be enabled")
    if str(approach_config["model"]) != TOKENPLAN_MODEL:
        raise ValueError("G4a approach model does not match frozen model")
    wall_timeout_seconds = int(
        approach_config["tokenplan_wall_timeout_seconds"]
    )
    if wall_timeout_seconds != TOKENPLAN_WALL_TIMEOUT_SECONDS:
        raise ValueError("TokenPlan wall timeout changed from frozen contract")

    approach_limits = ApproachLimits(
        max_model_calls=int(approach_config["maximum_model_calls"]),
        max_invalid_responses=int(
            approach_config["maximum_invalid_responses"]
        ),
        max_consecutive_same_action=int(
            approach_config["maximum_consecutive_same_action"]
        ),
        max_consecutive_blocked=int(
            approach_config["maximum_consecutive_blocked"]
        ),
        max_grasp_queries=int(
            approach_config["maximum_grasp_queries"]
        ),
        max_grasp_executions=int(
            approach_config["maximum_grasp_executions"]
        ),
        max_simulator_steps=int(
            approach_config["maximum_simulator_steps"]
        ),
        prompt_history_items=int(
            approach_config["prompt_history_items"]
        ),
        translations_after_out_of_reach=int(
            approach_config["translations_after_out_of_reach"]
        ),
        minimum_effective_translation_m=float(
            approach_config["minimum_effective_translation_m"]
        ),
    )
    base_cfg = approach_config["base"]
    base_macro_config = BaseMacroConfig(
        translation_command=float(base_cfg["translation_command"]),
        rotation_command=float(base_cfg["rotation_command"]),
        forward_slow_command=float(base_cfg["forward_slow_command"]),
        forward_slow_clearance_m=float(
            base_cfg["forward_slow_clearance_m"]
        ),
        forward_hard_stop_clearance_m=float(
            base_cfg["forward_hard_stop_clearance_m"]
        ),
        translation_steps=int(base_cfg["translation_steps"]),
        forward_slow_steps=int(base_cfg["forward_slow_steps"]),
        rotation_steps=int(base_cfg["rotation_steps"]),
        stall_warmup_steps=int(base_cfg["stall_warmup_steps"]),
        stall_window_steps=int(base_cfg["stall_window_steps"]),
        stall_speed_threshold=float(base_cfg["stall_speed_threshold"]),
        minimum_completed_translation_m=float(
            base_cfg["minimum_completed_translation_m"]
        ),
        minimum_partial_translation_m=float(
            base_cfg["minimum_partial_translation_m"]
        ),
        minimum_completed_rotation_deg=float(
            base_cfg["minimum_completed_rotation_deg"]
        ),
        minimum_partial_rotation_deg=float(
            base_cfg["minimum_partial_rotation_deg"]
        ),
    )
    command_frequency_hz = float(
        run_config["g2"]["command_frequency_hz"]
    )
    if not np.isfinite(command_frequency_hz) or command_frequency_hz <= 0:
        raise ValueError("base command frequency must be finite and positive")
    look_cfg = approach_config["look"]
    frozen_look_translates = {
        "look_up": float(look_cfg["translate_up"]),
        "look_center": float(look_cfg["translate_center"]),
        "look_down": float(look_cfg["translate_down"]),
    }
    for action, configured in frozen_look_translates.items():
        if configured != look_translate_for_action(action):
            raise ValueError("G4a approach look interval changed")

    artifact_root = Path(run_config["runtime"]["artifact_root"])
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"_g4a_approach_grasp_{os.getpid()}"
    )
    run_dir = artifact_root / run_id
    for relative in ("private", "public", "policy", "video", "canary"):
        (run_dir / relative).mkdir(parents=True, exist_ok=False)

    _configure_process_environment(run_dir)
    stage_path = run_dir / "canary" / "stages.jsonl"
    trace_path = run_dir / "private" / "physical_trace.jsonl"
    checkpoint_path = run_dir / "canary" / "gate_checkpoints.jsonl"
    _record_stage(stage_path, "RUN_DIRECTORY_READY")
    launch_contract = _validate_launch_contract()
    _write_json(run_dir / "private" / "launch_contract.json", launch_contract)
    _record_stage(stage_path, "ZERO_SCENE_LAUNCH_CONTRACT_PASS")

    registry = build_picking_up_toys_registry()
    approach_ledger = ApproachLedger(approach_limits)
    initial_manifest = {
        "run_id": run_id,
        "status": "RUNNING",
        "gates": ["G4a.approach", "G4a.grasp"],
        "task": "picking_up_toys",
        "mode": "train",
        "instance_id": 0,
        "simulator_seed": 0,
        "gpu": 0,
        "scene_load_budget": 1,
        "scene_loads_attempted": 0,
        "scored_main_protocol": False,
        "calibration_segment": True,
        "target_source": "public_rgbd_image_point",
        "point_source": "live_tokenplan_qwen3.8_max_head_rgb",
        "approach_model": TOKENPLAN_MODEL,
        "approach_prompt_mode": "strict_single_action_json",
        "privileged_target_pose": False,
        "privileged_postcondition_audit": True,
        "viewer_camera_control": False,
        "teleport_enabled": False,
        "wrapper": run_config["runtime"]["wrapper"],
        "effective_wrapper": EFFECTIVE_WRAPPER,
        "config_sha256": _sha256(CONFIG_PATH),
        "registry_sha256": registry.sha256(),
        "observation_schema_sha256": schema_sha256(),
        "embodied_source_sha256": _source_aggregate(
            PROJECT_ROOT / "src" / "behavior_hl" / "embodied"
        ),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_dir / "manifest.json", initial_manifest)

    context = multiprocessing.get_context("spawn")
    control = SharedCanaryControl(context)
    ready_queue = context.Queue()
    server_process = context.Process(
        target=server_process_main,
        args=(
            str(run_config["g1"]["server_host"]),
            int(run_config["g1"]["server_port"]),
            str(run_dir / "policy"),
            control,
            float(run_config["observation"]["minimum_depth_finite_ratio"]),
            ready_queue,
        ),
        daemon=False,
    )
    server_process.start()
    evaluator_context = None
    scene_loads_attempted = 0
    try:
        host = str(run_config["g1"]["server_host"])
        actual_port = int(ready_queue.get(timeout=20))
        _wait_for_port(host, actual_port)
        _record_stage(
            stage_path,
            "POLICY_SERVER_READY",
            host=host,
            port=actual_port,
        )

        _record_stage(stage_path, "OFFICIAL_RUNTIME_IMPORT_START")
        import torch as th
        from omegaconf import OmegaConf
        from omnigibson.action_primitives.starter_semantic_action_primitives import (
            StarterSemanticActionPrimitives,
        )
        from omnigibson.eval.evaluator import Evaluator
        from omnigibson.eval.utils.eval_utils import seed_everything
        from omnigibson.macros import gm
        from omnigibson.utils.usd_utils import RigidContactAPI

        expected_appdata = Path(
            os.environ["OMNIGIBSON_APPDATA_PATH"]
        ).resolve()
        if Path(gm.APPDATA_PATH).resolve() != expected_appdata:
            raise RuntimeError("OmniGibson appdata escaped isolation")
        _record_stage(stage_path, "OFFICIAL_RUNTIME_IMPORT_PASS")

        class ArmSelectableStarter(StarterSemanticActionPrimitives):
            def __init__(self, *args, selected_arm: str, **kwargs):
                self._selected_arm = selected_arm
                super().__init__(*args, **kwargs)

            @property
            def arm(self):
                return self._selected_arm

        gm.HEADLESS = True
        seed_everything(0)
        cfg = OmegaConf.create(
            {
                "env_wrapper": {"_target_": EFFECTIVE_WRAPPER},
                "policy_name": "behavior_hl_g4a_public_rgbd_canary",
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
                "max_steps": None,
                "write_video": True,
                "mode": "train",
                "seed": 0,
                "task": {"name": "picking_up_toys"},
                "robot": None,
            }
        )
        scene_loads_attempted = 1
        _record_stage(
            stage_path,
            "SCENE_CONSTRUCTION_START",
            scene_load_number=1,
        )
        evaluator_context = Evaluator(cfg)
        with nullcontext(evaluator_context.__enter__()) as evaluator:
            _record_stage(
                stage_path,
                "SCENE_CONSTRUCTION_PASS",
                scene_load_number=1,
            )
            timeout_steps = int(
                evaluator.env.task._termination_conditions["timeout"]._max_steps
            )
            if timeout_steps != 28_335:
                raise RuntimeError(
                    f"unexpected resolved timeout: {timeout_steps}"
                )
            if int(evaluator.robot.action_dim) != 23:
                raise RuntimeError("R1Pro action dimension is not 23")
            if str(g4a["hand"]) not in evaluator.robot.arm_names:
                raise RuntimeError("configured hand is not available on R1Pro")

            control.hold()
            evaluator.reset()
            evaluator.load_task_instance(0)
            evaluator.reset()
            _record_stage(stage_path, "SUITE_INITIALIZATION_PASS")

            suite_step = 0
            safety_reference_pose = _robot_pose(evaluator.robot)

            def one_step(
                segment: str,
                *,
                allow_base_motion: bool = False,
            ) -> dict[str, Any]:
                nonlocal suite_step
                terminated, truncated = evaluator.step()
                pose = _robot_pose(evaluator.robot)
                record = {
                    "suite_step": suite_step,
                    "segment": segment,
                    "robot_pose": pose,
                    "public_state": control.public_state(),
                    "base_contact": _base_contact(
                        evaluator,
                        RigidContactAPI,
                    ),
                    "trunk_arm_contact": _trunk_arm_contact(
                        evaluator,
                        RigidContactAPI,
                    ),
                }
                _append_jsonl(trace_path, record)
                suite_step += 1
                if terminated or truncated:
                    raise RuntimeError(
                        "task terminated during G4a grasp calibration"
                    )
                xy_drift = float(
                    np.linalg.norm(
                        np.asarray(pose["position"][:2])
                        - np.asarray(
                            safety_reference_pose["position"][:2]
                        )
                    )
                )
                tilt = max(abs(pose["roll"]), abs(pose["pitch"]))
                z_drift = abs(
                    pose["position"][2]
                    - safety_reference_pose["position"][2]
                )
                if (
                    (
                        not allow_base_motion
                        and xy_drift
                        > float(g4a["maximum_base_xy_drift_m"])
                    )
                    or tilt > float(g4a["maximum_roll_pitch_rad"])
                    or z_drift > float(g4a["maximum_z_drift_m"])
                    or bool(record["trunk_arm_contact"])
                ):
                    control.hold()
                    raise RuntimeError(
                        "G4a safety guard stopped manipulation: "
                        f"xy_drift={xy_drift:.6f}, tilt={tilt:.6f}, "
                        f"z_drift={z_drift:.6f}, "
                        "trunk_arm_contact="
                        f"{record['trunk_arm_contact']}"
                    )
                return record

            video_path = run_dir / "video" / "official_rollout.mp4"
            evaluator.start_recording(
                str(video_path),
                rate=int(run_config["runtime"]["video_fps"]),
            )
            try:
                camera_role = str(g4a["camera"])
                camera_name = evaluator.robot_camera_names[camera_role]
                sensor_name = camera_name.split("::")[1]
                intrinsic = _array(
                    evaluator.robot.sensors[sensor_name].intrinsic_matrix
                )
                if intrinsic.shape != (3, 3):
                    raise RuntimeError("camera intrinsic matrix is not 3x3")
                _write_json(
                    run_dir / "public" / "camera_calibration.json",
                    {
                        "camera": camera_role,
                        "intrinsic_matrix": intrinsic.tolist(),
                        "source": (
                            "fixed onboard camera calibration; no object state"
                        ),
                    },
                )

                action_history_path = (
                    run_dir / "public" / "approach_actions.jsonl"
                )
                approach_private_path = (
                    run_dir / "private" / "approach_grounding.jsonl"
                )

                def persist_public_result(
                    *,
                    item,
                    frame_sha256: str,
                ) -> None:
                    _append_jsonl(
                        action_history_path,
                        {
                            **item.to_dict(),
                            "source_head_rgb_sha256": frame_sha256,
                        },
                    )

                def execute_base_macro(
                    action_name: str,
                ) -> PublicActionResult:
                    progress = BaseMacroProgress(
                        action_name,
                        base_macro_config,
                    )
                    active_steps = 0
                    status = None
                    integrated_qvel = np.zeros(3, dtype=np.float64)
                    contact_detected = False
                    clearance_hard_stop = False
                    for _ in range(
                        base_macro_config.maximum_step_budget(action_name)
                    ):
                        public_state = control.public_state()
                        clearance = float(
                            public_state["head_forward_clearance_m"]
                        )
                        decision = base_macro_config.step_decision(
                            action_name,
                            head_forward_clearance_m=clearance,
                        )
                        if decision.hard_stop:
                            clearance_hard_stop = True
                            status = PublicStatus.BLOCKED
                            break
                        control.command_base(decision.command)
                        record = one_step(
                            f"g4a.approach.{action_name}",
                            allow_base_motion=True,
                        )
                        active_steps += 1
                        base_qvel = np.asarray(
                            record["public_state"]["base_qvel"],
                            dtype=np.float64,
                        )
                        integrated_qvel += (
                            base_qvel / command_frequency_hz
                        )
                        contact_detected = bool(record["base_contact"])
                        status = progress.observe(
                            base_qvel,
                            head_forward_clearance_m=float(
                                record["public_state"][
                                    "head_forward_clearance_m"
                                ]
                            ),
                            contact_detected=contact_detected,
                        )
                        if status is not None:
                            break
                    if status is None:
                        status = PublicStatus.BLOCKED
                    control.hold()
                    settle_steps = int(base_cfg["settle_steps"])
                    for _ in range(settle_steps):
                        one_step(
                            f"g4a.approach.{action_name}.settle",
                            allow_base_motion=True,
                        )
                    status = classify_macro_displacement(
                        action_name,
                        forward_m=float(integrated_qvel[0]),
                        left_m=float(integrated_qvel[1]),
                        yaw_deg=float(np.degrees(integrated_qvel[2])),
                        config=base_macro_config,
                        contact_detected=contact_detected,
                        clearance_hard_stop=clearance_hard_stop,
                        stall_detected=(
                            status is PublicStatus.BLOCKED
                            and not contact_detected
                            and not clearance_hard_stop
                        ),
                    )
                    return movement_result(
                        status=status,
                        simulator_steps=active_steps + settle_steps,
                        forward_m=float(integrated_qvel[0]),
                        left_m=float(integrated_qvel[1]),
                        yaw_deg=float(np.degrees(integrated_qvel[2])),
                    )

                def execute_look_macro(
                    action_name: str,
                ) -> PublicActionResult:
                    target = torso_profile_from_translate(
                        look_translate_for_action(action_name)
                    )
                    active_steps = 0
                    converged = False
                    for _ in range(int(look_cfg["maximum_steps"])):
                        control.command_torso(
                            target,
                            max_delta_per_step=float(
                                look_cfg["maximum_delta_per_step"]
                            ),
                        )
                        record = one_step(
                            f"g4a.approach.{action_name}",
                            allow_base_motion=True,
                        )
                        active_steps += 1
                        actual = np.asarray(
                            record["public_state"]["trunk_qpos"],
                            dtype=np.float32,
                        )
                        if float(np.max(np.abs(actual - target))) <= float(
                            look_cfg["tolerance_rad"]
                        ):
                            converged = True
                            break
                    control.hold()
                    settle_steps = int(look_cfg["settle_steps"])
                    for _ in range(settle_steps):
                        one_step(
                            f"g4a.approach.{action_name}.settle",
                            allow_base_motion=True,
                        )
                    return PublicActionResult(
                        status=(
                            PublicStatus.COMPLETED
                            if converged
                            else PublicStatus.JOINT_STALLED
                        ),
                        simulator_steps=active_steps + settle_steps,
                    )

                _record_stage(stage_path, "G4A_QWEN_APPROACH_START")
                snapshot_id = 0
                grounded = None
                reach = None
                snapshot_step = None
                selected_frame_sha256 = None

                def terminate_reach_query_budget() -> None:
                    control.hold()
                    _write_json(
                        run_dir / "public" / "approach_summary.json",
                        {
                            **approach_ledger.public_summary(),
                            "status": (
                                "REACH_QUERY_BUDGET_EXHAUSTED"
                            ),
                        },
                    )
                    _record_stage(
                        stage_path,
                        "G4A_REACH_QUERY_BUDGET_EXHAUSTED",
                        model_calls=approach_ledger.model_calls,
                        simulator_steps=(
                            approach_ledger.simulator_steps
                        ),
                        grasp_queries=approach_ledger.grasp_queries,
                    )
                    raise ApproachBudgetExceeded(
                        "grasp_at reach-query budget exhausted"
                    )

                while grounded is None:
                    if approach_ledger.grasp_query_budget_exhausted:
                        terminate_reach_query_budget()
                    snapshot_id += 1
                    control.hold()
                    control.request_public_snapshot(snapshot_id)
                    snapshot_step = one_step(
                        f"g4a.approach.snapshot_{snapshot_id:03d}",
                        allow_base_motion=True,
                    )
                    approach_ledger.note_simulator_steps(1)
                    snapshot_path = (
                        run_dir
                        / "policy"
                        / "captures"
                        / f"public_snapshot_{snapshot_id:03d}.npz"
                    )
                    snapshot = load_public_snapshot(snapshot_path)
                    _write_json(
                        run_dir
                        / "public"
                        / f"snapshot_metadata_{snapshot_id:03d}.json",
                        {
                            **public_snapshot_metadata(snapshot),
                            "head_forward_clearance_m": float(
                                snapshot_step["public_state"][
                                    "head_forward_clearance_m"
                                ]
                            ),
                        },
                    )

                    envelope = None
                    frame_sha256 = ""
                    while envelope is None:
                        envelope, snapshot, frame_sha256 = (
                            _run_tokenplan_approach_call(
                                run_dir=run_dir,
                                snapshot_path=snapshot_path,
                                registry=registry,
                                ledger=approach_ledger,
                                wall_timeout_seconds=wall_timeout_seconds,
                            )
                        )
                        if envelope is None:
                            _record_stage(
                                stage_path,
                                "G4A_QWEN_INVALID_RESPONSE",
                                model_calls=approach_ledger.model_calls,
                                invalid_responses=(
                                    approach_ledger.invalid_responses
                                ),
                            )

                    immediate_result = approach_ledger.begin_action(
                        envelope
                    )
                    if immediate_result is not None:
                        persist_public_result(
                            item=approach_ledger.history[-1],
                            frame_sha256=frame_sha256,
                        )
                        continue
                    if envelope.action in APPROACH_BASE_ACTIONS:
                        result = execute_base_macro(envelope.action)
                        item = approach_ledger.finish_action(result)
                        persist_public_result(
                            item=item,
                            frame_sha256=frame_sha256,
                        )
                        continue
                    if envelope.action in APPROACH_LOOK_ACTIONS:
                        result = execute_look_macro(envelope.action)
                        item = approach_ledger.finish_action(result)
                        persist_public_result(
                            item=item,
                            frame_sha256=frame_sha256,
                        )
                        continue
                    if envelope.action == "stop":
                        item = approach_ledger.finish_action(
                            PublicActionResult(
                                status=PublicStatus.COMPLETED,
                                simulator_steps=0,
                            )
                        )
                        persist_public_result(
                            item=item,
                            frame_sha256=frame_sha256,
                        )
                        _record_stage(
                            stage_path,
                            "G4A_QWEN_STOPPED_BEFORE_GRASP",
                            model_calls=approach_ledger.model_calls,
                            simulator_steps=(
                                approach_ledger.simulator_steps
                            ),
                        )
                        raise ApproachStopped(
                            "Qwen chose stop before a confirmed grasp"
                        )
                    if envelope.action != "grasp_at":
                        raise RuntimeError(
                            "approach phase dispatched an unknown action"
                        )

                    attempt = approach_ledger.grasp_queries
                    point_args = envelope.args
                    try:
                        candidate = ground_image_point(
                            snapshot,
                            camera=point_args.camera,
                            point=point_args.point,
                            intrinsic_matrix=intrinsic,
                            depth_radius_px=int(g4a["depth_radius_px"]),
                            minimum_depth_m=float(
                                g4a["minimum_depth_m"]
                            ),
                            maximum_depth_m=float(
                                g4a["maximum_depth_m"]
                            ),
                            maximum_depth_spread_m=float(
                                g4a["maximum_depth_spread_m"]
                            ),
                        )
                    except ValueError as exc:
                        _append_jsonl(
                            approach_private_path,
                            {
                                "attempt": attempt,
                                "status": "INVALID_DEPTH",
                                "error": str(exc),
                                "source_observation_step": snapshot_step[
                                    "suite_step"
                                ],
                            },
                        )
                        item = approach_ledger.finish_action(
                            PublicActionResult(
                                status=PublicStatus.INVALID_DEPTH,
                                simulator_steps=0,
                            )
                        )
                        persist_public_result(
                            item=item,
                            frame_sha256=frame_sha256,
                        )
                        if approach_ledger.grasp_query_budget_exhausted:
                            terminate_reach_query_budget()
                        continue

                    grounding_payload = {
                        "attempt": attempt,
                        "camera": candidate.camera,
                        "normalized_point": candidate.normalized_point,
                        "pixel_xy": candidate.pixel_xy,
                        "depth_m": candidate.depth_m,
                        "camera_xyz_m": candidate.camera_xyz_m,
                        "robot_base_xyz_m": candidate.robot_base_xyz_m,
                        "depth_sample_count": (
                            candidate.depth_sample_count
                        ),
                        "depth_spread_m": candidate.depth_spread_m,
                        "source_observation_step": snapshot_step[
                            "suite_step"
                        ],
                        "source_head_rgb_sha256": frame_sha256,
                    }
                    try:
                        candidate_reach = (
                            validate_local_manipulation_target(
                                candidate.robot_base_xyz_m,
                                maximum_horizontal_reach_m=float(
                                    g4a[
                                        "maximum_horizontal_reach_m"
                                    ]
                                ),
                                minimum_height_m=float(
                                    g4a["minimum_target_height_m"]
                                ),
                                maximum_height_m=float(
                                    g4a["maximum_target_height_m"]
                                ),
                            )
                        )
                    except ValueError as exc:
                        _append_jsonl(
                            approach_private_path,
                            {
                                **grounding_payload,
                                "status": "OUT_OF_REACH",
                                "error": str(exc),
                            },
                        )
                        item = approach_ledger.finish_action(
                            PublicActionResult(
                                status=PublicStatus.OUT_OF_REACH,
                                simulator_steps=0,
                            )
                        )
                        persist_public_result(
                            item=item,
                            frame_sha256=frame_sha256,
                        )
                        _record_stage(
                            stage_path,
                            "PUBLIC_RGBD_REACH_REJECT",
                            attempt=attempt,
                        )
                        if approach_ledger.grasp_query_budget_exhausted:
                            terminate_reach_query_budget()
                        continue

                    grounded = candidate
                    reach = candidate_reach
                    selected_frame_sha256 = frame_sha256
                    approach_ledger.note_grasp_execution()
                    _append_jsonl(
                        approach_private_path,
                        {
                            **grounding_payload,
                            "status": "PASS",
                            "reach": reach,
                        },
                    )

                _write_json(
                    run_dir / "public" / "approach_summary.json",
                    {
                        **approach_ledger.public_summary(),
                        "status": "REACH_PASS",
                        "selected_head_rgb_sha256": (
                            selected_frame_sha256
                        ),
                    },
                )
                _record_stage(
                    stage_path,
                    "G4A_QWEN_APPROACH_PASS",
                    model_calls=approach_ledger.model_calls,
                    simulator_steps=approach_ledger.simulator_steps,
                    grasp_queries=approach_ledger.grasp_queries,
                    grasp_executions=(
                        approach_ledger.grasp_executions
                    ),
                )
                safety_reference_pose = _robot_pose(evaluator.robot)
                world_point = robot_base_point_to_world(
                    grounded.robot_base_xyz_m,
                    robot_position=snapshot_step["robot_pose"]["position"],
                    robot_orientation_xyzw=snapshot_step["robot_pose"][
                        "orientation_xyzw"
                    ],
                )
                _record_stage(
                    stage_path,
                    "PUBLIC_RGBD_GROUNDING_PASS",
                    depth_m=grounded.depth_m,
                    horizontal_reach_m=reach["horizontal_reach_m"],
                )

                grasp_start_suite_step = suite_step
                _record_stage(stage_path, "CUROBO_INITIALIZATION_START")
                controller = ArmSelectableStarter(
                    evaluator.env,
                    evaluator.robot,
                    selected_arm=str(g4a["hand"]),
                    enable_head_tracking=False,
                    task_relevant_objects_only=False,
                    curobo_batch_size=1,
                )
                _record_stage(stage_path, "CUROBO_INITIALIZATION_PASS")

                last_action = None

                def execute_generator(label: str, generator) -> int:
                    nonlocal last_action
                    count = 0
                    for generated in generator:
                        action = _array(generated)
                        if action.shape != (23,) or not np.isfinite(
                            action
                        ).all():
                            raise RuntimeError(
                                f"{label} generated an invalid action"
                            )
                        control.command_direct_action(action)
                        last_action = action
                        one_step(label)
                        count += 1
                        if count > int(g4a["maximum_generator_steps"]):
                            raise RuntimeError(
                                f"{label} exceeded the frozen step budget"
                            )
                    return count

                hand = str(g4a["hand"])
                target_quat = th.tensor(
                    [0.0, 0.0, 0.0, 1.0],
                    dtype=th.float32,
                )
                fingertip_offset = float(
                    np.mean(
                        [
                            float(value)
                            for value in evaluator.robot.eef_to_fingertip_lengths[
                                hand
                            ].values()
                        ]
                    )
                )
                pregrasp_offset = fingertip_offset + float(
                    g4a["grasp_approach_clearance_m"]
                )
                grasp_position = th.tensor(
                    world_point,
                    dtype=th.float32,
                )
                pregrasp_position = grasp_position + th.tensor(
                    [0.0, 0.0, pregrasp_offset],
                    dtype=th.float32,
                )
                _write_json(
                    run_dir / "private" / "planned_grasp_geometry.json",
                    {
                        "world_point_m": world_point.tolist(),
                        "top_down_quaternion_xyzw": target_quat.tolist(),
                        "fingertip_offset_m": fingertip_offset,
                        "pregrasp_offset_m": pregrasp_offset,
                        "object_pose_or_extent_used": False,
                    },
                )

                _record_stage(stage_path, "G4A_OPEN_START")
                open_steps = execute_generator(
                    "g4a.open_gripper",
                    controller._execute_release(),
                )
                _record_stage(
                    stage_path,
                    "G4A_PREGRASP_START",
                    open_steps=open_steps,
                )
                pregrasp_steps = execute_generator(
                    "g4a.move_pregrasp",
                    controller._move_hand(
                        (pregrasp_position, target_quat),
                        lock_auxiliary_arm=True,
                    ),
                )
                _record_stage(
                    stage_path,
                    "G4A_APPROACH_START",
                    pregrasp_steps=pregrasp_steps,
                )
                approach_steps = execute_generator(
                    "g4a.approach",
                    controller._move_hand(
                        (grasp_position, target_quat),
                        motion_constraint=[1, 1, 1, 1, 1, 0],
                        lock_auxiliary_arm=True,
                    ),
                )
                _record_stage(
                    stage_path,
                    "G4A_CLOSE_START",
                    approach_steps=approach_steps,
                )
                close_steps = execute_generator(
                    "g4a.close_gripper",
                    controller._execute_grasp(),
                )
                if last_action is None:
                    raise RuntimeError("grasp generated no physical action")
                for _ in range(int(g4a["settle_steps"])):
                    control.command_direct_action(last_action)
                    one_step("g4a.post_close_settle")

                held_object = controller._get_obj_in_hand()
                held_before_lift = held_object
                if held_object is None:
                    raise RuntimeError(
                        "G4a grasp closed without private-confirmed held object"
                    )
                held_category = str(getattr(held_object, "category", ""))
                if held_category != "tennis_ball":
                    raise RuntimeError(
                        "G4a grasp held an unexpected object category"
                    )
                held_start_position = _array(
                    held_object.get_position_orientation()[0]
                )
                _record_stage(
                    stage_path,
                    "G4A_PRIVATE_GRASP_CONFIRMED",
                    category=held_category,
                    close_steps=close_steps,
                )

                lift_position = grasp_position + th.tensor(
                    [0.0, 0.0, float(g4a["post_grasp_lift_m"])],
                    dtype=th.float32,
                )
                lift_steps = execute_generator(
                    "g4a.post_grasp_lift",
                    controller._move_hand(
                        (lift_position, target_quat),
                        lock_auxiliary_arm=True,
                    ),
                )
                for _ in range(int(g4a["settle_steps"])):
                    control.command_direct_action(last_action)
                    one_step("g4a.post_lift_settle")
                held_after_lift = controller._get_obj_in_hand()
                if held_after_lift is not held_before_lift:
                    raise RuntimeError("held object was lost during physical lift")
                held_end_position = _array(
                    held_after_lift.get_position_orientation()[0]
                )
                object_lift = float(
                    held_end_position[2] - held_start_position[2]
                )
                if object_lift < 0.05:
                    raise RuntimeError(
                        f"held tennis ball did not lift enough: {object_lift}"
                    )

                grasp_public_result = PublicActionResult(
                    status=PublicStatus.GRASP_UNCERTAIN,
                    simulator_steps=(
                        suite_step - grasp_start_suite_step
                    ),
                    hand=hand,
                )
                grasp_history_item = approach_ledger.finish_action(
                    grasp_public_result
                )
                persist_public_result(
                    item=grasp_history_item,
                    frame_sha256=selected_frame_sha256,
                )
                private_result = {
                    "status": "PASS",
                    "held_category": held_category,
                    "assisted_grasp_confirmed": True,
                    "object_lift_m": object_lift,
                    "steps": {
                        "open": open_steps,
                        "pregrasp": pregrasp_steps,
                        "approach": approach_steps,
                        "close": close_steps,
                        "lift": lift_steps,
                    },
                    "target_from_public_rgbd": True,
                    "target_object_pose_used": False,
                    "approach_model_calls": (
                        approach_ledger.model_calls
                    ),
                    "approach_simulator_steps": (
                        approach_ledger.simulator_steps
                    ),
                }
                _write_json(
                    run_dir / "private" / "g4a_grasp_result.json",
                    private_result,
                )
                _write_json(
                    run_dir / "public" / "action_result.json",
                    {
                        "action": "grasp_at",
                        **grasp_public_result.to_dict(),
                    },
                )
                _write_json(
                    run_dir / "public" / "approach_summary.json",
                    {
                        **approach_ledger.public_summary(),
                        "status": "GRASP_CALIBRATION_PASS",
                        "selected_head_rgb_sha256": (
                            selected_frame_sha256
                        ),
                    },
                )
                _append_jsonl(
                    checkpoint_path,
                    {
                        "gate": "G4a.approach",
                        "status": "PASS",
                        "end_step": grasp_start_suite_step,
                        "scene_load_number": 1,
                        "result_path": "public/approach_summary.json",
                    },
                )
                _append_jsonl(
                    checkpoint_path,
                    {
                        "gate": "G4a.grasp",
                        "status": "PASS",
                        "end_step": suite_step,
                        "scene_load_number": 1,
                        "result_path": "private/g4a_grasp_result.json",
                    },
                )
                _record_stage(
                    stage_path,
                    "G4A_GRASP_PASS",
                    end_step=suite_step,
                    object_lift_m=object_lift,
                )
            finally:
                control.hold()
                evaluator.stop_recording()

            video = _validate_video(video_path)
            final_manifest = {
                **initial_manifest,
                "status": "PASS",
                "gate_status": {
                    "G3a": "PASS_FROM_PRIOR_RUN",
                    "G3": "PARTIAL",
                    "G4a.approach": "PASS",
                    "G4a.grasp": "PASS",
                    "G3b": "NOT_RUN",
                    "G4a.place": "NOT_RUN",
                },
                "scene_loads_attempted": scene_loads_attempted,
                "suite_steps": suite_step,
                "approach_model_calls": approach_ledger.model_calls,
                "approach_invalid_responses": (
                    approach_ledger.invalid_responses
                ),
                "approach_simulator_steps": (
                    approach_ledger.simulator_steps
                ),
                "resolved_timeout_steps": timeout_steps,
                "video_sha256": video["sha256"],
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "source_files": {
                    "behavior_head": _git_head(BEHAVIOR_ROOT),
                    "config_sha256": _sha256(CONFIG_PATH),
                    "runner_sha256": _sha256(Path(__file__)),
                },
            }
            _write_json(run_dir / "manifest.json", final_manifest)
            _record_stage(
                stage_path,
                "RESULTS_PERSISTED_BEFORE_EVALUATOR_EXIT",
                status="PASS",
            )
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
                "approach_model_calls": approach_ledger.model_calls,
                "approach_invalid_responses": (
                    approach_ledger.invalid_responses
                ),
                "approach_simulator_steps": (
                    approach_ledger.simulator_steps
                ),
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


def main() -> None:
    print(run_suite())


if __name__ == "__main__":
    main()
