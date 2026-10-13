"""Launch one isolated Action20 model-only debugging rollout.

This entrypoint deliberately skips the private Oracle and same-scene reset. It
retains the private official verifier for post-hoc Q scoring, so its artifacts
are useful for protocol and model-behavior debugging but are not Oracle-gated
benchmark evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from behavior_hl.embodied.evaluator_bridge.gripper_visual_profiles import (
    GRIPPER_VISUAL_PROFILE_INWARD_12CM,
)
from behavior_hl.embodied.evaluator_bridge.inside_task_profiles import (
    get_inside_task_profile,
)
from behavior_hl.embodied.evaluator_bridge.launch_b1_single_load import (
    build_production_bindings,
)
from behavior_hl.embodied.evaluator_bridge.predeclared_camera_config import (
    THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_marker_symbolic_rollout import (
    make_action20_model_only_debug_rollout_callback,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import (
    B1SingleLoadWorkerSpec,
    run_b1_single_load,
)


EXPECTED_MODEL = "qwen3.7-plus"
MODEL_PYTHON = Path(os.environ.get("ESI_MODEL_PYTHON", sys.executable))
DEFAULT_MAXIMUM_MODEL_CALLS = 300
MODEL_VIDEO_FPS = 4.0
MODEL_PHASE_NAME = "marker_symbolic_action20_model"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_model_only_video(
    run_dir: Path,
    *,
    fps: float = MODEL_VIDEO_FPS,
) -> Mapping[str, Any]:
    """Encode the runner's already-rendered visual index without sim steps."""

    if fps <= 0:
        raise ValueError("model-only video FPS must be positive")
    phase_dir = Path(run_dir) / "strategies" / MODEL_PHASE_NAME
    index_path = phase_dir / "visual_frame_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    records = payload.get("frames")
    if (
        not isinstance(records, list)
        or not records
        or payload.get("frame_count") != len(records)
    ):
        raise RuntimeError("model-only visual frame index is empty or malformed")
    phase_resolved = phase_dir.resolve()
    frame_paths: list[Path] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("model-only visual frame record is malformed")
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise RuntimeError("model-only visual frame has no path")
        source = Path(raw_path)
        source = source.resolve() if source.is_absolute() else (phase_dir / source).resolve()
        if phase_resolved not in source.parents or not source.is_file():
            raise RuntimeError("model-only visual frame escaped or is missing")
        expected_hash = record.get("png_sha256") or record.get("sha256")
        if expected_hash is not None and expected_hash != _sha256_file(source):
            raise RuntimeError("model-only visual frame hash mismatch")
        frame_paths.append(source)

    import imageio.v2 as imageio

    output = phase_dir / "episode.mp4"
    writer = imageio.get_writer(
        output,
        fps=float(fps),
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    try:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))
    finally:
        writer.close()
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("model-only episode MP4 was not produced")
    report = {
        "schema_version": "behavior2026.action20-model-only-video/v1",
        "render_only_capture": True,
        "additional_simulator_steps": 0,
        "frame_count": len(frame_paths),
        "fps": float(fps),
        "path": str(output),
        "sha256": _sha256_file(output),
    }
    report_path = phase_dir / "model_only_video_manifest.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--server-port", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument(
        "--task-name",
        default="clearing_food_from_table_into_fridge",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--instance-id", default=0, type=int)
    parser.add_argument(
        "--maximum-model-calls",
        default=DEFAULT_MAXIMUM_MODEL_CALLS,
        type=int,
    )
    return parser.parse_args(argv)


def _require_model_environment() -> None:
    if os.environ.get("BEHAVIOR_HL_TOKENPLAN_MODEL") != EXPECTED_MODEL:
        raise ValueError(
            f"BEHAVIOR_HL_TOKENPLAN_MODEL must be {EXPECTED_MODEL!r}"
        )
    if os.environ.get("BEHAVIOR_HL_TOKENPLAN_MAX_RETRIES") != "0":
        raise ValueError("BEHAVIOR_HL_TOKENPLAN_MAX_RETRIES must be 0")
    if os.environ.get("TOKEN_PLAN_BASE_URL") != (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ):
        raise ValueError(
            "TOKEN_PLAN_BASE_URL is not the frozen DashScope endpoint"
        )
    if not os.environ.get("TOKEN_PLAN_API_KEY"):
        raise ValueError("TOKEN_PLAN_API_KEY must be supplied at launch time")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _require_model_environment()
    if args.maximum_model_calls <= 0:
        raise ValueError("maximum-model-calls must be positive")
    profile = get_inside_task_profile(args.task_name)
    if profile.split != args.split or profile.instance_id != args.instance_id:
        raise ValueError(
            "worker split/instance does not match the frozen task profile"
        )
    worker = B1SingleLoadWorkerSpec(
        worker_id=args.worker_id,
        physical_gpu=args.physical_gpu,
        server_port=args.server_port,
        seed=args.seed,
        artifact_root=args.artifact_root,
        task_name=profile.task_name,
        split=profile.split,
        instance_id=profile.instance_id,
    )
    result = run_b1_single_load(
        bindings=build_production_bindings(),
        worker=worker,
        expected_task_id=profile.task_id,
        camera_profile=THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
        gripper_visual_profile=GRIPPER_VISUAL_PROFILE_INWARD_12CM,
        defer_inward_arm_presentation=True,
        strategy_sweep_callback=(
            make_action20_model_only_debug_rollout_callback(
                seed=worker.seed,
                profile=profile,
                model_python=MODEL_PYTHON,
                maximum_model_calls=args.maximum_model_calls,
                observation_profile=THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
                model_phase_gripper_presentation=True,
            )
        ),
    )
    materialize_model_only_video(result.run_dir)
    print(result.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
