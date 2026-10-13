"""Synthetic record replay through the frozen model loop, with no external I/O."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .bootstrap import PROFILE, REVISIONS, activate

FIXTURE = Path(__file__).resolve().parent / "data/examples/synthetic-marker-v1.json"


def run_replay(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Exercise real protocol, history, Marker transitions and output writers.

    The scene and provider replies are synthetic. No action was simulated and
    no official score is claimed. This is a packaging/protocol demonstration.
    """
    activate()
    from PIL import Image
    from model_client import run_model_episode
    from scene_protocol import _write_json
    fixture = json.loads(FIXTURE.read_text())
    history = config.get("run", {}).get("history_frames", 5)
    if type(history) is not int or history not in {5, 20}:
        raise ValueError("history_frames must be 5 or 20")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Replay output must be a new directory")
    output.mkdir(parents=True)
    scene_dir = output / "private/synthetic_scene"
    scene_dir.mkdir(parents=True)
    images = []
    for camera, color in (("head", "blue"), ("left_wrist", "red"), ("right_wrist", "green")):
        path = scene_dir / f"{camera}.png"
        Image.new("RGB", (16, 16), color).save(path)
        images.append({"camera": camera, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    class RecordedProvider:
        model = "synthetic-marker-replay"
        maximum_output_tokens = None
        cursor = 0

        def capability_preflight(self, *, artifact, **kwargs):
            report = {"status": "PASS", "mode": "offline_replay", "simulator_touched": False}
            _write_json(artifact, report)
            return report

        def complete(self, request, *, call_dir, seed):
            call_dir.mkdir(parents=True, exist_ok=False)
            action = fixture["actions"][self.cursor]
            self.cursor += 1
            response = {"choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{
                    "id": f"replay_{self.cursor}", "type": "function", "function": {
                        "name": "robot_action", "arguments": json.dumps(action)}}]}}]}
            _write_json(call_dir / "request.json", request)
            _write_json(call_dir / "response.json", response)
            return response

    class RecordedScene:
        scene_run_dir = scene_dir
        cursor = 0

        def request(self, command, **fields):
            if command == "status":
                return {"scene_resident": True, "session_valid": True, "episode_active": False,
                        "episode_id": None, "next_sequence": self.cursor}
            if command == "begin_episode":
                return {"observation": {"observation_id": "initial", "kind": "physical",
                                        "images": images, "marker_targets": []}}
            if command == "end_episode":
                _write_json(scene_dir / "end_episode.json", {"mode": "offline_replay", **fields})
                return {"status": "PASS"}
            if command != "step" or self.cursor >= len(fixture["steps"]):
                raise ValueError("Unexpected synthetic scene command")
            expected = {k: v for k, v in fixture["actions"][self.cursor].items() if k != "target_description"}
            if fields != {"sequence": self.cursor, **expected}:
                raise ValueError("Replay action differs from the fixed record")
            step = fixture["steps"][self.cursor]
            self.cursor += 1
            return {"action_result": step["action_result"], "observation": {
                "observation_id": f"replay_{self.cursor}", "kind": step["kind"],
                "images": images, "marker_targets": step["marker_targets"]}}

    _write_json(output / "run_record.json", {
        "schema_version": "esi-benchmark/run-v1", "mode": "offline_replay", "profile": PROFILE,
        "history_frames": history,
        "evaluation_variant": "frozen_history_5" if history == 5 else "history_20_variant_not_baseline",
        "source_revisions": REVISIONS, "fixture_id": fixture["fixture_id"],
        "public_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "task_identity": None, "requested_task_id": config.get("run", {}).get("task_id"),
        "evidence_scope": "synthetic protocol and artifact demonstration; no official task evaluation",
        "official_benchmark_eligible": False, "strict_production_eligible": False,
        "simulator_started": False, "model_requested": False, "official_score": None,
    })
    summary = run_model_episode(
        scene=RecordedScene(), provider=RecordedProvider(), experiment_id="synthetic_marker_v1",
        instruction=fixture["instruction"], task_id=0, task_name="synthetic_protocol_fixture",
        task_split="synthetic", task_instance_id="0", task_seed=0,
        instruction_provenance={"kind": "synthetic_fixture", "fixture_id": fixture["fixture_id"]},
        maximum_model_calls=30, maximum_action_decisions=150,
        physical_history_max=history, output_dir=output,
    )
    return {"output_dir": str(output), "mode": "offline_replay", "official_score": None, "summary": dict(summary)}
