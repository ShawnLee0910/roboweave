"""Thin public adapter for the frozen provider and episode implementation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlsplit

from .bootstrap import PROFILE, REVISIONS, activate

_GENERATION = {
    "maximum_output_tokens", "temperature", "top_p", "top_k",
    "repetition_penalty", "preserve_thinking", "reasoning_effort",
    "reasoning_context", "reasoning_mode", "reasoning_summary", "text_verbosity",
}
_MODEL_FIELDS = {"endpoint", "model", "api_key_env", "api", "timeout"} | _GENERATION


def validate_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate configuration without reading credentials or making requests."""
    if not isinstance(config, Mapping) or set(config) - _MODEL_FIELDS:
        raise ValueError("Unknown or invalid model configuration fields")
    result = dict(config)
    for name in ("endpoint", "model"):
        if not isinstance(result.get(name), str) or not result[name].strip():
            raise ValueError(f"model.{name} must be a nonempty string")
    endpoint = urlsplit(result["endpoint"])
    if not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError("Model endpoint must have a host and no credentials, query or fragment")
    key_env = result.get("api_key_env")
    if key_env is not None and (not isinstance(key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_env)):
        raise ValueError("api_key_env must name an environment variable; never put a key in JSON")
    result.setdefault("api", "chat_completions")
    result.setdefault("timeout", 300)
    if type(result["timeout"]) is not int:
        raise ValueError("Model timeout must be an integer in [1, 900]")
    # Reuse all frozen transport/generation checks without issuing a request.
    _provider(result)
    return result


def _provider(config: Mapping[str, Any]):
    activate()
    from model_client import OpenAICompatibleProvider
    return OpenAICompatibleProvider(
        model=config["model"], base_url=config["endpoint"],
        provider_api=config.get("api", "chat_completions"),
        timeout_seconds=config.get("timeout", 300),
        api_key_file=None,
        api_key_env=config.get("api_key_env") or "ESI_UNCONFIGURED_API_KEY",
        require_api_key=config.get("api_key_env") is not None,
        **{key: config[key] for key in _GENERATION if key in config},
    )


def create_provider(config: Mapping[str, Any]):
    """Construct the upstream transport; credentials are read only when sent."""
    validated = validate_model_config(config)
    provider = _provider(validated)
    if provider.requires_api_key:
        key_env = validated.get("api_key_env")
        if not key_env or not os.environ.get(key_env, "").strip():
            raise ValueError("The configured API key environment variable is unavailable")
    return provider


def run_episode(
    *, model_config: Mapping[str, Any], task: Mapping[str, Any],
    ready_path: Path, runtime_root: Path, output_dir: Path,
    experiment_id: str, history_frames: int = 5, output_reserved: bool = False,
) -> dict[str, Any]:
    """Run a frozen Action20 episode on a caller-owned, independently started scene.

    ``task`` contains task_id (numeric public handle), task_name, task_split,
    task_instance_id, task_seed, instruction and optional instruction_provenance.
    The caller owns the resident process and must always stop its own child.
    """
    activate()
    from model_client import run_model_episode
    from scene_protocol import SceneClient
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", experiment_id):
        raise ValueError("Experiment ID must be a safe path component")
    if type(task.get("task_id")) is not int or not 0 <= task["task_id"] <= 999:
        raise ValueError("Task public handle must be an integer in [0, 999]")
    if type(task.get("task_seed")) is not int or not 0 <= task["task_seed"] <= 0xFFFF_FFFF:
        raise ValueError("Task seed must be an unsigned 32-bit integer")
    for field in ("task_name", "task_split", "task_instance_id", "instruction"):
        if not isinstance(task.get(field), str) or not task[field].strip():
            raise ValueError(f"Frozen task is missing {field}")
    if type(history_frames) is not int or history_frames not in {5, 20}:
        raise ValueError("history_frames must be 5 or 20")
    if "budget" in task and not isinstance(task["budget"], Mapping):
        raise ValueError("Frozen task budget must be an object")
    for name, limit in (("maximum_action_decisions", 150), ("maximum_model_calls", 30000)):
        if type(task.get(name)) is not int or not 1 <= task[name] <= limit:
            raise ValueError(f"Frozen task requires {name} in [1, {limit}]")
    output_dir = Path(output_dir).resolve()
    if type(output_reserved) is not bool:
        raise ValueError("output_reserved must be boolean")
    if output_reserved:
        private = output_dir / "private"
        inputs = private / "inputs"
        if (not output_dir.is_dir() or set(output_dir.iterdir()) != {private}
                or private.is_symlink() or not private.is_dir()
                or set(private.iterdir()) != {inputs} or inputs.is_symlink() or not inputs.is_dir()):
            raise FileExistsError("Reserved output must contain only private/inputs")
    elif output_dir.exists():
        raise FileExistsError("Episode output must be a new directory")
    provider = create_provider(model_config)
    scene = SceneClient(
        ready_path=Path(ready_path), runtime_root=Path(runtime_root),
        expected_task_id=task["task_id"], expected_task_name=task["task_name"],
        expected_split=task["task_split"], expected_instance_id=task["task_instance_id"],
    )
    if not output_reserved:
        output_dir.mkdir(parents=True)
    record = {
        "schema_version": "esi-benchmark/run-v1", "mode": "live", "profile": PROFILE,
        "source_revisions": REVISIONS, "experiment_id": experiment_id,
        "public_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task_identity": {k: task[k] for k in ("task_id", "task_name", "task_split", "task_instance_id", "task_seed")},
        "model": validate_model_config(model_config), "history_frames": history_frames,
        "evaluation_variant": "frozen_history_5" if history_frames == 5 else "history_20_variant_not_baseline",
        "maximum_action_decisions": task["maximum_action_decisions"],
        "maximum_model_calls": task["maximum_model_calls"],
        "evidence_scope": "live_execution_requested; inspect final private score for success",
        "official_benchmark_eligible": False, "strict_production_eligible": False,
        "release_qualification": "candidate_live_pending; historical release gate remains unresolved",
    }
    if "budget" in task:
        record["declared_budget"] = dict(task["budget"])
        if "wall_clock_seconds" in task["budget"]:
            record["wall_clock_enforcement"] = "not_implemented_in_frozen_bench"
    (output_dir / "run_record.json").write_text(json.dumps(record, indent=2) + "\n")
    summary = run_model_episode(
        scene=scene, provider=provider, experiment_id=experiment_id,
        instruction=task["instruction"], task_id=task["task_id"],
        task_name=task["task_name"], task_split=task["task_split"],
        task_instance_id=task["task_instance_id"], task_seed=task["task_seed"],
        instruction_provenance=task.get("instruction_provenance"),
        maximum_action_decisions=task["maximum_action_decisions"],
        maximum_model_calls=task["maximum_model_calls"],
        physical_history_max=history_frames, output_dir=output_dir,
    )
    result = _archive_scene_result(scene.scene_run_dir, output_dir, summary, task)
    summary = {**summary, "candidate_result": result}
    from scene_protocol import _write_json
    _write_json(output_dir / "summary.json", summary)
    return {"output_dir": str(output_dir), "mode": "live", "summary": summary}


def recorded_score_result(score: Mapping[str, Any], episode_id: str) -> dict[str, Any]:
    """Interpret the original final record conservatively without recomputing Q."""
    if not isinstance(episode_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", episode_id):
        raise ValueError("Final score episode identity is invalid")
    if not isinstance(score, Mapping) or score.get("episode_id") != episode_id:
        raise ValueError("Final score episode identity mismatch")
    if type(score.get("official_success")) is not bool or type(score.get("verifier_ok")) is not bool:
        raise ValueError("Final score lacks boolean verifier status")
    q = score.get("official_q")
    if q is not None and (type(q) not in (int, float) or not 0 <= q <= 1 or not math.isfinite(q)):
        raise ValueError("Final score Q must be null or a finite number in [0, 1]")
    if "task_success" in score and type(score["task_success"]) is not bool:
        raise ValueError("Final task_success must be boolean")
    process = score.get("public_process_report")
    if process is not None and (not isinstance(process, Mapping)
            or type(process.get("operational")) is not bool
            or type(process.get("public_process_satisfied")) is not bool
            or not isinstance(process.get("violations"), list)
            or not isinstance(process.get("errors"), list)):
        raise ValueError("Final process report is malformed")
    valid = score["verifier_ok"] and score.get("integrity_failure") is None
    success = valid and score["official_success"] and score.get("task_success", True)
    if process is not None:
        success = success and process["operational"] and process["public_process_satisfied"] and not process["errors"] and not process["violations"]
    if score.get("native_q_alone_is_success") is False:
        success = success and score.get("physical_qualification") == "PASS"
    return {"status": "recorded", "episode_id": episode_id, "official_q": q,
            "official_success": bool(success), "recorded_official_success": score["official_success"],
            "verifier_ok": score["verifier_ok"], "integrity_valid": score.get("integrity_failure") is None,
            "task_success": score.get("task_success"), "termination_reason": score.get("termination_reason"),
            "public_process_satisfied": None if process is None else process["public_process_satisfied"],
            "physical_qualification": score.get("physical_qualification")}


def _archive_scene_result(scene_run_dir: Path, output_dir: Path, summary: Mapping[str, Any], task: Mapping[str, Any]) -> dict[str, Any]:
    from scene_protocol import _write_json
    root = Path(scene_run_dir).resolve()
    episode_id = summary.get("current_scene_episode_id")
    private = output_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    _write_json(private / "task_binding.json", dict(task))
    if episode_id is None:
        return {"status": "unavailable", "episode_id": None, "official_success": None,
                "official_q": None, "reason": "Upstream episode failed before establishing a score identity", "evidence": {}}
    if not isinstance(episode_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", episode_id):
        raise ValueError("Final scene episode identity is invalid")
    evidence = {}
    for name in ("final_score.json", "oracle_prepared_restore.json", "task_process_result.json"):
        source = root / "episodes" / episode_id / "private" / name
        if not source.exists():
            continue
        if source.is_symlink() or not source.resolve().is_relative_to(root) or not source.is_file() or source.stat().st_size > 512 * 1024:
            raise ValueError("Unsafe or oversized scene result evidence")
        raw = source.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Scene result evidence must be a JSON object")
        destination = private / ("scene_final_score.json" if name == "final_score.json" else name)
        if destination.exists():
            if destination.is_symlink() or destination.read_bytes() != raw:
                raise ValueError("Previously archived scene evidence differs")
        else:
            with destination.open("xb") as stream:
                stream.write(raw)
            destination.chmod(0o600)
        evidence[name] = {"source": str(source), "path": str(destination.relative_to(output_dir)),
                          "sha256": hashlib.sha256(raw).hexdigest()}
    if "final_score.json" not in evidence:
        return {"status": "unavailable", "episode_id": episode_id, "official_success": None,
                "official_q": None, "reason": "Scene did not produce a final score", "evidence": evidence}
    score = json.loads((private / "scene_final_score.json").read_text())
    result = recorded_score_result(score, episode_id)
    result["evidence"] = evidence
    return result
