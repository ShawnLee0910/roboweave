"""Private provider and episode implementation within one frozen worker."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlsplit

from .bootstrap import activate
from ..profiles import DEFAULT_PROFILE, source_revisions
from ..models import _archive_scene_result

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
    provider = _provider(result)
    if provider.requires_api_key and key_env is None:
        raise ValueError("This endpoint requires model.api_key_env")
    result["model"] = provider.model
    return result


def _provider(config: Mapping[str, Any]):
    activate()
    from model_client import OpenAICompatibleProvider
    return OpenAICompatibleProvider(
        model=config["model"], base_url=config["endpoint"],
        provider_api=config.get("api", "chat_completions"),
        timeout_seconds=config.get("timeout", 300),
        api_key_file=None,
        api_key_env=config.get("api_key_env") or "ROBOWEAVE_UNCONFIGURED_API_KEY",
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
    profile: str = DEFAULT_PROFILE,
    source_verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a frozen Action20 episode on a caller-owned, independently started scene.

    ``task`` contains task_id (numeric public handle), task_name, task_split,
    task_instance_id, task_seed, instruction and optional instruction_provenance.
    The caller owns the resident process and must always stop its own child.
    """
    activate(profile)
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
    for name, limit in (("maximum_action_decisions", 3000), ("maximum_model_calls", 30000)):
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
    from ..tasks import validate_model_task
    validate_model_task(profile, task)
    provider = create_provider(model_config)
    scene = SceneClient(
        ready_path=Path(ready_path), runtime_root=Path(runtime_root),
        expected_task_id=task["task_id"], expected_task_name=task["task_name"],
        expected_split=task["task_split"], expected_instance_id=task["task_instance_id"],
    )
    if not output_reserved:
        output_dir.mkdir(parents=True)
    record = {
        "schema_version": "roboweave/run-v1", "mode": "live", "profile": profile,
        "source_revisions": source_revisions(profile), "experiment_id": experiment_id,
        "source_verification": source_verification,
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
            record["wall_clock_enforcement"] = ("owned_common_safe_point_watchdog/model_wall_timeout-v1"
                                                if profile == "l2-collection37" else "not_implemented_in_frozen_bench")
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
        # Frozen Common launches pause on provider/protocol failure for every profile.
        pause_on_client_failure=True,
    )
    result = _archive_scene_result(scene.scene_run_dir, output_dir, summary, task, profile=profile)
    summary = {**summary, "candidate_result": result}
    from scene_protocol import _write_json
    _write_json(output_dir / "summary.json", summary)
    return {"output_dir": str(output_dir), "mode": "live", "summary": summary}


