"""Public model configuration and episode APIs with isolated frozen workers."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .profiles import DEFAULT_PROFILE

EVIDENCE_SCHEMA = "esi-benchmark/result-evidence-v2"


def validate_model_config(config: Mapping[str, Any], *, profile: str = DEFAULT_PROFILE) -> dict[str, Any]:
    """Validate with the selected frozen provider without issuing a request."""
    from .engine import call
    if not isinstance(config, Mapping):
        raise ValueError("Model configuration must be an object")
    return call("validate_model", {"model_config": dict(config)}, profile=profile)


def run_episode(
    *, model_config: Mapping[str, Any], task: Mapping[str, Any],
    ready_path: Path, runtime_root: Path, output_dir: Path,
    experiment_id: str, history_frames: int = 5, output_reserved: bool = False,
    profile: str = DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Run the frozen loop in a child; the caller owns its simulator process."""
    from .engine import call
    if not isinstance(model_config, Mapping) or not isinstance(task, Mapping):
        raise ValueError("Model configuration and task binding must be objects")
    return call("episode", {
        "model_config": dict(model_config), "task": dict(task),
        "ready_path": str(ready_path), "runtime_root": str(runtime_root),
        "output_dir": str(output_dir), "experiment_id": experiment_id,
        "history_frames": history_frames, "output_reserved": output_reserved,
    }, profile=profile)


def recorded_score_result(score: Mapping[str, Any], episode_id: str, *, profile: str = DEFAULT_PROFILE) -> dict[str, Any]:
    """Interpret the original final record conservatively without recomputing Q."""
    from .tasks import profile_for
    family = profile_for(profile)["family"]
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
    if family == "L1-P" or (family != "L2" and score.get("native_q_alone_is_success") is False):
        success = success and score.get("physical_qualification") == "PASS"
    v_report = score.get("v_protocol")
    if v_report is not None and (not isinstance(v_report, Mapping)
            or type(v_report.get("v_protocol_complete")) is not bool
            or ("operational" in v_report and type(v_report["operational"]) is not bool)):
        raise ValueError("Final V protocol report is malformed")
    if family == "L1-V":
        if v_report is None:
            raise ValueError("Final V score lacks its protocol report")
        success = success and v_report["v_protocol_complete"] and v_report.get("operational") is not False
    v_complete = None if v_report is None else v_report["v_protocol_complete"]
    if family == "L1-V":
        v_complete = v_complete and v_report.get("operational") is not False and valid
    physical = score.get("physical_template")
    if physical is not None and (not isinstance(physical, Mapping)
            or type(physical.get("task_success")) is not bool):
        raise ValueError("Final Physical template report is malformed")
    if family == "L2":
        if (type(score.get("task_success")) is not bool or not isinstance(score.get("status"), str)
                or score["status"] not in {"PASS", "FAIL"}):
            raise ValueError("Final L2 score lacks the native combined outcome")
        # L2 end_episode has already combined its canonical, Physical and V gates.
        success = score["task_success"] and score["status"] == "PASS" and score.get("integrity_failure") is None
        if v_report is not None and score.get("v_protocol_complete") is not v_report["v_protocol_complete"]:
            raise ValueError("Final L2 V completion fields disagree")
        if physical is not None and score.get("physical_template_complete") is not physical["task_success"]:
            raise ValueError("Final L2 Physical completion fields disagree")
        if success and ((v_report is not None and not v_report["v_protocol_complete"])
                        or (physical is not None and not physical["task_success"])
                        or not score["verifier_ok"]
                        or (process is not None and (not process["operational"] or not process["public_process_satisfied"]
                                                    or process["errors"] or process["violations"]))):
            raise ValueError("Final L2 combined outcome contradicts its evidence")
    return {"status": "recorded", "episode_id": episode_id, "official_q": q,
            "official_success": bool(success), "recorded_official_success": score["official_success"],
            "verifier_ok": score["verifier_ok"], "integrity_valid": score.get("integrity_failure") is None,
            "task_success": score.get("task_success"), "termination_reason": score.get("termination_reason"),
            "public_process_satisfied": None if process is None else process["public_process_satisfied"],
            "physical_qualification": score.get("physical_qualification"),
            "v_protocol_complete": v_complete,
            "physical_template_complete": None if physical is None else physical["task_success"]}


def _archive_scene_result(scene_run_dir: Path, output_dir: Path, summary: Mapping[str, Any], task: Mapping[str, Any],
                          *, profile: str = DEFAULT_PROFILE) -> dict[str, Any]:
    from .tasks import profile_for
    family = profile_for(profile)["family"]
    root = Path(scene_run_dir).resolve()
    episode_id = summary.get("current_scene_episode_id")
    private = output_dir / "private"
    private.mkdir(parents=True, mode=0o700, exist_ok=True)
    binding = private / "task_binding.json"
    temporary = binding.with_suffix(".json.tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        stream.write(json.dumps(dict(task), sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(binding)
    raw_binding = binding.read_bytes()
    evidence = {"task_binding.json": {"source": str(binding), "path": "private/task_binding.json",
                "sha256": hashlib.sha256(raw_binding).hexdigest(), "size_bytes": len(raw_binding)}}
    if episode_id is None:
        return {"status": "unavailable", "episode_id": None, "official_success": None,
                "official_q": None, "reason": "Upstream episode failed before establishing a score identity",
                "evidence_schema_version": EVIDENCE_SCHEMA, "evidence": evidence}
    if not isinstance(episode_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", episode_id):
        raise ValueError("Final scene episode identity is invalid")
    scene_private = root / "episodes" / episode_id / "private"
    names = ["final_score.json", "oracle_prepared_restore.json", "task_process_result.json",
             "task_process_events.jsonl", "task_evidence_index.jsonl", "physical_initial_binding.json"]
    if family in {"L1-V", "L2"}:
        names += ["v_collection_identity.json", "v_protocol_result.json", "v_marker_bindings.jsonl"]
        folders = ["v_intervention"] + (["l2_v"] if family == "L2" else [])
        for folder in folders:
            intervention = scene_private / folder
            if intervention.is_symlink():
                raise ValueError("Unsafe V intervention directory")
            if intervention.is_dir():
                for item in intervention.iterdir():
                    if item.suffix in {".json", ".jsonl"}:
                        names.append(str(item.relative_to(scene_private)))
                    if len(names) > 64:
                        raise ValueError("V archive file limit exceeded")
    for name in names:
        source = scene_private / name
        if not source.exists():
            continue
        limit = 32 * 1024 * 1024 if name.startswith(("v_", "l2_v/")) or source.suffix == ".jsonl" else 512 * 1024
        if source.is_symlink() or not source.resolve().is_relative_to(root) or not source.is_file() or source.stat().st_size > limit:
            raise ValueError("Unsafe or oversized scene result evidence")
        raw = source.read_bytes()
        rows = [row for row in raw.splitlines() if row.strip()] if source.suffix == ".jsonl" else [raw]
        types = (dict, list) if name == "l2_v/checkpoint_violations.json" else (dict,)
        if any(not isinstance(json.loads(row), types) for row in rows):
            raise ValueError("Scene result evidence has an invalid JSON structure")
        relative = ("scene_final_score.json" if name == "final_score.json" else
                    "v_scene/" + name if name.startswith("v_") else name)
        destination = private / relative
        destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        if destination.exists():
            if destination.is_symlink() or destination.read_bytes() != raw:
                raise ValueError("Previously archived scene evidence differs")
        else:
            with os.fdopen(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
                stream.write(raw)
        evidence[name] = {"source": str(source), "path": str(destination.relative_to(output_dir)),
                          "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    if "final_score.json" not in evidence:
        return {"status": "unavailable", "episode_id": episode_id, "official_success": None,
                "official_q": None, "reason": "Scene did not produce a final score",
                "evidence_schema_version": EVIDENCE_SCHEMA, "evidence": evidence}
    score = json.loads((private / "scene_final_score.json").read_text())
    result = recorded_score_result(score, episode_id, profile=profile)
    if family == "L1-V":
        if not {"v_collection_identity.json", "v_protocol_result.json"} <= set(evidence):
            raise ValueError("V result lacks its identity or protocol evidence")
        identity = json.loads((output_dir / evidence["v_collection_identity.json"]["path"]).read_text())
        report = json.loads((output_dir / evidence["v_protocol_result.json"]["path"]).read_text())
        if (identity.get("branch") != "D1" or not isinstance(identity.get("collection_config_key"), str)
                or not identity["collection_config_key"]
                or score.get("collection_config_key") != identity["collection_config_key"]
                or score.get("v_protocol") != report
                or ("collection_config_key" in task and task["collection_config_key"] != identity["collection_config_key"])):
            raise ValueError("V result identity or protocol binding mismatch")
        if (summary.get("v_protocol_complete") is not result["v_protocol_complete"]
                or summary.get("official_task_success") is not result["official_success"]):
            raise ValueError("V native model result disagrees with its archived proof")
    result.update(evidence=evidence, evidence_schema_version=EVIDENCE_SCHEMA)
    return result
