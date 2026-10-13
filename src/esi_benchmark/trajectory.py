"""Bounded inspection of local runs, without exporting private trajectory data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_SCORE_BYTES = 512 * 1024
_MODES = {"live", "offline_replay", "unknown_legacy"}
_VARIANT_HISTORY = {"frozen_history_5": 5, "history_20_variant_not_baseline": 20}


def _path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Artifact reference must be a relative path")
    path = root / relative
    if (".." in Path(relative).parts or "\\" in relative
            or not path.resolve().is_relative_to(root)
            or any(part.is_symlink() for part in (path, *path.parents) if part != root and root in part.parents)):
        raise ValueError("Artifact reference escaped the run directory")
    return path


def _json(raw, name):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"Duplicate JSON field in run artifact: {name}")
            value[key] = item
        return value
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON value")))
    except RecursionError as exc:
        raise ValueError(f"Run artifact exceeds JSON nesting limit: {name}") from exc


def _read(root: Path, relative: str) -> dict[str, Any]:
    path = _path(root, relative)
    if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"Missing or oversized run artifact: {relative}")
    value = _json(path.read_bytes(), relative)
    if not isinstance(value, dict):
        raise ValueError(f"Run artifact must be an object: {relative}")
    return value


def _evidence(root, candidate):
    from .models import EVIDENCE_SCHEMA
    schema = candidate.get("evidence_schema_version")
    if schema not in (None, EVIDENCE_SCHEMA):
        raise ValueError("Unsupported result evidence schema")
    records = candidate.get("evidence", {})
    if not isinstance(records, dict) or len(records) > 65:
        raise ValueError("candidate_result.evidence must be a bounded object")
    if schema == EVIDENCE_SCHEMA and "task_binding.json" not in records:
        raise ValueError("Result evidence lacks its archived task binding")
    retained = {"final_score.json", "task_binding.json", "task_process_result.json",
                "v_collection_identity.json", "v_protocol_result.json", "l2_v/final.json",
                "l2_v/checkpoint_violations.json"}
    documents = {}
    for name, evidence in records.items():
        if not isinstance(name, str) or not isinstance(evidence, dict):
            raise ValueError("candidate_result.evidence entries must be named objects")
        if schema == EVIDENCE_SCHEMA and (
                set(evidence) != {"source", "path", "sha256", "size_bytes"}
                or not isinstance(evidence["source"], str) or not evidence["source"]):
            raise ValueError("Scene evidence descriptor is malformed")
        suffix = Path(name).suffix
        if (suffix not in {".json", ".jsonl"} or Path(name).is_absolute()
                or ".." in Path(name).parts or "\\" in name):
            raise ValueError("Invalid scene evidence name")
        relative = evidence.get("path")
        path = _path(root, relative)
        expected = "private/" + ("scene_final_score.json" if name == "final_score.json" else
                                  "v_scene/" + name if name.startswith("v_") else name)
        if schema == EVIDENCE_SCHEMA and relative != expected:
            raise ValueError("Scene evidence name differs from its archived path")
        limit = MAX_JSON_BYTES if name.startswith(("v_", "l2_v/")) or suffix == ".jsonl" else MAX_SCORE_BYTES
        if not path.is_file() or path.stat().st_size > limit:
            raise ValueError(f"Archived scene evidence is missing or oversized: {name}")
        raw = path.read_bytes()
        if len(raw) > limit:
            raise ValueError(f"Archived scene evidence is oversized: {name}")
        digest = evidence.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Archived scene evidence hash mismatch: {name}")
        size = evidence.get("size_bytes")
        if (schema == EVIDENCE_SCHEMA or "size_bytes" in evidence) and (type(size) is not int or size != len(raw)):
            raise ValueError(f"Archived scene evidence size mismatch: {name}")
        if suffix == ".jsonl":
            count = 0
            for row in raw.splitlines():
                if row.strip():
                    if not isinstance(_json(row, name), dict):
                        raise ValueError(f"Scene JSONL evidence must contain objects: {name}")
                    count += 1
            documents[name] = count
        else:
            document = _json(raw, name)
            types = (dict, list) if name == "l2_v/checkpoint_violations.json" else (dict,)
            if not isinstance(document, types):
                raise ValueError(f"Invalid scene evidence structure: {name}")
            if name in retained:
                documents[name] = document
    return documents


def _v_identity(documents, score, candidate, record, summary, manifest):
    from .assets import asset_manifest
    from .models import EVIDENCE_SCHEMA
    from .profiles import source_revisions
    from .tasks import get_task
    if candidate.get("evidence_schema_version") != EVIDENCE_SCHEMA:
        raise ValueError("V identity verification requires result-evidence-v2")
    identity = documents.get("v_collection_identity.json")
    protocol = documents.get("v_protocol_result.json")
    binding = documents.get("task_binding.json")
    if not isinstance(identity, dict) or not isinstance(protocol, dict) or not isinstance(binding, dict):
        raise ValueError("V result lacks its identity, protocol or task binding evidence")
    identity_fields = {"collection_config_key", "task_spec_id", "package_id", "branch", "runtime_revision",
                       "historical_oracle_runtime_revision", "source_oracle_success", "runtime_validation_only",
                       "v_package_sha256"}
    if (set(identity) != identity_fields
            or any(not isinstance(identity[field], str) or not identity[field] for field in
                   ("collection_config_key", "task_spec_id", "package_id", "branch", "runtime_revision", "v_package_sha256"))
            or not re.fullmatch(r"[0-9a-f]{64}", identity["v_package_sha256"])
            or type(identity["source_oracle_success"]) is not bool
            or type(identity["runtime_validation_only"]) is not bool
            or (identity["historical_oracle_runtime_revision"] is not None and
                (not isinstance(identity["historical_oracle_runtime_revision"], str)
                 or not re.fullmatch(r"[0-9a-f]{40}", identity["historical_oracle_runtime_revision"])))):
        raise ValueError("V scene identity lacks its original frozen identity fields")
    key = identity.get("collection_config_key")
    if not isinstance(key, str) or not key:
        raise ValueError("V collection identity is missing")
    task = get_task(key, "L1-V")
    profile = record["profile"]
    seed = asset_manifest(profile)["inputs_by_task"][key]["seed"]
    revisions = source_revisions(profile)
    if (task["profile"] != profile or identity.get("branch") != "D1"
            or identity.get("task_spec_id") != task["runtime_instruction_id"]
            or identity.get("runtime_revision") != revisions["runtime"]
            or record.get("source_revisions") != revisions
            or binding.get("collection_config_key") != key or binding.get("v_identity") != identity
            or score.get("collection_config_key") != key or score.get("v_protocol") != protocol
            or ("branch" in protocol and protocol["branch"] != identity["branch"])):
        raise ValueError("V scene identity or protocol differs from its frozen task binding")
    expected = {"task_id": task["source"]["row_index"], "task_name": task["task_id"], "task_seed": seed}
    if any(type(binding.get(field)) is not type(value) or binding.get(field) != value for field, value in expected.items()):
        raise ValueError("V archived task differs from its frozen public identity")
    fields = ("task_id", "task_name", "task_split", "task_instance_id", "task_seed")
    run_identity = record.get("task_identity")
    if (not isinstance(run_identity, dict)
            or any(not isinstance(binding.get(field), str) or not binding[field] for field in ("task_split", "task_instance_id"))
            or any(field not in binding or type(run_identity.get(field)) is not type(binding[field])
                   or run_identity.get(field) != binding[field] or type(manifest.get(field)) is not type(binding[field])
                   or manifest.get(field) != binding[field] for field in fields)
            or binding.get("instruction") != task["instruction_en"]
            or manifest.get("instruction_sha256") != task["instruction_sha256"]):
        raise ValueError("V model task or instruction differs from the archived task binding")
    model = record.get("model")
    if (not isinstance(model, dict) or not isinstance(model.get("model"), str) or not model["model"]
            or model["model"] != manifest.get("model") or model["model"] != summary.get("model")
            or summary.get("current_scene_episode_id") != candidate.get("episode_id")
            or summary.get("v_protocol_complete") is not candidate.get("v_protocol_complete")
            or summary.get("official_task_success") is not candidate.get("official_success")):
        raise ValueError("V archived model result disagrees with its task/protocol evidence")


def inspect_run(run_dir: str | Path) -> dict[str, Any]:
    """Inspect public images/provenance and preserve explicitly incomplete runs.

    Legacy runs may omit run_record.json; their execution mode remains unknown.
    Private transcripts and raw provider responses are never returned.
    """
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise ValueError("Run directory does not exist")
    summary = _read(root, "summary.json")
    if not isinstance(summary.get("status"), str) or not summary["status"]:
        raise ValueError("Run summary has no terminal status")
    record = _read(root, "run_record.json") if _path(root, "run_record.json").is_file() else {}
    mode = record.get("mode", "unknown_legacy")
    if not isinstance(mode, str) or mode not in _MODES:
        raise ValueError("Unknown run evidence mode")
    variant = record.get("evaluation_variant", "unknown")
    if not isinstance(variant, str) or variant not in {*_VARIANT_HISTORY, "unknown"}:
        raise ValueError("Unknown evaluation variant")
    if variant in _VARIANT_HISTORY and summary.get("physical_history_max") not in (None, _VARIANT_HISTORY[variant]):
        raise ValueError("Evaluation variant disagrees with recorded physical history")
    for field in ("source_revisions", "task_identity", "model", "declared_budget"):
        if record.get(field) is not None and not isinstance(record[field], dict):
            raise ValueError(f"run_record.{field} must be an object or null")
    if "profile" in record and (not isinstance(record["profile"], str) or not record["profile"]):
        raise ValueError("run_record.profile must be a nonempty string")
    failure = None
    if _path(root, "candidate_error.json").is_file():
        error = _read(root, "candidate_error.json")
        for field in ("status", "stage", "error_type"):
            if not isinstance(error.get(field), str) or not error[field]:
                raise ValueError(f"candidate_error.{field} must be a nonempty string")
        failure = {field: error[field] for field in ("status", "stage", "error_type")}
    candidate = summary.get("candidate_result")
    documents = {}
    if candidate is not None:
        if (not isinstance(candidate, dict) or not isinstance(candidate.get("status"), str)
                or candidate["status"] not in {"recorded", "unavailable"}):
            raise ValueError("Candidate result is malformed")
        documents = _evidence(root, candidate)
        if candidate["status"] == "unavailable" and (
                candidate.get("official_success") is not None or candidate.get("official_q") is not None):
            raise ValueError("Unavailable result must not contain a score")
    result = {
        "schema_version": "esi-benchmark/trajectory-inspection-v1", "run_dir": str(root),
        "mode": mode, "evaluation_variant": variant, "client_status": summary["status"], "official_success": None,
        "official_q": None, "score_available": False,
        "model_calls": summary.get("model_calls"), "actions_sent_to_scene": summary.get("actions_sent_to_scene"),
        "macro_actions_resolved": summary.get("macro_actions_resolved"),
        "source_revisions": record.get("source_revisions"), "profile": record.get("profile"),
        "official_benchmark_eligible": record.get("official_benchmark_eligible") is True,
        "strict_production_eligible": record.get("strict_production_eligible") is True,
        "private_artifacts_present": any((root / name).exists() for name in ("private", "conversation_private.json", "provider_calls")),
    }
    if failure is not None:
        result["candidate_error"] = failure
    if not _path(root, "conversation_chatml.json").is_file():
        unavailable = _path(root, "conversation_unavailable.json")
        if unavailable.is_file():
            reason = _read(root, "conversation_unavailable.json").get("reason")
            if not isinstance(reason, str) or not reason:
                raise ValueError("Conversation-unavailable record lacks a reason")
        elif summary["status"] == "INITIALIZATION_FAILED":
            reason = "initialization_failed"
        elif failure is not None:
            reason = "candidate_failure"
        else:
            raise ValueError("Missing ChatML without an explicit initialization failure record")
        return {**result, "status": "INCOMPLETE", "messages": None, "images": None,
                "completeness": "candidate_failure" if failure is not None else (
                    "initialization_failed" if summary["status"] == "INITIALIZATION_FAILED" else "conversation_unavailable"),
                "reason": reason}
    chatml = _read(root, "conversation_chatml.json")
    manifest = _read(root, "client_manifest.json")
    if chatml.get("schema_version") != "behavior-hl-action20-chatml-v3" or not isinstance(chatml.get("messages"), list):
        raise ValueError("Unsupported or malformed ChatML trajectory")
    if not 1 <= len(chatml["messages"]) <= 100000:
        raise ValueError("Trajectory message count exceeds the inspection bounds")
    images = {}
    for message in chatml["messages"]:
        if (not isinstance(message, dict) or not isinstance(message.get("role"), str)
                or message["role"] not in {"system", "user", "assistant", "tool"}):
            raise ValueError("Malformed trajectory message")
        content = message.get("content")
        if not isinstance(content, list):
            if not isinstance(content, str):
                raise ValueError("Malformed trajectory message content")
            continue
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("Malformed trajectory content block")
            if item.get("type") == "image":
                relative = item.get("image")
                image = _path(root, relative)
                if not image.is_file() or image.stat().st_size > MAX_IMAGE_BYTES:
                    raise ValueError("Missing or oversized trajectory image")
                images[relative] = hashlib.sha256(image.read_bytes()).hexdigest()
    prompt_path = _path(root, "system_prompt.txt")
    if not prompt_path.is_file() or prompt_path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Missing or oversized system prompt")
    prompt_sha = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    prompt_provenance = manifest.get("prompt_provenance", {})
    if not isinstance(prompt_provenance, dict):
        raise ValueError("client_manifest.prompt_provenance must be an object")
    if prompt_sha != summary.get("prompt_sha256") or prompt_sha != prompt_provenance.get("rendered_prompt_sha256"):
        raise ValueError("Rendered system prompt provenance mismatch")
    if not isinstance(manifest.get("source_sha256"), dict) or not manifest["source_sha256"]:
        raise ValueError("Run lacks source hash provenance")
    if candidate is not None and candidate["status"] == "recorded":
        from .models import EVIDENCE_SCHEMA, recorded_score_result
        from .tasks import profile_for
        score = documents.get("final_score.json")
        if score is None:
            raise ValueError("Archived final score is missing")
        profile = record.get("profile", "l0-english-v2")
        family = profile_for(profile)["family"]
        if score.get("collection_config_key") is not None and family != "L1-V":
            raise ValueError("V collection evidence requires its frozen V profile")
        interpreted = recorded_score_result(score, candidate.get("episode_id"), profile=profile)
        if any(type(candidate.get(key)) is not type(value) or candidate.get(key) != value
               for key, value in interpreted.items()):
            raise ValueError("Candidate result differs from original final score")
        strict = candidate.get("evidence_schema_version") == EVIDENCE_SCHEMA
        for name, field in (("task_process_result.json", "public_process_report"), ("l2_v/final.json", "v_protocol")):
            required = strict and field in score and (field != "v_protocol" or family == "L2")
            if required and name not in documents:
                raise ValueError(f"Archived final score lacks supporting evidence: {name}")
            if name in documents and documents[name] != score.get(field):
                raise ValueError(f"Archived supporting evidence differs from final score: {name}")
        process = score.get("public_process_report")
        if strict and process is not None:
            for field, name in (("execution_events", "task_process_events.jsonl"),
                                ("existing_frame_references", "task_evidence_index.jsonl")):
                count = process.get(field)
                if type(count) is not int or count < 0 or documents.get(name, 0) != count:
                    raise ValueError(f"Archived process evidence count mismatch: {name}")
        if "l2_v/checkpoint_violations.json" in documents:
            protocol = score.get("v_protocol", {})
            if isinstance(protocol, dict) and "checkpoint_violations" in protocol and documents["l2_v/checkpoint_violations.json"] != protocol["checkpoint_violations"]:
                raise ValueError("Archived L2 checkpoint evidence differs from final score")
        if family == "L1-V":
            _v_identity(documents, score, candidate, record, summary, manifest)
    if failure is not None:
        return {**result, "status": "INCOMPLETE", "messages": len(chatml["messages"]), "images": len(images),
                "completeness": "candidate_failure", "reason": "Candidate execution failed; see candidate_error.json"}
    if mode == "live" and candidate is not None and candidate["status"] == "recorded":
        result.update(official_success=candidate["official_success"], official_q=candidate["official_q"], score_available=True)
    elif mode == "unknown_legacy":
        result.update(official_success=summary.get("official_task_success", summary.get("official_success")), official_q=summary.get("q_score"))
        result["score_available"] = type(result["official_success"]) is bool
    return {
        **result, "status": "PASS", "messages": len(chatml["messages"]), "images": len(images),
        "image_sha256": images, "source_hashes_recorded": len(manifest["source_sha256"]), "prompt_sha256": prompt_sha,
        "public_projection": "conversation_chatml.json and relative images; review model-authored text before sharing",
        "completeness": "bounded single-run ChatML and provenance inspection; private evaluator and source files are not re-executed",
    }


def summarize_runs(run_dirs: Iterable[str | Path]) -> dict[str, Any]:
    """Keep invalid/incomplete attempts visible and report both Live denominators."""
    runs = []
    for path in run_dirs:
        try:
            runs.append(inspect_run(path))
        except (ValueError, OSError) as exc:
            root = Path(path).resolve()
            mode = "unknown_legacy"
            variant = "unknown"
            try:
                record = _read(root, "run_record.json")
                recorded_mode = record.get("mode")
                recorded_variant = record.get("evaluation_variant")
                if isinstance(recorded_variant, str) and recorded_variant in _VARIANT_HISTORY:
                    variant = recorded_variant
                if isinstance(recorded_mode, str) and recorded_mode in _MODES:
                    mode = recorded_mode
            except (ValueError, OSError):
                pass
            runs.append({"run_dir": str(root), "mode": mode, "evaluation_variant": variant, "status": "INVALID", "error": str(exc),
                         "score_available": False, "official_success": None, "completeness": "invalid_artifacts"})
    if not runs:
        raise ValueError("Supply at least one run directory")
    counts = {mode: sum(run["mode"] == mode for run in runs) for mode in sorted(_MODES)}
    variant_counts = {variant: sum(run["evaluation_variant"] == variant for run in runs)
                      for variant in sorted({run["evaluation_variant"] for run in runs})}
    scored = [run for run in runs if run["mode"] == "live" and run["score_available"]]
    successes = sum(run["official_success"] is True for run in scored)
    live = counts["live"]
    return {
        "schema_version": "esi-benchmark/run-summary-v1", "runs": runs, "counts": counts,
        "evaluation_variant_counts": variant_counts, "mixed_evaluation_variants": len(variant_counts) > 1,
        "total_live_attempts": live, "scored_live_runs": len(scored), "successful_live_runs": successes,
        "success_rate_scored_live": successes / len(scored) if scored else None,
        "success_rate_all_live_attempts": successes / live if live else None,
        "live_score_coverage": len(scored) / live if live else None,
        "unscored_live_runs": live - len(scored),
        "invalid_runs": sum(run["status"] == "INVALID" for run in runs),
        "incomplete_runs": sum(run["status"] == "INCOMPLETE" for run in runs),
        "denominator_policy": "report all declared Live attempts and score-covered Live separately; offline/unknown excluded from Live rates",
    }


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """Validate recorded outputs; incomplete attempts are reported as INCOMPLETE."""
    return inspect_run(run_dir)
