"""Canonical, model-first artifact layout for ``bench_try`` model runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping


RUN_LAYOUT_VERSION = "behavior-hl-bench-try-run-layout/v1"
TASK25_ARTIFACT_KEY = "task25_clearing_food_from_table_into_fridge"
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
class RunArtifactLayoutError(ValueError):
    pass


def allocate_incomplete_run(
    runs_root: Path,
    *,
    model: str,
    task: str,
    experiment_id: str,
) -> Path:
    """Create one fresh run beneath the canonical ``incomplete`` bucket."""

    root = Path(runs_root).resolve()
    model_key = _component(model, field="model")
    task_key = _component(task, field="task")
    experiment_key = _component(experiment_id, field="experiment")
    task_root = root / model_key / task_key
    for directory in (
        task_root / "completed" / "success",
        task_root / "completed" / "failure",
        task_root / "incomplete",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    output = task_root / "incomplete" / experiment_key
    output.mkdir(parents=False, exist_ok=False)
    return output


def finalize_completed_run(
    output_dir: Path,
    *,
    summary: Mapping[str, Any],
) -> Path:
    """Atomically classify a terminal run after all artifacts are flushed."""

    output = Path(output_dir).resolve()
    if output.parent.name != "incomplete" or output.parent.parent.name == "":
        raise RunArtifactLayoutError("run is not in the canonical incomplete bucket")
    status = summary.get("status")
    official_task_success = (
        status == "DONE"
        and summary.get("terminated_by_official_success") is True
    )
    if summary.get('native_q_alone_is_success') is False:
        official_task_success = (
            status == 'DONE' and summary.get('private_score_received_from_scene') is True
            and summary.get('official_task_success') is True
            and summary.get('verifier_ok') is True
            and summary.get('physical_qualification') == 'PASS')
    if official_task_success:
        outcome = "success"
    elif status == "DONE":
        outcome = "failure"
    elif isinstance(status, str) and status.startswith("FAIL_"):
        outcome = "failure"
    else:
        raise RunArtifactLayoutError("summary has no recognized terminal status")
    task_root = output.parent.parent
    destination = task_root / "completed" / outcome / output.name
    if destination.exists():
        raise RunArtifactLayoutError("terminal run destination already exists")
    _write_json(
        output / "run_classification.json",
        {
            "schema_version": RUN_LAYOUT_VERSION,
            "classified_at": datetime.now(timezone.utc).isoformat(),
            "bucket": f"completed/{outcome}",
            "classification_basis": "official_task_success_required",
            "client_status": status,
            "official_task_success": official_task_success,
            "official_score_received": bool(
                summary.get("private_score_received_from_scene")
            ),
            "success_semantics": (
                "scene reported canonical official task success"
                if outcome == "success"
                else (
                    "model called done without canonical official task success"
                    if status == "DONE"
                    else "client reached a recorded terminal failure status"
                )
            ),
            "failure_reason": (
                "model_declared_done_without_official_success"
                if status == "DONE" and not official_task_success
                else None
            ),
        },
    )
    output.replace(destination)
    return destination


def _component(value: str, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_COMPONENT.fullmatch(value) is None:
        raise RunArtifactLayoutError(f"{field} artifact component is invalid")
    return value


def _write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
