from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


CAMPAIGN_MANIFEST_SCHEMA_VERSION = "behavior-hl-evaluation-campaign/v1"
MAX_CAMPAIGN_MANIFEST_BYTES = 4 * 1024 * 1024
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


class CampaignManifestError(ValueError):
    """The campaign manifest is missing, unsafe, or inconsistent."""


@dataclass(frozen=True)
class CampaignBinding:
    campaign_id: str
    manifest_path: str
    manifest_sha256: str
    experiment_id_prefix: str
    initial_view_mode: str
    scene_ready_path: str
    scene_run_dir: str
    instruction_review_sha256: str | None = None
    bench_commit: str | None = None
    scene_manifest_sha256: str | None = None

    def to_record(self) -> dict[str, Any]:
        record = {
            "campaign_id": self.campaign_id,
            "campaign_manifest_path": self.manifest_path,
            "campaign_manifest_sha256": self.manifest_sha256,
            "campaign_experiment_id_prefix": self.experiment_id_prefix,
            "campaign_initial_view_mode": self.initial_view_mode,
            "campaign_scene_ready_path": self.scene_ready_path,
            "campaign_scene_run_dir": self.scene_run_dir,
        }
        if self.instruction_review_sha256 is not None:
            record["campaign_instruction_review_sha256"] = (
                self.instruction_review_sha256
            )
        if self.bench_commit is not None:
            record["campaign_bench_commit"] = self.bench_commit
        if self.scene_manifest_sha256 is not None:
            record["campaign_scene_manifest_sha256"] = self.scene_manifest_sha256
        return record


def _read_manifest(path: Path) -> tuple[Mapping[str, Any], str, str]:
    target = Path(path).resolve()
    if not target.is_file() or target.is_symlink():
        raise CampaignManifestError("campaign manifest is missing or unsafe")
    descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        payload = os.read(descriptor, MAX_CAMPAIGN_MANIFEST_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(payload) > MAX_CAMPAIGN_MANIFEST_BYTES:
        raise CampaignManifestError("campaign manifest exceeds its bounded size")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignManifestError("campaign manifest is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise CampaignManifestError("campaign manifest root must be an object")
    return value, str(target), hashlib.sha256(payload).hexdigest()


def _safe_id(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise CampaignManifestError(f"{field} is invalid")
    return value


def _absolute_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise CampaignManifestError(f"{field} must be an absolute path")
    return str(Path(value).resolve())


def _reviewed_at(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise CampaignManifestError("instruction review timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CampaignManifestError("instruction review timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise CampaignManifestError("instruction review timestamp has no timezone")
    return value


def load_campaign_manifest(path: Path) -> tuple[Mapping[str, Any], str, str]:
    value, manifest_path, manifest_sha256 = _read_manifest(path)
    if value.get("schema_version") != CAMPAIGN_MANIFEST_SCHEMA_VERSION:
        raise CampaignManifestError("campaign manifest schema version is invalid")
    if value.get("status") != "FROZEN":
        raise CampaignManifestError("campaign manifest is not frozen")
    _safe_id(value.get("campaign_id"), field="campaign_id")
    prefix = value.get("experiment_id_prefix")
    if (
        not isinstance(prefix, str)
        or not prefix
        or len(prefix) > 96
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix) is None
    ):
        raise CampaignManifestError("experiment_id_prefix is invalid")
    counting = value.get("counting_contract")
    if (
        not isinstance(counting, Mapping)
        or counting.get("historical_runs_eligible") is not False
        or counting.get("campaign_binding_required") is not True
        or counting.get("exact_scene_binding_required") is not True
    ):
        raise CampaignManifestError("campaign counting contract is unsafe")
    initial = value.get("initial_view_contract")
    if (
        not isinstance(initial, Mapping)
        or initial.get("mode") != "official_reset"
        or initial.get("fresh_episode_required") is not True
        or initial.get("same_scene_reset_required") is not True
        or initial.get("fixture_id") is not None
        or initial.get("task_specific_station_before_observation") is not False
        or initial.get("pixel_hash_is_hard_gate") is not False
    ):
        raise CampaignManifestError("campaign initial-view contract is unsafe")
    models = value.get("models")
    if not isinstance(models, list) or not models or len(models) > 32:
        raise CampaignManifestError("campaign models must be a bounded non-empty list")
    model_keys: set[str] = set()
    for item in models:
        if not isinstance(item, Mapping):
            raise CampaignManifestError("campaign model entry must be an object")
        key = _safe_id(item.get("model_key"), field="model_key")
        _safe_id(item.get("provider_model"), field="provider_model")
        if key in model_keys:
            raise CampaignManifestError("campaign model identity is duplicated")
        model_keys.add(key)
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or not tasks or len(tasks) > 1_000:
        raise CampaignManifestError("campaign tasks must be a bounded non-empty list")
    task_ids: set[int] = set()
    ready_paths: set[str] = set()
    for item in tasks:
        if not isinstance(item, Mapping):
            raise CampaignManifestError("campaign task entry must be an object")
        task_id = item.get("task_id")
        if isinstance(task_id, bool) or not isinstance(task_id, int) or not 0 <= task_id <= 999:
            raise CampaignManifestError("campaign task_id is invalid")
        _safe_id(item.get("task_name"), field="task_name")
        split = item.get("task_split")
        instance = item.get("task_instance_id")
        seed = item.get("task_seed")
        if not isinstance(split, str) or not split or not isinstance(instance, str) or not instance:
            raise CampaignManifestError("campaign task split/instance is invalid")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFF_FFFF:
            raise CampaignManifestError("campaign task seed is invalid")
        _safe_id(item.get("observer_id"), field="observer_id")
        gpu = item.get("gpu")
        slot = item.get("resident_slot")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (gpu, slot)):
            raise CampaignManifestError("campaign GPU/slot identity is invalid")
        ready = _absolute_path(item.get("scene_ready_path"), field="scene_ready_path")
        _absolute_path(item.get("scene_run_dir"), field="scene_run_dir")
        scene_manifest_sha256 = item.get("scene_manifest_sha256")
        if (
            scene_manifest_sha256 is not None
            and (
                not isinstance(scene_manifest_sha256, str)
                or _SHA256.fullmatch(scene_manifest_sha256) is None
            )
        ):
            raise CampaignManifestError("campaign scene manifest digest is invalid")
        if task_id in task_ids or ready in ready_paths:
            raise CampaignManifestError("campaign task or scene binding is duplicated")
        task_ids.add(task_id)
        ready_paths.add(ready)
    review = value.get("instruction_review")
    if review is not None:
        if not isinstance(review, Mapping):
            raise CampaignManifestError("campaign instruction review is invalid")
        review_path = _absolute_path(review.get("path"), field="instruction_review.path")
        reviewer = review.get("reviewer")
        review_task_ids = review.get("task_ids")
        if (
            Path(review_path).parent != Path(manifest_path).parent
            or review.get("status") != "APPROVED"
            or not isinstance(review.get("sha256"), str)
            or _SHA256.fullmatch(str(review.get("sha256"))) is None
            or not isinstance(reviewer, str)
            or not reviewer.strip()
            or len(reviewer) > 256
            or review_task_ids != sorted(task_ids)
        ):
            raise CampaignManifestError(
                "campaign instruction review is not an exact local approval"
            )
        _reviewed_at(review.get("reviewed_at"))
    bench_source = value.get("bench_source")
    if bench_source is not None:
        if (
            not isinstance(bench_source, Mapping)
            or _absolute_path(bench_source.get("root"), field="bench_source.root")
            != bench_source.get("root")
            or not isinstance(bench_source.get("commit"), str)
            or _GIT_COMMIT.fullmatch(str(bench_source.get("commit"))) is None
            or bench_source.get("tracked_worktree_clean") is not True
        ):
            raise CampaignManifestError("campaign Bench source identity is invalid")
    return value, manifest_path, manifest_sha256


def bind_campaign_run(
    path: Path,
    *,
    experiment_id: str,
    provider_model: str,
    task_id: int,
    task_name: str,
    task_split: str,
    task_instance_id: str,
    task_seed: int,
    scene_ready_path: Path,
    formal: bool = False,
    instruction_provenance: Mapping[str, Any] | None = None,
    bench_root: Path | None = None,
    bench_commit: str | None = None,
    scene_manifest_sha256: str | None = None,
) -> CampaignBinding:
    manifest, manifest_path, manifest_sha256 = load_campaign_manifest(path)
    prefix = str(manifest["experiment_id_prefix"])
    if not experiment_id.startswith(prefix):
        raise CampaignManifestError("experiment ID is outside the frozen campaign prefix")
    if provider_model not in {
        str(item["provider_model"]) for item in manifest["models"]
    }:
        raise CampaignManifestError("provider model is outside the frozen campaign")
    matches = [
        item
        for item in manifest["tasks"]
        if item.get("task_id") == task_id
    ]
    if len(matches) != 1:
        raise CampaignManifestError("task is outside the frozen campaign")
    task = matches[0]
    exact = {
        "task_name": task_name,
        "task_split": task_split,
        "task_instance_id": task_instance_id,
        "task_seed": task_seed,
        "scene_ready_path": str(Path(scene_ready_path).resolve()),
    }
    for field, actual in exact.items():
        if task.get(field) != actual:
            raise CampaignManifestError(f"campaign {field} binding changed")
    ready_path = Path(exact["scene_ready_path"])
    if not ready_path.is_file() or ready_path.is_symlink():
        raise CampaignManifestError("campaign resident ready file is unavailable")
    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignManifestError("campaign resident ready file is unreadable") from exc
    if (
        not isinstance(ready, Mapping)
        or str(Path(str(ready.get("run_dir"))).resolve()) != task["scene_run_dir"]
    ):
        raise CampaignManifestError("campaign resident scene identity changed")
    review_sha256 = None
    frozen_bench_commit = None
    frozen_scene_manifest_sha256 = None
    if formal:
        review = manifest.get("instruction_review")
        bench_source = manifest.get("bench_source")
        if (
            not isinstance(review, Mapping)
            or not isinstance(bench_source, Mapping)
            or not isinstance(instruction_provenance, Mapping)
            or bench_root is None
            or bench_commit is None
            or scene_manifest_sha256 is None
        ):
            raise CampaignManifestError("formal campaign provenance is incomplete")
        expected_review = {
            "source_path": review.get("path"),
            "source_sha256": review.get("sha256"),
            "review_status": review.get("status"),
            "reviewer": review.get("reviewer"),
            "reviewed_at": review.get("reviewed_at"),
            "scope_task_ids": review.get("task_ids"),
        }
        if any(
            instruction_provenance.get(field) != expected
            for field, expected in expected_review.items()
        ):
            raise CampaignManifestError(
                "formal instruction review changed from the frozen campaign"
            )
        if (
            str(Path(bench_root).resolve()) != bench_source.get("root")
            or bench_commit != bench_source.get("commit")
        ):
            raise CampaignManifestError("formal Bench source identity changed")
        frozen_scene_manifest_sha256 = task.get("scene_manifest_sha256")
        if frozen_scene_manifest_sha256 != scene_manifest_sha256:
            raise CampaignManifestError("formal Simulator scene manifest changed")
        review_sha256 = str(review["sha256"])
        frozen_bench_commit = str(bench_source["commit"])
    return CampaignBinding(
        campaign_id=str(manifest["campaign_id"]),
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        experiment_id_prefix=prefix,
        initial_view_mode="official_reset",
        scene_ready_path=exact["scene_ready_path"],
        scene_run_dir=str(task["scene_run_dir"]),
        instruction_review_sha256=review_sha256,
        bench_commit=frozen_bench_commit,
        scene_manifest_sha256=frozen_scene_manifest_sha256,
    )


def campaign_manifest_sha256(path: Path) -> str:
    _, _, digest = load_campaign_manifest(path)
    if _SHA256.fullmatch(digest) is None:  # Defensive; hashlib is authoritative.
        raise CampaignManifestError("campaign manifest digest is invalid")
    return digest
