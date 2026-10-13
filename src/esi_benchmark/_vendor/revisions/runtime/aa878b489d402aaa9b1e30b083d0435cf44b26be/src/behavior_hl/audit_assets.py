"""Fast, simulator-free audit of the isolated five-task asset bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from behavior_hl.tasks import FIVE_TASKS


EXPECTED_BEHAVIOR_ASSET_VERSION = "3.9.0"
EXPECTED_ROBOT_ASSET_VERSION = "3.8.2"


def _read_version(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def _instance_directory(
    instances_root: Path,
    *,
    split_root: Path,
    scene_model: str,
    task_name: str,
) -> Path:
    return (
        instances_root
        / split_root
        / scene_model
        / "json"
        / f"{scene_model}_task_{task_name}_instances"
    )


def audit_data_root(data_root: Path) -> dict[str, Any]:
    """Return a JSON-safe report without importing OmniGibson or Isaac Sim."""

    data_root = data_root.expanduser().resolve()
    behavior_assets = data_root / "behavior-1k-assets"
    robot_assets = data_root / "omnigibson-robot-assets"
    instances_root = data_root / "2026-challenge-task-instances"
    behavior_version = _read_version(behavior_assets / "VERSION")
    robot_version = _read_version(robot_assets / "VERSION")

    errors: list[str] = []
    if behavior_version != EXPECTED_BEHAVIOR_ASSET_VERSION:
        errors.append(
            "behavior asset version mismatch: "
            f"expected {EXPECTED_BEHAVIOR_ASSET_VERSION}, got {behavior_version!r}"
        )
    if robot_version != EXPECTED_ROBOT_ASSET_VERSION:
        errors.append(
            "robot asset version mismatch: "
            f"expected {EXPECTED_ROBOT_ASSET_VERSION}, got {robot_version!r}"
        )

    tasks: list[dict[str, Any]] = []
    for spec in sorted(FIVE_TASKS.values(), key=lambda item: item.b100_index):
        template = (
            instances_root
            / "scenes"
            / spec.scene_model
            / "json"
            / f"{spec.scene_model}_task_{spec.name}_0_0_template.json"
        )
        train_dir = _instance_directory(
            instances_root,
            split_root=Path("scenes"),
            scene_model=spec.scene_model,
            task_name=spec.name,
        )
        public_dir = _instance_directory(
            instances_root,
            split_root=Path("scene_test") / "public",
            scene_model=spec.scene_model,
            task_name=spec.name,
        )
        train_count = sum(path.is_file() and path.name.endswith("-tro_state.json") for path in train_dir.iterdir()) if train_dir.is_dir() else 0
        public_count = (
            sum(path.is_file() and path.name.endswith("-tro_state.json") for path in public_dir.iterdir()) if public_dir.is_dir() else 0
        )
        task_errors: list[str] = []
        if not template.is_file():
            task_errors.append("missing scene template")
        if train_count == 0:
            task_errors.append("missing train instances")
        if public_count == 0:
            task_errors.append("missing public-test instances")
        errors.extend(f"{spec.name}: {message}" for message in task_errors)
        tasks.append(
            {
                "task": spec.name,
                "scene_model": spec.scene_model,
                "template": template.is_file(),
                "train_instances": train_count,
                "public_test_instances": public_count,
                "ready": not task_errors,
            }
        )

    return {
        "data_root": str(data_root),
        "behavior_asset_version": behavior_version,
        "robot_asset_version": robot_version,
        "ready": not errors,
        "tasks": tasks,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()
    report = audit_data_root(args.data_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
