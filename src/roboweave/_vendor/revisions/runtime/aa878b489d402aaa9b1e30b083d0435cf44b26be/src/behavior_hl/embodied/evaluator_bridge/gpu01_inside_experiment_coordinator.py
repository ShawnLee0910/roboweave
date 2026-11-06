"""Auditable GPU 0/1 coordinator for frozen Inside-task workers.

The coordinator owns only subprocesses it starts. It stages one worker per GPU,
waits for a real scene-ready lifecycle event, applies a conservative VRAM/PID
gate, and then admits a second independent scene on that GPU. Completed slots
may consume the frozen Wave 2 queue until the no-new-load deadline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Iterable

from behavior_hl.embodied.evaluator_bridge.inside_task_profiles import (
    get_inside_task_profile,
)


COORDINATOR_VERSION = "behavior2026.gpu01-inside-coordinator/v1"
OVERLAY_PYTHON = Path(
    os.environ.get("ESI_RUNTIME_PYTHON", sys.executable)
)


@dataclass(frozen=True, slots=True)
class QueueItem:
    task_name: str
    gpu: int
    port: int
    seed: int
    wave: int


WAVE_1 = (
    QueueItem("picking_up_toys", 0, 8810, 19506, 1),
    QueueItem("composting_waste", 0, 8811, 19507, 1),
    QueueItem("picking_up_trash", 1, 8812, 19508, 1),
    QueueItem("putting_dirty_dishes_in_sink", 1, 8813, 19509, 1),
)
WAVE_2 = (
    QueueItem("collecting_aluminum_cans", 0, 8810, 19510, 2),
    QueueItem("boxing_books_up_for_storage", 1, 8812, 19511, 2),
    QueueItem("clean_up_broken_glass", 0, 8811, 19512, 2),
    QueueItem("dispose_of_glass", 1, 8813, 19513, 2),
)


def _select_queue_items(
    *,
    gpus: Iterable[int],
    skipped_tasks: Iterable[str] = (),
) -> tuple[tuple[QueueItem, ...], tuple[QueueItem, ...], tuple[int, ...]]:
    """Return a validated, order-preserving recovery queue.

    ``--gpus`` limits all launch and admission loops to explicitly authorized
    physical GPUs. ``--skip-task`` preserves prior attempts while allowing a
    recovery coordinator to begin with the next unattempted frozen queue item.
    """

    selected_gpus = tuple(dict.fromkeys(gpus))
    if not selected_gpus or any(gpu not in (0, 1) for gpu in selected_gpus):
        raise ValueError(f"gpus must be a nonempty subset of (0, 1), got {selected_gpus!r}")
    skipped = frozenset(skipped_tasks)
    known_tasks = {item.task_name for item in (*WAVE_1, *WAVE_2)}
    unknown_skips = skipped - known_tasks
    if unknown_skips:
        raise ValueError(f"unknown skipped tasks: {sorted(unknown_skips)!r}")
    wave_1 = tuple(
        item for item in WAVE_1
        if item.gpu in selected_gpus and item.task_name not in skipped
    )
    wave_2 = tuple(
        item for item in WAVE_2
        if item.gpu in selected_gpus and item.task_name not in skipped
    )
    missing_initial = [
        gpu for gpu in selected_gpus
        if not any(item.gpu == gpu for item in wave_1)
    ]
    if missing_initial:
        raise ValueError(
            "each selected GPU needs at least one unskipped Wave 1 item; "
            f"missing={missing_initial!r}"
        )
    return wave_1, wave_2, selected_gpus


@dataclass(slots=True)
class OwnedWorker:
    item: QueueItem
    worker_id: str
    process: subprocess.Popen[str]
    stdout_stream: Any
    artifact_root: Path
    launched_at: float
    scene_ready: bool = False


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def _event(journal: Path, item: QueueItem, state: str, **extra: Any) -> None:
    _append(
        journal,
        {
            "schema_version": COORDINATOR_VERSION,
            "created_at": _utc(),
            "task": item.task_name,
            "gpu": item.gpu,
            "port": item.port,
            "seed": item.seed,
            "wave": item.wave,
            "state": state,
            **extra,
        },
    )


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _source_files(root: Path) -> Iterable[Path]:
    for relative in ("src", "scripts", "configs"):
        base = root / relative
        if base.is_dir():
            yield from (
                path for path in base.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    if (root / "pyproject.toml").is_file():
        yield root / "pyproject.toml"


def _require_task_instance_exists(*, data_root: Path, item: QueueItem) -> Path:
    """Fail before snapshot/scene work when a frozen profile has no TRO state."""

    profile = get_inside_task_profile(item.task_name)
    split_root = {
        "train": Path("scenes"),
        "public_test": Path("scene_test/public"),
        "hidden_test": Path("scene_test/private"),
    }.get(profile.split)
    if split_root is None:
        raise ValueError(f"unsupported task instance split: {profile.split!r}")
    stem = (
        f"{profile.scene_model}_task_{profile.task_name}_0_"
        f"{profile.instance_id}_template-tro_state.json"
    )
    path = (
        data_root
        / "2026-challenge-task-instances"
        / split_root
        / profile.scene_model
        / "json"
        / f"{profile.scene_model}_task_{profile.task_name}_instances"
        / stem
    )
    if not path.is_file():
        raise FileNotFoundError(
            "frozen task profile has no local 2026 TRO state: "
            f"task={profile.task_name}, split={profile.split}, "
            f"scene={profile.scene_model}, instance_id={profile.instance_id}, "
            f"expected_path={path}"
        )
    return path


def _freeze_source(source_root: Path, snapshot_root: Path) -> dict[str, Any]:
    if snapshot_root.exists():
        raise FileExistsError(snapshot_root)
    shutil.copytree(
        source_root,
        snapshot_root,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".runtime"),
    )
    records = []
    aggregate = hashlib.sha256()
    for path in sorted(_source_files(snapshot_root)):
        relative = str(path.relative_to(snapshot_root))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"relative_path": relative, "sha256": digest})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(digest.encode("ascii"))
    manifest = {
        "schema_version": "behavior2026.frozen-source/v1",
        "created_at": _utc(),
        "source_root": str(source_root),
        "snapshot_root": str(snapshot_root),
        "file_count": len(records),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": records,
    }
    (snapshot_root / "SOURCE_SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _nvidia_rows(gpu: int) -> list[dict[str, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    rows = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                rows.append({"pid": int(parts[0]), "used_memory_mib": int(parts[1])})
    return rows


def _gpu_free_mib(gpu: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return int(result.stdout.strip().splitlines()[0])


def _descends_from(pid: int, ancestor: int) -> bool:
    current = pid
    for _ in range(64):
        if current == ancestor:
            return True
        try:
            status = Path(f"/proc/{current}/status").read_text(encoding="utf-8")
        except OSError:
            return False
        parent = next(
            (line.split(":", 1)[1].strip() for line in status.splitlines() if line.startswith("PPid:")),
            "0",
        )
        if not parent.isdigit() or int(parent) <= 1:
            return False
        current = int(parent)
    return False


def _scene_ready(worker: OwnedWorker) -> bool:
    for stage_path in worker.artifact_root.glob(
        f"workers/{worker.worker_id}/*/lifecycle/stages.jsonl"
    ):
        if '"stage": "SCENE_CONSTRUCTION_PASS"' in stage_path.read_text(
            encoding="utf-8", errors="replace"
        ):
            return True
    return False


def _launch(
    *,
    item: QueueItem,
    snapshot_root: Path,
    experiment_root: Path,
    api_lock_root: Path,
    journal: Path,
) -> OwnedWorker:
    if not _port_available(item.port):
        raise RuntimeError(f"port {item.port} is not available")
    profile = get_inside_task_profile(item.task_name)
    worker_id = (
        f"wave{item.wave}_gpu{item.gpu}_{item.task_name}_seed{item.seed}"
    )
    artifact_root = experiment_root / "runs" / worker_id
    artifact_root.mkdir(parents=True, exist_ok=False)
    stdout_path = artifact_root / "worker_stdout.log"
    stream = stdout_path.open("w", encoding="utf-8")
    environment = os.environ.copy()
    for name in (
        "CUDA_VISIBLE_DEVICES", "VCUDA_PORT", "LOCAL_HOST_IP",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
        "TOKEN_PLAN_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY",
    ):
        environment.pop(name, None)
    bundle_root = snapshot_root.parent
    environment.update(
        {
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "OMNIGIBSON_GPU_ID": str(item.gpu),
            "OMNIGIBSON_HEADLESS": "True",
            "OMNIGIBSON_DATA_PATH": str(
                bundle_root / "BEHAVIOR-1K-2026-data"
            ),
            "BEHAVIOR_HL_PROJECT_ROOT": str(snapshot_root),
            "BEHAVIOR_HL_TOKENPLAN_MODEL": "bailian/qwen3.8-max-tp",
            "BEHAVIOR_HL_MODEL_BACKEND": "qoderclicn",
            "TOKEN_PLAN_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "TORCHDYNAMO_DISABLE": "1",
            "TORCHINDUCTOR_DISABLE": "1",
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
            "__NV_PRIME_RENDER_OFFLOAD": "1",
            "__VK_LAYER_NV_optimus": "NVIDIA_only",
            "LD_PRELOAD": str(bundle_root / "dlsym_shim.so"),
            "PYTHONPATH": f"{snapshot_root / 'src'}:{snapshot_root / 'scripts'}",
            "PYTHONDONTWRITEBYTECODE": "1",
            "BEHAVIOR_HL_TOKENPLAN_GLOBAL_LOCK": str(api_lock_root),
            "BEHAVIOR_HL_TOKENPLAN_GLOBAL_SLOTS": "2",
        }
    )
    command = [
        str(OVERLAY_PYTHON), "-m",
        "behavior_hl.embodied.evaluator_bridge.launch_b1_marker_symbolic_full_rollout",
        "--worker-id", worker_id,
        "--physical-gpu", str(item.gpu),
        "--server-port", str(item.port),
        "--seed", str(item.seed),
        "--task-name", item.task_name,
        "--split", profile.split,
        "--instance-id", str(profile.instance_id),
        "--artifact-root", str(artifact_root),
    ]
    process = subprocess.Popen(
        command,
        cwd=snapshot_root,
        env=environment,
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    _event(
        journal,
        item,
        "RUNNING_ORACLE",
        worker_id=worker_id,
        owned_pid=process.pid,
        artifact_root=str(artifact_root),
        command_without_secrets=command,
    )
    return OwnedWorker(item, worker_id, process, stream, artifact_root, time.monotonic())


def _second_slot_allowed(gpu: int, first: OwnedWorker) -> tuple[bool, dict[str, Any]]:
    rows = _nvidia_rows(gpu)
    unknown = [row for row in rows if not _descends_from(row["pid"], first.process.pid)]
    owned_used = sum(row["used_memory_mib"] for row in rows if row not in unknown)
    free = _gpu_free_mib(gpu)
    required = max(18 * 1024, int(1.25 * owned_used))
    report = {
        "gpu": gpu,
        "compute_rows": rows,
        "unknown_rows": unknown,
        "owned_used_mib": owned_used,
        "free_mib": free,
        "required_free_mib": required,
    }
    return not unknown and free >= required, report


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_process_group_exit(pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _process_group_alive(pgid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)
    return True


def _reap_orphaned_group(pgid: int) -> None:
    """Finish stopping only the recorded worker group after its parent exits."""

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if _wait_for_process_group_exit(pgid, 5.0):
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    _wait_for_process_group_exit(pgid, 5.0)


def _safe_stop(worker: OwnedWorker) -> None:
    if worker.process.poll() is None:
        try:
            os.killpg(worker.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            worker.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(worker.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker.process.wait(timeout=15)
    # A native Kit crash can orphan the policy server and resource tracker in
    # the owned process group. Confirm that only this recorded PGID is gone
    # before a replacement worker tries to reuse its port.
    _reap_orphaned_group(worker.process.pid)
    worker.stdout_stream.close()


def run(args: argparse.Namespace) -> int:
    selected_wave_1, selected_wave_2, selected_gpus = _select_queue_items(
        gpus=args.gpus,
        skipped_tasks=args.skip_tasks,
    )
    source_root = args.source_root.resolve()
    data_root = source_root.parent / "BEHAVIOR-1K-2026-data"
    for item in (*selected_wave_1, *selected_wave_2):
        _require_task_instance_exists(data_root=data_root, item=item)
    experiment_root = args.artifact_root.resolve() / args.experiment_id
    experiment_root.mkdir(parents=True, exist_ok=False)
    journal = experiment_root / "queue_journal.jsonl"
    snapshot_root = source_root.parent / f"BEHAVIOR-HL-{args.experiment_id}-snapshot"
    source_manifest = _freeze_source(source_root, snapshot_root)
    (experiment_root / "source_snapshot.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for item in (*selected_wave_1, *selected_wave_2):
        profile = get_inside_task_profile(item.task_name)
        _event(
            journal,
            item,
            "READY",
            scene_model=profile.scene_model,
            required_official_goal_leaves=profile.expected_goal_leaf_count,
            snapshot_sha256=source_manifest["aggregate_sha256"],
        )
    if args.dry_run:
        return 0

    start = time.monotonic()
    no_new_load_at = start + args.no_new_load_seconds
    stop_at = start + args.maximum_runtime_seconds
    api_lock_root = experiment_root / "api_slots"
    api_lock_root.mkdir()
    active: list[OwnedWorker] = []
    pending_wave2 = list(selected_wave_2)
    wave_1_by_gpu = {
        gpu: [item for item in selected_wave_1 if item.gpu == gpu]
        for gpu in selected_gpus
    }
    first_by_gpu = {gpu: items[0] for gpu, items in wave_1_by_gpu.items()}
    second_by_gpu = {
        gpu: items[1] for gpu, items in wave_1_by_gpu.items() if len(items) > 1
    }
    second_launched: set[int] = set()
    second_next_check = {gpu: 0.0 for gpu in selected_gpus}
    try:
        for gpu in selected_gpus:
            active.append(
                _launch(
                    item=first_by_gpu[gpu],
                    snapshot_root=snapshot_root,
                    experiment_root=experiment_root,
                    api_lock_root=api_lock_root,
                    journal=journal,
                )
            )
        while active and time.monotonic() < stop_at:
            for worker in list(active):
                if not worker.scene_ready and _scene_ready(worker):
                    worker.scene_ready = True
                    _event(journal, worker.item, "SCENE_READY", owned_pid=worker.process.pid)
                code = worker.process.poll()
                if code is None:
                    continue
                _safe_stop(worker)
                active.remove(worker)
                _event(
                    journal,
                    worker.item,
                    "PROCESS_COMPLETE" if code == 0 else "PROCESS_FAIL",
                    returncode=code,
                )
                if pending_wave2 and time.monotonic() < no_new_load_at:
                    replacement_index = next(
                        (index for index, item in enumerate(pending_wave2) if item.gpu == worker.item.gpu),
                        None,
                    )
                    if replacement_index is not None:
                        item = pending_wave2.pop(replacement_index)
                        try:
                            active.append(
                                _launch(
                                    item=item,
                                    snapshot_root=snapshot_root,
                                    experiment_root=experiment_root,
                                    api_lock_root=api_lock_root,
                                    journal=journal,
                                )
                            )
                        except Exception as exc:
                            _event(
                                journal,
                                item,
                                "LAUNCH_FAIL",
                                error_type=type(exc).__name__,
                                error=str(exc),
                            )
            for gpu in selected_gpus:
                now = time.monotonic()
                if (
                    gpu not in second_by_gpu
                    or gpu in second_launched
                    or now >= no_new_load_at
                    or now < second_next_check[gpu]
                ):
                    continue
                first = next((worker for worker in active if worker.item == first_by_gpu[gpu]), None)
                if first is None or not first.scene_ready:
                    continue
                admitted, report = _second_slot_allowed(gpu, first)
                _append(experiment_root / "gpu_admission.jsonl", {"created_at": _utc(), **report})
                if admitted:
                    try:
                        active.append(
                            _launch(
                                item=second_by_gpu[gpu],
                                snapshot_root=snapshot_root,
                                experiment_root=experiment_root,
                                api_lock_root=api_lock_root,
                                journal=journal,
                            )
                        )
                        second_launched.add(gpu)
                    except Exception as exc:
                        _event(
                            journal,
                            second_by_gpu[gpu],
                            "LAUNCH_FAIL",
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                        second_next_check[gpu] = now + 30.0
                else:
                    _event(journal, second_by_gpu[gpu], "BLOCKED_RESOURCE", **report)
                    second_next_check[gpu] = now + 30.0
            time.sleep(5)
        return 0
    finally:
        for worker in active:
            _safe_stop(worker)
            _event(journal, worker.item, "SAFE_STOPPED", owned_pid=worker.process.pid)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--maximum-runtime-seconds", type=int, default=14_400)
    parser.add_argument("--no-new-load-seconds", type=int, default=12_600)
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        choices=(0, 1),
        default=[0, 1],
        help="Physical GPUs authorized for this coordinator (default: 0 1).",
    )
    parser.add_argument(
        "--skip-task",
        dest="skip_tasks",
        action="append",
        default=[],
        choices=tuple(item.task_name for item in (*WAVE_1, *WAVE_2)),
        help="Preserved prior queue item to skip; repeat for multiple tasks.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
