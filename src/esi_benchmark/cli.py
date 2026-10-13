"""One public entry point for installation, resources, episodes and reports."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from . import __version__
from .bootstrap import PROFILE, REVISIONS, activate
from .config import load_config, validate_config


def _print(value):
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str))


def parser():
    result = argparse.ArgumentParser(description="Frozen ESI Benchmark public candidate (CPU replay and explicit Live execution)")
    result.add_argument("--version", action="version", version=__version__)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check installed CPU imports and packaged source layout; no device or network calls")
    config = commands.add_parser("config", help="Inspect configuration").add_subparsers(dest="operation", required=True)
    config.add_parser("validate").add_argument("--config", required=True, type=Path)
    tasks = commands.add_parser("tasks", help="List frozen public task definitions").add_subparsers(dest="operation", required=True)
    listing = tasks.add_parser("list")
    listing.add_argument("--family")
    show = tasks.add_parser("show")
    show.add_argument("task_id")
    show.add_argument("--family")
    assets = commands.add_parser("assets", help="Verify or import separately obtained evaluation inputs").add_subparsers(dest="operation", required=True)
    assets.add_parser("verify").add_argument("--config", required=True, type=Path)
    prepare = assets.add_parser("prepare")
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--source-root", action="append", default=[], metavar="NAME=PATH")
    prepare.add_argument("--mapping", type=Path)
    prepare.add_argument("--bundle", type=Path)
    run = commands.add_parser("run", help="Execute a synthetic CPU replay or one explicitly configured owned Live episode")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--mode", choices=("replay", "live"))
    run.add_argument("--output", type=Path, help="New result directory; an existing directory is never overwritten")
    run.add_argument("--dry-run", action="store_true", help="Validate inputs without starting a runtime or making a model request")
    reports = commands.add_parser("results", help="Validate and summarize local trajectory artifacts").add_subparsers(dest="operation", required=True)
    reports.add_parser("validate").add_argument("run_dir", type=Path)
    reports.add_parser("summarize").add_argument("run_dirs", nargs="+", type=Path)
    score = commands.add_parser("score", help="Replay the canonical verifier over a CPU evidence record; no Live claim")
    score.add_argument("record", type=Path)
    score.add_argument("--process-output", type=Path, help="Fresh directory for required process-audit artifacts")
    return result


def doctor():
    roots = activate()
    import Agent
    from behavior_hl.embodied.evaluator_bridge import l0_verifier_adapter
    packages = {}
    for name in ("esi-benchmark", "numpy", "Pillow", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "source checkout (not installed)" if name == "esi-benchmark" else "missing"
    return {"ok": all(value != "missing" for value in packages.values()), "profile": PROFILE,
            "python": sys.version.split()[0], "packages": packages, "source_revisions": REVISIONS,
            "source_roots": roots, "live_status": "not tested; separate simulator, assets and owned GPU required"}


def _fresh_output(config, requested):
    root = Path(config["run"]["output_root"])
    path = (requested or root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])).resolve()
    package = Path(__file__).resolve().parent
    if path == package or path.is_relative_to(package):
        raise ValueError("Output cannot be written into the installed source bundle")
    if path.exists():
        raise ValueError(f"Output already exists; choose a new directory: {path}")
    return path


def _stop_owned(process):
    # The launcher can exit before its policy server; the owned session still needs cleanup.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=1)
        return
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def _run(config, requested, dry_run):
    from .tasks import get_task
    task = get_task(config["run"]["task_id"], family=config["run"]["family"])
    output = _fresh_output(config, requested)
    if config["mode"] == "replay":
        if dry_run:
            return {"mode": "replay", "ok": True, "output_dir": output,
                    "evidence": "Synthetic protocol fixture; does not evaluate the selected catalog task"}
        from .offline import run_replay
        return run_replay(config, output)
    from .assets import materialize_task_inputs, resolve_task_inputs
    from .runtime import launch_runtime, runtime_plan, validate_owned_ready
    inputs = resolve_task_inputs(config, task)
    plan = runtime_plan(config, inputs)
    if dry_run:
        return {"mode": "live", "ok": True, "ready_path": plan["ready_path"],
                "output_dir": output, "preflight": plan.get("preflight"), "started": False,
                "scope": "Configuration, sources and resource availability only; task input relocation and native admission occur at run time"}
    output.mkdir(parents=True, exist_ok=False)
    process = None
    stage = "input_materialization"
    try:
        inputs = materialize_task_inputs(config, task, output / "private" / "inputs")
        plan = runtime_plan(config, inputs)
        stage = "scene_startup"
        process = launch_runtime(config, inputs)
        deadline = time.monotonic() + config["runtime"]["scene_load_timeout"]
        ready = Path(plan["ready_path"])
        while not ready.is_file():
            if process.poll() is not None:
                raise RuntimeError(f"Owned scene launcher exited with code {process.returncode}; see its launcher.log")
            if time.monotonic() >= deadline:
                raise TimeoutError("Owned scene load exceeded its configured deadline; no automatic retry")
            time.sleep(0.2)
        validate_owned_ready(plan, process.pid)
        stage = "model_episode"
        from .models import run_episode
        return run_episode(model_config=config["model"], task=inputs["model_task"],
                           ready_path=ready, runtime_root=Path(plan["runtime_root"]),
                           output_dir=output, experiment_id=output.name,
                           output_reserved=True,
                           history_frames=config["run"]["history_frames"])
    except (Exception, KeyboardInterrupt) as exc:
        failure = {"status": "INITIALIZATION_FAILED", "stage": stage,
                   "error_type": type(exc).__name__, "official_success": None, "q_score": None,
                   "termination_reason": "candidate_initialization_failure"}
        if not (output / "run_record.json").exists():
            (output / "run_record.json").write_text(json.dumps({
                "schema_version": "esi-benchmark/run-v1", "mode": "live", "profile": PROFILE,
                "source_revisions": REVISIONS, "experiment_id": output.name,
                "evaluation_variant": "frozen_history_5" if config["run"]["history_frames"] == 5 else "history_20_variant_not_baseline",
                "task_identity": {"task_id": task["task_id"], "family": task["family"]},
                "evidence_scope": "initialization_failed; no evaluation success claim",
                "official_benchmark_eligible": False, "strict_production_eligible": False,
            }, indent=2) + "\n")
        if not (output / "summary.json").exists():
            (output / "summary.json").write_text(json.dumps(failure, indent=2) + "\n")
            (output / "conversation_unavailable.json").write_text(json.dumps({
                "reason": "initialization_failed", "stage": stage, "error_type": type(exc).__name__,
            }, indent=2) + "\n")
        else:
            (output / "candidate_error.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    finally:
        if process is not None:
            _stop_owned(process)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor()
        elif args.command == "config":
            config = load_config(args.config)
            from .tasks import get_task
            get_task(config["run"]["task_id"], family=config["run"]["family"])
            result = {"ok": True, "profile": config["profile"], "mode": config["mode"]}
        elif args.command == "tasks":
            from .tasks import get_task, list_tasks
            result = list_tasks(family=args.family) if args.operation == "list" else get_task(args.task_id, family=args.family)
        elif args.command == "assets":
            from .assets import check_assets, prepare_assets
            config = load_config(args.config)
            if args.operation == "verify":
                result = check_assets(config)
            else:
                mappings = {}
                for entry in args.source_root:
                    if "=" not in entry:
                        raise ValueError("--source-root must be NAME=PATH")
                    name, value = entry.split("=", 1)
                    if not name or not value or name in mappings:
                        raise ValueError("Source roots require unique names and nonempty paths")
                    mappings[name] = str(Path(value).expanduser().resolve())
                mapping = json.loads(args.mapping.read_text()) if args.mapping else None
                result = prepare_assets(config, source_roots=mappings or None, mapping=mapping, bundle=args.bundle)
        elif args.command == "run":
            config = load_config(args.config)
            if args.mode:
                config["mode"] = args.mode
                validate_config(config)
            result = _run(config, args.output, args.dry_run)
        elif args.command == "results":
            from .trajectory import summarize_runs, validate_run
            result = validate_run(args.run_dir) if args.operation == "validate" else summarize_runs(args.run_dirs)
        elif args.command == "score":
            from .scoring import score_record
            if args.record.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("Score record exceeds 16 MiB")
            result = score_record(json.loads(args.record.read_text()), process_directory=args.process_output)
        else:
            raise ValueError("Unknown command")
        _print(result)
        if args.command == "results" and args.operation == "validate" and result.get("status") != "PASS":
            return 1
        return 1 if isinstance(result, dict) and result.get("ok") is False else 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"esi-benchmark: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
