"""Private, single-request backend. Never import this to activate public APIs."""
from contextlib import redirect_stdout
import importlib.metadata
import json
from pathlib import Path
import sys

from ..engine import MAX_MESSAGE_BYTES, SCHEMA, _read_json, validate_request
from ..profiles import source_revisions, source_roots, verify_sources
from .bootstrap import activate


def dispatch(request):
    validate_request(request)
    operation, profile, payload = (request[key] for key in ("operation", "profile", "payload"))
    sources = verify_sources(profile)
    activate(profile)
    if operation == "inspect":
        import Agent  # noqa: F401
        from behavior_hl.embodied.evaluator_bridge import l0_verifier_adapter  # noqa: F401
        packages = {}
        for name in ("esi-benchmark", "numpy", "Pillow", "PyYAML"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = "source checkout (not installed)" if name == "esi-benchmark" else "missing"
        return {"ok": all(value != "missing" for value in packages.values()), "profile": profile,
                "python": sys.version.split()[0], "packages": packages,
                "source_revisions": source_revisions(profile), "source_verification": sources,
                "source_roots": {name: str(path) for name, path in source_roots(profile).items()},
                "live_status": "not tested; separate simulator, assets and owned GPU required"}
    if operation == "validate_model":
        from .models import validate_model_config
        return validate_model_config(payload["model_config"])
    if operation == "replay":
        from .offline import run_replay
        return run_replay({**payload["config"], "profile": profile}, Path(payload["output_dir"]),
                          source_verification=sources)
    if operation == "score":
        from .scoring import score_record
        directory = payload["process_directory"]
        return score_record(payload["record"], process_directory=None if directory is None else Path(directory),
                            profile=profile)
    if operation == "task_info":
        from ..adapters import inspect_inputs
        return inspect_inputs(profile, payload["inputs"])
    if operation == "episode":
        from .models import run_episode
        paths = {key: Path(payload[key]) for key in ("ready_path", "runtime_root", "output_dir")}
        return run_episode(**{**payload, **paths}, profile=profile, source_verification=sources)
    raise ValueError("Unknown engine operation")


def main():
    try:
        raw = sys.stdin.buffer.read(MAX_MESSAGE_BYTES + 1)
        if len(raw) > MAX_MESSAGE_BYTES:
            raise ValueError("Engine request is too large")
        request = _read_json(raw)
        with redirect_stdout(sys.stderr):
            result = dispatch(request)
        response = {"schema_version": SCHEMA, "ok": True, "result": result, "error": None}
        status = 0
    except Exception as exc:
        response = {"schema_version": SCHEMA, "ok": False, "result": None,
                    "error": {"type": type(exc).__name__, "message": str(exc)}}
        status = 1
    sys.stdout.write(json.dumps(response, allow_nan=False, separators=(",", ":")) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
