"""Synthetic CPU replay through an isolated frozen worker."""
from pathlib import Path
from typing import Any, Mapping

from .profiles import DEFAULT_PROFILE

FIXTURE = Path(__file__).resolve().parent / "data/examples/synthetic-marker-v1.json"


def run_replay(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    from .engine import call
    return call("replay", {"config": dict(config), "output_dir": str(output_dir)},
                profile=config.get("profile", DEFAULT_PROFILE))
