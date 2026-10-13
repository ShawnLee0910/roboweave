"""Locate the installed frozen source trees without using a development checkout."""

from pathlib import Path
import os
from functools import lru_cache
import sys

from ..profiles import DEFAULT_PROFILE, source_roots

_active_profile = None


@lru_cache(maxsize=1)
def _source_modules(paths: tuple[Path, ...]):
    """Index only the installed bundle; loaded module origins are checked afresh."""
    modules = {}
    for root in paths:
        for source in root.rglob("*.py"):
            parts = source.relative_to(root).with_suffix("").parts
            package = source.stem == "__init__"
            if package:
                parts = parts[:-1]
            if not parts:
                continue
            modules[".".join(parts)] = (source.resolve(), source.parent.resolve() if package else None)
            for depth in range(1, len(parts)):
                modules.setdefault(".".join(parts[:depth]), (None, root.joinpath(*parts[:depth]).resolve()))
    return modules, frozenset(name.partition(".")[0] for name in modules)


def _check_loaded_sources(paths: tuple[Path, ...]):
    modules, owned_names = _source_modules(paths)
    for name, module in tuple(sys.modules.items()):
        if name.partition(".")[0] not in owned_names:
            continue
        expected = modules.get(name)
        valid = False
        if expected is not None and module is not None:
            source, package = expected
            try:
                origin = getattr(module, "__file__", None)
                locations = getattr(module, "__path__", None)
                actual_source = Path(origin).resolve() if origin is not None else None
                actual_locations = {Path(path).resolve() for path in locations} if locations is not None else None
                valid = (actual_source == source
                         and actual_locations == ({package} if package is not None else None))
            except (TypeError, ValueError, OSError):
                pass
        if not valid:
            raise RuntimeError(f"Refusing to mix another {name} source tree into this profile")


def activate(profile: str | None = None) -> dict[str, Path]:
    """Load exactly one frozen profile inside a private worker or upstream test."""
    global _active_profile
    selected = profile or _active_profile or DEFAULT_PROFILE
    roots = source_roots(selected)
    if _active_profile is not None and selected != _active_profile:
        raise RuntimeError("A native worker cannot switch source profiles")
    paths = (roots["bench_root"], roots["bench_root"] / "lab",
             roots["runtime_root"] / "src", roots["runtime_root"] / "scripts")
    for path in paths:
        if not path.is_dir():
            raise RuntimeError(f"Installed source bundle is incomplete: {path}")
    _check_loaded_sources(paths)
    for path in reversed(paths):
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    os.environ["BEHAVIOR_HL_PROJECT_ROOT"] = str(roots["runtime_root"])
    _active_profile = selected
    return roots
