"""Locate the installed frozen source trees without using a development checkout."""

from pathlib import Path
from functools import lru_cache
import sys

PROFILE = "l0-english-v2"
REVISIONS = {
    "bench": "5b8d2b37fd7d97b161e88de4d438ec789c78f9b4",
    "runtime": "9b992b6797330a836e9533e5409627bf63f7d94b",
}


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


def activate(profile: str = PROFILE) -> dict[str, Path]:
    if profile != PROFILE:
        raise ValueError(f"No executable source bundle for profile {profile!r}")
    vendor = Path(__file__).resolve().parent / "_vendor"
    roots = {name + "_root": vendor / name for name in REVISIONS}
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
    return roots
