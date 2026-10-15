"""Locate the installed frozen source trees without using a development checkout."""

from pathlib import Path
import os
from functools import lru_cache
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.util import spec_from_file_location
import sys

from ..profiles import DEFAULT_PROFILE, source_roots

_active_profile = None
_source_finder = None


@lru_cache(maxsize=1)
def _source_modules(paths: tuple[Path, ...]):
    """Index only the installed bundle; loaded module origins are checked afresh."""
    modules = {}
    for root in paths:
        for source in root.rglob("*.py"):
            if "__pycache__" in source.relative_to(root).parts:
                continue
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


class _SourceOnlyLoader(SourceFileLoader):
    def get_code(self, fullname):
        # Installed bytecode is not covered by the frozen source manifest.
        return self.source_to_code(self.get_data(self.path), self.path)


class _FrozenSourceFinder:
    def __init__(self, paths):
        self.modules, self.owned_names = _source_modules(paths)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] not in self.owned_names:
            return None
        if fullname not in self.modules:
            raise ModuleNotFoundError(f"Module is absent from the frozen source bundle: {fullname}", name=fullname)
        source, package = self.modules[fullname]
        locations = [str(package)] if package is not None else None
        if source is None:
            spec = ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = locations
            return spec
        return spec_from_file_location(fullname, source, loader=_SourceOnlyLoader(fullname, str(source)),
                                       submodule_search_locations=locations)


def _check_loaded_sources(paths: tuple[Path, ...]):
    modules, owned_names = _source_modules(paths)
    unverified_loader = None
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
                if source is not None and not isinstance(getattr(module, "__loader__", None), _SourceOnlyLoader):
                    unverified_loader = name
            except (TypeError, ValueError, OSError):
                pass
        if not valid:
            raise RuntimeError(f"Refusing to mix another {name} source tree into this profile")
    if unverified_loader is not None:
        raise RuntimeError(f"Refusing an unverified bytecode loader for {unverified_loader}")


def activate(profile: str | None = None) -> dict[str, Path]:
    """Load exactly one frozen profile inside a private worker or upstream test."""
    global _active_profile, _source_finder
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
    if _source_finder in sys.meta_path:
        sys.meta_path.remove(_source_finder)
    _source_finder = _FrozenSourceFinder(paths)
    sys.meta_path.insert(0, _source_finder)
    for path in reversed(paths):
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    os.environ["BEHAVIOR_HL_PROJECT_ROOT"] = str(roots["runtime_root"])
    _active_profile = selected
    return roots
