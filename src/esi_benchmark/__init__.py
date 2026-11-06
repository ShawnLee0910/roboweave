"""Compatibility imports for pre-RoboWeave clients and frozen source hooks."""

from importlib import import_module
import sys

_package = import_module("roboweave")
# Share module objects, especially the native loader's state, across both names.
for _name in (
    "tasks", "profiles", "engine", "config", "assets", "models", "scoring",
    "trajectory", "offline", "bootstrap", "runtime", "cli",
    "adapters", "adapters.amv", "adapters.physical", "_native", "_native.bootstrap",
):
    sys.modules[f"{__name__}.{_name}"] = import_module(f"roboweave.{_name}")
sys.modules[__name__] = _package
