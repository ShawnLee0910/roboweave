"""Locate the installed frozen source trees without using a development checkout."""

from pathlib import Path
import sys

PROFILE = "l0-english-v2"
REVISIONS = {
    "bench": "5b8d2b37fd7d97b161e88de4d438ec789c78f9b4",
    "runtime": "9b992b6797330a836e9533e5409627bf63f7d94b",
}


def activate(profile: str = PROFILE) -> dict[str, Path]:
    if profile != PROFILE:
        raise ValueError(f"No executable source bundle for profile {profile!r}")
    vendor = Path(__file__).resolve().parent / "_vendor"
    roots = {name + "_root": vendor / name for name in REVISIONS}
    paths = [roots["bench_root"], roots["bench_root"] / "lab",
             roots["runtime_root"] / "src", roots["runtime_root"] / "scripts"]
    for path in paths:
        if not path.is_dir():
            raise RuntimeError(f"Installed source bundle is incomplete: {path}")
    owned_modules = {"Agent": roots["bench_root"], "action20_protocol": roots["bench_root"],
                     "model_client": roots["bench_root"], "behavior_hl": roots["runtime_root"]}
    for name, root in owned_modules.items():
        module = sys.modules.get(name)
        if module is not None and not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError(f"Refusing to mix another {name} source tree into this profile")
    for path in reversed(paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return roots
