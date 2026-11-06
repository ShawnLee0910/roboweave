#!/usr/bin/env python3
"""Maintainer-only import of frozen revisions; never execute imported code.

Original repositories are read-only. The import refuses changed source trees and
existing destinations. Local paths identify provenance, not runtime dependencies.
"""

import argparse
import ast
import difflib
import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/roboweave"
BENCH_SOURCES = {
    "1d76bc5dc53b1995f2e7c301cb06569bcf0c4923": "/liyuanshuai/l1a30c",
    "fb1cf143f850fe43a04d25440f07cbfab4007093": "/liyuanshuai/l1r-m",
    "d8ae5151fadfb3338d122fdb12b26051ed753852": "/liyuanshuai/vguard-fix-20260924",
    "7ebc9c948a047ba7f51e0a1d2bd768b228fad5eb": "/liyuanshuai/p30high",
    "4c8364842157153875d098ebb529adcc0620a714": "/liyuanshuai/l2g25",
}
_BUNDLE = "/liyuanshuai/esibench_l20_bundle_20260526/"
RUNTIME_SOURCES = {
    "cd25f9ba1579bdca1f0dadf6fce7ad10974b8717": _BUNDLE + "l1a-runtime-cd25f9b",
    "918aefd40beb72368126cc3fa2891bbb732ad490": _BUNDLE + "am002-runtime-918aefd",
    "955585540d95cba02f672c0d3cdb94048db23bf2": _BUNDLE + "p28-inside",
    "c813a24f853bb90ba3772365ca5182ecb28c553b": _BUNDLE + "p28-ontop",
    "ab615d43d4ede7e6a50eaeda189077529bef361a": _BUNDLE + "l1-unified-scorefix-20260919",
    "aa878b489d402aaa9b1e30b083d0435cf44b26be": _BUNDLE + "l2-collection-runtime-20260922",
}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_archive(root, revision, names):
    raw = git(root, "archive", revision, "--", *names)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        result = {}
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile() or Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                raise ValueError("Source archive contains a non-regular or unsafe path")
            result[member.name] = archive.extractfile(member).read()
    return result


def read_records(kind):
    return json.loads((ROOT / f"provenance/imports_{kind}.json").read_text())


def bench_closure(files):
    selected = {row["source_path"] for row in read_records("bench")
                if "/test" not in row["source_path"]}
    selected &= files.keys()
    pending = list(selected)
    while pending:
        name = pending.pop()
        if not name.endswith(".py"):
            continue
        for node in ast.walk(ast.parse(files[name], filename=name)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    prefix = list(Path(name).parent.parts)
                    prefix = prefix[:len(prefix) - node.level + 1]
                    module = ".".join(prefix + ([module] if module else []))
                modules = [module, *[module + "." + alias.name for alias in node.names]]
            for module in modules:
                relative = module.replace(".", "/")
                for candidate in (relative + ".py", relative + "/__init__.py",
                                  "lab/" + relative + ".py", "lab/" + relative + "/__init__.py"):
                    if candidate in files and candidate not in selected:
                        selected.add(candidate)
                        pending.append(candidate)
    return {name: files[name] for name in sorted(selected)}


@lru_cache(maxsize=None)
def adaptation_blocks(old, adapted, context=3):
    before, after = old.splitlines(keepends=True), adapted.splitlines(keepends=True)
    blocks = []
    for group in difflib.SequenceMatcher(a=before, b=after).get_grouped_opcodes(context):
        original = "".join(before[group[0][1]:group[-1][2]])
        replacement = "".join(after[group[0][3]:group[-1][4]])
        blocks.append((original, replacement))
    return blocks


def apply_adaptation(old, adapted, target, name):
    """Apply reviewed candidate changes only where the old context matches exactly."""
    for context in (3, 1):
        result = target
        for original, replacement in adaptation_blocks(old, adapted, context):
            if result.count(original) != 1:
                break
            result = result.replace(original, replacement, 1)
        else:
            return result
    raise ValueError(f"Candidate adaptation requires review: {name}")


def replace_function(source, name, replacement):
    node = next(node for node in ast.parse(source).body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
    lines = source.splitlines(keepends=True)
    return "".join(lines[:node.lineno - 1]) + replacement + "\n" + "".join(lines[node.end_lineno:])


def adapt_scene_identity(source):
    source = source.replace('"""Small Git identity contract for one resident launch."""',
                            '"""Frozen bundled sources and exact external-checkout identity."""')
    head = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "git_source_head")
    lines = source.splitlines(keepends=True)
    original = """    revision = subprocess.run(
"""
    insert = '''    from roboweave.profiles import registered_revision, verify_bundle
    frozen_revision = registered_revision(source)
    if frozen_revision is not None:
        verify_bundle(source, frozen_revision)
        return frozen_revision
    if (source / "SOURCE_IDENTITY.json").exists():
        raise ValueError(f"unregistered frozen source root: {source}")
    top = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "--show-toplevel"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=15, check=False,
    )
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != source:
        raise ValueError(f"source root is not an independent Git checkout: {source}")
    revision = subprocess.run(
'''
    function = "".join(lines[head.lineno - 1:head.end_lineno])
    if function.count(original) != 1:
        raise ValueError("source identity adaptation requires its original Git implementation")
    source = replace_function(source, "git_source_head", function.replace(original, insert).rstrip() + "\n")
    original = '    commit, dirty = _cached_git_source_identity(str(source))\n'
    insert = '''    from roboweave.profiles import registered_revision
    if registered_revision(source) is not None or (source / "SOURCE_IDENTITY.json").exists():
        return {"root": str(source), "git_commit": git_source_head(source), "dirty": False}
    commit, dirty = _cached_git_source_identity(str(source))
'''
    if source.count(original) != 1:
        raise ValueError("source identity adaptation requires the original lookup")
    return source.replace(original, insert)


def adapt_projected_verifier_loader(source):
    original = "    spec = importlib.util.spec_from_file_location(module_name, path)\n"
    replacement = ("    from roboweave._native.bootstrap import _SourceOnlyLoader\n"
                   "    spec = importlib.util.spec_from_file_location(\n"
                   "        module_name, path, loader=_SourceOnlyLoader(module_name, str(path)))\n")
    if source.count(original) != 1:
        raise ValueError("projected verifier loader adaptation requires review")
    return source.replace(original, replacement)


_PROVIDER_REDACTION_VALUES = '''        redaction_values = (
            headers.get("x-api-key", ""),
            headers.get("Authorization", "").removeprefix("Bearer "),
        )
'''


def adapt_provider_error_requests(source):
    """Bind each revision's existing failure paths to the credential it sent."""
    start = '        try:\n            with self.opener.open(http_request, timeout=self.timeout_seconds) as response:\n'
    if source.count(start) != 1:
        raise ValueError("provider error redaction adaptation requires review")
    source = source.replace(start, _PROVIDER_REDACTION_VALUES + start)
    provider = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.ClassDef) and node.name == "OpenAICompatibleProvider")
    post = next(node for node in provider.body if isinstance(node, ast.FunctionDef) and node.name == "_post")
    calls = [node for node in ast.walk(post) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "ProviderRequestError"]
    if len(calls) not in {7, 10} or any(keyword.arg == "redaction_values" for node in calls for keyword in node.keywords):
        raise ValueError("provider failure paths changed; credential binding requires review")
    original = ast.get_source_segment(source, post)
    redacted = original
    for expression in ("exc.reason", "exc"):
        redacted = redacted.replace("{" + expression + "}",
            "{_redact_provider_error_text(str(" + expression + "), redaction_values)}")
    source = source.replace(original, redacted, 1)
    lines = source.splitlines(keepends=True)
    for call in sorted(calls, key=lambda node: node.end_lineno, reverse=True):
        index = call.end_lineno - 1
        indent = len(lines[index]) - len(lines[index].lstrip()) + 4
        lines.insert(index, " " * indent + "redaction_values=redaction_values,\n")
    return "".join(lines)


def adapt(kind, files):
    changes = {}
    for row in read_records(kind):
        name = row["source_path"]
        if name not in files or row["changes"] == "none":
            continue
        if kind == "bench" and name == "lab/scene_source_identity.py":
            continue  # Its manifest and external-checkout policy is applied below.
        if kind == "runtime" and name == "src/behavior_hl/embodied/evaluator_bridge/l0_taskspec_live_adapter.py":
            files[name] = adapt_projected_verifier_loader(files[name].decode()).encode()
            changes[name] = row["changes"]
            continue
        if kind == "bench" and name == "lab/scene_server.py":
            source = files[name].decode()
            cls = next(n for n in ast.parse(source).body
                       if isinstance(n, ast.ClassDef) and n.name == "ResidentInsideTaskScene")
            init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
            lines = source.splitlines(keepends=True)
            lines.insert(init.body[0].lineno - 2, "        runtime_registry: Path | None = None,\n")
            source = "".join(lines)
            source = source.replace("        self.run_dir = Path(run_dir).resolve()\n",
                                    '        self.run_dir = Path(run_dir).resolve()\n        self._runtime_registry = (BENCH_TRY_ROOT / "runtime" if runtime_registry is None else Path(runtime_registry)).resolve()\n', 1)
            for old, new in [('allowed_root=BENCH_TRY_ROOT / "runtime",', 'allowed_root=self._runtime_registry,'),
                             ('"ready_at": datetime.now(timezone.utc).isoformat(),', '"ready_at": datetime.now(timezone.utc).isoformat(),\n                    "pid": os.getpid(),'),
                             ('scene = ResidentInsideTaskScene(\n', 'scene = ResidentInsideTaskScene(\n            runtime_registry=Path(runtime_root).resolve().parent,\n')]:
                if source.count(old) != 1:
                    raise ValueError("scene runtime registry adaptation needs review")
                source = source.replace(old, new)
            files[name] = source.encode()
            changes[name] = row["changes"]
            continue
        old = git(row["source_root"], "show", row["source_revision"] + ":" + name).decode()
        adapted = (ROOT / row["destination"]).read_text()
        if kind == "bench" and name == "lab/model_client.py":
            # L2 has no Anthropic branch; adapt only failure paths that revision actually has.
            if adapted.count(_PROVIDER_REDACTION_VALUES) != 1:
                raise ValueError("reviewed provider credential binding is missing")
            adapted = adapted.replace(_PROVIDER_REDACTION_VALUES, "")
            adapted = re.sub(r"(?m)^ +redaction_values=redaction_values,\n", "", adapted)
            for expression in ("exc.reason", "exc"):
                adapted = adapted.replace("{_redact_provider_error_text(str(" + expression + "), redaction_values)}",
                                          "{" + expression + "}")
        if name == "scripts/tokenplan_resident_decision.py" and "if ENV_FILE.is_file()" in files[name].decode():
            source = files[name].decode()
            # The L2 loader already has environment fallback; retain that newer logic.
            node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "ENV_FILE" for t in n.targets))
            lines = source.splitlines(keepends=True)
            lines[node.lineno - 1:node.end_lineno] = ['ENV_FILE = (Path(os.environ["ESI_MODEL_ENV_FILE"]).expanduser()\n'
                                                    '            if os.environ.get("ESI_MODEL_ENV_FILE") else None)\n']
            files[name] = "".join(lines).replace("if ENV_FILE.is_file()", "if ENV_FILE is not None and ENV_FILE.is_file()").encode()
            changes[name] = row["changes"] + "; retain the newer L2 environment credential fallback"
            continue
        files[name] = apply_adaptation(old, adapted, files[name].decode(), name).encode()
        if kind == "bench" and name == "lab/model_client.py":
            files[name] = adapt_provider_error_requests(files[name].decode()).encode()
        changes[name] = row["changes"]
    if kind == "bench":
        name = "lab/model_client.py"
        source = files[name].decode()
        previous = 'PROJECT_ROOT = Path(os.environ.get("BEHAVIOR_HL_PROJECT_ROOT", str(Path(__file__).resolve().parents[2] / "runtime"))).resolve()'
        replacement = 'PROJECT_ROOT = Path(os.environ["BEHAVIOR_HL_PROJECT_ROOT"]).resolve()'
        if source.count(previous) != 1:
            raise ValueError("model Runtime root adaptation requires review")
        files[name] = source.replace(previous, replacement).encode()
        changes[name] = changes[name].replace(
            "rc3: honor explicit BEHAVIOR_HL_PROJECT_ROOT; otherwise use the installed default L0 Runtime bundle",
            "rc3: require the explicit selected BEHAVIOR_HL_PROJECT_ROOT")
        name = "lab/scene_source_identity.py"
        files[name] = adapt_scene_identity(files[name].decode()).encode()
        changes[name] = "verify registered frozen source manifests; preserve external official checkout Git identity only when its top-level exactly equals the supplied root"
        name = "lab/v_collection_handoff.py"
        if name in files:
            source = files[name].decode()
            source = replace_function(source, "verify_runtime", '''def verify_runtime(binding):
    from roboweave.profiles import verify_bundle
    import sys
    root = Path(binding['runtime']['root']).resolve()
    verify_bundle(root, binding['runtime']['commit'])
    module = importlib.import_module('behavior_hl.embodied.evaluator_bridge.v_intervention')
    for name, loaded in tuple(sys.modules.items()):
        if name == 'behavior_hl' or name.startswith('behavior_hl.'):
            origin = getattr(loaded, '__file__', None)
            if origin is None or not Path(origin).resolve().is_relative_to(root):
                raise ValueError('mixed V Runtime import roots')
    return module
''')
            files[name] = source.encode()
            changes[name] = "validate frozen Runtime manifest and every loaded Runtime module origin instead of unavailable checkout Git metadata"
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write new bundles after complete read-only validation")
    args = parser.parse_args()
    l0_identity = json.loads((PACKAGE / "_vendor/bench/SOURCE_IDENTITY.json").read_text())
    l0_row = next(row for row in read_records("bench") if row["source_path"] == "lab/scene_source_identity.py")
    imported = [{**l0_row, "changes": l0_identity["candidate_adaptations"][l0_row["source_path"]]}]
    prepared = []
    for kind, sources in (("bench", BENCH_SOURCES), ("runtime", RUNTIME_SOURCES)):
        for revision, root in sources.items():
            head = git(root, "rev-parse", "HEAD").decode().strip()
            dirty = git(root, "status", "--porcelain=v1", "--untracked-files=no").decode().splitlines()
            if head != revision or dirty:
                raise ValueError(f"Frozen source checkout changed: {kind} {revision}")
            if kind == "bench":
                names = git(root, "ls-tree", "-r", "--name-only", revision).decode().splitlines()
                names = [p for p in names if (p.startswith("lab/") or "/" not in p)
                         and (p.endswith(".py") or p in {"action20_bare_marker_schema.json", "action20_tool_schema.json", "action20_core_instructions.txt"})]
                files = bench_closure(read_archive(root, revision, names))
            else:
                resources = [r["source_path"] for r in read_records(kind)
                             if not r["source_path"].startswith(("src/", "tests/"))]
                files = read_archive(root, revision, ["src/behavior_hl", *resources])
            source_scopes = (["src/behavior_hl", "scripts", "configs"] if kind == "runtime"
                             else ["*.py", "lab/*.py"])
            unexpected = git(root, "ls-files", "--others", "--exclude-standard", "--", *source_scopes).decode().splitlines()
            if unexpected:
                raise ValueError("Untracked source content needs explicit provenance review")
            if any(Path(name).name in {"vray_maps.mdl", "vray_materials.mdl"} for name in files):
                raise ValueError("Restricted VRay dependencies must remain external")
            original = {name: sha(data) for name, data in files.items()}
            changes = adapt(kind, files)
            for name, data in files.items():
                if name.endswith(".py"):
                    ast.parse(data, filename=name)
            relative = Path("_vendor/revisions") / kind / revision
            destination = PACKAGE / relative
            if args.write and destination.exists():
                raise ValueError(f"Refusing to overwrite existing bundle: {kind} {revision}")
            identity = {"schema_version": "roboweave/source-identity-v1", "upstream_revision": revision,
                        "source_observation": {"root": root, "head": head, "tracked_dirty": dirty},
                        "files": {name: sha(data) for name, data in sorted(files.items())},
                        "candidate_adaptations": changes}
            files["SOURCE_IDENTITY.json"] = (json.dumps(identity, indent=2, ensure_ascii=False) + "\n").encode()
            prepared.append((kind, revision, relative, files))
            for name, digest in original.items():
                imported.append({"source_root": root, "source_revision": revision, "source_path": name,
                                 "source_sha256": digest, "destination": str(Path("src/roboweave") / relative / name),
                                 "changes": changes.get(name, "none")})
            print(f"{kind} {revision}: {len(original)} files, {sum(map(len, files.values()))} bytes, {len(changes)} adapted")
    if not args.write:
        return
    registry = {"schema_version": "roboweave/source-bundles-v1", "bundles": {"bench": {}, "runtime": {}}}
    for kind in registry["bundles"]:
        path = PACKAGE / "_vendor" / kind / "SOURCE_IDENTITY.json"
        identity = json.loads(path.read_text())
        registry["bundles"][kind][identity["upstream_revision"]] = {"path": f"_vendor/{kind}", "manifest_sha256": sha(path.read_bytes())}
    for kind, revision, relative, files in prepared:
        for name, data in files.items():
            target = PACKAGE / relative / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        registry["bundles"][kind][revision] = {"path": relative.as_posix(), "manifest_sha256": sha(files["SOURCE_IDENTITY.json"])}
    (PACKAGE / "data/source_bundles.json").write_text(json.dumps(registry, indent=2) + "\n")
    (ROOT / "provenance/imports_profiles.json").write_text(json.dumps(imported, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
