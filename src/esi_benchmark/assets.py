"""Hash-checked private input import and immutable per-run path relocation."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
from importlib.resources import files

from .tasks import profile_for


def asset_manifest(profile="l0-english-v2"):
    profile_for(profile)
    resource = files("esi_benchmark").joinpath(f"data/assets/{profile}.json")
    if not resource.is_file():
        raise ValueError(f"No asset manifest for profile {profile}")
    return json.loads(resource.read_text())


def _sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _relative(value):
    value = str(value)
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Unsafe relative asset path: {value}")
    return Path(*path.parts)


def _within(root, relative):
    root = Path(root).absolute()
    path = root / _relative(relative)
    if root.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Asset path escapes its root or root is a symlink")
    for current in (path, *path.parents):
        if current.is_symlink():
            raise ValueError("Symlink asset paths are not permitted")
    return path


def _path(root, row):
    return _within(root, Path("files") / _relative(row["source_root"]) / _relative(row["source_path"]))


def _private_directory(path, *, exist_ok=True):
    if not path.parent.exists():
        _private_directory(path.parent, exist_ok=False)
    path.mkdir(mode=0o700, exist_ok=exist_ok)
    status = path.lstat()
    if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.geteuid():
        raise ValueError("Private input directory must be owned by the current user")
    path.chmod(0o700)


def _private_open(path, *, exclusive=True):
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
    fd = os.open(path, flags | (os.O_EXCL if exclusive else 0), 0o600)
    try:
        status = os.fstat(fd)
        if not stat.S_ISREG(status.st_mode) or status.st_uid != os.geteuid() or status.st_nlink != 1:
            raise ValueError("Private input file must be a singly linked file owned by the current user")
        os.fchmod(fd, 0o600)
        return os.fdopen(fd, "r+b")
    except BaseException:
        os.close(fd)
        raise


def _private_write(path, data, *, exclusive=False):
    with _private_open(path, exclusive=exclusive) as stream:
        stream.truncate(0)
        stream.write(data)


def _selected(config, task=None):
    manifest = asset_manifest(config.get("profile", "l0-english-v2"))
    if task is None:
        return manifest, manifest["files"]
    if task["profile"] != manifest["profile"]:
        raise ValueError("Task belongs to a different version profile")
    return manifest, [row for row in manifest["files"] if task["task_id"] in row["task_ids"]]


def check_assets(config, task=None):
    """Check all profile assets, or just the explicitly supplied task's inputs."""
    manifest, rows = _selected(config, task)
    root = Path(config["assets"]["root"])
    missing, invalid = [], []
    for row in rows:
        try:
            path = _path(root, row)
            if not path.is_file():
                missing.append(row["key"])
            elif path.stat().st_size != row["bytes"] or _sha(path) != row["sha256"]:
                invalid.append(row["key"])
        except (OSError, ValueError):
            invalid.append(row["key"])
    return {"ok": not missing and not invalid, "profile": manifest["profile"],
            "checked": len(rows), "missing": missing, "invalid": invalid,
            "scope": task["task_id"] if task else "all", "root": str(root)}


def prepare_assets(config, source_roots=None, mapping=None, bundle=None, task=None):
    """Copy authorized local files only after every selected source passes its hash.

    ``mapping`` maps logical asset keys to explicit source files. A bundle has the
    same ``files/<source_root>/<source_path>`` layout as an imported asset root.
    This function never downloads files, extracts archives, or modifies sources.
    """
    manifest, rows = _selected(config, task)
    root = Path(config["assets"]["root"]).absolute()
    if source_roots is not None and not isinstance(source_roots, dict):
        raise ValueError("source_roots must map root names to local directories")
    if mapping is not None and not isinstance(mapping, dict):
        raise ValueError("mapping must map asset keys to local files")
    roots = {**config["assets"].get("source_roots", {}), **(source_roots or {})}
    mapping = mapping or {}
    if any(not isinstance(value, (str, Path)) for value in [*roots.values(), *mapping.values()]):
        raise ValueError("Asset source paths must be strings or paths")
    unknown = set(roots) - set(manifest["source_roots"])
    unknown_keys = set(mapping) - {row["key"] for row in manifest["files"]}
    if unknown or unknown_keys:
        raise ValueError("Unknown asset source root or asset mapping key")
    marker = _within(root, ".esi-benchmark-assets.json")
    if root.is_symlink():
        raise ValueError("Asset destination must not be a symlink")
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise ValueError("Refusing a nonempty destination without an ESI asset ownership marker")
    if marker.exists():
        if marker.is_symlink() or json.loads(marker.read_text()).get("profile") != manifest["profile"]:
            raise ValueError("Asset destination ownership/profile mismatch")
    candidates = []
    for row in rows:
        if row["key"] in mapping:
            source = Path(mapping[row["key"]]).absolute()
            source_root = source.parent
            source = _within(source_root, source.name)
        elif bundle is not None:
            source_root = Path(bundle).absolute()
            source = _path(source_root, row)
        elif row["source_root"] in roots:
            source_root = Path(roots[row["source_root"]]).absolute()
            source = _within(source_root, row["source_path"])
        else:
            raise ValueError(f"No source supplied for {row['source_root']} ({row['key']})")
        if root.resolve().is_relative_to(source_root.resolve()) or source_root.resolve().is_relative_to(root.resolve()):
            raise ValueError("Asset destination overlaps a source tree")
        if not source.is_file() or source.stat().st_size != row["bytes"] or _sha(source) != row["sha256"]:
            raise ValueError(f"Asset missing or SHA256/size mismatch: {row['key']}")
        destination = _path(root, row)
        if destination.exists() and (not destination.is_file() or _sha(destination) != row["sha256"]):
            raise ValueError(f"Refusing to overwrite a different imported asset: {row['key']}")
        candidates.append((row, source, destination))
    _private_directory(root)
    _private_write(marker, (json.dumps({"schema_version": 1, "profile": manifest["profile"]}) + "\n").encode())
    copied = 0
    for row, source, destination in candidates:
        if destination.exists():
            with _private_open(destination, exclusive=False):
                pass
            continue
        _private_directory(destination.parent)
        created = False
        try:
            with source.open("rb") as incoming:
                with _private_open(destination) as outgoing:
                    created = True
                    shutil.copyfileobj(incoming, outgoing)
            if _sha(destination) != row["sha256"]:
                raise ValueError(f"Source changed during import: {row['key']}")
        except BaseException:
            if created:
                destination.unlink(missing_ok=True)
            raise
        copied += 1
    receipt = {"schema_version": 1, "profile": manifest["profile"],
               "files": {row["key"]: {"path": str(_path(root, row).relative_to(root)), "sha256": row["sha256"]}
                         for row in manifest["files"] if _path(root, row).is_file()}}
    _private_write(_within(root, "import_receipt.json"), (json.dumps(receipt, indent=2) + "\n").encode())
    return {**check_assets(config, task), "copied": copied}


def _suffix(row):
    return row["source_root"] + "/" + row["source_path"]


def _match_path(value, rows):
    if not isinstance(value, str):
        return None
    normalized = os.path.normpath(value).replace("\\", "/")
    for row in rows:
        suffix = _suffix(row)
        if normalized == suffix or normalized.endswith("/" + suffix):
            return row
    return None


def resolve_task_inputs(config, task):
    """Return verified private inputs; these dictionaries must never reach models."""
    manifest, rows = _selected(config, task)
    report = check_assets(config, task)
    if not report["ok"]:
        raise ValueError(f"Task assets are incomplete: {len(report['missing'])} missing, {len(report['invalid'])} invalid")
    root = Path(config["assets"]["root"]).absolute()
    if task["profile"] != "l0-english-v2":
        from .adapters import resolve_inputs
        return resolve_inputs(config, task, manifest, rows, root)
    release = next(row for row in rows if "release_tasks" in row["roles"])
    source_tasks = json.loads(_path(root, release).read_text())
    matches = [row for row in source_tasks if row["runtime_instruction_id"] == task["runtime_instruction_id"]]
    if len(matches) != 1:
        raise ValueError("Frozen task identity does not resolve uniquely")
    raw = matches[0]
    if raw["instruction_en"] != task["instruction_en"] or any(raw["budget"].get(k) != v for k, v in task["budget"].items()):
        raise ValueError("Frozen task public instruction or budget does not match the catalog")
    def ref(value):
        row = _match_path(value["path"], rows)
        if row is None:
            raise ValueError("Runtime input reference is outside the selected asset manifest")
        return str(_path(root, row))
    source_json = ref(raw["source"])
    source = json.loads(Path(source_json).read_text())
    package_json = ref(source["package"])
    package = json.loads(Path(package_json).read_text())
    task_specs_ref = source.get("task_specs_ref") or package["task_specs"]
    identity = source["bench_instruction_binding"]["source_task_identity"]
    model_task = {"profile": task["profile"], "task_id": task["source"]["row_index"], "task_name": task["runtime_instruction_id"],
                  "task_split": identity["split"], "task_instance_id": str(identity["instance_id"]),
                  "task_seed": source["seed"], "instruction": task["instruction_en"],
                  **task["budget"], "budget": dict(task["budget"]), "instruction_provenance": {
                      "profile": task["profile"], "source_sha256": release["sha256"],
                      "source_task_identity": dict(identity),
                      "instruction_sha256": task["instruction_sha256"]}}
    return {"profile": task["profile"], "entrypoint": "prepared", "asset_root": str(root), "raw_task": raw,
            "instruction_id": task["runtime_instruction_id"], "model_task": model_task,
            "task_json": str(_path(root, release)), "source_json": source_json,
            "handoff_json": ref(raw["handoff"]), "prepared_index_json": ref(raw["prepared_index"]),
            "task_specs_jsonl": ref(task_specs_ref),
            "benchmark_package_json": package_json,
            "mapping": {_suffix(row): str(_path(root, row)) for row in rows}, "source": source}


def _project(value, row, instruction_id, task_id=None):
    """Select a frozen record without changing any of its semantic fields."""
    if "selection" in row:
        selection = row["selection"]
        if selection is None:
            return value
        ids = selection["ids"].get(task_id)
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
            raise ValueError("Asset projection lacks an unambiguous selected identity")
        wanted = set(ids)
        kind = selection["kind"]

        def selected(items, field):
            result = [item for item in items if item.get(field) in wanted]
            if len(result) != len(wanted) or {item[field] for item in result} != wanted:
                raise ValueError("Asset projection identities do not resolve uniquely")
            return result

        if kind == "release_tasks":
            return selected(value, selection.get("field", "runtime_instruction_id"))
        if kind == "task_specs":
            return selected(value, "task_spec_id")
        if kind == "registration_catalog":
            field = selection.get("field", "task_spec_id")
            if all(field not in item and "candidate_id" in item for item in value["tasks"]):
                field = "candidate_id"
            chosen = selected(value["tasks"], field)
            return {**value, "tasks": chosen, "task_count": len(chosen)}
        if kind == "package":
            if not wanted <= set(value["task_spec_ids"]):
                raise ValueError("Package projection is absent from its frozen identities")
            return {**value, "task_spec_ids": [item for item in value["task_spec_ids"] if item in wanted],
                    "task_count": len(wanted)}
        if kind == "handoff":
            samples = selected(value["samples"], "task_spec_id")
            names = {item["package_name"] for item in samples}
            packages = [item for item in value["packages"] if item["name"] in names]
            if len(packages) != len(names):
                raise ValueError("Handoff projection package identity is ambiguous")
            return {**value, "samples": samples, "packages": packages, "count": len(samples)}
        if kind == "prepared_index":
            return {**value, "instructions": {key: value["instructions"][key] for key in ids}}
        if kind == "prepared_oracle_states":
            return {**value, "items": selected(value["items"], "task_spec_id")}
        if kind == "v_success_index":
            return {**value, "rows": selected(value["rows"], "collection_config_key")}
        if kind == "v_compatibility":
            return {**value, "configurations": {key: value["configurations"][key] for key in ids}}
        raise ValueError(f"Unknown asset projection kind: {kind}")
    roles = row["roles"]
    if "release_tasks" in roles:
        return [item for item in value if item["runtime_instruction_id"] == instruction_id]
    if "task_specs" in roles:
        selected = [item for item in value if item["task_spec_id"] == instruction_id]
        if len(selected) != 1:
            raise ValueError("TaskSpecs do not contain exactly one selected identity")
        return selected
    if "registration_catalog" in roles:
        selected = [item for item in value["tasks"] if item.get("task_spec_id", item.get("candidate_id")) == instruction_id]
        if len(selected) != 1:
            raise ValueError("Registration does not contain exactly one selected identity")
        return {**value, "tasks": selected, "task_count": len(selected)}
    if "package" in roles:
        if instruction_id not in value["task_spec_ids"]:
            raise ValueError("Package does not contain the selected identity")
        return {**value, "task_spec_ids": [instruction_id], "task_count": 1}
    if "handoff" in roles:
        samples = [item for item in value["samples"] if item["task_spec_id"] == instruction_id]
        if len(samples) != 1:
            raise ValueError("Handoff does not contain exactly one selected identity")
        packages = [item for item in value["packages"] if item["name"] == samples[0]["package_name"]]
        if len(packages) != 1:
            raise ValueError("Handoff package identity is ambiguous")
        return {**value, "samples": samples, "packages": packages, "count": 1}
    if "prepared_index" in roles:
        return {**value, "instructions": {instruction_id: value["instructions"][instruction_id]}}
    return value


def materialize_task_inputs(config, task, output_dir):
    """Create a private, single-task runnable copy and a complete change receipt.

    Imported bytes stay immutable. Only source paths, their file hashes and
    aggregate membership/counts change. Goal, process, initial-state, identity,
    seed and model budget fields are retained from their exact frozen records.
    """
    resolved = resolve_task_inputs(config, task)
    manifest, rows = _selected(config, task)
    source_root = Path(config["assets"]["root"]).absolute()
    output = Path(output_dir).absolute()
    if output.is_symlink() or output.resolve().is_relative_to(source_root.resolve()) or source_root.resolve().is_relative_to(output.resolve()):
        raise ValueError("Materialized inputs must be separate from imported assets")
    if output.exists():
        raise ValueError("Materialization needs a new output directory")
    # Check parents before writing, so a run cannot escape through a symlink.
    for parent in output.parents:
        if parent.is_symlink():
            raise ValueError("Materialization output ancestors must not be symlinks")
    _private_directory(output, exist_ok=False)
    by_key = {row["key"]: row for row in rows}
    destinations = {row["key"]: _path(output, row) for row in rows}
    states, hashes, records = {}, {}, []
    instruction_id = task["runtime_instruction_id"]
    handoff = next((row for row in rows if "handoff" in row["roles"]
                    and str(_path(source_root, row)) == resolved.get("handoff_json")), None)
    for row in rows:
        if row.get("handoff_relative_path"):
            if handoff is None:
                raise ValueError("Relative handoff dependency lacks a selected handoff")
            destinations[row["key"]] = _within(destinations[handoff["key"]].parent, row["handoff_relative_path"])

    def build(key):
        if states.get(key) == "done":
            return hashes[key]
        if states.get(key) == "visiting":
            raise ValueError(f"Cyclic hashed asset references: {key}")
        states[key] = "visiting"
        row, destination = by_key[key], destinations[key]
        original = _path(source_root, row)
        changes = []

        def reference_for(value):
            key = row.get("path_bindings", {}).get(value) if isinstance(value, str) else None
            if key is not None:
                if key not in by_key:
                    raise ValueError("Relative asset reference is outside the selected manifest")
                return by_key[key]
            return _match_path(value, rows)

        def rewrite(value, pointer=""):
            if isinstance(value, list):
                return [rewrite(item, f"{pointer}/{index}") for index, item in enumerate(value)]
            if isinstance(value, dict):
                changed = {k: rewrite(v, pointer + "/" + k.replace("~", "~0").replace("/", "~1")) for k, v in value.items()}
                reference = reference_for(value.get("path"))
                if reference and "sha256" in value:
                    digest = build(reference["key"])
                    if changed["sha256"] != digest:
                        changes.append({"pointer": pointer + "/sha256", "kind": "relocated_file_hash", "old": changed["sha256"], "new": digest})
                        changed["sha256"] = digest
                return changed
            if isinstance(value, str):
                reference = reference_for(value)
                replacement = str(destinations[reference["key"]]) if reference else value
                asset_anchor = "/BEHAVIOR-1K-2026-data/"
                if pointer == "/runtime/root" and "v_collection_binding" in row["roles"]:
                    from .profiles import source_roots
                    replacement = str(source_roots(task["profile"])["runtime_root"])
                elif not reference and asset_anchor in value and value.startswith("/"):
                    data_root = config.get("runtime", {}).get("data_root")
                    if not data_root:
                        raise ValueError("Task input contains external assets; set runtime.data_root before materialization")
                    replacement = str(_within(Path(data_root), value.split(asset_anchor, 1)[1]))
                elif not reference and value.startswith("/"):
                    normalized = os.path.normpath(value)
                    candidates = []
                    for root_name in manifest["source_roots"]:
                        anchor = "/" + root_name
                        index = normalized.find(anchor + "/")
                        if index >= 0 or normalized.endswith(anchor):
                            index = index if index >= 0 else len(normalized) - len(anchor)
                            candidates.append((index, normalized[index + 1:]))
                    if candidates:
                        suffix = min(candidates)[1]
                        replacement = str(_within(output, Path("files") / _relative(suffix)))
                if replacement != value:
                    changes.append({"pointer": pointer, "kind": "path_relocation", "old": value, "new": replacement})
                return replacement
            return value

        if original.suffix in (".json", ".jsonl"):
            raw = original.read_text()
            value = [json.loads(line) for line in raw.splitlines() if line.strip()] if original.suffix == ".jsonl" else json.loads(raw)
            projected = _project(value, row, instruction_id, task["task_id"])
            if projected != value:
                changes.append({"pointer": "", "kind": "single_task_selection", "instruction_id": instruction_id,
                                "rule": "Select the exact frozen task record and its aggregate identity/count metadata."})
            transformed = rewrite(projected)
            if "selection" not in row and "prepared_index" in row["roles"] and "handoff_sha256" in transformed:
                if handoff is None:
                    raise ValueError("Prepared index lacks a selected handoff")
                digest = build(handoff["key"])
                changes.append({"pointer": "/handoff_sha256", "kind": "relocated_file_hash", "old": transformed["handoff_sha256"], "new": digest})
                transformed["handoff_sha256"] = digest
            for binding in row.get("hash_bindings", []):
                if "task_ids" in binding and task["task_id"] not in binding["task_ids"]:
                    continue
                pointer = binding["pointer"]
                if not isinstance(pointer, str) or not pointer.startswith("/"):
                    raise ValueError("Asset hash binding requires an absolute JSON pointer")
                parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]
                node = transformed
                try:
                    for part in parts[:-1]:
                        node = node[int(part)] if isinstance(node, list) else node[part]
                    leaf = int(parts[-1]) if isinstance(node, list) else parts[-1]
                    old = node[leaf]
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    raise ValueError("Selected asset hash binding is absent after projection") from exc
                digest = build(binding["key"])
                encoding = binding.get("encoding", "file")
                if encoding == "canonical_json":
                    bound = json.loads(destinations[binding["key"]].read_text())
                    digest = hashlib.sha256(json.dumps(bound, ensure_ascii=False, sort_keys=True,
                                                       separators=(",", ":"), allow_nan=False).encode()).hexdigest()
                elif encoding != "file":
                    raise ValueError("Unsupported asset hash binding encoding")
                node[leaf] = digest
                changes.append({"pointer": pointer, "kind": "relocated_file_hash" if encoding == "file" else "relocated_canonical_json_hash",
                                "old": old, "new": digest})
            data = ("".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in transformed)
                    if original.suffix == ".jsonl" else json.dumps(transformed, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        else:
            data = original.read_bytes()
        _private_directory(destination.parent)
        _private_write(destination, data, exclusive=True)
        hashes[key] = hashlib.sha256(data).hexdigest()
        states[key] = "done"
        records.append({"key": key, "original_sha256": row["sha256"], "derived_sha256": hashes[key],
                        "derived_path": str(destination.relative_to(output)), "changes": changes})
        return hashes[key]

    for row in rows:
        build(row["key"])
    receipt = {"schema_version": 1, "profile": task["profile"], "task_id": task["task_id"],
               "instruction_id": instruction_id, "files": records,
               "semantics": "Single-task selection and file relocation only; no goal/process/initial-state/budget changes."}
    receipt_path = output / "materialization_receipt.json"
    _private_write(receipt_path, (json.dumps(receipt, indent=2) + "\n").encode(), exclusive=True)
    original_to_derived = {str(_path(source_root, row)): str(destinations[row["key"]]) for row in rows}
    def relocate(value):
        if isinstance(value, str):
            return original_to_derived.get(value, value)
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        return value
    result = relocate(resolved)
    result["mapping"] = {_suffix(row): str(destinations[row["key"]]) for row in rows}
    if "source_json" in result:
        result["source"] = json.loads(Path(result["source_json"]).read_text())
    result["materialization_receipt"] = str(receipt_path)
    return result
