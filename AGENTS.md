# Candidate workspace

Only this directory may be modified. Original source trees, assets, services,
environments and run outputs are read-only. Do not start GPU, simulator, model
requests, SSH, resident services or queues during preparation. Do not publish.

Multiple agents share this workspace. Respect file ownership; never revert
another agent's changes. The coordinator owns packaging, CLI, configuration,
bootstrap, integration reports and shared interfaces.

Preserve frozen evaluation semantics. Record every imported file's source
revision, source SHA256, and candidate path. Untracked source content must be
hashed; do not silently substitute HEAD for a task's frozen revision.

Architecture: `src/esi_benchmark` is a thin public interface. `_vendor/bench`,
`_vendor/runtime` and revision bundles retain upstream layouts as package data.
Public APIs use fixed JSON operations in isolated native child processes; only
the private bootstrap adds the selected immutable roots to those child import
paths. No mutable symlinks to original repositories. Default profile is
`l0-english-v2`: Bench 5b8d2b37fd7d97b161e88de4d438ec789c78f9b4,
Runtime 9b992b6797330a836e9533e5409627bf63f7d94b.

All generated files, temporary files, isolated environments and outputs must
remain beneath this candidate directory. Use `PYTHONDONTWRITEBYTECODE=1` for
source comparisons. Tests are CPU-only; offline fixtures are never Live proof.

Source import records: each worker writes `provenance/imports_<owner>.json`
as a list of objects with `source_root`, `source_revision` (or null),
`source_path` (relative), `source_sha256`, `destination` (candidate relative),
and `changes` (short explanation or `none`). Do not store secrets or patches
containing secrets. Preserve source file attribution and applicable licenses.
