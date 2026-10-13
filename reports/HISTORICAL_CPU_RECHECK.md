# Historical Action CPU failures: independent recheck

The **four exact historical cases pass on both the original frozen Runtime and
the candidate in this isolated CPU environment**. This is new, environment-scoped
evidence. It does **not** rewrite the historical FAIL, establish its cause,
demonstrate a bug fix, or release the original launch gate.

The original L0 English-v2 `runtime_lock.json` still records:

- `action_targeted_cpu_tests.status = FAIL`: 185 passed, 4 failed.
- Isolated historical repeat: 4 failed, 10 passed, 31 deselected.
- Historical `a5bf13a` comparison: the same assertions also had 4 failures.
- `launch_gate = BLOCKED_UNRESOLVED_BASELINE_CPU_FAILURES`.

The source lock was read only. Its SHA256 before and after this recheck was
`89b89919ec2cb91a1236d07249da051c4ddcbefe9961584ecc887c41d3302cd9`.

| Parametrized case | Original `9b992b6` | Candidate |
| --- | --- | --- |
| `hidden_center-False` | PASS | PASS |
| `hidden_center-True` | PASS | PASS |
| `hidden_coarse_hits-False` | PASS | PASS |
| `hidden_coarse_hits-True` | PASS | PASS |

All four belong to
`test_action20_surface.py::test_on_top_provider_preserves_surface_evidence_through_publication`.
Here `False` and `True` are the upstream `physical` fixture parameter. Both runs
use synthetic arrays and mocked ray/scene data; neither parameter launches a
physical simulator.

The reference was recreated from Runtime commit
`9b992b6797330a836e9533e5409627bf63f7d94b` with read-only `git show` calls.
The original Python source tree, original test file, and required YAML/JSON
configuration were copied beneath the candidate's `.tmp/`. No MDL or simulator
asset was copied. The original test file SHA256 is
`8e0d1ccdf2e4005a3208e9c3561ee83725c8cab00625653c916fc0947021c9c9`.

The selected test function and its `_local_surface_world` / `_surface_hit`
helpers have identical AST hashes in original and candidate tests. The runtime
functions `_action20_on_top_surface_regions` and `on_top_snapshot_provider` also
have identical AST hashes. No assertion or runtime algorithm was modified for
this check. The relocated test file has unrelated source-layout adjustments;
those do not touch these cases.

Two separate Python processes imported their respective source trees. Import
origin assertions confirmed that all 177 loaded `behavior_hl` modules came from
the selected reference or candidate root. Both processes used the candidate's
independent `.venv`: Python 3.12.9, NumPy 1.26.4, Pillow 11.3.0, pytest 8.4.2 and
PyYAML 6.0.3. External pytest plugin autoload and bytecode/cache writing were
disabled. The JSON report records the exact four node IDs, commands, stdout,
configuration/source hashes, loaded-source hashes and environment versions.

The first reference collection attempt lacked three profile JSON dependencies.
Those original files were added from the same frozen commit before the final
recheck. That preparation error did not execute the four assertions and is not
counted as a historical-case result. All reference copies and temporary outputs
were removed after the completed runs.

[HISTORICAL_CPU_RECHECK.json](HISTORICAL_CPU_RECHECK.json) is the full evidence
record. Five imported candidate files differ at file level because of documented
portability changes; their paths are listed there and their changes are recorded
in [imports_runtime.json](../provenance/imports_runtime.json). The tested surface
sampling and publication functions are unchanged.

The historical environment was not reconstructed and the old failure's root
cause remains unknown. The original freeze/launch gate remains historical
evidence and is not automatically cleared by this report. No GPU, simulator,
resident, model request, task preparation, current evaluation process, or Live
benchmark was started or contacted.
