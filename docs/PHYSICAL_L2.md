# Physical L1 and L2 collection profiles

These bindings preserve four separate frozen profiles. They do not promote the older
L2 proposal audit to a runnable task collection and do not reuse the latest runtime
for every task. The L2 collection has 37 tasks: 23 Symbolic and 14 Physical.

| Profile | Tasks | Bench revision | Runtime revision | Initial-state route |
| --- | ---: | --- | --- | --- |
| `l1-p-inside5` | 5 | `7ebc9c948a047ba7f51e0a1d2bd768b228fad5eb` | `955585540d95cba02f672c0d3cdb94048db23bf2` | Prepared Physical restore |
| `l1-p-ontop5` | 5 | `7ebc9c948a047ba7f51e0a1d2bd768b228fad5eb` | `c813a24f853bb90ba3772365ca5182ecb28c553b` | Prepared Physical restore |
| `l1-p-additions20` | 20 | `7ebc9c948a047ba7f51e0a1d2bd768b228fad5eb` | `ab615d43d4ede7e6a50eaeda189077529bef361a` | Prepared Physical restore |
| `l2-collection37` | 37 | `4c8364842157153875d098ebb529adcc0620a714` | `aa878b489d402aaa9b1e30b083d0435cf44b26be` | Per-task Symbolic restore or Physical reconstruction |

The family adapter is `src/roboweave/adapters/physical.py`. Its first operation,
`resolve_inputs`, only reads verified private files and the selected public task.
Its second operation, `inspect_inputs`, runs after the profile has been activated
in an isolated worker and calls that profile's original handoff validators and
native argument builders. P30 additionally invokes the original model-start
admission and Physical-kwargs checks using the frozen handoff preparation-only
decision. It supplies no synthetic Live restore evidence. It does not import or
launch OmniGibson.

## Frozen source and public instructions

P30 is the three task/review pairs under the benchmark's
`benchmark/L1-P-FINAL/model_batch_20260924/{inside,ontop,additions}`. L2 public
instructions and task handles come from the frozen
`l2_37_gpt6max_pair_19slot_20260925.inputs` campaign and its approved English review.
The L2 input operator profile is **not distributed or imported as an asset**.
Only its per-task budgets and handoff/index bindings are projected into the asset
manifest. These fields were checked for agreement across all frozen lanes. The
original operator profile's SHA256 records the projection's provenance.

L2's approved English review intentionally differs from some source-language
TaskSpecs. The adapter verifies the review's exact task handle/name, approved
status and English text, and preserves the TaskSpec text and its own hash.
It does not rewrite the TaskSpec, registration, process contract or goal formula.
P30 still requires its source TaskSpec public instruction to equal its approved
English instruction.

See [the import record](../provenance/imports_physical_l2.json), the per-file
SHA256 manifests in `src/roboweave/data/assets/`, and the L2 catalog delta in
`src/roboweave/data/tasks/l2-collection37.json`. `tools/build_physical_l2_manifests.py`
is a maintainer-only, bounded reader of the original inputs, not an installation
step or an external user's dependency. It does not traverse artifact directories.

## Two different Physical contracts

P30 requires a derived `PREPARED_MODEL_REPLAY` package, `model_rollout_allowed=true`
and `RESTORE_AND_VERIFY_NO_RECONSTRUCTION`. The native route restores the frozen
initial state and rechecks admission. The adapter retains the package's original
`live_run_ready`, promotion label and restore verification status. Snapshot
presence is not a new claim of Live qualification.

L2 Physical requires `PENDING_NEW_LIVE_READBACK`,
`RECONSTRUCT_AND_VERIFY_DONOR_PHYSICAL_CONTRACT` and `donor_success_inherited=false`.
The original `load_physical_instruction` validator checks those fields, the
Physical execution policy, the template and registration identities. Its native
arguments use the original physical collection service callback: that callback
prepares with `model_prepared_states=None`, performs fresh construction and
Physical qualification, and does not execute an Oracle strategy. A Symbolic
prepared-world restore must never substitute for this route. The prepared index
is retained as provenance and camera/contract evidence; it is not permission to
restore the L2 Physical world.

The supported model loop is the frozen Common harness. Replacing it with a
historical AgentCLI harness is a separate experimental condition, even when the
model name, task ID and action budget are identical.

## Budgets, observations and history

| Profile | Action decisions | Model calls | Declared wall-clock budget | Seed |
| --- | ---: | ---: | ---: | ---: |
| P Inside5 | 12–15 | 300 | 3600 s | 20260914 |
| P OnTop5 | 12–54 | 300 | 3600 s | 20260914 |
| P Additions20 | 12–78 | 300 | 3600 s | 20260914 |
| L2 collection37 | 35–300 | 105–3810 | 7200 s | 19506 |

Each exact per-task value is frozen in the catalog and input manifest. Ranges are
informational, not default budgets. The inputs retain the frozen five-turn history,
`r1pro_three_rgb_head_eval`, overhead off, Full scene mode, and Symbolic zero
physics steps or Physical stepping according to their original handoff.

The declared wall-clock fields are preserved. The frozen Common task loop does
not by itself implement the complete historical dispatcher/watchdog policy.
A local process timeout is not proof of equivalent end-to-end wall-clock scoring;
that policy remains a separately declared validation item. This release work
neither invents a new timeout rule nor silently treats an external watchdog as
part of the model loop.

## Private inputs and portability

Manifests describe required inputs; they do not contain private TaskSpecs,
registration records, scene states, simulator asset binaries or raw trajectories.
An authorized asset owner supplies the exact files by source-root mapping or
logical asset key. Hash mismatches fail before import. Imported originals remain
immutable; each run receives derived files with a recorded projection/relocation
receipt and updated dependent hashes.

The closure includes task/review metadata, handoff, package, TaskSpecs, runtime
registration, BDDL, initial template/TRO, template contract, prepared index/bundle
and any exact donor proof files actually read by the native validators. For a
`prepared_oracle_states` index, the donor success/prepared/camera/construction
records remain evidence only. They are not rewritten as successful target runs.
The selected target contract hash is updated only to account for its recorded
path relocation. The restricted BEHAVIOR asset tree remains a separately acquired
external dependency configured by `runtime.data_root`.

## Verification and remaining work

This change ran 13 small distributable adapter tests and checked all 67 bindings
against their respective frozen Bench `compile_handoff`, native argument builders
and `load_prepared` functions on CPU. All 67 passed. These checks read metadata;
they do not load a simulator, restore a scene or make a model request. The source
file/hash closure was checked while constructing each manifest. A further
[67-task relocation check](../reports/rc3/PHYSICAL_L2_INPUT_CHECK.json) imported
only the exact manifest files, materialized each task and passed the frozen
`inspect_inputs` and `load_registered_l0_tasks` validators under a guard that
forbade original source-tree and external-asset reads. All 905 derived-file
receipts were checked: reversing only recorded path/hash changes reproduced the
projected source JSON exactly. Original imported hashes remained unchanged, and
all temporary private inputs were deleted.

Per the current task scope, independent environment reproduction and Live checks
are deferred. A later owned environment must establish P30 restore admission,
L2 Symbolic restore, L2 Physical fresh construction/qualification, model-visible
camera parity, Action20 execution, process/history acceptance, trajectory outputs
and the explicitly chosen wall-clock policy. Historical source success and these
CPU checks are not new Live results for the public candidate.
