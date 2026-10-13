# Frozen L1 A/M/V adapters

The three profiles preserve separate Bench revisions and two Runtime revisions.
They share the public CLI and private asset importer, not a floating Runtime.

| Profile | Tasks | Bench | Runtime | Entry |
| --- | ---: | --- | --- | --- |
| `l1-a30` | 30 | `1d76bc5dc53b1995f2e7c301cb06569bcf0c4923` | `cd25f9ba1579bdca1f0dadf6fce7ad10974b8717` | Prepared handoff |
| `l1-m30-public-v2` | 30 | `fb1cf143f850fe43a04d25440f07cbfab4007093` | `918aefd40beb72368126cc3fa2891bbb732ad490` | Prepared handoff |
| `l1-v30` | 30 | `d8ae5151fadfb3338d122fdb12b26051ed753852` | `918aefd40beb72368126cc3fa2891bbb732ad490` | V collection binding |

The source of this selection is the frozen AMV90 release; its location and full
SHA256 are in [imports_amv.json](../provenance/imports_amv.json). M's task and
instruction-review aliases are resolved once to approved ordinary files; the
freeze hashes still match, and both declared and resolved paths are recorded.
The public catalog
retains the approved release instructions and per-task budgets. A and V use 100
action decisions; M preserves its individual 12–150 decision budgets. All use
1,000 model calls, a declared 3,600-second budget and five history frames. The
historical direct model loop does not enforce a strict total wall-clock deadline;
this integration does not silently redefine that behavior.

## Data flow

1. `assets prepare` verifies each authorized local file before private import.
2. A task-specific materialization relocates file references, selects exact task
   records and updates the corresponding byte hashes. Every change is recorded.
3. `adapters.amv.resolve_inputs` checks the frozen release row and approved public
   instruction review without importing Bench or Runtime.
4. A separate source-profile worker calls `inspect_inputs`. It runs the original
   static handoff/V loaders and validates the complete prepared state.
5. The explicit owned Live launcher can then pass the resulting arguments to the
   frozen Bench entry point. Static checks do not prove simulator restoration.

M has no `raw.source` file. Its native split and instance are obtained from the
original `compile_handoff`, not guessed from its public task index. Public model
handles remain 0–29 in each release; native task IDs and V collection IDs remain
separate identities. V `model_task.task_name` is the public collection key, matching
the frozen scene status; `instruction_id`, prepared state and
`v_identity.task_spec_id` retain the native TaskSpec identity.

M registration also verifies its frozen prepared-Oracle index, donor success,
robot/camera preparation and initial construction proof. Relocated JSON is compact
so deep evidence does not exceed the original 2 MiB loader limit merely because
of pretty printing. That limit is unchanged.

A derived challenge can rely on a control task's approved preparation without
claiming that the challenge itself was solved. Its `prepared_derivation` and
complete donor handoff are retained. The donor handoff is never selected using
the challenge ID.

V passes `--v-collection-binding`, the native instruction ID and the public
collection configuration ID; it does not masquerade as a prepared handoff. Its
original `load_binding`, `native_source_identity` and `load_package` validate the
complete goal, intervention class, trigger timing, historical source and approved
Runtime compatibility. The binding Runtime root is relocated explicitly to the
installed source bundle and the original `verify_runtime` validates its complete
source identity before CPU admission. `complete_goal_sha256` and intervention semantics are not
rewritten. File hashes in compatibility records follow the relocated index and
package. R4 destination variants also recompute the canonical JSON hash of the
relocated candidate contract because that contract contains relocated file
references; the receipt distinguishes this from a byte hash change.

## Private resources and dependency boundaries

The installed manifests contain only logical roots, relative names, sizes,
hashes and task membership. They contain no prepared state, hidden goal, licensed
scene content, API keys or raw trajectory data. The user must provide authorized
local copies using explicit source roots, asset-key mappings or a prepared bundle.
Importing a manifest does not download anything and never connects to the
historical SSH endpoints mentioned by private evidence.

AMV dependency generation is explicit and bounded: `tools/build_amv_asset_manifests.py`
starts at the frozen release and follows executable references. It does not scan
artifact trees. Prepared bundles are self-contained; their historical `sources`
are provenance, not remote recovery instructions. V's frozen loader does read
specific native-world receipts, successful preparation, run manifests and task
protocol files. Those required private files are listed by hash, not distributed.
Full run directories, images, action trajectories and training exports are not
included. Licensed BEHAVIOR assets must still be obtained separately.

A manifest row's `selection` explicitly distinguishes executable records from
opaque donor evidence. `hash_bindings` describes hashes stored without adjacent
path fields. The materializer must reject missing selectors, missing binding
pointers and cyclic file-hash dependencies; it must not weaken source validators
when relocation fails.

## Scoring and remaining verification

A/M retain the original process/history auditing and canonical goal verification.
V retains its lifecycle result and protocol completion separately from canonical
Q. Goal Q alone must not trigger V auto-termination. V protocol incompleteness is
an invalid formal run, and V checkpoint/rollback remains forbidden. An incomplete
process cannot become successful because only its final static goal holds.

The distributable tests cover public identity/review mismatches, exact per-task
budgets, absence of parent-process source activation, and manifest dependency and
donor selection integrity. The release's overall validation report and
[AMV input check](../reports/rc3/AMV_INPUT_CHECK.json) record the integration checks.
All 90 tasks passed their applicable static contracts across the initial and
targeted repair checks; V30 was rechecked with the installed-Runtime binding.
No Live scene, model request or new reproduction environment
was run for this change. Independent future Live acceptance must verify initial
state restoration, action/Marker transitions, A/M process gates, each supported V
intervention class and timing, terminal result evidence, and owned resource cleanup.
