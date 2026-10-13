# Frozen sources and candidate changes

Directory dates and the current development HEAD are not execution provenance.
Eight task profiles select 13 installed source bundles. The complete current
mapping is in [Support matrix](SUPPORT_MATRIX.md) and the installed
`data/source_bundles.json`; each root has its own `SOURCE_IDENTITY.json`.
New profile imports use the exact specified commit from a matching clean source
checkout, with relevant untracked source checks and per-file original hashes.

The following table is the **initial L0 import observation**, retained as
historical provenance. Its execution binding follows L0's `runtime_lock.json`:

| Component | Imported execution revision | Observed source checkout HEAD |
| --- | --- | --- |
| Bench | `5b8d2b37fd7d97b161e88de4d438ec789c78f9b4` | `a623762b2cf1b5f02025e6f1cd48cd80f426c7c5` |
| Runtime | `9b992b6797330a836e9533e5409627bf63f7d94b` | same |
| Task repository | Final inputs frozen individually by SHA256 | `1665ac1b0d1863cc642165deceb020eedd0db17e` |

The observed Bench HEAD changes movement tolerance, provider envelope handling,
provider adapters and Physical result classification relative to the task-frozen
revision. It was not substituted. Its uncommitted DeepSeek experiments are outside
this profile and are recorded without importing their logic. The task repository's
old compiler and tests have uncommitted changes; the candidate consumes the actual
frozen final definitions instead of regenerating them with a different compiler.
Many final task files are untracked upstream: Git HEAD alone cannot identify them.

The source repository paths, HEADs, tracked working-tree status and relevant hashes
are in [source_roots.json](../provenance/source_roots.json). Each imported code,
resource and original test has a source path, full revision or working-file status,
SHA256 and candidate destination:

- [Bench imports](../provenance/imports_bench.json)
- [Runtime imports](../provenance/imports_runtime.json)
- [Task inputs and initial public projections](../provenance/imports_tasks.json)
- [Additional frozen Bench/Runtime profiles and L0 identity delta](../provenance/imports_profiles.json)
- [A/M/V resource bindings](../provenance/imports_amv.json)
- [Physical/L2 resource and public-input bindings](../provenance/imports_physical_l2.json)
- [Original source preservation checks](../provenance/source_preservation.json)

This is a new Git repository with no imported internal history or remote.
There are no source-tree symlinks. Registered source manifests retain upstream
revisions separately from packaging adaptations and pin every imported file.
The L0 roots remain under `_vendor/bench` and `_vendor/runtime`; additional
snapshots are under `_vendor/revisions/<kind>/<full-revision>`. Shared revisions
are stored once, without sharing mutable run state.

`profiles.source_roots`, `source_revisions` and `verify_sources` are pure public
source APIs. Public model/scoring/replay calls use isolated workers and leave
the caller's `sys.path` unchanged. Inside a worker, activation rejects foreign
origins for every owned Bench, `lab`, `behavior_hl` and runtime-script module;
namespace package locations are checked too. A private worker cannot switch
profiles, while one public caller may operate on multiple profiles through
separate workers. `bootstrap.activate()` is now a compatibility path lookup,
not permission to import bare legacy modules in a caller.

A registered bundle's manifest must exist and match its registry digest and
files. Missing manifests never fall back to the candidate's parent Git HEAD.
External official source identity is still checked with Git, but its reported
top-level must exactly equal the given source root. Model run records retain the
selected revisions, source-verification receipt and prompt/implementation hashes.
Asset imports verify raw bytes before generating private relocation copies; their
receipts retain original and derived identities. A derived path does not relabel
an original run or prepared state as having used new code.

Portability changes make paths, credentials, dependency locations, owned runtime
directories and result collection explicit. Core executors, Marker transitions,
canonical predicate kernels and process verifiers remain upstream code. See
[behavior equivalence](BEHAVIOR_EQUIVALENCE.md) for independent fixed-input checks,
[the runtime](RUNTIME.md) for deployment adaptations, and [source limitations](../reports/SOURCE_LIMITATIONS.md)
for the original L0 discrepancies. L2's separate Common watchdog and its
limits are documented in [Runtime](RUNTIME.md). Historical operational scripts are
not public entry points, even when preserved as dependencies of runtime modules.

The source manifest is evidence for this local candidate, not permission to
redistribute the code or assets. See [license status](../LICENSE_STATUS.md) and
[distribution boundaries](DISTRIBUTION.md).
