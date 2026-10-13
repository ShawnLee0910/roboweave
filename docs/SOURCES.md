# Frozen sources and candidate changes

Directory dates and the current development HEAD are not execution provenance.
The executable L0 English-v2 profile follows its task `runtime_lock.json`:

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
- [Task inputs and public projections](../provenance/imports_tasks.json)
- [Original source preservation checks](../provenance/source_preservation.json)

This is a new Git repository with no imported internal history or remote. There
are no source-tree symlinks. Runtime `SOURCE_IDENTITY.json` hashes its installed
files and retains the upstream revision separately from packaging adaptations.
Model runs record source hashes, the rendered prompt hash and wrapper hashes.
Asset imports verify raw bytes before generating private relocation copies; their
receipts retain original and derived identities. A derived path does not relabel
an original run or prepared state as having used new code.

Portability changes make paths, credentials, dependency locations, owned runtime
directories and result collection explicit. Core executors, Marker transitions,
canonical predicate kernels and process verifiers remain upstream code. See
[behavior equivalence](BEHAVIOR_EQUIVALENCE.md) for independent fixed-input checks,
[the runtime](RUNTIME.md) for deployment adaptations, and [source limitations](../reports/SOURCE_LIMITATIONS.md)
for discrepancies in the original contracts. Historical operational scripts are
not public entry points, even when preserved as dependencies of runtime modules.

The source manifest is evidence for this local candidate, not permission to
redistribute the code or assets. See [license status](../LICENSE_STATUS.md) and
[distribution boundaries](DISTRIBUTION.md).
