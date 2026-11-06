# RoboWeave

RoboWeave evaluates embodied agents through frozen tasks, Action20, public visual
observations and canonical goal/process verification. Version `0.1.0rc5` provides
one installable package and command-line workflow for **257 task configurations**:
L0 (100), L1 A/M/V/P (30 each), and L2 collection (37).

**This is a privately shared review candidate. Public release and Live validation
remain pending.** Private evaluation inputs and licensed simulator assets are
obtained separately. Project licensing and redistribution remain unresolved;
see [license status](LICENSE_STATUS.md) and [distribution boundaries](docs/DISTRIBUTION.md).

The prior rc4 repair pass added source-only loading, strict task/result binding,
concurrent asset-import protection and credential-safe error diagnostics. The
review and CPU acceptance evidence are retained in [the rc4 report](reports/rc4/RESULT.md).

Version rc5 adopts the RoboWeave package, Python namespace and CLI. The
[rename and compatibility notes](docs/NAMING.md) describe legacy input support;
[current verification](reports/rc5/RESULT.md) is separate from historical receipts.

The repository connects the existing implementations through explicit modules:

```text
CLI / Python APIs
  ├─ tasks + profiles       public definitions and frozen source selection
  ├─ assets + adapters      checked private inputs and family-specific preparation
  └─ engine                 one isolated worker per operation/profile
       ├─ original Bench    model adapter, observations, Action20 and episode loop
       └─ original Runtime  execution, canonical goals and process/history audit
             └─ results    trajectories, scores, evidence hashes and run identity
```

The public process does not install legacy modules into its `sys.path`. Each
worker selects an exact Bench/Runtime pair from **13 source bundles**; shared
revisions are stored once. The source trees retain their original layout and
algorithms. [Architecture](docs/ARCHITECTURE.md) explains the interfaces and
ownership boundaries.

| Capability | Current scope |
| --- | --- |
| Task execution integration | Eight profiles: L0, A, M, V, three P subpackages, and L2 collection. Family-specific preparation, native admission, model episodes and result archival are connected. |
| Source identity | Exact upstream commits, per-file hashes, candidate adaptations and loaded-module origin checks. No original checkout or mutable source symlink is required. |
| CPU workflow | Installation checks, task enumeration, configuration validation, synthetic protocol replay and supplied-record scoring. |
| Process/history records | v1 snapshots and v2 per-event truth records call the original process auditor. Missing required evidence fails closed. |
| Historical L2 proposals | A separate 36-proposal authoring audit; not executable task entries. |
| Real-world evidence | No simulator, GPU evaluation, provider request or independent simulator environment reproduction was performed in this preparation. |

`execution_supported` means the code path and frozen input adapter are present.
It does not establish external-resource availability, successful resets, Live
completion or comparability with a historical model campaign. In particular,
the candidate's common Bench harness does not reproduce every historical native
model runner or scheduler. See the [support matrix](docs/SUPPORT_MATRIX.md).

Clone the private repository with an authorized GitHub account:

```bash
git clone https://github.com/ShawnLee0910/roboweave.git
cd roboweave
```

From the repository root, the CPU example uses an independent environment:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install '.[test]'
roboweave doctor
roboweave doctor --profile l2-collection37
roboweave config validate --config configs/replay.json
roboweave tasks list
roboweave tasks list --family L1-V
roboweave tasks list --family L2
roboweave run --config configs/replay.json --mode replay --output outputs/quickstart
roboweave results validate outputs/quickstart
roboweave results summarize outputs/quickstart
roboweave score examples/history-score-record.json --process-output outputs/history-example
```

Use fresh output directories. The replay uses synthetic images, scripted replies
and the original conversation loop; the history example supplies predicate
truth. Neither evaluates a real catalog task or proves physics or task success.

For an independently provisioned real run, select matching `profile`,
`run.family` and `run.task_id` values in a private copy of
`configs/live.example.json`, import that profile's authorized inputs, validate
the configuration and assets, and run `--mode live --dry-run` before launching.
The template intentionally needs real paths, device ownership and explicit
license acceptance. [Quickstart](docs/QUICKSTART.md), [Installation](docs/INSTALLATION.md)
and [Assets](docs/ASSETS.md) describe the complete sequence.

| Topic | Documentation |
| --- | --- |
| Task identity, instructions, budgets and version coverage | [Contract](docs/BENCHMARK_CONTRACT.md), [tasks](docs/TASKS.md), [support matrix](docs/SUPPORT_MATRIX.md) |
| Actions, coordinates and Marker stages | [Action20](docs/ACTION20.md) |
| Initial state, launch ownership and budget enforcement | [Runtime](docs/RUNTIME.md) |
| Goals, process/history constraints and outcomes | [Scoring](docs/SCORING.md) |
| Provider setup, public inputs and model history | [Models](docs/MODELS.md) |
| Trajectories, images and result interpretation | [Trajectory format](docs/TRAJECTORIES.md) |
| Actual CPU evidence and remaining checks | [Validation](docs/VALIDATION.md), [behavior equivalence](docs/BEHAVIOR_EQUIVALENCE.md) |
| Full source identities and adaptations | [Sources](docs/SOURCES.md), [bundle registry](src/roboweave/data/source_bundles.json), [profile imports](provenance/imports_profiles.json) |
| External dependencies and remaining acceptance | [Distribution](docs/DISTRIBUTION.md), [third-party notices](THIRD_PARTY_NOTICES.md), [Live plan](docs/LIVE_VALIDATION.md) |

Historical rc1/rc2 reports remain as dated evidence, including the
[rc2 fixes](docs/FIXES_RC2.md). Their counts and L0-only coverage are not evidence
for the additional rc3 profiles. Current validation reports separate source and
CPU checks from outstanding environment, private-resource and Live acceptance.
