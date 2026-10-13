# ESI Benchmark — local release candidate

ESI evaluates embodied agents through frozen tasks, the Action20 interface,
public visual observations and canonical task/process verification. This
candidate provides an installable package, one command-line workflow, source
provenance, CPU examples and a path to independently validate real simulator
execution.

**This is a local review candidate, not an authorized public release.** ESI
source licensing and the distribution of private task-preparation inputs remain
owner decisions. Simulation assets are obtained separately under their own terms.
See [license status](LICENSE_STATUS.md) and
[distribution boundaries](docs/DISTRIBUTION.md).

Version `0.1.0rc2` fixes the eight issues identified in the second review:
credential redirects, incomplete process evidence, private input permissions,
concurrent import cleanup, mixed source modules, hidden archival failures,
malformed result isolation and incomplete source distributions. See
[the fix record](docs/FIXES_RC2.md) and [validation evidence](docs/VALIDATION.md).

The architecture keeps the existing executor and evaluator: a thin public package
coordinates the frozen task catalog, model adapter, Bench episode loop, Runtime
and trajectory artifacts. [Architecture](docs/ARCHITECTURE.md) explains the module
boundaries and execution flow.

| Capability | Candidate scope |
| --- | --- |
| CPU installation, configuration, catalog, Action20 parsing, observations, episode protocol and synthetic replay | Implemented; see the [validation record](docs/VALIDATION.md) for actual checks and limitations. |
| Task catalog and Live integration | L0 English-v2 has 100 public definitions. Every Live run must pass the frozen private-input checks and supply its exact preparation resources and external dependencies. |
| Canonical scoring | Existing final-goal and process implementations are retained. Offline checks do not prove simulator behavior. |
| Other families | L1 A/M/V and P are catalog/audit entries; L2 proposal metadata is audit-only. These are not routed through the L0 runtime. |
| Actual simulator/model execution during this preparation | Not performed. Follow the [independent Live acceptance plan](docs/LIVE_VALIDATION.md). |

The L0 profile freezes Bench
`5b8d2b37fd7d97b161e88de4d438ec789c78f9b4` and Action/Simulator/Oracle
`9b992b6797330a836e9533e5409627bf63f7d94b`. These are task-specific bindings,
not a claim that the source directories' latest commits ran every historical
experiment. The [support matrix](docs/SUPPORT_MATRIX.md),
[source snapshot](provenance/source_roots.json) and per-file `provenance/imports_*.json`
records preserve the distinctions, including imported uncommitted task content.
The [per-task CPU resource report](reports/TASK_ASSET_CPU_VALIDATION.md) records
materialization readiness separately from Live success. A 100-row catalog is not
evidence that 100 scenes or episodes passed.

From this repository root, use Python 3.11 or newer for CPU checks:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install '.[test]'
esi-benchmark doctor
esi-benchmark config validate --config configs/replay.json
esi-benchmark tasks list --family L0
esi-benchmark run --config configs/replay.json --mode replay --output outputs/quickstart
esi-benchmark results validate outputs/quickstart
esi-benchmark results summarize outputs/quickstart
```

Replay is a small, synthetic Marker transaction through the frozen conversation
loop. It needs no GPU, simulator assets, model account or network model request.
Its artifacts demonstrate the software workflow; they are not official benchmark
scores. Use a new output directory for each attempt.

Continue with [Quickstart](docs/QUICKSTART.md) for the complete command sequence,
[installation and external dependencies](docs/INSTALLATION.md) for the separate
Live environment, and [asset preparation](docs/ASSETS.md) for checked resource
import.

| Topic | Documentation |
| --- | --- |
| Task identity, instructions, budgets and preparation | [Contract](docs/BENCHMARK_CONTRACT.md), [tasks](docs/TASKS.md), [support matrix](docs/SUPPORT_MATRIX.md) |
| Action names, parameters and Marker stages | [Action20](docs/ACTION20.md) |
| Runtime, initial state and owned scene lifecycle | [Runtime](docs/RUNTIME.md) |
| Final goals, process constraints and outcomes | [Scoring](docs/SCORING.md) |
| Provider setup, public input and history | [Models](docs/MODELS.md) |
| Trajectories, images, provenance and result interpretation | [Trajectory format](docs/TRAJECTORIES.md) |
| CPU evidence and source/candidate comparisons | [Validation](docs/VALIDATION.md), [behavior equivalence](docs/BEHAVIOR_EQUIVALENCE.md) |
| Source identity and independent review | [Sources](docs/SOURCES.md), [review findings and fixes](docs/REVIEW.md) |
| Release conditions and remaining acceptance | [Distribution](docs/DISTRIBUTION.md), [third-party notices](THIRD_PARTY_NOTICES.md), [Live plan](docs/LIVE_VALIDATION.md) |

The frozen task records declare `wall_clock_seconds: 3600`, but the original Bench
episode loop does not enforce that field. The candidate retains the original
action/call budgets and termination behavior; strict historical outer-controller
wall-clock reproduction is unsupported. See [source limitations](reports/SOURCE_LIMITATIONS.md).

No original environment, resident, queue or historical result is needed for the
CPU workflow. No original Git history, credentials, restricted assets or bulk
trajectories are included. Live also requires separately authorized NVIDIA MDL
modules, beyond pip packages and the official dataset. Future Live claims must report the exact supported
profile, environment and evidence rather than treating a synthetic replay as an
end-to-end benchmark run.
