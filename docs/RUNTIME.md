# Runtime and one-scene ownership

The public runtime is a thin launcher around the frozen implementation. It does
not replace Action20, Marker selection, preparation, reset, geometry, collision
checks, or episode verification.

```text
roboweave run
  assets.materialize_task_inputs      exact, hashed, single-task input copy
  runtime.runtime_plan                CPU file/configuration checks
  runtime.launch_runtime              one new process and one private workspace
    lab.package_scene_server          prepared / V / Physical native route
      launch_resident_scene_oracle_batch(model_callback=serve)
        make_scene_server_callback    existing Unix-socket protocol
  models.run_episode                  existing provider + episode loop
  original scene final_score.json     private evaluator result
```

Eight profiles have execution integration for the 257 public configurations.
`l0-english-v2` remains the default; A/M/V, the three P subpackages and L2 collection
each select their own frozen Bench/Runtime pair. [Support matrix](SUPPORT_MATRIX.md)
records the full revisions. Historical preparation evidence may name another
revision and is never relabelled as a new candidate execution. The separate
L2 proposal authoring snapshot is not executable.

## Layout and identity

All seven Runtime bundles preserve `src/behavior_hl`, required `configs`,
`scripts` and `pyproject.toml`; all six Bench bundles preserve their original
module layouts. The L0 roots remain `_vendor/bench` and `_vendor/runtime`; other
revisions are under `_vendor/revisions/<kind>/<full-revision>`. No original
working tree or mutable symlink is needed.

`profiles.source_roots(profile)` is a pure lookup. Public APIs keep the caller's
module search path unchanged; only a private, single-profile worker activates
legacy imports. Registered `SOURCE_IDENTITY.json` manifests pin upstream revisions,
every bundled file hash and candidate adaptations. Manifest absence, modified
sources and foreign loaded-module origins are rejected. An enclosing candidate
Git commit cannot stand in for Bench or Runtime provenance. External official
sources keep their independent exact-root Git identity checks.

The original official and projected verifier files are unchanged. Adaptations
are limited to explicit paths, optional host compatibility preload, explicit EULA
acceptance, installed-source identity, and placement of licensed material modules.
The original Action20 and scoring algorithms remain in the vendor tree.

The default public route does not run legacy canaries, coordinators, batch
schedulers, SSH, or the legacy `scene_server.main` lane-manifest entry point.
Those Python modules are preserved because the production runtime shares imports
with them; they are not separately supported public commands.

## External environment

Prepare a **new** Python 3.11 environment and the exact external checkout described
in the installation guide. The frozen dependency is BEHAVIOR commit
`802356a83d210a42bb6bfde93544bda4e33124e1`, OmniGibson 3.9.0, and Isaac Sim 5.1.0.
The public launcher requires a clean independent external checkout and checks that
OmniGibson and BDDL resolve inside it. Official assets remain externally acquired:
`behavior-1k-assets/VERSION` must be `3.9.0`, robot assets `3.8.2`, with
`2026-challenge-task-instances` available. Version markers are an installation
check, not proof that every geometry asset or task can load.

Set these paths in the JSON configuration:

| Field | Required content |
| --- | --- |
| `runtime.python` | The new environment's `bin/python`; the launcher preserves this path, including a normal venv symlink |
| `runtime.overlay_root` | The new environment prefix; must agree with the running child interpreter |
| `runtime.behavior_root` | The pinned external BEHAVIOR Git checkout |
| `runtime.data_root` | Externally licensed assets, with the version files above |
| `runtime.toolchain_root` | Matching CUDA 12.8 toolkit prefix containing `bin/nvcc` |
| `runtime.mdl_root` | Authorized external VRay MDL directory described below |
| `runtime.vulkan_icd` | Optional explicit NVIDIA Vulkan ICD JSON; otherwise the two standard `/usr/share` and `/etc` NVIDIA locations are checked |
| `runtime.host_dlsym_shim` | Optional, user-supplied host compatibility shim; omitted on the standard external environment |
| `runtime.workspace` | A fresh child of `run.output_root`, separate from all source, environment, and asset roots |

A configured host shim must exist and be preloaded before Python starts. No
internal shim binary is bundled. Its omission is an explicit deployment
adaptation for standard environments, still pending Live qualification. The
original GPU mapping and source/root checks remain active.

The material adapter imports two NVIDIA modules. Their headers reserve NVIDIA's
rights and prohibit redistribution without express permission, so this candidate
**does not include them**. Obtain them through your authorized simulator/material
installation or applicable license, and set `runtime.mdl_root` to their directory:

| File | Required SHA256 |
| --- | --- |
| `vray_maps.mdl` | `279d9e8ded03c2db928033c843b40a04b104f3fd6fc7ccd284aadc6459a2aa1c` |
| `vray_materials.mdl` | `eac4af88c7e39cdba091bf2e76c20adc3f9c9cd65d6d044479065537f70de696` |

The runner-owned wrapper `behavior_hl_vray_roughness.mdl` retains its attribution.
Only an explicit Live run copies the supplied licensed modules and the wrapper
to that run's private `workspace/materials`. It does not copy them into the
installed package. No asset download or license acceptance is automated.

## Launch and lifecycle

Configuration requires a physical GPU index **and** its explicitly owned UUID,
an explicit unprivileged server port, `max_scene_loads: 1`, and explicit acceptance
of the external simulator EULA after reading its terms. `runtime_plan()` does not
start a process, call a model, query GPUs, bind a port, or create files. It first
binds the public task handle, instruction, budget and selected family to the
frozen catalog, then checks source hashes, dependency file identities, input
availability, version markers, and output isolation. Before process creation and
again in the child, the materialization receipt must match the task and profile.
A plan is not a Live pass. Keep the workspace path short
enough for the Unix socket address limit; preflight rejects an overlong address
before loading a scene. CLI `--dry-run` does not materialize private inputs and
therefore reports resource/configuration checks, not native task admission.

Actual launch exclusively creates a new workspace. It creates independent AppData,
XDG and temporary directories there, removes inherited simulator tuning overrides,
internal Python paths, proxies, the explicitly selected model credential and
the supported providers' default credential variables from the simulator child,
and starts a new process session. Before scene construction, the child verifies
Python/dependency origins, physical GPU UUID, and local port availability. The
underlying simulator deliberately uses the physical GPU index without
`CUDA_VISIBLE_DEVICES` masking to preserve Vulkan/CUDA alignment.

The child selects the original `package_scene_server` route for the task. Prepared
routes retain handoff, seed, camera, robot/joint, exact instruction and zero
pre-model Oracle/physics checks before restoring the frozen state. V additionally
uses its exact collection binding, D1 intervention and inherited prepared bundle.
P retains its native physical admission. L2's 14 `physical_collection` entries
instead reconstruct and verify the declared world; their camera/preparation
bundle is historical evidence, not a state-restoration substitute. The route is
frozen per task, with no automatic pose-search or alternative-source fallback.

A ready record is accepted only from the new workspace, with its exact Unix socket,
contained run directory and child PID. The parent does not connect to another
resident. It waits only until the configured scene-load deadline and always stops
its own process group afterward. Failure preserves the launch log and independent
output evidence; no automatic second scene load is attempted.

## Task budgets and L2 interruption

Action-decision and model-call budgets come from the selected task. Provider
request timeout, scene-load timeout and model-worker wall time are different
clocks. L0/L1 retain the source limitation that their declared 3,600-second field
is not consumed by the common episode loop; no new global cutoff is added.

For `l2-collection37` only, the engine implements the historical Common
`model_wall_timeout/v1` safe-point policy. The frozen task threshold is 7,200
seconds. Time starts at **model worker process creation**, measured from Linux
`/proc/<pid>/stat` start ticks and `/proc/uptime`; scene loading is excluded.
The original 14,400-second outer-watchdog bound is retained as policy metadata,
not a second unconditional kill deadline.

After expiry, the observer checks the exact owned run, task/model/experiment
identity and latest provider-call artifacts. It permits one `SIGINT` to the
owned model process group only for `STARTING_PROVIDER_CALL`, `WAITING_PROVIDER`
or `RETRYING_PROVIDER`. It refreshes phase/ownership evidence before signaling
and rejects a process that ignores `SIGINT`. Missing or malformed evidence,
provider preflight, protocol/response processing and observed in-flight actions
do not permit a watchdog interruption. There is no watchdog escalation to
`SIGTERM`/`SIGKILL`; a run can therefore exceed its declared threshold when no
verified safe point is available. Caller cancellation and cleanup of owned
processes remain separate lifecycle operations.

`private/model_wall_timeout.json` records the threshold, elapsed time, observed
phase, process/run identity and request/signal timestamps. The run record uses
`wall_clock_enforcement: owned_common_safe_point_watchdog/model_wall_timeout-v1`.
The original L2 loop retains `pause_on_client_failure=True`, its
`PAUSED_INTERRUPTED` status and available checkpoint; the wrapper does not create
a successful or zero-valued final score for the interruption. The historical
scheduler recorded such a signalled attempt as `FAILED` unless the model had
already reached official success. Result inspection therefore reports the
validated watchdog record as `wall_timeout`, and the run summary counts a
signalled attempt without a score in `wall_timeout_failed_live_runs` and
`success_rate_scored_or_wall_timeout_live`. The scored-only rate is unchanged.

The phase observation and signal are not an atomic simulator lock. Their original
race remains, and Live interruption has not been tested in this preparation.
This policy covers the Common harness, not historical AgentCLI/native runners
or complete scheduler reproduction. See [Architecture](ARCHITECTURE.md) for
source provenance and the worker boundary.

## CPU evidence and remaining Live acceptance

CPU tests cover installed resources, source identity, strict parameter and Marker
contracts, geometric execution with small test objects, projected/official scoring,
process/history audits, deadlines, and supplied-record replay. Original integration
tests requiring two undistributed historical task packages are explicitly skipped.
No offline test certifies simulator rendering, an actual scene reset, GPU execution,
a model request, or task success.

Before advertising Live reproducibility, use separate owned hardware and a fresh
workspace to validate installation/import origins; exercise the supported
family-specific routes (including registered/TaskSpec L0, A/M process, V
intervention, P qualification and L2 prepared/fresh-world routes); compare
state/camera attestations;
exercise each Marker stage and error feedback; run an explicitly configured model;
finish and archive its complete conversation, images, native score and source
receipts. Check success, failure, budget exhaustion and process-gated termination
where those task families are declared executable. Keep the one-load/no-retry
budget and exact task contracts. This candidate preparation performed none of
those Live steps.
