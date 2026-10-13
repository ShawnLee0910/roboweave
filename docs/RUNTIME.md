# Runtime and one-scene ownership

The public runtime is a thin launcher around the frozen implementation. It does
not replace Action20, Marker selection, preparation, reset, geometry, collision
checks, or episode verification.

```text
esi-benchmark run
  assets.materialize_task_inputs      exact, hashed, single-task input copy
  runtime.runtime_plan                CPU file/configuration checks
  runtime.launch_runtime              one new process and one private workspace
    lab.package_scene_server          exact handoff + prepared-state restore
      launch_resident_scene_oracle_batch(model_callback=serve)
        make_scene_server_callback    existing Unix-socket protocol
  models.run_episode                  existing provider + episode loop
  original scene final_score.json     private evaluator result
```

The executable profile is `l0-english-v2`: Bench
`5b8d2b37fd7d97b161e88de4d438ec789c78f9b4` and Runtime
`9b992b6797330a836e9533e5409627bf63f7d94b`. Historical evidence may name other
revisions. A prepared-state source revision is not relabelled as the execution
revision. See the task support matrix and provenance files for individual task
readiness and source distinctions.

## Layout and identity

`_vendor/runtime` preserves `src/behavior_hl`, required `configs`, `scripts`, and
`pyproject.toml`. The installed package supplies these files; no symlink to an
internal working tree is required. `bootstrap.activate()` adds only the installed
vendor roots. `SOURCE_IDENTITY.json` records the upstream revision, every imported
runtime file's candidate hash, and path/deployment adaptations. Public preflight
checks those hashes. Project identity uses the same manifest, and execution
identity does not mistake an enclosing candidate Git commit for the upstream
runtime revision.

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
start a process, call a model, query GPUs, bind a port, or create files. It checks
source hashes, dependency file identities, input availability, version markers,
and output isolation. A plan is not a Live pass. Keep the workspace path short
enough for the Unix socket address limit; preflight rejects an overlong address
before loading a scene. CLI `--dry-run` does not materialize private inputs and
therefore reports resource/configuration checks, not native task admission.

Actual launch exclusively creates a new workspace. It creates independent AppData,
XDG and temporary directories there, removes inherited simulator tuning overrides,
internal Python paths, proxies and model credentials from the simulator child,
and starts a new process session. Before scene construction, the child verifies
Python/dependency origins, physical GPU UUID, and local port availability. The
underlying simulator deliberately uses the physical GPU index without
`CUDA_VISIBLE_DEVICES` masking to preserve Vulkan/CUDA alignment.

The child uses the original `package_scene_server` path. It validates the handoff,
prepared bundle hash, seed, camera profile, robot/joint identity, exact source
instruction, and zero pre-model Oracle/physics actions. It restores the recorded
initial state through the original checks. It does not regenerate an initial
state, request a pose-search fallback, or run an Oracle for the model.

A ready record is accepted only from the new workspace, with its exact Unix socket,
contained run directory and child PID. The parent does not connect to another
resident. It waits only until the configured scene-load deadline and always stops
its own process group afterward. Failure preserves the launch log and independent
output evidence; no automatic second scene load is attempted.

## CPU evidence and remaining Live acceptance

CPU tests cover installed resources, source identity, strict parameter and Marker
contracts, geometric execution with small test objects, projected/official scoring,
process/history audits, deadlines, and supplied-record replay. Original integration
tests requiring two undistributed historical task packages are explicitly skipped.
No offline test certifies simulator rendering, an actual scene reset, GPU execution,
a model request, or task success.

Before advertising Live reproducibility, use separate owned hardware and a fresh
workspace to validate installation/import origins; load one admitted fixed-pool
and one admitted registered L0 task; compare restored state/camera attestation;
exercise each Marker stage and error feedback; run an explicitly configured model;
finish and archive its complete conversation, images, native score and source
receipts. Check success, failure, budget exhaustion and process-gated termination
where those task families are declared executable. Keep the one-load/no-retry
budget and exact task contracts. This candidate preparation performed none of
those Live steps.
