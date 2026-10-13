# Frozen-source limitations and integration findings

These findings distinguish source behavior from changes made to package it. No original source, task or process was modified. This report concerns the explicitly selected L0 English-v2 profile, not every historical campaign.

## Declared wall-clock budget is not an episode deadline

The frozen task input declares `wall_clock_seconds: 3600` for each of the 100 tasks. The frozen operator reads the task's budget at `scene_backfill.py:1993`, passes action-decision and model-call limits at lines 1994–2005, and exports `ESI_MODEL_WALL_SECONDS` at lines 2070–2072. It does not pass that value as the native model client's episode deadline.

The selected Bench revision has no tracked Python consumer of `ESI_MODEL_WALL_SECONDS`; the complete `lab/model_client.py` likewise has no episode wall-clock deadline. It does enforce per-request transport timeouts and the action/model-call accounting. The independent operator declares `MODEL_WALL_TIMEOUT_SECONDS = 4 * 60 * 60` at line 102. Its `_wall_timeout_evidence` function at lines 2236–2267 checks the owned model process's elapsed time and a safe interruption phase. Its caller at lines 3533–3550 passes the controller timeout. That timer belongs to the model process, not to scene loading or reset.

The candidate preserves the original model client's episode behavior. `run_record.json` retains the complete declared budget and labels the wall-clock enforcement as `not_implemented_in_frozen_bench`. It does not add a new 3,600-second stopping rule. Reproduction of the original outer controller's four-hour interruption behavior is outside the candidate's supported single-task integration.

Evidence identities:

| Source | Revision / identity | SHA256 |
|---|---|---|
| `/liyuanshuai/esi_operators/l0-reuse-root-20260923/scene_backfill.py` | Frozen sidecar identified by file hash; not assumed to be a clean Bench commit | `9d50f1a270961f8baf3e5cfad86adb768b612af5803354cc50b4325d9adf070a` |
| `/liyuanshuai/bench-l0-reuse-root-20260923/lab/model_client.py` | Bench `5b8d2b37fd7d97b161e88de4d438ec789c78f9b4` | `2224523570db9ae882b18233210a401aea0cd0dcadf493ad079bc13752611c48` |
| `benchmark/L0-100-FINAL/revisions/english_v2/tasks.json` | Untracked frozen task file under Benchmark HEAD `1665ac1b0d1863cc642165deceb020eedd0db17e` | `5daca755447954c2b1bd2c9a0cafb9cf5e92c78f8cf470666ffee1531dfbd969` |

The paths above are read-only provenance, not runtime dependencies of the candidate.

## History depends on the frozen caller

The same operator explicitly passes `--history-frames 5` for model calls at line 2002 and for other agent launchers at line 2024. The Bench function's alternative/default history capacity is not evidence that a historical L0 run used it. Reproduction requires the frozen five-frame setting; an explicitly selected alternative must be recorded as a different observation-history condition.

## The original dependency manifest missed implicit runtime files

The original dependency manifest passes its own 715 recorded runtime hashes. Two aliases normalize to 713 unique files. This does not prove a closed native-loader dependency set.

At Bench revision `5b8d2b37fd7d97b161e88de4d438ec789c78f9b4`, `lab/instruction_handoff.py` selects two kinds of additional files:

- Seven `perception114` tasks use the optional `handoff.parent / esi_artifacts/instruction88_inputs_20260907/perception_frozen/package.json` path. That file exists in their actual frozen source layouts and points to a separately frozen preparation package. Loading only the original package would incorrectly produce `PENDING_LIVE_GEOMETRY`; changing that marker to `FROZEN` would bypass the real source contract. The candidate imports the original fallback package and its exact task/registration references and recreates the same relative lookup in the private per-run directory.
- Four nonlegacy tasks in the `fixed_l0_pool` package derive a registration-catalog path from `runtime.expansion_root`, scene model and `benchmark/l0/expansion_v1/official_compatible`. Their scene catalogs and selected BDDL/template/TRO/contract files were also absent from the original manifest.

The candidate's manifest adds **55 exact files, 2,246,417 bytes**, for these 11 tasks. Every reference hash was checked against the real frozen file. Only metadata is redistributed. The added closure does not create a new task, change a goal or claim a new Live pass. The lookup source SHA256 is `745e80bebbefb5e56ac806c7b9cb2d52a61895a984c73a5be73ab18536591dd5`; it is recorded in `supplemental_closure` with the pinned Bench revision.

## Other evidence boundaries

The original L0 runtime lock records `acceptance.launch_gate = BLOCKED_UNRESOLVED_BASELINE_CPU_FAILURES` and four Action surface-evidence failures. In the candidate's independent CPU environment, the unchanged `test_on_top_provider_preserves_surface_evidence_through_publication` assertions pass for all four formerly failing parameter combinations: `hidden_center-False`, `hidden_center-True`, `hidden_coarse_hits-False` and `hidden_coarse_hits-True`. The current evidence is [pytest.xml](pytest.xml). This does not establish the root cause in the old environment, change the immutable source lock or provide Live acceptance.

The source preparation entry point, `scripts/prepare_l0_100_final.py`, reads the lock at line 67, validates source identity/file hashes at lines 68–82, and reports the gate at lines 96 and 113. It does not branch on that gate; line 114 invokes the dispatcher's compile-only `prepare`. Neither the frozen `scene_backfill.py` nor `bench_route_entry.py` consumes `launch_gate` or `runtime_lock`. The release README separately requires the blockers to be resolved and deployment, provider, ownership and experiment authorization checks before a formal start. Thus the historical label is an explicit release/review constraint, not a previously enforced model-client Boolean that this packaging may silently redefine. A new candidate validation receipt must keep its own identity; it does not rewrite the historical gate.

The source lock also explicitly lacks all-100 prepared-reset and new runtime/Bench combination Live evidence. Current CPU input checks cannot resolve those physical-evidence gaps. No simulator, model request or existing worker was used for this preparation.

The FINAL catalog has no demonstrated, frozen train/dev/hidden-test partition. Underlying simulator `split` values are preserved for lookup, not promoted into a new dataset split. Private preparation and simulator-asset redistribution rights remain unconfirmed, and the candidate does not grant them.
