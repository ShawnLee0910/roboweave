# Task definitions and release identity

The candidate contains a public evaluation catalog of **220 tasks**: L0 English-v2 (100) and L1 A, M, V and P (30 each). The executable integration is limited to the frozen L0 profile and requires separately authorized preparation inputs. L1 entries support inspection and version auditing; their runtimes are not installed. L2 has 36 research proposals with unequal readiness, and contributes no executable catalog entries.

```bash
esi-benchmark tasks list --family L0
esi-benchmark tasks show am200v1_a001_b2_control --family L0
```

Python provides `list_tasks(family=None)`, `get_task(task_id, family=None)` and `profile_for(task)`. A lookup that is ambiguous across families requires `family`; it never silently chooses a version. L0's public `task_id` is its frozen `runtime_instruction_id`. Other families retain their public `instruction_id`, which matters for V branches that share an underlying runtime task. Both original identity fields remain available.

## What a public record contains

A record contains only identity, family, approved English instruction, instruction SHA256, scene model, action mode, exact version profile, source file identity/hash and declared budget. Budget fields are `maximum_action_decisions`, `maximum_model_calls`, `repeats_per_model` when present, and `wall_clock_seconds`.

The catalog excludes preparation state, goals, verifier scopes, object poses, Oracle evidence, human trajectories and private handoff contents. The model receives its approved instruction and the public observation protocol described in the model and Action20 documentation; passing the entire private task input to a model is unsupported.

Catalog metadata does not define a new train/dev/hidden-test partition. No such partition was found frozen for these FINAL releases. An underlying simulator task's `split` field is retained internally for the original runtime lookup and does not create a Benchmark training split.

## L0 English-v2

The source is `benchmark/L0-100-FINAL/revisions/english_v2/`. Its `tasks.json` SHA256 is `5daca755447954c2b1bd2c9a0cafb9cf5e92c78f8cf470666ffee1531dfbd969`. It has the same 100 approved identities as the previous FINAL release, with 14 revised English instructions and 86 retained instructions. The candidate uses the paired English-v2 inputs and does not relabel older results.

The runtime is `9b992b6797330a836e9533e5409627bf63f7d94b`; Bench is `5b8d2b37fd7d97b161e88de4d438ec789c78f9b4`. Source `tasks.json` and the FINAL release metadata were untracked in the source repository: the repository HEAD is context, while file SHA256 values freeze their actual contents. See [the source import records](../provenance/imports_tasks.json).

L0 contains **62 registered-task routes and 38 TaskSpec routes**. The original package, source, handoff, preparation index and prepared bundle determine the route, initial state, task identity and verifier input. The integration restores the frozen preparation state. It does not regenerate a scene or replace a historical package with a newer compiler's output.

All 100 share these frozen conditions: Symbolic action mode, full scene loading, `r1pro_three_rgb_head_eval`, overhead view off, zero extra physics steps per action, and automatic interaction approach disabled. Seeds are read from the frozen private inputs. Per-task action and model-call budgets are preserved; the first task, for example, allows 33 action decisions and 330 model calls.

**Wall-clock limitation:** the frozen task inputs declare 3,600 seconds, but the locked model client does not consume the operator's `ESI_MODEL_WALL_SECONDS` variable. The original controller separately observes the model process against a default four-hour timeout, excluding scene loading. The candidate retains the model client's episode behavior and records the declared budget without inventing a new 3,600-second termination rule. Strict reproduction of the old outer controller's timeout is not supported. Action-decision and model-call limits remain enforced. See [source limitations](../reports/SOURCE_LIMITATIONS.md).

## Goals, process and termination

The official goal verifier and any declared process/history contract come from the selected private task record. A static goal result cannot satisfy an independently required process/history contract. Import and relocation retain those records field-for-field; they do not lower thresholds, discard requirements or replace official verification with a CPU projection.

The task catalog itself is not a verifier. Scoring and result interpretation are documented separately. Historical Q1 summary overrides for particular L1 releases are not adopted as the default scoring policy.

## Resource preparation and reproducibility

[ASSETS.md](ASSETS.md) explains the private import boundary. The original dependency manifest lists 715 runtime references (97,983,298 bytes including aliases), representing 713 unique files after normalizing two `..` aliases. The candidate additionally records 55 files required by frozen native-loader implicit lookups, which the original manifest omitted. With the exact top-level task input, the importer verifies 769 unique files. Historical evidence entries are excluded.

For each episode, a fresh private input directory selects the exact task from aggregate package, TaskSpecs, registry, handoff and preparation-index files. Path references and dependent file hashes are updated there. Every original-to-derived SHA256 and path change is recorded. The selected task's instruction, identity, initial state, goal and process records are preserved. Imported originals are never rewritten.

CPU checks on these inputs are not simulation evidence. See [the validation report](../reports/TASK_ASSET_CPU_VALIDATION.md) and [the support matrix](SUPPORT_MATRIX.md) for the exact evidence boundary.
