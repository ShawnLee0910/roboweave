# Task definitions and release identity

The public evaluation catalog has **257 configurations**: L0 English-v2 (100),
L1 A/M/V/P (30 each), and L2 collection (37). Eight profiles connect those
definitions to exact source bundles and private-input adapters. The separate
36-proposal L2 authoring audit is not part of the executable catalog. Every
profile remains Live-unverified; see [Support matrix](SUPPORT_MATRIX.md).

```bash
roboweave tasks list
roboweave tasks list --family L1-A
roboweave tasks list --family L1-M
roboweave tasks list --family L1-V
roboweave tasks list --family L1-P
roboweave tasks list --family L2
roboweave tasks show am200v1_a001_b2_control --family L0
roboweave tasks show l2r_am_02_20260916_w0_challenge --family L2
```

Python provides `list_tasks(family=None)`, `get_task(task_id, family=None)` and
`profile_for(task)`. Ambiguous IDs require an explicit family. A configuration's
`profile`, `run.family` and `run.task_id` must agree. For example,
`l1-p-inside5` is not interchangeable with `l1-p-ontop5` even though both have
family `L1-P` and share a Bench revision.

## Public and private identity

Public records contain approved English instructions and their SHA256, task and
runtime IDs, family, scene model, action mode, exact profile, source metadata and
declared budgets. Original public and runtime identity fields are retained;
V configurations may share a runtime task while having distinct collection
identities. Seeds, preparation state, goals, verifier scopes, object poses,
Oracle evidence and private handoffs belong to evaluator inputs, not public
model observations.

The catalog defines no new train/dev/hidden-test partition. An underlying
simulator `split` is retained for its original lookup; it is not a newly assigned
Benchmark data split. Importing a release does not convert historical trajectories
into training data or authorize publication of hidden evaluation inputs.

## Frozen family routes

L0 comes from `benchmark/L0-100-FINAL/revisions/english_v2/`. The source
`tasks.json` hash is
`5daca755447954c2b1bd2c9a0cafb9cf5e92c78f8cf470666ffee1531dfbd969`.
Its 100 approved IDs retain 86 earlier instructions and 14 revised English
instructions. The task files were untracked in the source repository; file
hashes freeze those contents independently of repository HEAD. Its 62 registered
routes and 38 TaskSpec routes retain their original package and preparation.

A/M/V use `benchmark/L1-AMV90-RELEASE/20260924_v1/RELEASE.json`, with distinct
Bench bindings and the M public-v2 instruction set. A/M restore their frozen
prepared state and retain declared process/history programs. V additionally
binds the exact collection key, D1 branch, intervention package and compatible
Runtime; the adapter checks inherited prepared state and native V lifecycle.

P uses the `inside`, `ontop` and `additions` subpackages under
`benchmark/L1-P-FINAL/model_batch_20260924`. Their counts are 5, 5 and 20,
respectively. Each subpackage retains its distinct Physical Runtime and original
model-start/qualification contract.

L2 collection is the frozen 37-entry collection, with 23 Symbolic entries and
14 Physical entries. It includes an approved model-facing English overlay; the
original TaskSpec wording remains in private source evidence. The 23 prepared
routes and 14 native `physical_collection` routes are selected per task.
Physical collection reconstructs and verifies the declared world; its historical
camera/preparation bundle is evidence, not a symbolic state-restoration recipe.
The exact task list defines support; it does not enable arbitrary combinations
from the separate L2 proposal audit.

## Budgets, goals and termination

`maximum_action_decisions`, `maximum_model_calls`, `repeats_per_model` when
present, and `wall_clock_seconds` retain their frozen values. There is no global
150-decision replacement: non-L2 Bench versions have an implementation ceiling
of 150; the L2 Bench ceiling is 3,000, with each task's smaller declared budget
still governing the episode. The first L0 task allows 33 decisions and 330 model
calls; L2 declarations use a 7,200-second wall-clock budget rather than the
L0/L1 declaration of 3,600 seconds. [Runtime](RUNTIME.md) distinguishes declared
budgets from implemented timers and records their enforcement.

Final goals and declared process/history constraints are separate gates. Static
success cannot bypass no-manipulation rules, ordering, state protection, dynamic
references or history bindings. V protocol completion and Physical qualifications
remain family-specific. A canonical Q of 1 is not universal task success.
The candidate calls the selected original evaluator; it does not lower thresholds
or adopt historical summary overrides as a new scoring policy.

## Inputs and evidence

Each execution profile has its own metadata-only resource manifest. Authorized
private files are hashed on import, copied into an independent asset directory,
then selected and relocated into a fresh private per-run directory. Path and
dependent-hash changes are recorded; instruction, initial state, goal, process,
seed, budget and version meaning are preserved. See [Assets](ASSETS.md).

CPU parsing, materialization and native metadata admission do not establish
actual resets, rendering, action execution or goal completion. The candidate's
common model harness also does not claim to reproduce historical native model
experiments. [Validation](VALIDATION.md) identifies current checks separately
from historical L0 reports and outstanding Live acceptance.
