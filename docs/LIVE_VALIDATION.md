# Independent Live validation plan

No GPU scene, resident service or model request was started to prepare this
candidate. CPU tests and synthetic replay establish software behavior only.
Installation of the external simulator stack and actual scene execution remain
separate acceptance items.

Run this plan on a newly allocated machine and an isolated environment, after
obtaining the required asset and task-preparation permissions. Do not use an
existing evaluation resident or queue to validate the candidate.

Before launching, freeze a validation record containing:

- Candidate commit and wheel SHA256, installed package list and the exact frozen
  Bench/Runtime/profile identities.
- OS, Python, driver, CUDA, GPU UUID, simulator and asset versions; record physical
  GPU identity rather than assuming a portable numeric device index.
- Task ID, exact public instruction, initial-state and dependency hashes, split
  label as actually declared, seed, action mode, action budget, history policy,
  retry limits and termination policy.
- A new output root, writable runtime directory and worker-private AppData/cache
  directory; no writable cache sharing or mutable links to an old deployment.
- An explicit scene-load/retry budget and the profile's actual termination policy.
  L0/L1 declare `wall_clock_seconds: 3600` without enforcing it in the original
  model loop. L2 Common uses its historical 7,200-second model-process watchdog:
  interrupt only at a verified safe provider phase, retaining the original
  interrupted result and the watchdog receipt. It excludes scene load and is not
  a hard upper bound during actions or unknown phases. Start with one owned
  worker and one scene load; inspect failure evidence before authorizing a retry.

Verify that the selected GPU is available to this validation and has no active
reservation or quarantine. A free-memory reading is not evidence of ownership.
Check for existing processes before launch; do not stop unknown processes.

Use the [installation instructions](INSTALLATION.md), validate the configuration,
import the exact preparation package, and verify all resource hashes. Select one
of the eight frozen execution profiles in [the support matrix](SUPPORT_MATRIX.md).
An initial L0 smoke task,
`am200v1_a001_b2_control`, is an example candidate; its small dependency closure
still needs official external assets and its private preparation inputs.
Also obtain the required licensed NVIDIA material modules and verify the hashes
in [Installation](INSTALLATION.md). Pip installation and dataset access alone do
not satisfy this dependency.

The acceptance sequence is:

1. **Cold installation:** install the built wheel into a new environment with no
   original RoboWeave paths on `PYTHONPATH`. Check command help, imports, profile identity
   and resource verification before opening the simulator.
2. **Initial state:** load the frozen task once. Preserve the materialized scene
   configuration, seed, initial observations and evaluator identity. Verify the
   declared initial-state contract, task identity and all object bindings.
3. **Action semantics:** execute a fixed legal action prefix that exercises a
   navigation action, one two-stage Marker action and one three-stage placement
   where the task permits it. Check parameter coordinates, Marker lifetime,
   blocked feedback, macro-action accounting and unchanged failure categories.
4. **Observation boundary:** save the actual provider request bodies locally.
   Verify current composite views, configured public history, image hashes and
   omission of private goals, object state, verifier outputs and Oracle content.
   Use a deterministic scripted provider first, then a separately authorized model
   endpoint; record transport and exact model identity.
5. **Scoring and termination:** cover success, failed final goal, budget exhaustion
   and protocol failure. For a task with declared process constraints, verify
   begin/step/end history and a deliberate process violation. Check official
   success, Q, process verdict and failure classification independently.
6. **Artifacts:** validate every image reference, relative path and SHA256; inspect
   private/public separation, summary provenance and branch preservation. Compare
   recorded prompt requests, not just the chronological ChatML export.
7. **Parity:** run the same frozen configuration and fixed input prefix using an
   independently installed reference snapshot, also on isolated resources.
   Compare parsing, public input construction, action counts, verifier decisions,
   terminal reason and artifact schema. Record tolerances for simulator state
   only where the original contract defines them; do not invent score tolerances.
8. **Coverage:** extend from a smoke task to every task and declared dependency/
   scene combination in the selected profile before claiming full-profile Live
   reproduction. The executable catalog has 257 tasks across eight profiles;
   validation of one profile cannot stand in for another. Preserve load failures
   and unsuccessful runs in the denominator.

A run passes only if its declared contract passes and its provenance is complete.
`Q=1` alone is not a pass, a mock verifier is not a canonical verifier, and a
successful scene load is not a completed episode. Do not change thresholds,
retries, initial state or alternate supports to turn an invalid result into a pass.

On failure, retain raw events, scene logs, materialized configuration and source
identity. Classify installation/resource failure, protocol failure, executor
failure, process failure and final-goal failure separately. Correct candidate
integration bugs and rerun only the affected checks under a recorded new attempt;
do not rewrite the first attempt or old source records.

The resulting report must separate completed CPU checks, installation checks,
one-task Live checks and full-profile Live coverage. In particular, validate P's
restore-and-verify gate separately from L2 Physical's reconstruct-and-verify gate,
and cover V identity/intervention/protocol evidence and L2 history constraints.
Agent CLI harnesses and resume/recovery are outside the Common entrypoint and
require their own frozen implementation and acceptance evidence.
