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
- An explicit scene-load/retry budget and a separately reported operational time
  limit, if one is imposed. The frozen task's `wall_clock_seconds: 3600` is not
  enforced by the source episode loop; a new watchdog would create a distinct
  execution policy, not a reproduction of that field. Start with one owned
  worker and one scene load; inspect failure evidence before authorizing a retry.

Verify that the selected GPU is available to this validation and has no active
reservation or quarantine. A free-memory reading is not evidence of ownership.
Check for existing processes before launch; do not stop unknown processes.

Use the [installation instructions](INSTALLATION.md), validate the configuration,
import the exact preparation package, and verify all resource hashes. Launch only
the supported `l0-english-v2` profile. The first task,
`am200v1_a001_b2_control`, is an example candidate; its small dependency closure
still needs official external assets and its private preparation inputs.
Also obtain the required licensed NVIDIA material modules and verify the hashes
in [Installation](INSTALLATION.md). Pip installation and dataset access alone do
not satisfy this dependency.

The acceptance sequence is:

1. **Cold installation:** install the built wheel into a new environment with no
   original ESI paths on `PYTHONPATH`. Check command help, imports, profile identity
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
8. **Coverage:** extend from the smoke task to all 100 supported tasks and the
   declared dependency/scene combinations before claiming full-profile Live
   reproduction. Preserve load failures and unsuccessful runs in the denominator.

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
one-task Live checks and full-profile Live coverage. New task families, Physical
execution modes, Agent CLI harnesses and resume/recovery require their own frozen
profiles and acceptance evidence before being advertised as supported.
