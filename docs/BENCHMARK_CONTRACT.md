# Candidate Benchmark contract

This is the contract entry point for the local public candidate. Its default is the frozen **L0 English-v2** profile in [SUPPORT_MATRIX.md](SUPPORT_MATRIX.md). The source project's earlier contract remains useful historical context, but its old candidate pools, static milestones and protocol examples do not replace the selected FINAL inputs.

## Frozen identity and task boundaries

A task is identified by family, public catalog ID, original instruction ID, runtime instruction ID, exact source file SHA256 and version profile. [TASKS.md](TASKS.md) documents all fields. Source directory dates and current repository HEADs do not establish a historical execution version. The imported code commits, source-file hashes and private per-run derivation receipt together identify an experiment.

L0 is baseline embodied execution. L1 A/M/V/P and L2 compositions retain their documented family definitions and different frozen sources; this release's executable runtime is not silently reused for them. The public catalog is an evaluation catalog. No new train/dev/hidden-test split or training-data export is introduced.

A task's approved English instruction, initial preparation state, scene, seed, goal, required process/history, decision budget and model-call budget are frozen inputs. Resource relocation may change filesystem references and dependent file hashes, and a per-run aggregate may select one exact task. It must preserve the selected record's non-path semantics. A repair to a goal, process condition, initial state, instruction or threshold requires a new profile and evidence; it is not a packaging operation.

## Agent interface and information boundary

The current default uses the native Action20 interface described in [ACTION20.md](ACTION20.md), with the vendored `action20_tool_schema.json`, core instructions and parser. Older `reasoning_action` wrappers or early milestone examples are historical protocols. They are not an alternative default unless an exact frozen adapter expressly admits them.

Action names, arguments, allowed numeric values, coordinate conventions and temporary Marker selection stages remain those of the pinned parser and executor. The candidate neither rewrites the executor nor treats a failed selection as a successful action. Marker stages and observations are checked by the original protocol logic.

Only approved public instructions, onboard observations, documented execution feedback and the configured public history reach the model. L0's frozen operator uses five-frame history. Private task specifications, verifier state, exact hidden poses, Oracle plans and prepared bundles are evaluator inputs. They must never be included wholesale in prompts, model-visible tool responses or a public training export. The candidate contains no raw training-data export path.

## Runtime, budget and termination

L0 uses Symbolic mode, full scene loading, `r1pro_three_rgb_head_eval`, overhead view off, zero extra physics steps per action and automatic interaction approach disabled. Preparation is restored from the exact imported bundle. A new independent process and output directory are required; this candidate does not take over a resident queue or historical campaign.

Action-decision and model-call limits are retained per task and enforced by the pinned episode implementation. Model `done`, exhaustion, protocol failure and runtime failure retain their original meanings. A model declaring completion is not proof of goal success. Retries or a fresh attempt must preserve their own identity and evidence rather than overwrite an earlier result.

The declared 3,600-second wall-clock field has no consumer in the frozen model client. Its original outer controller had a separate four-hour model-process timeout. The candidate records this discrepancy and does not silently invent a new stopping rule; strict reproduction of that outer timeout is unsupported. See [SOURCE_LIMITATIONS.md](../reports/SOURCE_LIMITATIONS.md).

## Goal and process acceptance

The selected canonical verifier owns final-state truth. [SCORING.md](SCORING.md) describes the implemented score and process interfaces. Full official-goal options, polarity, thresholds, initial credit and failure classification retain their pinned implementations. A CPU projection is not a replacement for official Live verification.

When a task declares a process/history contract, acceptance includes the required begin/step/end or history evidence. Static final-goal success alone cannot erase a process failure. When no such contract is declared, a wrapper does not invent one. Historical family-specific Q1 report overrides remain historical reporting rules, not the default verifier.

A trajectory and its score must retain task/profile identity, source revisions, public conversation/actions, image references, budgets, termination reason, private verifier provenance and result status at their proper visibility boundary. Missing, invalid, unrun and failed evidence are not reported as successful evaluation.

## Reproducibility and release evidence

The normal path is install and inspect, prepare authorized resources, validate the configuration and task inputs, run an explicitly selected independent episode, then validate and summarize artifacts. The complete commands are in [QUICKSTART.md](QUICKSTART.md); assets are governed by [ASSETS.md](ASSETS.md).

The historical publication gate remains `BLOCKED_UNRESOLVED_BASELINE_CPU_FAILURES`; current passing CPU evidence is a separate candidate receipt. `official_benchmark_eligible` and `strict_production_eligible` remain false. A future independent validation run is not automatically a formal benchmark result.

The candidate supports CPU replay and source/contract checks without simulator assets. CPU results do not certify scene loading, reset fidelity, rendering, collision behavior, action effectiveness or official Live completion. The per-task input audit and independent Live plan separate those claims. Code and asset redistribution still require applicable permissions; this local candidate is not a publication authorization.

Historical source contract: `action20-capability-benchmark/docs/BENCHMARK_CONTRACT.md`, Benchmark source HEAD `1665ac1b0d1863cc642165deceb020eedd0db17e`, file SHA256 `43bb2d7cbf1e496390fedc08fa4d2874cd6dcc9eaf5793ae5d3eb17112f8ee19`. The source's training/private-data boundary and evidence separation are retained; its earlier master53/100-candidate milestone references are not relabeled as this FINAL release.
