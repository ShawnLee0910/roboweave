# Trajectory and result artifacts

Each run uses a fresh output directory. The candidate keeps the original
trajectory formats; it does not export or normalize internal training data.
The bundled example is hand-authored synthetic data with generated solid-color
RGB images, declared CC0-1.0 in its fixture metadata.

| Artifact | Purpose and access boundary |
| --- | --- |
| `run_record.json` | Candidate profile, source revisions, execution mode and task/model configuration; records API-key variable names, never key values. |
| `client_manifest.json` | Original prompt/tool/source hashes, task binding, budgets, history, provider and generation settings. Contains local paths/private task metadata. |
| `system_prompt.txt`, `action20_tool_schema.json` | Exact material used in this episode. |
| `conversation_chatml.json` | Readable model-visible projection, schema `behavior-hl-action20-chatml-v3`; image references are relative paths. Model-authored text/reasoning can still require review before sharing. |
| `conversation_private.json` | Complete internal event transcript, including private contract and raw provider/reasoning material. Not a public model input. |
| `observations/<id>/*.png` | Materialized RGB frames and `three_view_composite.png`. Copied/hardlinked from owned run outputs with original hash checks. |
| `action_after/*.png` | Chronological post-action composites. Recovery branches have separate directories. |
| `provider_calls/call_####/` | Request/response, attempts, error and token/timing evidence. Treat as private. |
| `summary.json` | Original terminal classification and counters; the candidate may add `candidate_result` without replacing original fields. |
| `candidate_error.json` | Candidate execution or score-archive failure after the original summary was written; records the failure stage and exception type without replacing the original summary. |
| `private/scene_final_score.json` | Byte-for-byte archived resident score after episode end, when available. Never added to model observations. |
| `private/task_binding.json`, `private/oracle_prepared_restore.json` | Frozen identity/preparation evidence for Live runs when present. |
| `private/task_process_result.json`, `private/task_process_events.jsonl`, `private/task_evidence_index.jsonl` | Original process report, per-action audit receipts and frame references. |
| `private/physical_initial_binding.json` | Original Physical admission/binding record when emitted by the selected runtime. |
| `private/v_scene/` | V collection identity, protocol result, marker bindings and intervention records, kept private. |
| `private/l2_v/` | L2 V session records, including its native combined-protocol final report and checkpoint violations. |

`DONE` means the model ended its episode. It does not itself mean official task
success. `FAIL_BUDGET_EXHAUSTED`, internal call-cap failures, client/provider
errors, resume statuses and scene reload requirements remain distinguishable.
`macro_actions_resolved`, `actions_sent_to_scene`, `scene_step_attempts`, and
`model_calls` count different things; do not replace one with another.

For Live results, `candidate_result` references the original final-score path,
archived relative path and SHA256. Its interpretation requires verifier validity,
no execution-integrity failure, the original success flag and any task/process/
Physical qualification gate. `official_q` is copied unchanged; Q=1 is never used
to invent success. Missing final evidence is `unavailable`, not zero or success.
The original score has no mandatory schema-version field. This candidate does
not rewrite it to a new scoring format. L2 keeps its native combined
`task_success`/`status`, Physical-template and V-protocol gates; the P30
`physical_qualification` field is not imposed on L2.

New rc3 archives add `candidate_result.evidence_schema_version:
esi-benchmark/result-evidence-v2`. Each `evidence` entry records its original
source, fixed archive-relative `path`, `sha256` and `size_bytes`. The generated
`task_binding.json` is included in this inventory, including when no final score
was produced. These descriptors do not alter the original score or raw JSONL.

`trajectory.validate_run(path)` (also `inspect_run`) reads only the named run's
bounded JSON files, validates ChatML structure, checks local image containment
and existence, hashes images, checks rendered prompt provenance, and validates
every archived evidence reference when present. Missing files, changed hashes
or lengths, unsafe paths (including intermediate symlinks), malformed JSON/JSONL,
duplicate JSON fields and non-finite numbers are rejected. Native JSONL rows must
be objects; the known L2 `checkpoint_violations.json` file may be an array.
The final score and ordinary small private records remain bounded to 512 KiB;
JSONL and V/L2 auxiliary records are bounded to 32 MiB, with at most 65 evidence
entries. Only explicit references are read. Inspection does not traverse arbitrary
output trees or follow image references outside the run. Its output excludes the
private transcript and provider contents. Current image hashes are inventory
hashes; source image hashes are verified during materialization. Inspection alone
does not prove a historic file was never modified.

For v2 process evidence, the archived process report must equal the report in
`final_score.json`, and its event/frame counters must match the archived JSONL
row counts. L2's archived `l2_v/final.json` must equal the native final score's
V report; checkpoint-violation records are checked against that report when it
contains the corresponding field. These are consistency checks, not a rerun of
the process, physics or visual verifier.

V30 additionally requires its collection-identity and protocol files. The
protocol must equal the original final score's report. The original nine-field
scene identity must equal the archived task binding, and its collection key,
D1 branch, native TaskSpec identity and runtime revision must match the frozen
profile/catalog. The public scene/model `task_name` is the collection key;
`v_identity.task_spec_id` is the native donor task name. The task handle, seed,
approved instruction, run-record task identity and client-manifest task/model
must agree. A different auxiliary identity or report cannot override the score
by merely updating its hash.

Legacy evidence entries without the v2 schema remain readable without a
mandatory `size_bytes`; all references they do declare still undergo containment,
size-bound, hash and format checks. L0 historical records need no migration.
The stronger V30 task/model binding requires v2; an older V-shaped record that
lacks this evidence is rejected rather than presented as a fully checked V run.
Self-contained hashes cannot authenticate an attacker rewriting every record
consistently; preserve the original run and source manifests for provenance.

`trajectory.summarize_runs(paths)` accepts explicit run directories. It reports
Live, synthetic replay and unknown legacy modes separately. It publishes both
all-Live-attempt and score-covered denominators, score coverage, and unscored
Live attempts. `evaluation_variant_counts` separates `frozen_history_5` from
`history_20_variant_not_baseline`; `mixed_evaluation_variants` flags mixed inputs.
Missing legacy variant metadata stays `unknown`. Explicit initialization/conversation failures are `INCOMPLETE`;
malformed runs are individually `INVALID` and do not erase other runs from the
report. Neither is reported as a completed trajectory check. Legacy runs without
`run_record.json` have `unknown_legacy` mode rather than being assumed Live.
This is a per-run compatibility check, not a data export tool.

A valid `candidate_error.json` makes inspection `INCOMPLETE` even if the original
client status is `DONE` or a score was previously archived. The report preserves
the original client status and failure `status`, `stage` and `error_type`; it does
not count this attempt as scored or successful. A missing final score alone can
still pass the trajectory/provenance check with `score_available: false`.
Malformed nested result, evidence, prompt-provenance or run-record objects are
validation errors, reported per run as `INVALID` by the summarizer. Summary and
other source artifacts are never rewritten during inspection.

A replay run records `mode: offline_replay`, `official_score: null`,
`simulator_started: false` and `model_requested: false`. The complete original
model loop still emits the normal artifacts. Its `DONE` status verifies protocol
completion only. Live acceptance requires separately supplied simulator assets,
a new owned environment and the documented Live checklist.
