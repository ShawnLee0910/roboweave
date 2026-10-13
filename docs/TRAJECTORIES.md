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
| `private/scene_final_score.json` | Byte-for-byte archived resident score after episode end, when available. Never added to model observations. |
| `private/task_binding.json`, `private/oracle_prepared_restore.json` | Frozen identity/preparation evidence for Live runs when present. |

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
not rewrite it to a new scoring format.

`trajectory.validate_run(path)` (also `inspect_run`) reads only the named run's
bounded JSON files, validates ChatML structure, checks local image containment
and existence, hashes images, checks rendered prompt provenance, and validates
archived final-score fields/hashes when present. It does not traverse arbitrary
output trees or follow image references outside the run. Its output excludes the
private transcript and provider contents. Current image hashes are inventory
hashes; source image hashes are verified during materialization. Inspection alone
does not prove a historic file was never modified.

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

A replay run records `mode: offline_replay`, `official_score: null`,
`simulator_started: false` and `model_requested: false`. The complete original
model loop still emits the normal artifacts. Its `DONE` status verifies protocol
completion only. Live acceptance requires separately supplied simulator assets,
a new owned environment and the documented Live checklist.
