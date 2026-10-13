# Canonical scoring and process evidence

Live scoring remains inside the original evaluator. The model receives only the
public observation/action feedback; private goal truth, object bindings and score
files are not sent to the provider. `models.run_episode` archives the native
`final_score.json` unchanged, with its hash, after the episode.

## Distinct scoring routes

`OfficialPredicateVerifier` re-evaluates grounded predicate options. A completed
option yields success and Q=1. Otherwise, for each option it counts leaves that
were false initially and true now, divides by that option's total leaf count, and
takes the maximum. Initial truth, option order and leaf identities are frozen.
World-integrity failure makes the verifier invalid and leaves Q unavailable.

`ProjectedGoalVerifier` serves `official_projection` and `canonical_fallback`.
It applies the declared quantifier, scope/destination equivalence, identity-sensitive
initial credit, stability window and regression accounting. A goal already true at
initialization is not a newly completed task. Missing truth or invalid clauses fail
closed. A requested `done` before success is recorded as `early_done`. It does not
replace an `official_full` verifier.

Spatial truth is still provided by the original relation readers. In particular,
`official` and `action20_symbolic` are different predicate modes: they must not be
silently exchanged. CPU predicate records provide truth values; they do not
recompute scene geometry or establish that those values were true in a simulation.

## Process and termination gates

Declared process/history contracts require the original begin/step/end audit.
Successful final static predicates alone cannot bypass no-manipulation rules,
placement prerequisites, dynamic references, ordered milestones, state protection
or history bindings. Failed, uncertain and selection-only steps do not establish
a verified successful transition. The original process verifier preserves the raw
canonical result and gates task success with an operational, satisfied process
report. A canonical Q of 1 is not, by itself, task success.

Native `final_score.json` contains `official_success`, `official_q`, `verifier_ok`,
`termination_reason`, and integrity information. Some process/physical routes add
`task_success`, `canonical_final_success`, `public_process_report` or qualification
fields. The candidate never infers success from Q. A missing/invalid native score
is reported as missing/invalid, not converted into a zero or pass.

Termination and budgets are managed by the frozen episode implementation. The
runtime's `TerminationFacts` classifier requires exactly one asserted terminal
cause; it does not conceal ambiguity through implicit precedence. Per-task action
and model-call limits come from the frozen task configuration, not one global
example value.

## CPU supplied-record API

Run the distributable example:

```bash
esi-benchmark score examples/score-record.json
```

Or use `esi_benchmark.scoring.score_record(record)`. Input schema is
`esi-benchmark/score-record-v1`; result schema is `esi-benchmark/score-result-v1`,
with `mode: cpu-record-replay` and `truth_source: supplied-records`.

For projected routes, supply `goal_spec`, `initial_truth`, optional
`stability_window` and ordered `snapshots`. Each truth item has exactly
`predicate`, `target`, `destination` (string or null), and boolean `value`.
A snapshot supplies `truth` and optional `done_requested`.

For `official_full`, supply `initial_options` and ordered snapshots containing
`options`: nonempty lists of nonempty boolean leaf lists in the same frozen
option/leaf order. A snapshot may declare `integrity_ok: false`, which produces
an invalid result with unavailable Q. This low-level input assumes the supplied
leaf positions retain their original identity; the Live verifier additionally
checks grounded predicate identities against the actual evaluator.

The result retains all snapshots, the final `canonical` result, optional process
report, and gated `final` result. With `requires_process: true`, missing process
evidence fails closed. To replay an audit, pass an explicit fresh directory via
`--process-output` or `process_directory`. The `process` object supplies:

- `roles` and the original `requirements`;
- `initial_truth` and `final_truth`, each a list of `{ "atom": <original atom>, "value": <boolean> }`;
- ordered `events`, each with nonempty string `action` and `status`, `before`,
  `after`, and a nonnegative integer or null `action_index`.

Both event snapshots must be objects with `held_roles` (distinct declared roles),
`prerequisites` (one boolean per before-placement rule), and
`protected_relations` (boolean readbacks). A placement contract also requires
`placement_goals`, with one boolean per rule. State protection requires
`protected_states` covering the declared protected states. Protection readback
counts must remain unchanged throughout the record. The number of protected
relations comes from the private initial recipe, not the public requirements;
the replay cannot independently certify that recipe or the supplied truth.

Current-reference contracts require one `current_references` entry per rule,
each containing `source_matches` and `target_matches` as distinct roles in that
rule's domain. Milestone programs require each declared boolean in `milestones`,
plus the `_unary` role readbacks when the original program uses a unary-state
extension. An optional `selected_roles` list retains the native protected-pick
receipt. Missing snapshots, missing required fields, wrong types and changing
protection readback counts raise `ValueError`; they cannot produce a passing
score. Structural validation runs before event receipts are written. Valid
records retain the original audit's success, violation and verifier-error
classification and the unmodified canonical Q.

`history_program` contracts are rejected by this CPU API: their original audit
re-reads predicate truth during each event, while `score-record-v1` has only
initial and final atom truth. Those records cannot establish a complete history
audit. The original Live history verifier and its contract are unchanged; a
future replay format needs explicit per-event predicate evidence before this
route can support history programs.

The wrapper invokes the original `ProcessAudit`; it writes its original private
event receipts into that fresh directory. This API replays supplied audit facts.
It does not create missing evidence, reconstruct hidden state from RGB, or certify
dynamic-goal truth not present in the record. The native Live composition remains
authoritative for a task with a process/history contract.
