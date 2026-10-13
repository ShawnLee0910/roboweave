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

Or use `esi_benchmark.scoring.score_record(record)`. Input schemas are
`esi-benchmark/score-record-v1` and `esi-benchmark/score-record-v2`; result schema is `esi-benchmark/score-result-v1`,
with `mode: cpu-record-replay` and `truth_source: supplied-records`.
An optional top-level `profile` chooses a frozen source bundle. The Python API
also accepts `profile=...`; a conflicting record profile is rejected. The
default remains `l0-english-v2`. Profile selection does not establish that a
record came from a real task or scene. The public API invokes the selected
native scoring bundle in a separate CPU process; importing it does not modify
the caller's module search path or load another profile's runtime. Python return
values now use the same JSON types as the CLI: in particular, numeric keys in
`process.current_reference_placements` are strings. This is an API serialization
change from rc2; native scoring and CLI JSON values are unchanged.

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

Both versions keep the canonical scoring inputs above unchanged. The result
retains all snapshots, the final `canonical` result, optional process
report, and gated `final` result. With `requires_process: true`, missing process
evidence fails closed. To replay an audit, pass an explicit fresh directory via
`--process-output` or `process_directory`.

### v1: recorded process snapshots

Existing v1 input remains supported. Its `process` object supplies:

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

`history_program` contracts remain rejected for v1: their original audit
re-reads predicate truth during each event, while `score-record-v1` has only
initial and final atom truth. Those records cannot establish a complete history
audit. Do not relabel an old v1 record as v2 without collecting its missing
per-event evidence.

### v2: predicate readbacks at each event

Run the synthetic, distributable history example into a fresh directory:

```bash
esi-benchmark score examples/history-score-record.json --process-output outputs/history-example
```

The example declares an actual-first-subject binding, an open-until-stored
invariant, and ordered milestones. It supplies small boolean readbacks and no
scene, private asset or recorded model trajectory. A passing result demonstrates
CPU record processing only.

The v2 `process` object has exactly these fields:

| Field | Meaning |
| --- | --- |
| `roles`, `requirements` | Declared role identities and the unchanged native process contract. |
| `initial_truth` | `{ "atom": <native predicate atom>, "value": <boolean> }` rows available to audit construction. |
| `final_truth` | Atom truth available to the final native report, including dynamically selected final requirements. |
| `protected_relations` | Exact private initial-recipe atoms passed to the native audit for protected objects. |
| `protected_state_relations` | Exact protected-state atoms; they must cover the declared state-protection roles. |
| `placement_targets` | Exact target atom for each `before_placement` rule, in rule order. |
| `events` | Complete process event sequence in original order. |

The three private atom lists are constructor inputs to upstream
`create_process_audit` / `ProcessAudit`, not new scoring rules. Public
`requirements` alone does not contain the initial recipe or resolve final
placement references. Supply these lists from the same private contract;
use empty lists when the contract has no corresponding rule. Predicate atoms
retain their exact native shape, including `desired` when present; replay keys
use that full atom, and never guess a role or substitute an alias.

Each event has exactly `event_id`, `action`, `status`, `action_index`, `before`
and `after`. `event_id` is the one-based audit ordinal and must be consecutive.
It detects gaps, duplicate ordinals or changed ordinal order in the supplied sequence.
`action_index` remains the original nonnegative integer or null; several audit
events may share an action index. Each phase contains:

```json
{
  "held_roles": ["item_a"],
  "truth": [
    {"atom": {"predicate": "ontop", "subject_role": "item_a", "object_role": "tray", "desired": true}, "value": false}
  ]
}
```

`held_roles` contains zero or one declared role, reflecting the native single
held object. A pre-`pick` phase may additionally supply `selected_roles`, the
original protected-object selection receipt. Every truth value is a boolean;
atom identities must refer to declared roles. The truth list must cover each
predicate the original audit actually requests at that phase. Unknown fields,
handwritten `history`, `reached`, binding candidates or `milestones` are rejected.

Replay calls the original implementations in this order:

1. Construct `ProcessAudit` with `initial_truth`. Milestone initialization retains
   its original initial-readback behavior; history bindings are not initialized
   from a preferred subject or destination.
2. Select the event's `before.truth` and call `ProcessAudit.snapshot` with the
   recorded held-object identity.
3. Select `after.truth`, call `snapshot` again, then call `observe` while that
   post-action truth is still active. This matters because history milestones
   and placement guards re-read predicates inside `observe`.
4. After all events, select `final_truth` and call `report`. History expressions
   and cycle-order final states use this final readback.

Consequently, history candidates, actual subject and destination bindings,
first complete relation maps, maintenance windows, predecessor gates, milestone
cycles and dynamic final states are computed by the frozen implementation.
The wrapper does not rebuild those algorithms. Failed, uncertain and marker-only
events keep the original classifications and cannot acquire a successful binding.
Only features accepted by the selected runtime are available: for example, the
older `l1-p-inside5` and `l1-p-ontop5` bundles reject the example's newer subject
binding/maintenance contract. Choosing v2 does not substitute a newer verifier.

Missing required truth raises `ValueError`, including missing truth encountered
inside a native snapshot or final history report. Malformed structures and
event identities are checked before writing receipts. Missing evidence found
while executing a later readback can leave earlier private receipts in the
output directory; retain them and retry only into a new directory. Supplied
event numbers cannot certify that an external recorder actually captured every
real execution, and the replay does not prove geometry, image evidence or
canonical option identity against a simulator.

The wrapper invokes the original `ProcessAudit`; it writes its original private
event receipts into that fresh directory. This API replays supplied audit facts.
It does not create missing evidence, reconstruct hidden state from RGB, or certify
dynamic-goal truth not present in the record. The native Live composition remains
authoritative for a task with a process/history contract.
