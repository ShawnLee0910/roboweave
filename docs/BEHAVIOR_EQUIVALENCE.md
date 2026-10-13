# Fixed-input behavior equivalence

The retained L0 reference suite contains **75 fixed-input CPU comparison cases**
from the frozen L0 English-v2 implementation. Its historical passing receipt is
preserved; [Validation](VALIDATION.md) identifies the latest candidate run.
This suite does not establish equivalence for all eight rc3 profiles. The reference was independently extracted with
`git show`, run in another Python process, and compared with the installed
candidate. Expected results are not hand-authored and were not generated from
the candidate implementation.

| Component | Frozen source revision |
| --- | --- |
| Bench / model loop | `5b8d2b37fd7d97b161e88de4d438ec789c78f9b4` |
| Action / Runtime / verifier | `9b992b6797330a836e9533e5409627bf63f7d94b` |

The source closure contains 66 loaded Python/resource files. Full reference and
candidate SHA256 values are in [the machine-readable report](../provenance/equivalence.json).
The source file changes encountered by these probes are confined to
`lab/model_client.py` and `lab/scene_protocol.py`; the import manifests describe
their portability changes. No original source worktree was imported or modified.

| Case group | Count | Compared behavior |
| --- | ---: | --- |
| Parser | 24 | Action names, target identity, optional field normalization, movement float tolerance, numeric turn strings, repeat limits, native reasoning, lossless tool envelope, strict rejection |
| Public tool result | 8 | Remaining budget, stage validation, rounded odometry and bounds, blocked categories |
| Public observation | 5 | Depth median/validity, out-of-image anchors, public Marker fields and private-field rejection |
| Conversation | 13 | Initial full model request; two-call pick; three-call inside placement; Marker namespaces; 5-frame physical history and expired selection rendering |
| Budget | 2 | Insufficient-budget cancellation leaves state unchanged; authorized cancellation consumes semantic actions |
| Termination | 2 | Completed `done`; official-success terminal ledger |
| Trajectory | 4 | Full private transcript, selected-target audit evidence, ChatML/image references, restore to the same model request |
| Scoring | 8 | Official all/any-option scoring, initial-credit exclusion; projected stability, regression and early-done behavior; fail-closed invalid readers/routes |
| Process | 4 | Legal placement, prerequisite violation remains after repair, protected-object pick, unknown blocked release |
| Model loop | 5 | Production loop with recorded scene/provider replies: done, a complete three-call Marker action on one action budget, protocol retry without budget consumption, blocked-streak stop, model-call limit |

Each conversation and model-loop comparison includes the full request structure,
tool schema, public history, counters, termination fields and relevant ledger.
Comparison is exact after only these portability normalizations:

1. Temporary output roots, installed source roots and the upstream configured
   `PROJECT_ROOT` are replaced by stable labels. The original model loop embeds
   an internal absolute source path in the private episode contract; relocating
   that field is intentional and does not alter model-visible information.
2. PNG data URLs are compared by decoded image mode, size and pixel SHA256. This
   avoids treating an encoder/compression change as a visual change. Other image
   metadata, relative references and transcript fields remain in the comparison.

No budget, status, score, stage, exception message, model text or event is dropped
from compared outputs. Normalization itself has a regression check demonstrating
that budget, status and Q changes remain detectable.

Run against an installed candidate from this repository checkout:

```bash
python tools/check_equivalence.py
python -m pytest tests/test_equivalence.py -q
```

The command uses only the installed package and the small distributable inputs
and golden records in `tests/fixtures/equivalence/`; original source paths and Git
repositories are not required. A maintainer with the frozen Git objects can
recreate the independent reference:

```bash
python tools/check_equivalence.py \
  --source-roots /path/to/bench-git /path/to/runtime-git \
  --report provenance/equivalence.json
```

`--refresh-golden` additionally replaces the golden record **from the frozen
reference subprocess**. It requires both source roots and never updates source
repositories. Temporary source copies and outputs are kept inside `.tmp/` and
removed after the command. Python bytecode writing is disabled for subprocesses.

The fixture inputs are synthetic and follow scenarios in upstream
`lab/test_agent_target_description.py`, `lab/test_agent_marker_cancellation.py`,
`lab/test_model_protocol_retry.py`, `tests/embodied/test_l1_process_audit.py` and
the verifier implementations. They contain no simulator assets, private task
contracts, original trajectories or real provider responses.

These checks establish CPU behavior preservation for the selected source profile
and samples. They do **not** prove simulator geometry, visual rendering in Kit,
asset admission, live object binding, endpoint transport, full task success,
all failure/recovery paths, or any other task family's historical source profile.
These process cases exercise the L0 bundle's available CPU auditor. The separate
rc3 family adapters, source snapshots and v2 history-record tests are outside this
75-case reference comparison; their support and acceptance are documented in
[Support matrix](SUPPORT_MATRIX.md) and [Validation](VALIDATION.md). See the Live validation checklist for the
remaining independent environment acceptance work.
