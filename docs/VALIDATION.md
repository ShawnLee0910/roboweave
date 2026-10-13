# Candidate validation

The candidate is installed and CPU-validated. **No GPU, simulator scene, actual
model request, SSH, existing resident, service or scheduler was used.** Neither
these checks nor the synthetic example constitute a Live benchmark pass.

| Acceptance item | Evidence |
| --- | --- |
| Isolated installation | Editable development install and separate regular wheel install; Python 3.12.9, NumPy 1.26.4, Pillow 11.3.0, PyYAML 6.0.3 |
| Relocation | Wheel commands ran from an unrelated directory with `PYTHONPATH` unset; imports resolved inside the new wheel environment |
| Public command chain | Help, doctor, config validation, L0 enumeration, replay, trajectory validation/summary and canonical CPU scoring passed |
| Task/resource closure | **100/100** actual L0 inputs: 62 registered and 38 TaskSpec; **769/769** raw hashes unchanged; 1,834 per-task dependency instances compared |
| Protocol, runtime and wrappers | **928 distinct tests have passing evidence; 3 explicitly skipped**; detailed accounting below |
| Frozen-source behavior | **75/75** independent fixed-input comparisons passed from both the checkout and installed wheel |
| Historical failing assertions | Original frozen Runtime **4/4 passed**, candidate **4/4 passed** in this CPU environment; old source gate unchanged |
| Installed runtime identity | **303** candidate source/resource hashes verified, with upstream revision recorded separately |
| Distribution checks | No bundled restricted MDL modules, credentials found by the bounded scan, external source symlinks, simulator assets or original trajectories |

Machine evidence: [wheel acceptance](../reports/WHEEL_VALIDATION.json),
[wheel equivalence](../reports/WHEEL_EQUIVALENCE.json),
[source equivalence](../provenance/equivalence.json),
[task/resource audit](../reports/TASK_ASSET_CPU_VALIDATION.md),
[historical CPU recheck](../reports/HISTORICAL_CPU_RECHECK.md),
[source preservation](../provenance/source_preservation.json),
[source audit](../provenance/source_audit.json).

## Test accounting

The integrated run after the last module changes recorded 927 passed, 3 skipped
and one test-harness failure: its long pytest temporary directory exceeded the
Unix socket address limit before the ownership check could run. The test name
was shortened without changing its assertions; its targeted recheck passed with
the same long `--basetemp`. All 928 executable cases therefore have passing
observations after fixes. The original failed log is preserved:

- [Integrated log](../reports/pytest.log) and [JUnit](../reports/pytest.xml)
- [Socket recheck](../reports/pytest-socket-recheck.log) and [JUnit](../reports/pytest-socket-recheck.xml)
- [Final affected integration checks](../reports/integration-final.log): 33 passed

The three skips are explicit:

1. The optional complete asset test needs `ESI_BENCHMARK_ASSET_ROOT`. The actual
   100-task audit was performed separately; private temporary inputs were then deleted.
2. One upstream process test needs an undistributed historical eight-task package.
3. One upstream lifecycle test needs an undistributed historical six-task package.

The process kernels and synthetic tests run normally. These skips do not claim
that the two historical packages passed. Tests came from the frozen revisions;
path adaptations and operational-test substitutions are recorded in the import
manifests. No scoring or action assertion was relaxed to obtain a pass.

Reproduce the distributable checks:

```bash
python -m pip install '.[test]'
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -ra --basetemp=.tmp/p
python tools/check_equivalence.py
python tools/audit_candidate.py
```

The [CPU environment record](../provenance/cpu-environment.txt) lists installed
versions. It is not a lock for the external Python 3.11 simulator environment.
The official installer contains some dependency ranges and moving references;
Live acceptance must record its own complete package/build inventory.

## Evidence boundaries

Replay uses synthetic images and recorded replies through the original model
loop and returns no official score. Predicate-record scoring runs the original
kernels on supplied truth; it cannot infer that truth from RGB. Resource checks
establish identity, relocation and native-loader compatibility, not restoration,
rendering, collisions, action execution or live task success.

The source declares a 3,600-second budget that its model loop does not enforce.
The outer operator's separate four-hour policy is outside this candidate. The
historical release gate remains blocked. Current CPU results do not rewrite it,
identify the old environment's failure cause, grant formal result eligibility or
clear Live requirements. Runs retain false formal-eligibility flags; an original
evaluator success flag is a different concept.

L1 A/M/V/P and L2 have catalog/version-audit support only. Source licensing,
private preparation distribution, authorized NVIDIA MDL modules and simulator
assets remain external release conditions. See [Live acceptance](LIVE_VALIDATION.md),
[distribution boundaries](DISTRIBUTION.md) and [source limitations](../reports/SOURCE_LIMITATIONS.md).
