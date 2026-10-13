# Supported profiles and evidence boundary

The public catalog contains **257 configurations across eight executable
profiles**. Each selects its own frozen Bench and Runtime, family adapter and
private-resource manifest. There are six Bench bundles and seven Runtime bundles;
the common M/V Runtime and common P Bench are stored once.

| Profile | Catalog | Mode | Runtime revision | Bench revision |
| --- | ---: | --- | --- | --- |
| `l0-english-v2` | 100 | Symbolic | `9b992b6797330a836e9533e5409627bf63f7d94b` | `5b8d2b37fd7d97b161e88de4d438ec789c78f9b4` |
| `l1-a30` | 30 | Symbolic | `cd25f9ba1579bdca1f0dadf6fce7ad10974b8717` | `1d76bc5dc53b1995f2e7c301cb06569bcf0c4923` |
| `l1-m30-public-v2` | 30 | Symbolic | `918aefd40beb72368126cc3fa2891bbb732ad490` | `fb1cf143f850fe43a04d25440f07cbfab4007093` |
| `l1-v30` | 30 | Symbolic | `918aefd40beb72368126cc3fa2891bbb732ad490` | `d8ae5151fadfb3338d122fdb12b26051ed753852` |
| `l1-p-inside5` | 5 | Physical | `955585540d95cba02f672c0d3cdb94048db23bf2` | `7ebc9c948a047ba7f51e0a1d2bd768b228fad5eb` |
| `l1-p-ontop5` | 5 | Physical | `c813a24f853bb90ba3772365ca5182ecb28c553b` | `7ebc9c948a047ba7f51e0a1d2bd768b228fad5eb` |
| `l1-p-additions20` | 20 | Physical | `ab615d43d4ede7e6a50eaeda189077529bef361a` | `7ebc9c948a047ba7f51e0a1d2bd768b228fad5eb` |
| `l2-collection37` | 37 | 23 Symbolic, 14 Physical | `aa878b489d402aaa9b1e30b083d0435cf44b26be` | `4c8364842157153875d098ebb529adcc0620a714` |

All eight have source selection, checked resource import/materialization,
native metadata admission, common Bench model integration and score archival.
**Live validation is `NOT_RUN` for every profile.** Implementation support does
not grant formal benchmark or production eligibility. Current evidence belongs
in [Validation](VALIDATION.md); older L0-only receipts remain historical.

| Family-specific boundary | Preserved behavior |
| --- | --- |
| L0 | English-v2 identities; 62 registered-task and 38 TaskSpec routes; exact prepared inputs. |
| A/M | Their frozen public instructions, process/history contracts and prepared state; no substitution of L0 code. |
| V | Explicit collection identity, D1 branch, intervention package and prepared bundle; protocol completion remains a success gate. |
| P | Three distinct Runtime revisions, native model-start admission and physical qualification; Q alone is insufficient. |
| L2 collection | Approved public English overlay, task-specific mode and composite verification. The 14 Physical entries reconstruct their declared world instead of restoring a symbolic prepared state. |

The candidate uses the selected Bench's **common** model harness. Historical
native model runners, prompts, controllers, retries and result files remain
separate provenance. The historical CODEX/AgentCLI harness is not exposed by this
candidate. A candidate run is not automatically a reproduction of
those model experiments. Five history frames is the frozen common-harness
setting; 20 frames is an explicit variant.

L0/A/M/V/P contracts declare 3,600 seconds, but their frozen common Bench loop
does not consume that declaration; strict historical outer-controller timeout
reproduction remains unsupported. L2 collection declares 7,200 seconds and has
its own documented timeout boundary. See [Runtime](RUNTIME.md) for enforcement
and result classification rather than assuming one global episode deadline.

`l2-proposals` remains a separate **audit-only** authoring snapshot: 36 proposals,
three complete packages, six Base-only entries and 27 blocked entries. Its
Runtime reference is `e75bcd742ee093e5e9d64f8079d44b11fcc4d9b1` and compiler reference
is `37160fd489b56cbd893299c47a16d23f60b77c46`. It has no executable Bench binding and
cannot be passed to `run` or `doctor`. It is neither an alias nor a predecessor
version label for the 37 collection entries; authoring restrictions in that
older audit do not define the collection's exact supported combinations.
Arbitrary new task or axis combinations remain outside the frozen catalog.

Bindings come from the per-family release/handoff records, not directory dates
or current development HEADs. See [task provenance](../provenance/imports_tasks.json),
[source imports](../provenance/imports_profiles.json), and the installed
`data/tasks/profiles.json` / `data/source_bundles.json`. Bundled roots are verified
against their registered manifest and file hashes; external official checkouts
must have their own exact Git root.

The L0 historical publication gate remains
`BLOCKED_UNRESOLVED_BASELINE_CPU_FAILURES`; new CPU evidence does not rewrite it.
External asset access, redistribution permission, independent environment setup,
reset fidelity and complete Live acceptance remain separate requirements. No
GPU evaluation, model request, resident connection or existing campaign was used
for this candidate preparation.
