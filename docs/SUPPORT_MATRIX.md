# Supported profiles and evidence boundary

The candidate ships one runtime implementation, the explicitly frozen L0 English-v2 combination. Other families are represented by public task definitions and exact version metadata. They are not redirected to the L0 runtime.

| Profile | Catalog | Runtime revision | Bench revision | Candidate capability |
|---|---:|---|---|---|
| `l0-english-v2` | 100 | `9b992b6797330a836e9533e5409627bf63f7d94b` | `5b8d2b37fd7d97b161e88de4d438ec789c78f9b4` | Installed CPU protocol/scoring support; private-input import and explicit single-task Live integration; Live unverified |
| `l1-a30` | 30 | `cd25f9ba1579bdca1f0dadf6fce7ad10974b8717` | `1d76bc5dc53b1995f2e7c301cb06569bcf0c4923` | Catalog/version audit only |
| `l1-m30-public-v2` | 30 | `918aefd40beb72368126cc3fa2891bbb732ad490` | `fb1cf143f850fe43a04d25440f07cbfab4007093` | Catalog/version audit only |
| `l1-v30` | 30 | `918aefd40beb72368126cc3fa2891bbb732ad490` | `d8ae5151fadfb3338d122fdb12b26051ed753852` | Catalog/version audit only |
| `l1-p-inside5` | 5 | `955585540d95cba02f672c0d3cdb94048db23bf2` | `7ebc9c948a047ba7f51e0a1d2bd768b228fad5eb` | Physical catalog/version audit only |
| `l1-p-ontop5` | 5 | `c813a24f853bb90ba3772365ca5182ecb28c553b` | `7ebc9c948a047ba7f51e0a1d2bd768b228fad5eb` | Physical catalog/version audit only |
| `l1-p-additions20` | 20 | `ab615d43d4ede7e6a50eaeda189077529bef361a` | `7ebc9c948a047ba7f51e0a1d2bd768b228fad5eb` | Physical catalog/version audit only |
| `l2-proposals` | 0 executable | `e75bcd742ee093e5e9d64f8079d44b11fcc4d9b1` | Separate compiler `37160fd489b56cbd893299c47a16d23f60b77c46` | Proposal and version audit only |

A/M/V bindings come from `benchmark/L1-AMV90-RELEASE/20260924_v1/RELEASE.json`. P comes from the `inside`, `ontop` and `additions` subpackages under `benchmark/L1-P-FINAL/model_batch_20260924`. The exact source files and hashes are in [imports_tasks.json](../provenance/imports_tasks.json). None of these profiles is inferred from a directory date or the current repository HEAD.

L2 has 36 proposals: three complete packages, six Base-only entries and 27 blocked entries in the surveyed intake. Its documented authoring routes are `AM_SYMBOLIC`, `MV_BASE_SYMBOLIC`, `MV_D0_D1_SYMBOLIC_UNARY_MILESTONES` and `MP_PHYSICAL_ONTOP_HISTORY`. Physical+V and arbitrary unvalidated axis combinations are unsupported. Existing canaries do not certify all 36 proposals, and this candidate does not expose them as 36 runnable tasks.

For L0, `execution_supported: true` in profile metadata means the implementation and input adapter are present. It does not mean the external assets are installed, a GPU is authorized, the historical acceptance gate was cleared or all 100 tasks passed Live. The source runtime lock explicitly records four pre-existing Action CPU failures and missing all-100 initial-state and new-combination Live evidence. The same four formerly failing assertions pass in the candidate independent CPU suite; this does not diagnose the original environment or rewrite its frozen gate. Current test evidence is reported separately.

The frozen `wall_clock_seconds: 3600` declaration lacks a consumer in the original model client. The original controller's separate default four-hour model-process timeout is outside this candidate's single-task integration. Strict outer-controller wall-clock reproduction remains unsupported; the candidate preserves this limitation rather than silently adding a new evaluation rule. See [SOURCE_LIMITATIONS.md](../reports/SOURCE_LIMITATIONS.md).

No GPU, simulation, model request, resident service, scheduler or existing campaign was started for this release preparation. Source assets and original results remain unchanged.

The historical publication gate remains `BLOCKED_UNRESOLVED_BASELINE_CPU_FAILURES`. The candidate provides an independent validation entry point; it does not grant formal benchmark or production eligibility. Profile metadata sets `official_benchmark_eligible: false` and `strict_production_eligible: false`. Before formal use, the release owner must review a new source-identity/CPU receipt, complete the independent Live plan and all-100 reset evidence, verify deployment and provider integration, establish resource ownership and experiment authorization, and resolve redistribution permissions. None of those steps rewrites the historical lock or turns this local candidate into a formal published result.
