# RoboWeave rc5

The project is **RoboWeave**. Version `0.1.0rc5` uses `roboweave` for the
installable distribution, primary command, Python package, examples and current
documentation. The private repository is
[ShawnLee0910/roboweave](https://github.com/ShawnLee0910/roboweave).

The old CLI and public Python imports remain compatibility aliases. Old config
and scoring-record schemas are accepted without modifying the supplied records;
old and new v2 trajectory evidence receive the same strict binding, hash, size
and result checks. [Naming notes](../../docs/NAMING.md) list the precise boundary.

## Verification

- **1608 distinct CPU cases have passing evidence; 3 are skipped** for the same
  unavailable private input fixtures. Grouped run accounting is in
  [VALIDATION_SUMMARY.json](VALIDATION_SUMMARY.json); repeated cases are not added.
- The first new history compatibility test omitted the required fresh output
  directory. The implementation rejected it correctly; the test was corrected,
  and all **11 naming/compatibility cases pass**.
- The distribution test initially compared against a checkout receiving new
  reports during its build. It now checks the copied build snapshot; the
  **source-archive/extracted-checks/wheel regression passes**. Initial failures
  remain in their original receipts.
- The final wheel has **2484 package files identical to the checked source**.
  Its independent installation passes **22 checks across all 8 execution
  profiles**, including old/new commands and module entry points, package
  resources, configuration, scoring/history and synthetic episode validation.
  The installed catalog still contains **257 task configurations**.
- Frozen source bundles, task/asset data, fixed equivalence inputs and earlier
  provenance/reports retain their original contents. See
  [FROZEN_CONTENT_VERIFICATION.json](FROZEN_CONTENT_VERIFICATION.json).
- Current source and reachable candidate-history scans found no credential
  patterns or private runtime payload paths. No credentials are included in the
  audit output. Pattern scanning does not prove absence of every possible secret.
- All **157 checked local documentation links** resolve.

Task definitions, action/observation contracts, budgets and scoring remain
frozen. This is CPU/package evidence; no GPU, simulator scene or real-model
request was run for the rename. Live acceptance and public-release permission
remain separate from the authorized private GitHub upload.
