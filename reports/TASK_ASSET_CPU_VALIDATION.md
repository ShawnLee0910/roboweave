# Task and asset CPU validation

**100 of 100 L0 English-v2 tasks passed the real-input CPU audit.** These are offline contract and file checks; no simulator, GPU, model request, resident connection or official Live evaluation was used.

The detailed per-task result is [TASK_ASSET_CPU_VALIDATION.json](TASK_ASSET_CPU_VALIDATION.json).

| Check | Result |
|---|---:|
| Public catalog identities and exact approved instruction projection | 220 records: L0 100; A/M/V/P 30 each |
| Original frozen runtime dependency references, including aliases | 715/715 SHA256 matched |
| Unique files represented by those runtime references | 713; 97,073,009 bytes |
| Additional exact implicit-loader dependencies | 55; 2,246,417 bytes |
| Total imported files, including the original release task input | 769/769 verified before and after the audit |
| Per-task materialization and strict non-path semantic comparison | 100/100 |
| Per-task handoff compilation and native argument construction | 100/100 |
| Prepared bundle identity, seed and observation-profile validation | 100/100 |
| Registered-task route loaded with selected BDDL/template/TRO/contract files | 62/62 |
| Fixed TaskSpec route resolves the exact selected record | 38/38 |
| Model binding instruction, string instance ID and per-task budget | 100/100 |
| Per-task dependency instances checked, counting shared files again across tasks | 1,834 |
| Small distributable task/import regression tests | 14 passed |

For semantic comparison, the audit parsed each materialized JSON/JSONL, reversed every recorded path and dependent-hash rewrite, and compared it to the exact original selected record. Only aggregate membership/count selection was allowed. Non-JSON files were compared byte-for-byte. Original imported files were hashed again after the full audit.

The audit exposed two real dependency omissions in the original manifest: the handoff-relative perception-frozen package and scene-derived expansion registration catalogs. Their original files existed. The candidate adds their exact metadata/hash closure and preserves the same native lookup; it does not change a pending state flag or weaken an acceptance gate. Details are in [SOURCE_LIMITATIONS.md](SOURCE_LIMITATIONS.md).

Private preparation inputs were temporarily copied to an ignored candidate-local audit directory. That directory and all per-task derivatives were deleted after successful validation. No private asset contents were retained as test fixtures or added to the distributable repository. The ordinary tests use small synthetic fixtures.

To repeat the complete CPU audit after separately importing authorized assets:

```bash
ESI_BENCHMARK_ASSET_ROOT=/path/to/authorized/import \
  .venv/bin/python -m pytest tests/test_assets_authorized.py \
  --basetemp .tmp/authorized-audit -q
```

The optional test is skipped unless the asset-root variable is explicitly supplied. It reads the imported originals, writes and deletes private task copies under its test directory, and makes no simulator or network calls.

This result does not prove actual scene loading, prepared reset fidelity, image correctness, action execution, collision behavior or Live verifier outcomes. The candidate CPU suite separately passes the four assertions that failed in the historical source environment, without identifying that old failure's root cause or rewriting its frozen acceptance gate. This asset audit does not turn the declared wall-clock field into an enforced deadline. Live validation and redistribution permissions remain separate requirements.
