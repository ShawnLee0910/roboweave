# rc2 review fixes

This local revision addresses the eight findings from the second review. It
retains the L0 source profile and its task contracts. No simulator, model endpoint
or existing resident was used. The original repositories remain read-only.

| Finding | Correction | Regression coverage |
| --- | --- | --- |
| Provider redirects forwarded credentials | The shared provider opener refuses all automatic redirects, including same-origin redirects. Users configure the final endpoint explicitly. | Chat Completions, Responses and Anthropic; 301/302/303/307/308, cross-origin and HTTPS downgrade, plus successful normal requests |
| Missing process evidence passed scoring | Validate the required snapshot fields and contract-specific boolean readbacks before calling the original process auditor. | Missing/null/empty/wrong-type evidence, protection, placement, current reference, milestones; normal pass, violation and unavailable outcomes |
| Private inputs acquired broad permissions | New private directories use 0700 and files/receipts use 0600 independently of umask. Reused owned destinations are tightened; unsafe ownership and hard links are rejected. | Multiple umasks, existing owned imports, untouched sources and rejected unsafe destinations |
| Failed exclusive create deleted another importer's file | Cleanup runs only after this invocation successfully creates the destination. | A competing creator, source-open failure, copy failure and changed source hash |
| Loaded foreign modules changed evaluation behavior | Check all loaded bundled module names, exact source files and package search paths on activation; never replace loaded modules silently. | Foreign parser with altered tolerance, Runtime verifier, mixed namespaces, valid namespace and repeat activation |
| Candidate archival failure appeared as PASS | Inspect `candidate_error.json`, retain the original client status and report the candidate failure as INCOMPLETE without a usable score. | Actual score-identity mismatch, existing score, CLI nonzero validation and unchanged original files |
| One malformed result crashed a batch summary | Validate nested types and normalize JSON parsing failures at the file boundary; mark the affected run INVALID while retaining the others. | Malformed evidence/provenance/metadata, huge numeric Q, excessive JSON nesting and mixed valid/invalid runs |
| Source archive omitted verification material | Include tests, fixtures, tools and reviewed reports in sdist; supply test build dependencies; exclude compiled local caches from wheels. | Build, extract and run the previously missing checks; build a wheel with a synthetic stale cache and verify its exclusion |

The canonical action executors and process/predicate algorithms are unchanged.
The HTTP redirect change is an intentional transport security correction; it is
recorded in the Bench import manifest. Rejection of incomplete replay evidence is
a correction at the public input boundary, not a relaxed or rewritten verifier.

`score-record-v1` has no per-event predicate truth for `history_program`. Such
records now fail explicitly instead of using only initial truth to approximate
the history. The original Live history verifier remains present. This CPU format
must be extended with complete evidence before that kind of history can be
replayed through it.

For the latest test, wheel and source-archive evidence, see
[Validation](VALIDATION.md). The previous rc1 logs remain available as historical
evidence; they are not presented as proof of these fixes. The rc2 source tree,
wheel and source archive use version `0.1.0rc2`.

Live acceptance, author licensing, external asset permissions and historical
release gates are unchanged by these fixes. Other task families still have only
the catalog/version-audit scope described in the support matrix.
