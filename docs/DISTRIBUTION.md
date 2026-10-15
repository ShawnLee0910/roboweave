# Distribution and information boundaries

This repository is a local candidate. Public release awaits an ESI license and
approval of task and evaluator content; see [LICENSE_STATUS](../LICENSE_STATUS.md).
Technical completeness and publication clearance are separate checks.

| Material | Local candidate treatment | Public-release boundary |
| --- | --- | --- |
| Public CLI, selected Bench sources, Runtime sources, tests and documentation | Included with source identity and hashes | Requires ESI rights-holder authorization; preserve authorship. |
| Public task catalog | Task identity, public instruction, scene identity, declared budget, action mode and version profile | Review the exact allowlist before release; do not infer a hidden-test split. |
| Evaluator implementation | Included so execution and scoring can be audited | Code may be released only under approved terms. Its availability does not authorize revealing hidden task inputs to a model. |
| Task preparation data | Imported separately through checked manifests where available | Initial states, goals, private scopes, poses and verifier inputs need a separate release decision. |
| Simulation assets and keys | External; never copied into this candidate | Obtain from the official provider under its own terms. |
| NVIDIA MDL material modules | External; exact hashes checked before private Live staging | `vray_maps.mdl` and `vray_materials.mdl` require an applicable express NVIDIA license; neither is bundled. |
| Historical trajectories and model credentials | Excluded from distribution | Bounded source inspection is not permission to export data. |
| CPU examples | Synthetic observations and scripted provider responses | Can exercise software contracts; they are not benchmark scores or real robot/simulator evidence. |
| Source audit records | Local source paths, revisions and hashes | Review operational identifiers before publishing; paths are provenance, not runtime dependencies. |

The model may receive the frozen public instruction, permitted current RGB
observation, the configured public history, Action20 schema and public action
feedback. It must not receive preparation files, Oracle plans, privileged object
state, hidden goals, verifier internals, or raw private trajectory metadata.
The public task projection and public observation builder enforce different
boundaries: a task row is not an observation, and a private episode ledger is not
a prompt.

The catalog is an **evaluation catalog**. The inspected final packages did not
establish an authoritative train/development/hidden-test partition. This candidate
does not manufacture one. Eight frozen profiles now provide executable integration
for the 257 catalog configurations, while the separate L2 proposal authoring
snapshot remains audit-only. Integration and CPU evidence do not grant a license
to distribute private resources or establish Live/official benchmark eligibility.

OmniGibson software and the BEHAVIOR Data Bundle use different licenses. The
official data agreement restricts use and redistribution, including keys. Use
the [official download process](https://behavior.stanford.edu/getting_started/installation.html)
and review the [agreement in the pinned installer](https://github.com/StanfordVL/BEHAVIOR-1K/blob/802356a83d210a42bb6bfde93544bda4e33124e1/setup.sh).
CuRobo also has its own non-commercial research/evaluation condition; see
[third-party notices](../THIRD_PARTY_NOTICES.md).

The default workflow excludes internal SSH registrations, resident scheduling,
service takeover, bulk history migration, image-upload proxies and credentials.
It does not export a training dataset. Keep provider secrets in the configured
environment variable, and review any real run's private artifacts before sharing
them. Provider error text and generated content may themselves contain sensitive
information even when configuration templates are safe.

For an authorized public release, build a fresh artifact from the reviewed
candidate file set; do not upload the working directory with its local virtual
environments, imported private resources, outputs or provenance containing
operational details. Review the built wheel and source archive as well as the
repository. A clean Git working tree alone is not a distribution review.

`tools/audit_candidate.py` scans the declared source/data/documentation trees and
the top-level README, build metadata and license-status files. Root-directory
symlinks, including dangling ones, are rejected. Findings report file locations
and types without printing matched credential values. The scan also lists
historical absolute paths in JSON/YAML and other distributed file types for
reachability review; a listed provenance path is not itself a runtime dependency.
