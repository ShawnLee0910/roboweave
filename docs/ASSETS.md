# External assets and private evaluation inputs

The repository ships public task metadata, source code and small synthetic CPU
fixtures. Real scene/object assets, prepared simulator state, hidden goals,
Oracle evidence and raw trajectories are not distributed. A public catalog can
support inspection and protocol replay; execution additionally requires the
selected profile's authorized private resources.

Obtain the simulator and dataset under their maintainers' terms, following
[Installation](INSTALLATION.md). Dataset access does not authorize redistribution
of ESI preparation inputs. There is no public download endpoint for these
historical private packages in this candidate. Obtain the exact package from
its owner; missing private resources are not replaced by an approximate reset.

## Per-profile manifests

Every executable profile has a metadata-only manifest at
`esi_benchmark/data/assets/<profile>.json`. It records logical keys, source-relative
identities, byte counts, SHA256, task membership and relocation bindings.

| Profile | Manifest file records |
| --- | ---: |
| `l0-english-v2` | 769 |
| `l1-a30` | 211 |
| `l1-m30-public-v2` | 303 |
| `l1-v30` | 325 |
| `l1-p-inside5` | 52 |
| `l1-p-ontop5` | 52 |
| `l1-p-additions20` | 202 |
| `l2-collection37` | 353 |

Counts are per profile and may include files shared with another profile; they
are not counts of separately licensed simulator assets. The L0 manifest includes
its previously audited implicit-loader dependency closure. New family manifests
also retain explicit relative-path and dependent-hash bindings; a hash cannot be
updated merely to admit an unmatched resource.

Set `assets.root` to an independent directory for the selected profile. The
importer writes an ownership marker, `import_receipt.json` and checked copies
under `files/<source_root>/<source_path>`. A destination owned by another profile
is rejected. It also rejects traversal, symlink paths, hash/size mismatches,
source/destination overlap, unowned nonempty directories and conflicting bytes.
All selected sources are validated before import begins. Sources are never
modified; the importer neither downloads nor extracts archives.
Concurrent imports lock the destination directory before reading its ownership
marker, and serialize profile claims, copies and receipt updates.

On Linux, private directories use `0700`; files, markers and receipts use `0600`
regardless of umask. Existing source/parent permissions are not changed. Reusing
a matching import preserves its ownership checks and tightens selected private
permissions. Nonregular or hard-linked destination files are rejected. An
exclusive-create collision preserves the other writer's file.

## Import and validate

First copy and complete `configs/live.example.json` as described in
[Quickstart](QUICKSTART.md). Select matching `profile`, `run.family` and
`run.task_id`, then use one of the three import forms:

```bash
# An authorized directory with files/<root>/<path> layout.
esi-benchmark assets prepare --config configs/live.local.json --bundle /path/to/authorized-bundle

# Map all logical source roots to relocated local directories; repeat as needed.
esi-benchmark assets prepare --config configs/live.local.json \
  --source-root action20-capability-benchmark=/path/to/authorized-source-copy

# Map every logical asset key to its exact local file.
esi-benchmark assets prepare --config configs/live.local.json --mapping /path/to/local-mapping.json

esi-benchmark assets verify --config configs/live.local.json
```

The single-root command illustrates the syntax, not a complete multi-root
package. Supply every root/key required by the selected manifest, or use a
complete bundle. A mapping is a JSON object from manifest keys to absolute local
filenames. Keep real configurations, mappings and imported files out of public
release artifacts. Original machine paths are not required at runtime.

CLI preparation and verification operate on the complete selected profile.
Python callers can pass `task=get_task(task_id, family)` to `prepare_assets` or
`check_assets` for one task. Missing/invalid reports identify logical keys rather
than printing private contents.

## Per-run materialization and native admission

`resolve_task_inputs(config, task)` validates imported originals and returns
private evaluator paths plus a bounded public model binding. The complete return
object must never be sent to a model.

`materialize_task_inputs(config, task, new_output_directory)` selects the exact
records from aggregate TaskSpecs, package, registry, handoff, preparation index
or V binding, and relocates them into a new private directory. It updates file
and canonical-JSON hashes only where the manifest declares a dependent binding.
`materialization_receipt.json` records original/derived hashes, task selections
and every path/hash rewrite. Directories use `0700`, files use `0600`, and an
existing materialization directory is refused.
Each file's exact read bytes are checked again before transformation, so a change
after the initial asset check cannot be recorded as a frozen source.

The selected profile's isolated worker then invokes its original metadata
validators. A/M retain preparation and process contracts; V additionally validates
collection identity, D1 intervention and inherited preparation; P invokes its
physical admission. L2 retains approved public English separately from source
TaskSpec text. Its 14 Physical entries use native fresh-world construction,
while the 23 Symbolic entries use prepared routes. Relocation does not silently
convert one route or action mode into another.
Before native admission, the public inspection entry checks the model handle,
instruction, budgets and instruction provenance against the installed catalog,
then verifies the selected native instruction and V collection identity.

Task instructions, initial-state meaning, objects, seed, goal/process contracts,
budgets and original version evidence remain frozen. Private source inputs are
not rewritten, and older preparation provenance is not relabelled as a candidate
execution. Materialization receipts themselves may contain private path metadata;
keep `assets/local/` and each run's `private/` tree out of public exports.

## What validation establishes

Hash checks, synthetic importer tests, private-input materialization and native
metadata admission are separate from simulator proof. They can reject missing
or changed contracts but cannot establish reset fidelity, rendering, collision
behavior or Live completion. [Validation](VALIDATION.md) records actual rc3
coverage. The retained [L0 input report](../reports/TASK_ASSET_CPU_VALIDATION.md)
is historical evidence for that profile, not an all-family rc3 receipt. Independent
environment and Live reproduction were deferred for this preparation.
