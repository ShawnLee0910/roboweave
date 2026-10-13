# External assets and private evaluation inputs

The repository contains public task metadata, code and small CPU fixtures. It does not contain BEHAVIOR/OmniGibson scene or object assets, prepared simulator state, private goals, Oracle evidence or raw trajectories. A public catalog is enough to inspect task definitions and run protocol replay; real evaluation requires authorized external resources.

Obtain the simulator and dataset through their maintainers' installation and license process described in [INSTALLATION.md](INSTALLATION.md). Dataset access does not automatically authorize redistribution of derived preparation inputs. This candidate has no public download endpoint for the historical private preparation package. Obtain that package from its owner under applicable terms; do not reconstruct hidden fields from the catalog or treat missing assets as a successful run.

## Exact local import

The installed manifest `esi_benchmark/data/assets/l0-english-v2.json` is metadata-only. It lists logical keys, relative source identity, SHA256, byte counts, roles and task membership. Its 715 source runtime references normalize to 713 unique files (97,073,009 bytes); the exact release `tasks.json` is one additional required file. CPU compilation exposed 55 additional files (2,246,417 bytes) used by the frozen loader's implicit perception-frozen and scene-registration lookups. Their metadata and hashes are included as `supplemental_closure`, for **769 total imported files**. The 310 historical evidence entries in the source dependency manifest are excluded. Full simulator asset trees remain separate and are not covered by these 713 hashes.

Set `assets.root` in an independent configuration to an empty directory owned by this project. The importer creates `.esi-benchmark-assets.json`, `import_receipt.json` and immutable copies under `files/<source_root>/<source_path>`. It refuses unknown keys, traversal, symlink paths, hash/size mismatches, source/destination overlap, unowned nonempty destinations and overwriting different imported bytes. All selected source files pass validation before any destination content is created. It never downloads, extracts archives or writes to a supplied source tree.

On the supported Linux platform, import roots and newly created directories use owner-only `0700` permissions; copied files, markers and receipts use `0600`, independent of the process umask. Reusing an import with a matching ownership marker tightens the root and selected files to those modes. Destinations must belong to the current OS user; hard-linked or nonregular destination files are rejected before their permissions or contents can be changed. Existing parent directories and source permissions are left unchanged. Exclusive-create conflicts preserve the competing file; failed copies remove only a destination successfully created by that call.

There are three supported import forms:

```bash
# A directory supplied by the authorized package owner, with files/<root>/<path> layout.
esi-benchmark assets prepare --config configs/live.example.json --bundle /path/to/authorized-bundle

# Map every logical source root to a relocated local copy (repeat this option as needed).
esi-benchmark assets prepare --config configs/live.example.json \
  --source-root action20-capability-benchmark=/path/to/authorized-source-copy

# Map each asset key to its exact local file; useful when the owner reorganized the files.
esi-benchmark assets prepare --config configs/live.example.json --mapping /path/to/local-mapping.json

esi-benchmark assets verify --config configs/live.example.json
```

The one-root example is illustrative: a complete import needs every root listed in the manifest, unless a bundle or explicit per-file mapping supplies the remaining files. A mapping is a JSON object such as `{"l0-release-tasks": "/path/to/authorized/tasks.json", "<other-manifest-key>": "/path/to/authorized/file"}`. Put real local paths in your own untracked configuration or mapping file; the public manifest does not require the original machine's absolute paths.

Python callers can prepare or check one task by passing `task=get_task(task_id, "L0")` to `prepare_assets` or `check_assets`. CLI verification checks the entire frozen profile. Missing and invalid results report logical keys and never dump private file contents.

## Per-run materialization

`resolve_task_inputs(config, task)` verifies the imported originals and returns private paths plus the model's public instruction/budget binding. Its full return object is an evaluator input, not a model observation.

`materialize_task_inputs(config, task, new_output_directory)` creates private per-run copies. It selects the exact task record from aggregate TaskSpecs, registry, package, handoff and preparation index, and rewrites paths into that new directory. Dependent file SHA256 references and the handoff-index binding are refreshed. Every input's original and derived hash, each path/hash rewrite and aggregate task selection appears in `materialization_receipt.json`.

The new materialization root and directories use `0700`; inputs and the receipt use `0600`. An existing output directory is still refused, and no imported source file is chmodded or rewritten.

The task's selected initial state, objects, goal, process/history contract, seed, instruction and budget retain their frozen meanings. The importer never changes the original preparation package. Runtime provenance that refers to older source versions remains historical evidence; it is not relabeled as execution under the candidate runtime.

Private input directories and materialization receipts may contain hidden evaluation state and historical path metadata. Keep `assets/local/` and per-run `private/` directories out of any public release or shared trajectory export. Public trajectories must use the observation boundary and artifact validation described in the trajectory documentation.

## What CPU checks establish

The release preparation rehashed all 715 source runtime references, including their aliases, and tested the importer with small distributable fixtures. The real-input CPU report records per-task materialization, semantic comparison, handoff compilation, preparation validation and registration/TaskSpec checks. These checks can detect missing files and changed contracts; they cannot establish actual reset fidelity, render correctness, collision behavior, successful actions or official Live completion. See [TASK_ASSET_CPU_VALIDATION.md](../reports/TASK_ASSET_CPU_VALIDATION.md) and the independent Live validation plan.
