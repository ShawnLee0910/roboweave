# RoboWeave naming and compatibility

The project is **RoboWeave**. Starting with `0.1.0rc5`, the distribution,
command and Python package are all `roboweave`. Install from this repository or
its built wheel; these instructions do not imply a PyPI publication.

```bash
roboweave doctor
python -m roboweave --help
```

```python
from roboweave.tasks import list_tasks
from roboweave.scoring import score_record
```

Use a fresh virtual environment as shown in [Installation](INSTALLATION.md).
When upgrading an existing candidate-only environment, remove the old
distribution before installing this checkout:

```bash
python -m pip uninstall esi-benchmark
python -m pip install .
```

RoboWeave itself provides the compatibility imports. Do not keep both
distributions installed: they would both own the `esi_benchmark` package path.

## Existing users and recorded evidence

- The `esi-benchmark` command remains an alias for the same CLI.
- Imports from the old `esi_benchmark` public modules resolve to the same module
  objects as `roboweave`. The frozen sources' original profile and loader imports
  use this compatibility package, so their source bytes and hashes stay intact.
- New configs and outputs use `roboweave/...` schema identifiers. Config v1 and
  scoring-record v1/v2 also accept the corresponding `esi-benchmark/...` input
  identifiers. Normalization does not mutate the caller's scoring record.
- Both `roboweave/result-evidence-v2` and `esi-benchmark/result-evidence-v2` retain
  the same strict task binding, evidence hash, size and outcome checks. Old data
  is not downgraded to the permissive legacy format.
- Existing imported-asset ownership markers retain `.esi-benchmark-assets.json`
  so the same private directory remains protected against conflicting imports.

The model-key example uses `ROBOWEAVE_MODEL_API_KEY`. Existing configurations
continue to use their explicitly configured `model.api_key_env`. Native Runtime
environment variables such as `ESI_DATA_ROOT` are frozen integration contracts
and retain their names.

## Provenance

The source-bundle registry, task catalog, asset manifests, frozen source files,
fixed equivalence inputs and earlier reports retain their original contents.
Old provenance destinations beginning with `src/esi_benchmark/` refer to the
rc1–rc4 layout; the same frozen files now live under `src/roboweave/`.
The rc5 migration receipt records that mapping and verifies their hashes.

This migration changes names and accepts explicit compatibility aliases. Task
definitions, initial states, action and observation contracts, budgets and
scoring rules remain governed by the frozen profiles.
