# Quickstart

Start in the candidate repository root. The first workflow is CPU-only and can
run without an asset download or model account. For package requirements and
wheel installation, see [Installation](INSTALLATION.md).

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install '.[test]'
esi-benchmark --help
esi-benchmark doctor
```

Inspect a configuration and the frozen task catalog:

```bash
esi-benchmark config validate --config configs/replay.json
esi-benchmark tasks list --family L0
esi-benchmark tasks show am200v1_a001_b2_control --family L0
```

The catalog shows permitted public task metadata. It does not expose the private
initial state or verifier inputs. The 100-task L0 profile has a specific source
binding; another family's catalog entry is not a supported L0 run.

Run and inspect the included synthetic example:

```bash
esi-benchmark run --config configs/replay.json --mode replay --output outputs/quickstart
esi-benchmark results validate outputs/quickstart
esi-benchmark results summarize outputs/quickstart
```

Use an unused output directory. The fixture exercises a three-stage placement
Marker transaction followed by `done`, using synthetic RGB and scripted model
responses through the actual Bench episode loop. Its result records identify it
as replay. It cannot establish official task success, physics, scene recovery or
network provider compatibility. See [Trajectory format](TRAJECTORIES.md) for the
artifact layout and public/private fields.

The command-line help documents individual options:

```bash
esi-benchmark run --help
esi-benchmark assets prepare --help
esi-benchmark results validate --help
esi-benchmark score --help
python -m pytest
```

To replay a canonical verifier evidence record, use `esi-benchmark score RECORD`.
For a record that declares a process/history contract, use `esi-benchmark score
RECORD --process-output NEWDIR` to write its process audit to a fresh directory.
The input format and evidence limits are described in [Scoring](SCORING.md).
This evaluates recorded truth/history; it does not acquire simulator truth.

For a real task on separately allocated resources, first complete the
[external installation](INSTALLATION.md), [asset requirements](ASSETS.md) and
[Live validation preparation](LIVE_VALIDATION.md). Copy the safe template:

```bash
cp configs/live.example.json configs/live.local.json
```

Edit `configs/live.local.json` before using it. Configure the new environment's
Python, official BEHAVIOR checkout and data root, the authorized `mdl_root`, runtime toolchain paths, a new
workspace, an owned GPU UUID, a free port, and your explicit model endpoint and
model ID. Read the relevant third-party terms before setting `accept_eula`.
The shipped example deliberately has placeholder paths/identity and
`accept_eula: false`; it is not immediately runnable. Relative configuration paths
resolve from the configuration file's directory. A CLI `--output` path is supplied
for the current invocation; use an absolute path when invoking outside the repo.

Obtain the exact task-preparation bundle from a source authorized to distribute
it. The candidate does not download it automatically or regenerate an approximate
initial state. A bundle must match the documented manifest layout and file hashes:

```bash
esi-benchmark assets prepare --config configs/live.local.json --bundle /absolute/path/to/authorized-bundle
esi-benchmark assets verify --config configs/live.local.json
esi-benchmark config validate --config configs/live.local.json
esi-benchmark run --config configs/live.local.json --mode live --dry-run
```

For multiple source roots or an explicit file mapping, use `assets prepare
--source-root KEY=PATH` or `--mapping FILE` as described in [Assets](ASSETS.md).
Missing resources are expected until they have been supplied. A hash mismatch
must be resolved against the frozen package, not by editing the expected hash.

The model credential belongs in the environment variable named by
`model.api_key_env` (the template uses `ESI_MODEL_API_KEY`). Supply it through your
normal secret mechanism; do not put a real key in the JSON file or commit it.
Transport options and a provider's expected tool-call response are documented in
[Models](MODELS.md). Dry-run validation is not a provider request or Live proof.

Only after the independent machine, resources and model usage are ready, start
the owned run:

```bash
esi-benchmark run --config configs/live.local.json --mode live --output outputs/live-example
esi-benchmark results validate outputs/live-example
esi-benchmark results summarize outputs/live-example
```

The run preserves the frozen action/call budgets and verifier contract. The
declared `wall_clock_seconds: 3600` has no consumer in the source episode loop;
this candidate does not add a new 3600-second cutoff or reproduce the historical
outer scheduler's separate timeout. See [source limitations](../reports/SOURCE_LIMITATIONS.md).
Inspect final-goal
success, Q, process verdict and termination reason separately; `Q=1` is not a
universal success rule. Retain unsuccessful attempts and raw evidence. The
candidate's preparation did not execute this Live sequence, and completing one
task does not establish coverage of all 100 tasks.
