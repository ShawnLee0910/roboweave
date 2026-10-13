# Installation and environment preparation

CPU inspection/replay and simulator execution have separate dependencies. Install
them in new environments. Do not update an environment used by an existing
evaluation service.

For CPU usage, Python 3.11 or newer is required. NumPy, Pillow and PyYAML are
declared in `pyproject.toml`; the optional `test` extra adds pytest and the imported
tests' image/transport dependencies. No CUDA,
OmniGibson import, API key or model request is needed for the replay example.

From the candidate root:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install '.[test]'
esi-benchmark doctor
python -m pytest
```

To review the built distribution in a second environment:

```bash
python -m pip wheel --no-deps --wheel-dir dist .
python3.11 -m venv .venv-wheel
. .venv-wheel/bin/activate
python -m pip install dist/esi_benchmark-0.1.0rc1-py3-none-any.whl
esi-benchmark --help
esi-benchmark doctor
```

The wheel includes the vendored code and JSON/text resources, so it must work
without the original source trees on `PYTHONPATH`. Review the actual build and
installation checks in [Validation](VALIDATION.md). Package version ranges are
compatibility declarations, not a fully locked transitive environment. Preserve
`python -m pip freeze`, wheel hashes and Python/platform versions when reproducing
the reported CPU environment.

The frozen Live dependency target is:

| Component | Target |
| --- | --- |
| BEHAVIOR source | `802356a83d210a42bb6bfde93544bda4e33124e1` |
| Python | 3.11 |
| OmniGibson | 3.9.0 |
| Isaac Sim | 5.1.0 |
| PyTorch / CUDA toolkit | 2.7.0 / 12.8 |
| CuRobo | `78612f45cef52c3fa0298de243a54cd7ca614414` |
| BEHAVIOR assets / robot assets | 3.9.0 / 3.8.2 |

These pins come from the official snapshot's
[installer](https://github.com/StanfordVL/BEHAVIOR-1K/blob/802356a83d210a42bb6bfde93544bda4e33124e1/setup.sh),
[OmniGibson package declaration](https://github.com/StanfordVL/BEHAVIOR-1K/blob/802356a83d210a42bb6bfde93544bda4e33124e1/OmniGibson/setup.py)
and [asset constants](https://github.com/StanfordVL/BEHAVIOR-1K/blob/802356a83d210a42bb6bfde93544bda4e33124e1/OmniGibson/omnigibson/utils/asset_utils.py).
They describe the intended dependency target; this candidate has not installed
or exercised that stack during preparation. Do not substitute the current
BEHAVIOR default branch or a newer release and claim the same frozen experiment.

Use Linux and an NVIDIA device supported by the selected Isaac Sim release.
Consult the [official installation requirements](https://behavior.stanford.edu/getting_started/installation.html)
for driver, operating-system and hardware prerequisites; upstream minimum memory
figures do not guarantee enough memory for every ESI task. Record the real GPU
UUID, driver and available resources in the validation receipt.

On the independent Live validation machine, prepare a new checkout and a new
environment. The following downloads/installations are future setup instructions,
not operations performed while preparing this candidate:

```bash
mkdir -p external environments
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git external/BEHAVIOR-1K
git -C external/BEHAVIOR-1K checkout --detach 802356a83d210a42bb6bfde93544bda4e33124e1
conda create --prefix "$PWD/environments/live" python=3.11 pip 'setuptools>=71,<81' wheel -c conda-forge
conda activate "$PWD/environments/live"
cd external/BEHAVIOR-1K
./setup.sh --omnigibson --bddl --primitives --dataset --cuda-version 12.8
cd ../..
python -m pip install '.[live]'
```

The `live` extra installs `websockets>=13,<16` and `msgpack>=1,<2` for the scene
transport. It does not install the simulator, CUDA, private task inputs or licensed
material modules.

Provide a matching CUDA toolkit/compiler before installing primitives. Review
the official installer and accept its terms yourself; the example does not
automatically accept licenses. The installer modifies its active environment,
which is why it must be the new isolated one. It may perform first-import setup;
do not run it in an existing resident's environment.

Asset acquisition is separate from task preparation. The official downloader
supplies licensed simulator datasets; ESI's manifest supplies the exact
preparation dependencies for the frozen task. Point `runtime.data_root` at the
actual official dataset location, set `runtime.behavior_root` to the pinned
checkout, and import authorized task resources using [Assets](ASSETS.md). Do not
copy a decryption key or a deployed machine's asset/cache tree into a release.

The runtime's material wrapper also imports two separately licensed NVIDIA
modules: `vray_maps.mdl` and `vray_materials.mdl`. Obtain the exact files from an
authorized NVIDIA/Omniverse installation or its authorized distributor under an
applicable license; this candidate supplies no public download or redistribution
grant. Put them in a private directory and configure `runtime.mdl_root`. The
preflight requires these exact SHA256 values:

| File | SHA256 |
| --- | --- |
| `vray_maps.mdl` | `279d9e8ded03c2db928033c843b40a04b104f3fd6fc7ccd284aadc6459a2aa1c` |
| `vray_materials.mdl` | `eac4af88c7e39cdba091bf2e76c20adc3f9c9cd65d6d044479065537f70de696` |

Only an explicit Live launch copies those checked files into that run's private
workspace. They are not in the repository or wheel. Their source headers require
an express NVIDIA license for use/distribution; see
[Third-party notices](../THIRD_PARTY_NOTICES.md). If you cannot legally obtain the
files, CPU replay remains available but the frozen Live path is unavailable.

Complete the paths in `configs/live.example.json` in a private local copy.
`runtime.python` identifies the new environment; `workspace` is a new child of
`run.output_root` owned by this run. Set `toolchain_root` to the CUDA 12.8 prefix
containing `bin/nvcc`, and `overlay_root` to the new environment prefix containing
`bin/python`. A host loader shim is optional; omit `host_dlsym_shim` on a standard
installation. If the NVIDIA Vulkan ICD is outside the standard locations, set
`runtime.vulkan_icd`. See [Runtime](RUNTIME.md) for the loader checks.
The template is deliberately invalid until real paths, ownership and explicit
terms acceptance are supplied. The first CPU doctor check does not certify those
external paths or allocate a GPU.

Before any launch, run configuration/resource validation and the Live dry-run
from [Quickstart](QUICKSTART.md). Keep AppData and outputs independent for each
worker, keep the one-load budget for the initial check, and follow the
[Live acceptance plan](LIVE_VALIDATION.md). A dependency install, a passed dry-run
and a successful simulator episode are three different evidence levels.

Licensing conditions are listed in [Third-party notices](../THIRD_PARTY_NOTICES.md).
ESI has not assigned a project license in this candidate; installation instructions
do not authorize redistribution.
