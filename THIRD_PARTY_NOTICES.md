# Third-party notices and external dependencies

This inventory distinguishes ESI source provenance from externally installed
software and assets. It is not a replacement for the actual dependency licenses.
No OmniGibson dataset, decryption key, NVIDIA binary/material module, or CuRobo source is bundled
by this candidate.

| Component | Attribution and terms | Candidate relationship |
| --- | --- | --- |
| ESI Bench, Action/Runtime/Oracle, task contracts | Original ESI authors; project license not found in the inspected source roots | Selected source and contracts are preserved with file-level provenance. Public distribution needs owner authorization; see [license status](LICENSE_STATUS.md). |
| OmniGibson 3.9.0 | Copyright (c) 2023 Stanford Vision and Learning Group; MIT | External dependency. [License at the pinned BEHAVIOR snapshot](https://github.com/StanfordVL/BEHAVIOR-1K/blob/802356a83d210a42bb6bfde93544bda4e33124e1/OmniGibson/LICENSE). |
| BDDL | Copyright (c) 2021 Stanford Vision and Learning Lab; MIT | External dependency installed from the same BEHAVIOR snapshot. [License](https://github.com/StanfordVL/BEHAVIOR-1K/blob/802356a83d210a42bb6bfde93544bda4e33124e1/bddl3/LICENSE). |
| BEHAVIOR Data Bundle | Independent data license with third-party content restrictions | Download from the official provider after accepting its terms. Do not redistribute data or keys. [Pinned installer and data agreement](https://github.com/StanfordVL/BEHAVIOR-1K/blob/802356a83d210a42bb6bfde93544bda4e33124e1/setup.sh). |
| NVIDIA Isaac Sim 5.1.0 | NVIDIA software license agreement | External installation with separate agreement. [NVIDIA agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/). |
| `vray_maps.mdl`, `vray_materials.mdl` | NVIDIA 2023 copyright headers; use/distribution requires an express NVIDIA license | Required external material modules. Not redistributed. Obtain legally from an authorized installation/distributor; [Installation](docs/INSTALLATION.md) lists exact hashes and the `runtime.mdl_root` configuration. |
| CuRobo | Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES; NVIDIA License | External dependency pinned to `78612f45cef52c3fa0298de243a54cd7ca614414`. Its use restriction is non-commercial research or evaluation; it is not covered by OmniGibson's MIT license. [Pinned license](https://github.com/StanfordVL/curobo/blob/78612f45cef52c3fa0298de243a54cd7ca614414/LICENSE). |
| Python packages installed by pip | Respective package authors and licenses | Installed separately; consult installed distribution metadata and each package's notices. The project does not relicense these dependencies. |

The pinned BEHAVIOR repository is
`802356a83d210a42bb6bfde93544bda4e33124e1`. Its code licenses and data agreement
are separate. The data agreement permits non-commercial academic research within
OmniGibson and prohibits redistribution of data and keys. Consult the linked
agreement itself before downloading, using, or sharing any derived material.

The CPU fixture uses synthetic images and synthetic responses created for the
candidate. It does not contain rendered BEHAVIOR scenes or copied historical
trajectory images. Synthetic fixture outputs must not be reported as simulator
results.

Attribution to external research software does not imply endorsement of this
candidate. Preserve all applicable notices when building a future authorized
release.
