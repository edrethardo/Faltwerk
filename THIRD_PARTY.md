# Third-party content

This repository contains code plus one demo GIF (`docs/media/`). No scene data and no trained models from third parties are redistributed here. The GIF shows renderings derived from third-party material and is licensed **CC BY-NC-SA 4.0**, not MIT; attribution and the list of changes are in [`docs/media/README.md`](docs/media/README.md).

| Source | License | How it is used |
|---|---|---|
| **Neural 3D Video** - Li et al., CVPR 2022 | CC BY-NC 4.0 | Public benchmark for research evaluation. Downloaded separately by the user; only aggregate metrics are quoted. Where a still from the dataset appears in a write-up it is credited to the authors and marked "research evaluation only". The demo GIF in `docs/media/` shows renderings of models trained on the scene "sear_steak" (attribution there). |
| Vendor demo content (a streaming service's demo clip) | evaluation only | Used locally to compare rate and visual quality. **Not** redistributed and not shown in public material. |
| **QUEEN** - Girish et al., NeurIPS 2024 | CC BY-NC-SA 4.0 | Published numbers quoted with citation; released models not redistributed. The demo GIF in `docs/media/` shows the released sear_steak model and two modified versions of it (re-fitted + re-encoded, and fine-tuned); it is shared under CC BY-NC-SA 4.0 with attribution. No QUEEN code is included. |
| **3DGStream** - Sun et al., CVPR 2024 | MIT | Published numbers quoted with citation. |
| **4DGC** - Hu et al., CVPR 2025 | see paper | Published numbers quoted with citation. |
| **HiCoM** - Gao et al., NeurIPS 2024 | see repository | Published numbers quoted with citation. |

The MIT license of this repository covers our own source code only. Anyone running it on third-party data must comply with that data's license separately.

## Prior art and conventions (no third-party code included)

All source code in this repository was written for Faltwerk. It follows published methods and conventions; no source files or code fragments from the projects below are copied or vendored.

| Work | License of their code | What we follow |
|---|---|---|
| **3D Gaussian Splatting** - Kerbl et al., SIGGRAPH 2023 | Gaussian-Splatting License (Inria/MPII, non-commercial) | The splatting method itself (EWA projection of 3D Gaussians, SH colour, front-to-back alpha blending), re-implemented from the paper's equations in our own WGSL renderer. |
| **gsplat** - Ye et al., nerfstudio | Apache-2.0 | Attribute conventions (wxyz quaternions, log scales, logit opacity, SH layout, near plane, alpha threshold 1/255) so that our renderer matches gsplat renders. gsplat is a separate dependency of the (non-public) training pipeline, not part of this repository. |
| **ryg_rans** - F. Giesen | public domain | Byte-wise rANS state layout (`codec/rans.py`), re-implemented in Python/TypeScript. |
| Real spherical-harmonics constants | - | Standard mathematical normalisation constants. |
