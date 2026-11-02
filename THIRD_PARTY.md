# Third-party content

This repository contains source code only. No scene data, no trained models and no renderings of third-party material are redistributed here.

## Datasets and models used at run time (not included)

| Source | License | How it is used |
|---|---|---|
| **Neural 3D Video** – Li et al., CVPR 2022 | CC BY-NC 4.0 | Public research benchmark. `train/llff/prep_frames.py` converts a scene you downloaded yourself into the GT layout of `docs/recipe.md`; only aggregate metrics may be quoted, with attribution and "research evaluation only". Frames, renders or models derived from it must not be redistributed under MIT. |
| **gsplat** – Ye et al., nerfstudio | Apache-2.0 | Rasteriser of the training and evaluation scripts (`train/`), installed separately (`requirements-train.txt`). |
| **PyTorch** | BSD-3-Clause | Training, installed separately. |
| **COLMAP** (Docker image `colmap/colmap`) | BSD-3-Clause | Sparse frame-0 points for the LLFF init (`train/llff/colmap_init.sh`), run separately. |
| **LPIPS** – Zhang et al., CVPR 2018 (`lpips` package) | BSD-2-Clause | Perceptual metric and optional fine-tune loss; weights are downloaded by the package. |
| **Depth Anything V2** (via `transformers`) | Small: Apache-2.0; Base/Large: CC BY-NC 4.0 | Optional monocular depth for the frame-0 fit (`train/llff/p15_prep.py mono`, not used by the standard recipe). Choose the model whose license fits your use (`--model`). |
| **OpenCV** (DIS optical flow) | Apache-2.0 | Optional GT optical flow for flow guidance (`train/flow/gt_flow.py`). |

The MIT license of this repository covers our own source code only. Anyone running it on third-party data or with third-party models must comply with their licenses separately.

## Prior art and conventions (no third-party code included)

All source code in this repository was written for Faltwerk. It follows published methods and conventions; no source files or code fragments from the projects below are copied or vendored.

| Work | License of their code | What we follow |
|---|---|---|
| **3D Gaussian Splatting** – Kerbl et al., SIGGRAPH 2023 | Gaussian-Splatting License (Inria/MPII, non-commercial) | The splatting method itself (EWA projection of 3D Gaussians, SH colour, front-to-back alpha blending), re-implemented from the paper's equations in our own WGSL renderer; the 3DGS PLY attribute layout read by `train/rig/rig_lib.py`. |
| **gsplat** – Ye et al., nerfstudio | Apache-2.0 | Attribute conventions (wxyz quaternions, log scales, logit opacity, SH layout, near plane, alpha threshold 1/255) so that our renderer matches gsplat renders. |
| **ryg_rans** – F. Giesen | public domain | Byte-wise rANS state layout (`codec/rans.py`), re-implemented in Python/TypeScript. |
| **LLFF / Neural 3D Video camera convention** – Mildenhall et al. 2019 | – | `poses_bounds.npy` conversion to OpenCV cameras in `train/llff/prep_frames.py`. |
| Real spherical-harmonics constants | – | Standard mathematical normalisation constants. |
