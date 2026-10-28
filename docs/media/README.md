# Media in this folder

These files are **not** covered by the repository's MIT license.

## `faltwerk_queen_demo.gif`

Work-in-progress comparison on one scene (64-frame excerpt, front view, then a slow orbit beyond the capture rig).
Three rows, all rendered with the Faltwerk WebGPU viewer:

1. **QUEEN (published model)** – the released pre-trained model, decoded and rendered unchanged.
2. **QUEEN → Faltwerk encoder** – QUEEN's Gaussians re-fitted to Faltwerk's per-section curves and encoded with the
   Faltwerk encoder, no retraining.
3. **QUEEN + fine-tune → Faltwerk encoder** – as (2), plus a short fine-tune (3000 iterations per 32-frame section) on
   the original frames.

Rates are all bytes needed for the 64 frames at 30 fps (QUEEN: 157 Mbit/s without its dense keyframe). This is a
research teaser, not a benchmark; quality metrics will be published once verified.

**License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)** (adapted material, ShareAlike).
Research and evaluation only, non-commercial.

**Attribution and changes**

- Scene: *Neural 3D Video* dataset, scene "sear_steak" – T. Li et al., "Neural 3D Video Synthesis from Multi-View
  Video", CVPR 2022, <https://github.com/facebookresearch/Neural_3D_Video>, licensed
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Not original frames: all rows are renderings of
  models trained on the dataset.
- Model: *QUEEN* pre-trained N3DV model (release v1.0-compressed) – S. Girish et al., "QUEEN: QUantized Efficient
  ENcoding of Dynamic Gaussians for Streaming Free-viewpoint Videos", NeurIPS 2024,
  <https://github.com/NVlabs/queen>, licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  **Modified:** rows 2 and 3 are re-fitted to a different representation and re-encoded; row 3 is additionally
  fine-tuned. Row 1 is QUEEN's model rendered with our renderer.
- No endorsement by the dataset or model authors, NVIDIA or Meta is implied. The licensed material is provided "as is"
  by its licensors (see the respective license texts for the disclaimer of warranties).
- No QUEEN code is part of this repository; the dataset and the models are not redistributed here.

If you appear in this material or hold rights to it and want it removed, please open an issue.
