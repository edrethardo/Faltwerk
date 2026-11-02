# Faltwerk

**Faltwerk** is a codec for 4D Gaussian Splatting: it turns a dynamic, multi-view Gaussian-splat reconstruction into a compact bitstream (the Faltwerk format, `.falt`) that a WebGPU player streams and plays in the browser with free camera movement. This repository holds the codec, the training that produces its input, the standard recipe that runs both on a new scene, the viewer and the tests.

The method folds space and time into a small number of curves per splat and then removes the remaining redundancy:

- **Curve representation.** Every time-varying attribute (position, rotation, scale, opacity, base colour) is a curve with a handful of knots per 32-frame section, trained jointly per section instead of per frame. Every section can be keyed, so random access stays possible.
- **Colour codebooks.** The higher SH bands are vector-quantised per key interval, with a small SH degree per splat.
- **Spatial ordering** of newborn splats (Morton order) plus optional temporal / k-NN prediction of knot deltas.
- **Rules instead of hand values.** The standard recipe derives quantisation steps, splat budgets, near planes and sample spacing from the data (camera footprint, teacher statistics, mesh geometry); one constant set per source.
- **Three sources.** Camera images of a forward-facing rig (LLFF-style, e.g. the Neural 3D Video benchmark), a matted multi-camera rig with a per-frame teacher model, or animated meshes from 3D software (splats generated directly from the geometry, no training).
- **WebGPU player** (TypeScript, bit-exact with the Python reference decoder): curves evaluated on the GPU, progressive section streaming, a side-by-side comparison viewer and a WebXR mode.

## Repository layout

```
codec/               Python reference encoder/decoder of the .falt bitstream (rANS, curves, codebooks, prediction)
train/               training that produces the curves the codec encodes
  e3_track.py        per-frame tracking model, renderer and metrics (gsplat)
  e7_curves_train.py section-wise curve training (the trainer of the images/llff source)
  llff/              forward-facing rigs (N3DV-style): frame/camera prep, COLMAP init, init fit, training, decoded eval
  rig/, teacher/     matted rigs with a per-frame teacher: stage 1 tracking, stage 2 curve fit + fine-tune, eval
  mesh/              animated meshes -> splats (mesh2gs.py)
  recipe/            standard-recipe rules, the one encoder for every stream, splat budget from a rate target
  ext/, flow/        trainer extensions (adaptive knots, optical-flow guidance)
tools/               faltwerk_recipe.py (entry point), recipe_standard.json (constants), synthetic test scene, secret scan
web/                 WebGPU decoder, player and comparison viewer (see web/README.md)
tests/               Python tests (bit-exact round trips, golden digests, recipe rules, mesh recipe end to end)
docs/                format-falt.md (bitstream, German), recipe.md (standard recipe and data layouts)
```

## Quick start

Codec and tests (CPU, Python >= 3.10 with numpy and zstandard):

```bash
pip install numpy zstandard pytest
python -m pytest -q tests                         # bit-exact round trips + a mesh-recipe run on a synthetic scene
python -m codec.measure --help                    # encode a trained run directly
```

Viewer (Node >= 23):

```bash
cd web && npm ci && npm test && npm run build
python3 serve_viewer.py <data dir>                # http://127.0.0.1:8790, see web/README.md
```

## Standard recipe: scene -> Faltwerk stream

One entry point per source, no per-scene tuning values. A scene file holds only facts (data paths, frame count, camera kind, whether a teacher exists); the rules derive everything else. Details, data layouts and the list of rules: [`docs/recipe.md`](docs/recipe.md).

```bash
# animated meshes (runs on a CPU; synthetic example scene)
python3 tools/make_synthetic_mesh.py work/synthetic
python3 tools/faltwerk_recipe.py --source mesh --scene work/synthetic/scene.json --out work/runs/synthetic --run

# camera images (GPU with CUDA, torch and gsplat); N3DV must be downloaded separately, see docs/recipe.md
python3 tools/faltwerk_recipe.py --source images --scene train/recipe/scenes/n3dv_flame_steak.json --out work/runs/flame
bash work/runs/flame/recipe.sh
```

`recipe.sh` spells out every command of the chain and `recipe.json` records every derived value. `--rate-target <Mbit/s>` sets a delivery rate (splats are pruned to fit it and the kept splats are fine-tuned again); without it nothing is pruned.

The GPU stages need PyTorch, [gsplat](https://github.com/nerfstudio-project/gsplat) 1.5 and a CUDA GPU (24 GB recommended); see `requirements-train.txt`.

## Status

Research code, moving quickly. The bitstream format is documented and versioned (`docs/format-falt.md`, in German); the TypeScript decoder is bit-exact with the Python reference on synthetic fixtures. Older streams written as `.csgs` (magic `CSGS`) are still read.

## Credits / Contributors

Faltwerk by Aaron Nowak ([@edrethardo](https://github.com/edrethardo)) with James Kane ([@jameskane05](https://github.com/jameskane05)) – co-developer, testing and feedback. See [`CONTRIBUTORS.md`](CONTRIBUTORS.md).

## License

MIT for the code, see `LICENSE`. Third-party datasets and models are **not** part of this repository and not covered by the MIT license, see `THIRD_PARTY.md`.

## Citation

See `CITATION.cff`.
