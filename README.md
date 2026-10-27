# Faltwerk

**Faltwerk** is a codec for 4D Gaussian Splatting: it turns a dynamic, multi-view Gaussian-splat reconstruction into a compact bitstream that a WebGPU decoder can stream and play in a browser, with free camera movement.

The method (`CurveStream-GS`) folds space and time into a small number of spline curves per splat and then removes the remaining redundancy:

- **Curve representation.** Every time-varying attribute (position, rotation, scale, opacity, base colour) is a piecewise curve with a handful of knots per 32-frame section, trained jointly per section instead of per frame. Every section is keyed, so random access stays possible.
- **Colour codebooks.** The higher SH bands (about 40 % of the naive bitrate) are vector-quantised per key interval, with a small SH degree per splat.
- **Spatial ordering** of newborn splats (Morton order) plus optional temporal / k-NN prediction of knot deltas.
- **Background alpha** penalty: stops splats from living "for free" over a known background and showing up as a fringe in front of hair and faces.
- **Flow guidance:** the ground-truth optical flow of the training views regularises the projected motion of the splats, cutting temporal flicker at edges without softening the image.
- **Matte and near-plane pruning** during training removes ghost splats in front of the camera.
- **Quality-driven rate control:** encoder parameters are chosen per key interval so that every section reaches a chosen quality step (four steps, from "transparent" to compact).

## Repository layout

```
codec/               Python reference encoder/decoder for the .csgs bitstream (rANS, curves, codebooks)
web/                 TypeScript/WebGPU decoder and the side-by-side comparison viewer
  src/               decoder (csgs, rans, bspline, knn, splq), renderer, camera, i18n
  test/              node tests with synthetic bitstream fixtures (no scene data)
  bench/             headless browser benchmarks
tests/               Python tests (bit-exact round trips, golden digests)
docs/format-csgs.md  the .csgs container format (versions 1-6, flags)
tools/               small utilities
```

## Try it

```bash
python -m codec.measure --help      # encode a trained run
python -m pytest tests -q           # bit-exact round-trip tests

cd web && npm ci && npm test && npm run build
python3 serve_viewer.py             # http://127.0.0.1:8811/compare.html
```

WebGPU is required (Chrome/Edge, Safari 18+, or a recent browser with WebGPU enabled).

## Status

Research code, moving quickly. The bitstream format is documented and versioned; the TypeScript decoder is bit-exact with the Python reference on synthetic fixtures. The viewer plays a scene from `.csgs` sections with a shared camera across panes, a timeline, a difference view against a reference image and per-pane quality labels.

## License

MIT for the code, see `LICENSE`. Third-party datasets and models are **not** part of this repository and not covered by that license, see `THIRD_PARTY.md`.

## Citation

See `CITATION.cff`.
