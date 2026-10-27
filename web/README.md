# Faltwerk viewer (WebGPU)

The viewer decodes `.csgs` bitstreams in the browser and plays them as 2-4 panes with **one shared camera** and **one timeline**, so two codecs or two quality steps can be compared side by side.

## Build and run

```bash
npm ci
npm test          # node tests: bit-exact decoder against the Python reference (synthetic fixtures)
npm run build     # bundles src/ into dist/
python3 serve_viewer.py [root] [port]      # serves dist/ + scene data, default http://127.0.0.1:8811
```

`serve_viewer.py` serves a static directory and writes `manifest.json` from `recipe.json` with `bench/make_viewer_data.py`.

## Scene recipe

`viewer.recipe.example.json` shows the shape: scenes with a camera file, a frame count and one or more panes (`csgs` for our bitstream, `splq` for quantised splat dumps of other methods), plus optional per-scene parameters such as the near plane.

URL parameters: `scene=<id>`, `panes=<id,id>`, `snap=<camera>`, `frame=<n>`, `pause`, `dpr=<max>`, `lang=de|en`.

## Files

| File | Purpose |
|---|---|
| `src/csgs.ts` | `.csgs` container decoder (versions 1-6, flags SN_VQ / pred / follow / rescale / B-spline) |
| `src/rans.ts` | range decoder with adaptive models, bit-exact to `codec/rans.py` |
| `src/bspline.ts`, `src/knn.ts` | B-spline knot evaluation, deterministic k-NN and Morton order |
| `src/splq.ts` | `.splq` format for quantised splat dumps of other methods |
| `src/renderer.ts`, `src/shaders.ts` | WebGPU rasteriser (curves evaluated on the GPU, near-plane culling, difference view) |
| `src/compare.ts`, `src/camera.ts` | panes, shared camera, timeline, snapping to a test camera |
| `src/i18n.ts` | UI language from the browser (`?lang` overrides) |
| `test/` | node tests; fixtures are generated synthetically, no scene data |
| `bench/` | headless benchmark and check scripts |

## Notes

* WebGPU is required. On iOS use Safari with WebGPU enabled.
* Decoding runs in a worker; a 32-frame section stays well below the section duration, so playback is real time.
* The viewer is a research tool: it expects scene data (cameras, `.csgs` sections, optional reference images) in the served directory.
