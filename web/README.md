# Faltwerk viewer (WebGPU)

The player decodes Faltwerk sections (`.falt`, legacy `.csgs`; both may be mixed in one stream) in a worker with a TypeScript port of the Python reference decoder (bit-exact), uploads them to the GPU and evaluates the curves per frame in a compute pass (projection, SH degree 3, bitonic depth sort, front-to-back blending). Own renderer, no third-party runtime library. `compare.html` plays 2-4 panes with **one shared camera** and **one timeline**, so two codecs or two quality steps can be compared side by side.

## Build and run

```bash
npm ci
npm test            # node tests: bit-exact decoder against the Python reference (synthetic fixtures)
npm run typecheck
npm run build       # dist/ (index.html, compare.html, main.js, worker.js, compare.js)
python3 serve_viewer.py <root> [port]           # static server, http://127.0.0.1:8790, no-cache headers
python3 serve_viewer.py <root> [port] --https   # binds ALL interfaces with a self-signed cert (WebXR on a headset in your LAN)
```

WebGPU needs a secure context (HTTPS or `localhost`). Use `--https` only on a trusted network.

The served `<root>` is a directory you assemble: the built viewer (`web/dist/*`), the `.falt` sections and a scene file (`scene.json` for `index.html`) or a manifest (`manifest.json` for `compare.html`). Scene data never lives in this repository.

## Synthetic demo (no data download, runs on CPU)

From the repository root, with a Python that has `numpy scipy zstandard pillow`:

```bash
python tools/make_synthetic_mesh.py work/syn --frames 64                        # bouncing sphere, ribbon, far wall
python tools/faltwerk_recipe.py --source mesh --scene work/syn/scene.json --out work/run --py "$(which python)"
bash work/run/recipe.sh                                                          # mesh -> splats -> .falt (~20 s, CPU)
(cd web && npm ci && npm run build)
python web/make_scene.py work/root work/run/shaded_p0_fg/enc/falt               # root = viewer + data/ + scene.json
python web/serve_viewer.py work/root                                             # http://127.0.0.1:8790/index.html
```

`make_scene.py` reads first frame and frame count from each section header; `--center/--up/--front/--dist/--elev/--fovy` set the orbit camera (defaults fit the synthetic scene). The far wall is a separate stream (`work/run/shaded_p0_bd/enc/falt`).

**Without a GPU:** Chromium can run WebGPU on the CPU (SwiftShader). Add `offscreen=1`: SwiftShader has no usable swap chain, without it the device is lost after the first frame. Expect a few frames per second (one GPU pass takes ~120 ms on the CPU for ~3k splats).

```bash
chromium --enable-unsafe-webgpu --enable-unsafe-swiftshader --use-webgpu-adapter=swiftshader \
  "http://127.0.0.1:8790/index.html?scene=scene.json&offscreen=1"
```

In offscreen mode the canvas stays empty; `window.__p5.snapshot()` (browser console) draws the current frame on top.

## Scene data and the comparison manifest

The viewer expects scene data in the served directory (never in this repository): per scene a camera file, the `.falt` sections of each pane and optional reference stills.

```bash
python3 bench/make_viewer_data.py gt <gt_dir> <root>/<scene>/gt --cams 0 --frames 64      # reference stills (JPEG)
python3 bench/make_viewer_data.py manifest <root>/recipe.json                            # -> <root>/manifest.json
```

`viewer.recipe.example.json` shows the recipe format: scenes with `cameras`, `frames`, optional `near` (use 0.9 x min LLFF near for LLFF data, as in training), `gt` stills, `default_snap`, and panes (`falt` for our streams, `splq` for quantised per-frame splat dumps of other methods) with rate/metric labels read from run JSONs.

URL parameters (`compare.html`): `scene=<id>`, `panes=<id,id>`, `snap=<camera>`, `frame=<n>`, `pause`, `dpr=<max>`, `diff=0|1|4` (difference view against the reference still), `manifest=<url>`, `lang=de|en`, `offscreen=1` (headless without a swapchain). `index.html`: `scene=<scene.json>`, `n=<splats>` (stress test), `testcam=<i>`, `t=<frame>`, `az`, `zoom`.

Controls: drag = orbit, wheel / two fingers = zoom and pan, double tap = reset, space = play/pause, slider = time, camera menu = snap to a test camera (+ reference still).

## Files

| File | Purpose |
|---|---|
| `src/falt.ts`, `src/rans.ts` | container decoder (versions 1-6: SN_VQ, pred, follow, rescale, B-spline) and rANS, bit-exact to `codec/falt.py` / `codec/rans.py`; `src/csgs.ts` = old module name, re-exports |
| `src/bspline.ts`, `src/knn.ts` | B-spline knot evaluation, deterministic causal k-NN and Morton order |
| `src/pack.ts` | section -> GPU layout, CPU mirror of the curve evaluation |
| `src/shaders.ts`, `src/renderer.ts` | WGSL (curve evaluation, projection/SH, sort, raster, difference blit) and the WebGPU pipelines |
| `src/main.ts`, `src/worker.ts` | single-scene player: progressive loading (current + 2 sections decoded ahead), orbit camera |
| `src/compare.ts`, `src/camera.ts` | comparison viewer: per-pane fetch queues, shared clock, snapping to a test camera |
| `src/splq.ts` | `.splq` frames (16 B per splat, quantised per-frame dumps of other methods) |
| `src/xr.ts`, `src/recolor.ts` | WebXR immersive-ar mode (two-hand pinch, depth occlusion), automatic colour from the camera image |
| `src/i18n.ts` | UI language from the browser (`?lang` overrides) |
| `test/` | `node --test`; fixtures are generated synthetically (`make_fixture.py`, `make_splq_fixture.py`), no scene data. Tests against real sections run only when `REF_DIR` / `S1_REF_DIR` / `S6_REF_DIR` point to them |
| `bench/` | headless Chromium benchmark (`bench.py`, Playwright), gsplat reference render + PSNR (`ref_render.py`), decode-speed probes, manifest/still builder (`make_viewer_data.py`) |
