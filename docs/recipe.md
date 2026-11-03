# Standard recipe

`tools/faltwerk_recipe.py` is the single entry point from a scene to a Faltwerk stream. It writes `<out>/recipe.sh` (every command of the chain spelled out, re-runnable: finished stages are skipped) and `<out>/recipe.json` (scene facts, constants, which rules ran where). `--run` executes `recipe.sh` directly.

```
faltwerk_recipe.py --source images|mesh --scene <scene.json> --out <run dir> [--rate-target MBIT] [--run]
                   [--reuse <stage>=<dir> ...] [--cfg tools/recipe_standard.json] [--py <python>]
                   [--write <local dir>]   # recipe.sh/json also here when --out is a path on another (GPU) machine
```

- **No per-scene tuning values.** A scene file holds only data paths and facts. All constants live in `tools/recipe_standard.json`, one set per source; the rules in `train/recipe/recipe_rules.py`, `train/recipe/mesh_std.py`, `train/recipe/budget.py` and `train/recipe/teacher_stats.py` turn measured data into the numbers each stage needs. A constant is changed only after re-checking every scene of its source.
- **`--rate-target <Mbit/s>`** is a delivery value (e.g. for a web or headset player), not a scene value. Without it nothing is pruned (quality first). With it, a per-splat contribution ranking is scored on every frame, a splat budget is searched with the standard encoder until the rate fits, and the kept splats are fine-tuned again (prune, then fine-tune: pruning alone strips half-transparent surface splats and lets darker inner splats show through).
- **`--reuse <stage>=<dir>`** takes a finished stage whose configuration is identical to the standard.
- **One encoder** for every stream of every source: `train/recipe/encode.py` (dead / giant / runaway cut, deep floor, cull floor, position step from the camera footprint, SH degree from the data). Output: `<enc>/falt/s<index:04d>.falt` + `<enc>/falt.json` + `measure.json`.
- Always look at results on a coloured background too (the evaluations report red/white measures): black or near-transparent splats are invisible on black.

## Sources

| `--source` | data | chain |
|---|---|---|
| `images`, cameras with `near`/`far` (`kind: llff`) | forward-facing multi-camera video, e.g. Neural 3D Video | sparse COLMAP depth -> frame-0 fit (`train/llff/p15_init.py`) -> section training (`train/llff/n3dv_train.py` = `train/e7_curves_train.py` with the LLFF near plane) -> standard encode -> decoded eval (`n3dv_eval.py`), LPIPS, red/white (`n3dv_rw.py`) |
| `images`, matted rig + per-frame teacher (`kind: rig`, `teacher`) | multi-camera rig with a foreground matte and one 3DGS PLY per frame fitted to it | stage 1 tracking against the teacher (`train/teacher/stage1.py`) -> stage 2 curve fit (`stage2_fit.py`) -> fine-tune (`stage2_ft.py`) -> [rate target: contrib -> budget -> fine-tune of the kept splats] -> standard encode -> decoded eval incl. red/white (`train/rig/rig_eval.py`) |
| `mesh` | animated meshes exported per frame from 3D software | per geometry piece: mesh rules (sample spacing from the pixel footprint, per-mesh caps, backdrop split) -> `train/mesh/mesh2gs.py` (splats directly from the surfaces, no training) -> standard encode per stream |

A rig without a teacher is not part of the standard recipe.

## Scene files

Examples: `train/recipe/scenes/`. Paths are relative to the repository root (the generated script `cd`s there) or absolute.

```json
{"name": "flame_steak", "kind": "llff", "frames": 64, "gt": "work/data/gt/n3dv_flame_steak_1352"}
{"name": "my_rig", "kind": "rig", "frames": 64, "gt": "<GT dir>", "teacher": "<dir>/{:05d}.ply", "teacher_frames": 50}
{"name": "my_meshes", "looks": ["shaded"],
 "geom": [{"dir": "<geom dir>", "f0": 0, "frames": 120, "cams": "<cams.json>"}]}
```

`kind` is optional (detected from `cameras.json`: cameras with `near` = `llff`). For `mesh`, several geometry pieces (e.g. two segments of one export) can be listed; the sample spacing is pooled over all of them. `looks`: `shaded` (view-independent clay shading from the vertex normals), `sil` (flat mesh colour).

## Data layouts

### GT directory (`images`)

```
<gt>/cameras.json
<gt>/f<frame:03d>/c<camera:02d>.png        RGB (llff) or RGBA with RGB premultiplied by the matte (rig)
<gt>/colmap/sparse_txt/, <gt>/init_points.npz, <gt>/sparse_depth.npz     llff only, written by the prep steps below
```

`cameras.json`: `{"frames": N, "fps": 30, "width": W, "height": H, "cameras": [{"name": "cam00", "split": "test"|"train", "R": [[...]], "t": [...], "fx", "fy", "cx", "cy", "near", "far"}, ...]}` – OpenCV world-to-camera `R`, `t`; `near`/`far` only for LLFF-style data. The test cameras are held out for the decoded evaluation. For LLFF, set `width` and `height`; legacy square inputs may use `res` instead. Rig/teacher inputs additionally require top-level `W` and `H` with the same dimensions (the rig reader uses these keys).

Neural 3D Video (CC BY-NC 4.0, download it yourself from the authors' release): put the scene's `cam*.mp4` and `poses_bounds.npy` under `work/data/n3dv/<scene>/`, then

```bash
python train/llff/prep_frames.py --scene flame_steak --frames 64     # -> work/data/gt/n3dv_flame_steak_1352/ (cam00 = test)
bash train/llff/colmap_init.sh work/data/gt/n3dv_flame_steak_1352   # COLMAP (Docker) frame-0 points -> init_points.npz
```

The recipe runs the sparse-depth step (`p15_prep.py sparse`) itself.

### Teacher (`images`, rig)

One binary little-endian 3DGS PLY per frame (`x y z f_dc_* opacity scale_* rot_*`, INRIA conventions), pattern with `{:05d}`; `teacher_frames` = how many frames have one (later frames train against the GT only).

### Mesh geometry (`mesh`)

```
<root>/geom/meta.json        {"meshes": {"<id>": {"path": "<name>", "color": [r, g, b]}}, "film_cams": [{"pos": [x,y,z], "fov": deg}, ...]}
<root>/geom/topo.npz         tris_<id>_<nv>: int32 [n_tris, 3]  (one entry per mesh id and vertex count)
<root>/geom/f<frame:04d>.npz pos_<id>, nrm_<id>: float32 [nv, 3] world positions and vertex normals of the meshes visible in the frame
<root>/geom/flags.json       optional: {"<id>": {"ontop": true}} for meshes drawn without depth test (decals)
<root>/owners.json           {"<id>": "<owner>"}: groups meshes (a static, far owner becomes the backdrop stream)
<cams.json>                  {"H": <image height>, "cams": [{"pos": [x,y,z], "fov": deg} | {"track": [{"pos", "fov"}, ...]}]}
```

Every contiguous run of frames in which a mesh id is present becomes one model with its own lifetime. `tools/make_synthetic_mesh.py <dir>` writes a complete example (a bouncing sphere, a waving ribbon born later, a far static wall) and its `scene.json`; `tests/test_recipe_mesh_smoke.py` runs the whole mesh chain on it.

## Rules (replacing hand values)

| quantity | rule | constant |
|---|---|---|
| position step | 0.5 px at the median depth of the visible splats in the test cameras (images) or at the figures' pixel footprint (mesh); `qpos = step / extent` | `qstep_px` 0.5 |
| clean encode | dead (opacity < 1/255 at every knot), deep floor and cull floor for every stream | – |
| giant / runaway cut | sigma_max > 0.5 R95, path > 3 R99 – only for scenes with a matte (object scenes; far/large splats are the background elsewhere) | `giant_rel`, `runaway_rel` |
| SH degree | highest band with energy | – |
| splat budget | from the rate target: secant search over the standard encode, aim 0.97 x delivery rate, then fine-tune the kept splats | – |
| near plane | 0.9 x min camera `near` (LLFF), else 0.1 x min camera distance | – |
| random background + alpha loss | on whenever the GT has a matte | – |
| teacher swap tau / curve tolerance | 2.06 x median teacher sigma / 1.43 x 10th-percentile teacher sigma; extent = visible frame-0 teacher | `P43_SWAP_TAU_REL`, `TOLP_REL` |
| mesh sample spacing | `spacing_px` x pixel footprint at the median camera distance of the moving meshes, pooled over the clip; never below 1 px at a mesh's closest view | `spacing_px` 2.1 |
| backdrop stream | static meshes whose closest camera distance > 10 x figure distance get their own stream (keeps the k-NN grid of the figures compact) | `backdrop_dist_rel` 10 |
| thin open meshes | width w = 2A/P_open < 3h: spacing w/3 (floor 0.5 px in the closest view), sampled at their largest size, covariance follows the deformation, 1-sigma inset from the open edge | `fix` in `recipe_standard.json` |
| per-mesh cap | memory constant per mesh | `max_per_mesh` 25 000 |

The calibrated constants (teacher tau, curve tolerance, `spacing_px`) were each tuned on a single scene of their kind and are the first candidates to revisit on new data.
