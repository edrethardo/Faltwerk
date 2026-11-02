#!/usr/bin/env python3
"""standard recipe, source "mesh": rules for mesh2gs (train/mesh/mesh2gs.py, ) instead of hand values
(sample spacing, per-mesh caps, a backdrop stream chosen by name).

Rules (constants from tools/recipe_standard.json "mesh"):
  d_fig   = median over the MOVING meshes of their closest camera distance (film camera per frame + every listed camera)
  h       = spacing_px * pixel footprint at d_fig (footprint = 2 d tan(fov/2) / H of the GT)
  h_mesh  = max(h, spacing_px * footprint at the mesh's own closest camera distance)  -> far meshes coarser
  cap     = min(max_per_mesh, area / h_mesh^2) per mesh (max_per_mesh = memory constant, the same for every scene)
  backdrop= meshes that never move AND whose closest camera distance > backdrop_dist_rel * d_fig -> own stream
            (codec/knn.py grid: far scenery in the figures' stream makes the k-NN grid pathological, )

  mesh_std.py stats --three <geom> --frames N [--f0 F] --cams cams.json [--cfg recipe_standard.json]  -> JSON
  mesh_std.py run --part fg|bd --three <geom> --frames N [--f0 F] --cams cams.json --out <run> --look sil|shaded
              [mesh2gs args ...]     mesh2gs with h, per-mesh caps and the part's meshes (env P42_CODE = dir of mesh2gs)
"""
import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.environ.get('P42_CODE', os.path.join(ROOT, 'train', 'mesh')))
import mesh2gs as m2g                                       # noqa: E402


def cam_list(cams_json, gd, f0, T):
    """[(pos [3], fov_deg, frames or None)] from cams.json (static cams, 'track' lists) + meta.json film_cams."""
    out = []
    c = json.load(open(cams_json))
    H = float(c.get('H', 720))
    for x in c.get('cams', []):
        if 'pos' in x:
            out.append((np.asarray(x['pos'], float), float(x.get('fov', 30.0)), None))
        elif isinstance(x.get('track'), list):
            tr = x['track']
            P = np.array([t['pos'] for t in tr if isinstance(t, dict) and 'pos' in t], float)
            fov = float(next((t.get('fov', 30.0) for t in tr if isinstance(t, dict)), 30.0))
            if len(P):
                out.append((P, fov, 'track'))
    meta = json.load(open(os.path.join(gd, 'meta.json')))
    fc = meta.get('film_cams') or []
    if fc:
        P = np.array([x['pos'] for x in fc], float)
        out.append((P, float(fc[0].get('fov', 30.0)), 'track'))
    return out, H


def mesh_dist(md, cams, f0):
    """camera distances to the mesh surface (centroid distance minus the mesh radius, >= 1 cm): the MEDIAN over all
    (camera, frame) pairs (typical view: drives the sample spacing) and the minimum (closest view)."""
    V = md['V']                                              # [nT, nv, 3], frames md f0..f1 (segment-local)
    cen = V.mean(1)                                          # [nT, 3]
    rad = float(np.linalg.norm(V[V.shape[0] // 2] - cen[V.shape[0] // 2], axis=-1).max())
    ds = []
    for P, fov, kind in cams:
        if kind == 'track':                                  # per-frame camera (local frame t -> track index f0 + t)
            idx = np.clip(np.arange(md['f0'], md['f1']) + f0, 0, len(P) - 1)
            ds.append(np.linalg.norm(cen - P[idx], axis=-1))
        else:
            ds.append(np.linalg.norm(cen - P[None], axis=-1))
    ds = np.maximum(np.concatenate(ds) - rad, 0.01)
    fov = cams[0][1] if cams else 30.0
    return float(np.median(ds)), float(ds.min()), fov, rad


def rules(models, cams, H, cfg, f0, pool=None):
    c = cfg['mesh']
    rows = []
    for md in models:
        V = md['V']
        move = float(np.abs(V - V[:1]).max()) if V.shape[0] > 1 else 0.0
        d, dmin, fov, rad = mesh_dist(md, cams, f0)
        fp = 2 * d * math.tan(math.radians(fov) / 2) / H
        _, area = m2g.tri_frames(V[[V.shape[0] // 2]][:, md['tris']].astype(np.float64))
        rows.append({'name': md['name'], 'owner': md.get('owner'), 'f0': md['f0'], 'f1': md['f1'], 'dist': d, 'dist_min': dmin,
                     'footprint': fp, 'move': move, 'radius': rad, 'area': float(area.sum())})
    ref = list(rows)
    for p in pool or []:                                     # the whole clip: every geometry piece's meshes (one h for all)
        ref += json.load(open(p))['meshes']
    mv = [r for r in ref if r['move'] > 0.01 * max(r['radius'], 1e-3)]
    d_fig = float(np.median([r['dist'] for r in mv])) if mv else float(np.median([r['dist'] for r in ref]))
    fp_fig = float(np.median([r['footprint'] for r in mv])) if mv else float(np.median([r['footprint'] for r in ref]))
    h = c['spacing_px'] * fp_fig
    for r in rows:
        r['backdrop'] = bool(r['move'] <= 0.01 * max(r['radius'], 1e-3) and r['dist_min'] > c['backdrop_dist_rel'] * d_fig)
        r['h_mesh'] = max(h, c['spacing_px'] * r['footprint']) if r['backdrop'] else h   # far scenery: >= spacing_px at its distance
        # figures: only the memory cap (as before); a tighter area/h^2 cap would bind and cancel the thin-part densification
        # (curv-dens) and the grow-ratio sampling of mesh2gs. Backdrop: the coarser spacing of its own distance.
        r['cap'] = int(min(c['max_per_mesh'], max(16, r['area'] / r['h_mesh'] ** 2))) if r['backdrop'] else int(c['max_per_mesh'])
    return {'d_fig': d_fig, 'footprint_fig': fp_fig, 'h': h, 'H': H, 'meshes': rows,
            'backdrop_owners': sorted({r['owner'] for r in rows if r['backdrop']} - {r['owner'] for r in rows if not r['backdrop']}),
            'n_backdrop_meshes': sum(r['backdrop'] for r in rows)}


def main():
    mode = sys.argv[1]
    argv = sys.argv[2:]

    def opt(k, d=None):
        return argv[argv.index(k) + 1] if k in argv else d
    gd, T, f0, cams_p = opt('--three'), int(opt('--frames')), int(opt('--f0', 0)), opt('--cams')
    cfg = json.load(open(opt('--cfg', os.path.join(ROOT, 'tools', 'recipe_standard.json'))))
    models = m2g.models_three(gd, T, None, f0)
    cams, H = cam_list(cams_p, gd, f0, T)
    pool = [p for p in (opt('--pool') or '').split(',') if p and os.path.exists(p)]
    R = rules(models, cams, H, cfg, f0, pool)
    if mode == 'stats':
        if opt('--json'):
            json.dump(R, open(opt('--json'), 'w'))
        own = {}
        for r in R['meshes']:
            o = own.setdefault(r['owner'], {'n': 0, 'area': 0.0, 'dist_min': 1e9, 'dist_med_max': 0.0, 'move_max': 0.0, 'backdrop': True, 'cap_sum': 0})
            o['n'] += 1; o['area'] += r['area']; o['dist_min'] = min(o['dist_min'], r['dist_min']); o['move_max'] = max(o['move_max'], r['move'])
            o['dist_med_max'] = max(o['dist_med_max'], r['dist'])
            o['backdrop'] &= r['backdrop']; o['cap_sum'] += r['cap']
        print(json.dumps({k: v for k, v in R.items() if k != 'meshes'} | {'owners': own}, indent=1))
        return
    # run: mesh2gs on the part's meshes with h and the per-mesh caps
    part = opt('--part', 'fg')
    bd = set(R['backdrop_owners'])
    owners = sorted({r['owner'] for r in R['meshes'] if r['owner'] is not None})
    sel = [o for o in owners if (o in bd) == (part == 'bd')]
    if not sel:
        print(f'MESHSTD part {part}: no meshes'); return
    c = cfg['mesh']
    fix = c['fix'].split()
    # decal stream 'dt': thin open meshes (w = 2A/P_open < thin_open x h, mesh2gs --thin-open) get their own
    # stream, encoded with the position step of THEIR closest view (0.5 px there). In the figure stream the step is 0.5 px
    # at the median figure distance = 3.3 mm in D1, 4x the smile splats' sigma: the codec broke the 15-splat grin into
    # 3-5 pieces and would scatter the dense new one the same way.
    k_thin = float(fix[fix.index('--thin-open') + 1]) if '--thin-open' in fix else 0.0
    dt_ids = []
    if k_thin > 0:
        for md in models:
            if md.get('owner') in bd:
                continue
            V = md['V']; mi = V.shape[0] // 2
            _, ar = m2g.tri_frames(V[[mi]][:, md['tris']].astype(np.float64))
            if m2g.open_width(md['tris'], np.asarray(V[mi], np.float64), float(ar.sum())) < k_thin * R['h']:
                dt_ids.append(md['name'].split(':')[0][1:])
    dt_ids = sorted(set(dt_ids))
    dt_rows = [r for r in R['meshes'] if r['name'].split(':')[0][1:] in dt_ids]
    R['dt_ids'] = dt_ids
    R['footprint_dt'] = min(r['footprint'] * r['dist_min'] / max(r['dist'], 1e-9) for r in dt_rows) if dt_rows else None
    idsel = []
    if part == 'dt':
        if not dt_ids:
            print('MESHSTD part dt: no meshes'); return
        sel = sorted({r['owner'] for r in dt_rows if r['owner'] is not None})
        idsel = ['--mesh-ids', ','.join(dt_ids)]
    elif part == 'fg' and dt_ids:
        idsel = ['--skip-mesh-ids', ','.join(dt_ids)]
    caps = {r['name']: r['cap'] for r in R['meshes']}
    # --thin-open: spacing floor 0.5 px at the mesh's closest camera distance (finer is invisible)
    hfl = {r['name']: 0.5 * r['footprint'] * r['dist_min'] / max(r['dist'], 1e-9) for r in R['meshes']}
    orig = m2g.sample_points

    def sample_points(md, a, rng):                           # per-mesh cap instead of --max-per-model / --cap-owner
        a.cap_owner = {md.get('owner'): caps.get(md['name'], a.max_per_model)}
        a.h_floor = hfl.get(md['name'], 0.0)
        return orig(md, a, rng)
    m2g.sample_points = sample_points
    skip = {'--part', '--cams', '--cfg', '--pool', '--json'}
    rest, i = [], 0
    while i < len(argv):
        if argv[i] in skip:
            i += 2; continue
        rest.append(argv[i]); i += 1
    flags = os.path.join(gd, 'flags.json')                     # depthTest-off meshes (ba3_export.py flags)
    if '--ontop' in fix:
        if os.path.exists(flags):
            fix.insert(fix.index('--ontop') + 1, flags)
            fix += ['--ontop-cams', cams_p]                    # the other camera passes see the decal too
        else:
            fix.remove('--ontop')
    if part == 'bd':                                         # backdrop lives shorter than the piece: the stream ends with it
        rows = [r for r in R['meshes'] if r['owner'] in sel]
        last = max(r['f1'] for r in rows)
        if last < T:
            rest[rest.index('--frames') + 1] = str(last)
        R['bd_frames'] = [min(r['f0'] for r in rows), last]
    if part == 'dt':                                         # decal stream: from the first to the last decal frame
        first, last = min(r['f0'] for r in dt_rows), max(r['f1'] for r in dt_rows)
        rest[rest.index('--frames') + 1] = str(last - first)
        if '--f0' in rest:
            rest[rest.index('--f0') + 1] = str(f0 + first)
        else:
            rest += ['--f0', str(f0 + first)]
        R['dt_frames'] = [first, last]; R['dt_f0'] = f0 + first
    sys.argv = ['mesh2gs.py'] + rest + ['--h', f"{R['h']:.6g}", '--sigma', str(c['sigma']), '--K', str(c['K']), '--L', str(c['L']),
                                        '--only', ','.join(sel)] + fix + idsel
    os.makedirs(opt('--out'), exist_ok=True)
    json.dump(R | {'part': part, 'owners': sel}, open(os.path.join(opt('--out'), f'mesh_rules_{part}.json'), 'w'), indent=1)
    print('MESHSTD', part, 'h', R['h'], 'owners', sel, 'argv', ' '.join(sys.argv[1:]), flush=True)
    m2g.main()


if __name__ == '__main__':
    main()
