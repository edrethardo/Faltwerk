"""standard recipe: the RULES that replace the scene-specific hand values.

Every function takes measured data (curves, cameras, meshes) and returns plain numbers; the constants live in
tools/recipe_standard.json (one set for all scenes of a source). Nothing here knows a scene name.

Rules (see docs/recipe.md):
  content radius   R95 = 95th percentile distance of the visible splats from their median position
  giant cut        sigma_max > GIANT_REL * R95                       (replaces --smax 0.5 m)
  runaway cut      mean leaves RUNAWAY_REL * R99 around the median    (replaces --rmax 5 m)
  position step    step = QSTEP_REL * sigma_fine, sigma_fine = 10th percentile of sigma_max of the visible splats;
                   qpos = step / ext_kept                             (replaces qpos 0.0015 / 1e-4 / 1.4e-5)
  SH degree sent   highest band with energy                           (replaces --sn-deg 0/3)
  near plane       0.9 * min(camera near) or NEAR_REL * min camera distance to the content
"""
import json
import math

import numpy as np

DEFAULTS = {'visible_opa': 0.1, 'giant_rel': 0.5, 'runaway_rel': 3.0, 'qstep_rel': 0.3, 'near_rel': 0.1}


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def curve_stats(path, keep=None):
    """per-id statistics over all sections of a curves.npz (hat/bspline knots, log scales, opacity logits)."""
    z = np.load(path)
    gs = sorted(int(k[1:-4]) for k in z.files if k.startswith('g') and k.endswith('_ids'))
    smax = {}; omax = {}; pmin = {}; pmax = {}; p0 = {}
    shE = None
    for g in gs:
        ids = z[f'g{g}_ids']
        S = z[f'g{g}_scales'].astype(np.float32)            # [n, K, 3] log sigma
        O = z[f'g{g}_opacities'].astype(np.float32)         # [n, K] logit
        M = z[f'g{g}_means'].astype(np.float32)             # [n, K, 3]
        s = np.exp(S.max(axis=(1, 2)))
        o = _sig(O.max(axis=1))
        lo = M.min(axis=1); hi = M.max(axis=1); c = M[:, M.shape[1] // 2]
        for name, arr, red in (('s', s, np.maximum), ('o', o, np.maximum)):
            dct = smax if name == 's' else omax
            # vectorised merge through a dense table indexed by id
            dct.setdefault('ids', []).append(ids); dct.setdefault('v', []).append(arr)
        pmin.setdefault('v', []).append(lo); pmax.setdefault('v', []).append(hi); p0.setdefault('v', []).append(c)
        pmin.setdefault('ids', []).append(ids)
    ids = np.concatenate(smax['ids']); n = int(ids.max()) + 1
    S = np.zeros(n, np.float32); O = np.zeros(n, np.float32)
    np.maximum.at(S, ids, np.concatenate(smax['v'])); np.maximum.at(O, ids, np.concatenate(omax['v']))
    LO = np.full((n, 3), np.inf, np.float32); HI = np.full((n, 3), -np.inf, np.float32); C = np.zeros((n, 3), np.float32)
    lo = np.concatenate(pmin['v']); hi = np.concatenate(pmax['v']); c = np.concatenate(p0['v'])
    for k in range(3):
        np.minimum.at(LO[:, k], ids, lo[:, k]); np.maximum.at(HI[:, k], ids, hi[:, k])
    C[ids] = c                                              # any section's mid knot (last wins): a representative position
    present = np.zeros(n, bool); present[ids] = True
    if keep is not None:
        present &= keep[:n]
    vis = present & (O >= DEFAULTS['visible_opa'])
    med = np.median(C[vis], axis=0)
    d = np.linalg.norm(C - med, axis=1)
    move = np.linalg.norm(HI - LO, axis=1)
    lo_all = LO[present].min(0); hi_all = HI[present].max(0)
    summ = {
        'n_ids': int(present.sum()), 'n_visible': int(vis.sum()), 'sections': len(gs),
        'sigma_p10_p50_p90_vis': [float(np.percentile(S[vis], q)) for q in (10, 50, 90)],
        'sigma_max_vis': float(S[vis].max()),
        'R50_R95_R99': [float(np.percentile(d[vis], q)) for q in (50, 95, 99)],
        'median': med.tolist(), 'ext_bbox': float(np.linalg.norm(hi_all - lo_all)),
        'moving_share_vis': float((move[vis] > 0.01 * np.percentile(d[vis], 95)).mean()),
    }
    return {'summary': summ, 'S': S, 'O': O, 'C': C, 'LO': LO, 'HI': HI, 'present': present, 'vis': vis, 'd': d}


def derive_encoder(st, cfg=None):
    """encoder-side rules from curve statistics -> dict (smax, rmax, qpos, step, ext_kept)."""
    c = dict(DEFAULTS); c.update(cfg or {})
    s = st['summary']
    R95, R99 = s['R50_R95_R99'][1], s['R50_R95_R99'][2]
    smax = c['giant_rel'] * R95
    rmax = c['runaway_rel'] * R99
    pres = st['present']
    lo = st['LO']; hi = st['HI']; med = np.asarray(s['median'])
    far = np.maximum(np.linalg.norm(lo - med, axis=1), np.linalg.norm(hi - med, axis=1))
    kept = pres & (st['S'] <= smax) & (far <= rmax)
    if not (kept & st['vis']).any():                       # a small decal that travels far relative to its own
        kept = pres                                        # size (the smile stream): every splat is a 'runaway'
    ext = float(np.linalg.norm(hi[kept].max(0) - lo[kept].min(0)))
    sig_fine = float(np.percentile(st['S'][kept & st['vis']], 10))
    step = c['qstep_rel'] * sig_fine
    return {'smax': smax, 'rmax': rmax, 'ext_kept': ext, 'sigma_fine': sig_fine, 'step': step, 'qpos': step / ext,
            'giant_n': int((pres & (st['S'] > smax)).sum()), 'runaway_n': int((pres & (far > rmax)).sum())}


def footprint_from_cams(cams_json, pos, which='test'):
    """metres per pixel at the content: median depth of the visible splats (pos [N, 3]) in the chosen cameras / fx
    (w2c R, t; pinhole fx). 'test' cameras if the file marks any, else all."""
    m = json.load(open(cams_json))
    cams = [c for c in m.get('cameras', []) if isinstance(c, dict) and 'R' in c and 'fx' in c]
    sel = [c for c in cams if c.get('split') == which] or cams
    fps = []
    P = np.asarray(pos, float)
    for c in sel:
        z = (P @ np.asarray(c['R'], float).T + np.asarray(c['t'], float))[:, 2]
        z = z[z > 0]
        if len(z):
            fps.append(float(np.median(z)) / float(c['fx']))
    return float(np.median(fps)) if fps else None


def sn_degree(tracks_path):
    """highest SH band with energy in tracks.npz shN [N, 15, 3] (0 when shN is absent or all zero)."""
    z = np.load(tracks_path, mmap_mode='r')
    if 'shN' not in z.files:
        return 0
    sh = np.abs(np.asarray(z['shN'], np.float32))
    for deg, (a, b) in ((3, (8, 15)), (2, (3, 8)), (1, (0, 3))):
        if sh[:, a:b].max() > 1e-4:
            return deg
    return 0


def near_plane(cams_json, content_center=None, near_rel=DEFAULTS['near_rel']):
    """0.9 * min camera 'near' (LLFF bounds) if the cameras carry one, else near_rel * min camera distance."""
    m = json.load(open(cams_json))
    cams = m.get('cameras', m.get('cams', []))
    nears = [c['near'] for c in cams if isinstance(c, dict) and 'near' in c]
    if nears:
        return 0.9 * min(nears)
    if content_center is None:
        return 0.01
    ds = []
    for c in cams:
        if not isinstance(c, dict) or 'R' not in c:
            continue
        R = np.asarray(c['R'], float); t = np.asarray(c['t'], float)
        ds.append(float(np.linalg.norm(R @ np.asarray(content_center) + t)))   # w2c: camera-space z of the centre
    return near_rel * min(ds) if ds else 0.01


def mesh_spacing(px_m, cfg=None):
    """mesh source: surface sample spacing h = SPACING_PX * pixel footprint at the median camera distance."""
    c = {'spacing_px': 1.0}; c.update(cfg or {})
    return c['spacing_px'] * px_m


def pixel_footprint(dist, fov_v_deg, H):
    return 2.0 * dist * math.tan(math.radians(fov_v_deg) / 2.0) / H
