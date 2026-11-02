#!/usr/bin/env python3
"""train the E7/E8 curve model (best P2 setting) on an N3DV scene. GPU, train venv, under the GPU lock.

Thin driver over train/e7_curves_train.py: the only differences to the synthetic reference runs are
  * GT dir via E3_GT (cameras.json with width/height, cam00 = test), real background (no mask, full-image PSNR counts),
  * frame-0 init from the COLMAP sparse points of the train cameras (<gt>/init_points.npz, colmap_init.sh) instead of
    back-projected depth: e3.init_points is replaced here, everything else (fit_densify, curves, lifetimes) is e7's.
Settings are passed as E7_* environment variables by tools/faltwerk_recipe.py (K=5, E7_SOFT_KNOT0=1, E7_CURV=3e-5,
E7_CURV_ATTRS=means,quats,scales,sh0, E7_GROW=4e-4, L=32, refresh 8). Defaults of e7/e3 stay untouched.
  E3_GT=<gt_dir> E7_RUN=<run_dir> E7_K=5 ... python n3dv_train.py
"""
import functools, json, math, os, sys
import numpy as np
import torch

assert os.environ.get('E3_GT'), 'E3_GT (N3DV GT dir) required'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import e3_track as e3                 # noqa: E402  reads E3_GT/cameras.json at import
import e7_curves_train as e7          # noqa: E402


def knn_scale(pts, k=3):
    """3DGS init: log(mean distance to the k nearest neighbours), computed in chunks on the GPU (N3DV depth spans 6..300
    units, so e3's single global scale ext/sqrt(n) would be wrong for most of the scene)."""
    X = torch.tensor(pts, device=e3.dev)
    out = torch.empty(len(X), device=e3.dev)
    for i in range(0, len(X), 4096):
        d = torch.cdist(X[i:i + 4096], X)
        out[i:i + 4096] = d.topk(k + 1, largest=False).values[:, 1:].mean(1)
    return torch.log(out.clamp_min(1e-7)).cpu().numpy()


def init_points_colmap(gt0):
    """COLMAP sparse points (train cams, frame 0) -> e3 init dict + scene extent (2 x 3DGS cameras_extent, colmap_model)."""
    d = np.load(os.path.join(e3.GT, 'init_points.npz'))
    pts, cols, ext = d['xyz'].astype(np.float32), d['rgb'].astype(np.float32) / 255.0, float(d['ext'])
    if len(pts) > e3.N_INIT:
        sel = np.random.default_rng(0).choice(len(pts), e3.N_INIT, replace=False)
        pts, cols = pts[sel], cols[sel]
    n = len(pts)
    p = {'means': pts, 'quats': np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
         'scales': np.repeat(knn_scale(pts)[:, None], 3, 1).astype(np.float32),
         'opacities': np.full((n,), -2.0, np.float32),
         'sh0': ((cols - 0.5) / 0.28209479).astype(np.float32)[:, None],
         'shN': np.zeros((n, (e3.SH + 1) ** 2 - 1, 3), np.float32)}
    print(f'init: {n} COLMAP points, ext {ext:.3f}, {len(e3.train)} train cams, test {e3.test}, {e3.W}x{e3.H}', flush=True)
    return {k: torch.tensor(v, device=e3.dev) for k, v in p.items()}, ext


e3.init_points = init_points_colmap

# Near-plane culling from the dataset's LLFF bounds (0.9 x the smallest near bound over all cameras): without it the
# frame-0 fit (17 views) put huge free floaters right in front of the held-out camera, outside every training frustum
# (train views 36-39 dB, cam00 14.7 dB). The same near plane is used in n3dv_eval.py.
NEAR = 0.9 * min(c['near'] for c in e3.cams)
if not getattr(e3, '_n3dv_near_patched', False):          # p12_init imports this module too: patch once
    e3.rasterization = functools.partial(e3.rasterization, near_plane=NEAR)
    e3._n3dv_near_patched = True
print(f'near plane {NEAR:.3f} (LLFF bounds)', flush=True)

# frame-0 fit with the init/densification switches of p12_init.py, e.g.
#   P12_INIT="--strategy mcmc --cap 300000 --init colmap+random"
# Default (unset) = the P6 behaviour above (e3.fit_densify first=True on the COLMAP points). The section training (e7)
# is untouched either way; the fit is cached in $E7_RUN/init.pt like before.
P12 = os.environ.get('P12_INIT')
if P12:
    import shlex
    sys.modules.setdefault('n3dv_train', sys.modules[__name__])   # p12_init imports n3dv_train: reuse this instance
    import p12_init
    _a = p12_init.parse(['--out', os.environ.get('E7_RUN', '.')] + shlex.split(P12))
    _rng = np.random.default_rng(_a.seed)
    _orig_fit = e3.fit_densify

    def _p12_init_points(gt0):
        p, ext, _ = p12_init.init_points(gt0, _a, _rng)
        return p, ext

    def _p12_fit(p, ids, gt, iters, ext, first):
        if not first:
            return _orig_fit(p, ids, gt, iters, ext, first)
        log = {'curve': []}
        out = p12_init.fit(p, gt, ext, _a, log)
        json.dump({'config': vars(_a), 'curve': log['curve']}, open(os.path.join(_a.out, 'p12_init_curve.json'), 'w'), indent=1)
        return out, torch.arange(out['means'].shape[0], device=e3.dev)

    e3.init_points = _p12_init_points
    e3.fit_densify = _p12_fit
    print(f'P12 init switches: {P12}', flush=True)

# per-train-camera exposure/colour correction learned by p15_init.py --expo (<fit>/expo.pt). With
# P15_EXPO=<expo.pt> every render of a TRAIN camera in the section training (and the refresh fits) goes through that
# fixed affine map before the loss, so the sequence keeps the camera-independent colours of the frame-0 fit. Test
# cameras have zero rows (identity, bit-exact); n3dv_eval.py renders without it. Default (unset): unchanged.
P15_EXPO = os.environ.get('P15_EXPO')
if P15_EXPO and not getattr(e3, '_p15_expo_patched', False):
    _E = torch.load(P15_EXPO, map_location=e3.dev)['E'].to(e3.dev)
    assert bool((_E[e3.test] == 0).all()), 'exposure rows of the test cameras must be zero'
    _orig_render = e3.render

    # P15_EXPO_LEARN=<lr>: the map keeps learning during the section training (starts from the frame-0 map, Adam step
    # after every iteration, gauge: mean over the train cameras removed, test rows exactly zero), so slow exposure
    # changes of the video are absorbed by the cameras instead of the splat colours. Default (unset): fixed map.
    _LR = float(os.environ.get('P15_EXPO_LEARN', '0'))
    if _LR > 0:
        _E = torch.nn.Parameter(_E.clone())
        _eopt = torch.optim.Adam([_E], lr=_LR)
        _tr = torch.zeros(_E.shape[0], 1, 1, device=e3.dev); _tr[list(e3.train)] = 1.0

    def _expo_rows():
        if _LR <= 0:
            return _E
        return (_E - (_E * _tr).sum(0, keepdim=True) / _tr.sum()) * _tr        # gauge-fixed, test rows 0

    _orig_render = e3.render
    # P15_EXPO_STOP=<frac>: the learned map only steps during the first frac x E7_ITS iterations of each
    # section, then stays fixed while the splats converge (diagnosis: with a constant lr 1e-3 the map keeps moving after
    # the splat learning rates have decayed and cam00 (identity) drifts by up to 2 dB late in the section). Default 0 =
    # unchanged (steps all section long).
    _STOP = float(os.environ.get('P15_EXPO_STOP', '0'))
    _nstep = [0]

    def _render_expo(p, ids, deg=None):
        if _LR > 0 and _E.grad is not None:               # backward of the previous iteration has run: step the map
            if _STOP <= 0 or _nstep[0] < _STOP * e7.ITS:
                _eopt.step()
            _nstep[0] += 1
            _eopt.zero_grad(set_to_none=True)
        rgb, alpha, info = _orig_render(p, ids, deg)
        Ei = _expo_rows()[torch.as_tensor([int(i) for i in ids], device=rgb.device)]
        A = torch.eye(3, device=rgb.device) + Ei[:, :, :3]
        return ((rgb[..., None, :] * A[:, None, None]).sum(-1) + Ei[:, None, None, :, 3]).clamp(0, 1), alpha, info

    e3.render = _render_expo
    e3._p15_expo_patched = True
    if _STOP > 0:
        _ts_expo = e7.train_section

        def _train_section_expo(*a, **kw):
            _nstep[0] = 0                                    # new section: the map learns again
            return _ts_expo(*a, **kw)

        e7.train_section = _train_section_expo
        print(f'P15 exposure map steps only during the first {_STOP} x E7_ITS of each section', flush=True)
    print(f'P15 exposure correction of the train cams from {P15_EXPO}' + (f' (learned, lr {_LR})' if _LR > 0 else ' (fixed)'), flush=True)

# near-prune IN the section training (owner finding: ghost splats in front of the cameras that show up in
# in-between views). P15_NEARPRUNE=<f>: splats whose mean lies in a train camera's frustum closer than f x its LLFF near
# bound, or within a ball of that radius around a train camera, die at that frame. Applied after every refine step (the
# optimiser then re-fits without them) and once more at the end of each section (every 2nd frame). Default (unset): off.
P15_NP = float(os.environ.get('P15_NEARPRUNE', '0'))
# diagnosis: P15_NEARPRUNE_BALL=0 drops the ball term (splats beside/behind a train camera, outside its
# frustum, are kept); default 1 = unchanged.
P15_NP_BALL = os.environ.get('P15_NEARPRUNE_BALL', '1') == '1'
if P15_NP > 0 and not getattr(e7, '_p15_np_patched', False):
    _CAM = []
    for _i in e3.train:
        _c = e3.cams[_i]
        _R = torch.tensor(_c['R'], device=e3.dev, dtype=torch.float32); _t = torch.tensor(_c['t'], device=e3.dev, dtype=torch.float32)
        _CAM.append((_R, _t, -(_R.T @ _t), _c['fx'], _c['fy'], _c['cx'], _c['cy'], P15_NP * _c['near']))

    @torch.no_grad()
    def _np_mask(means):
        m = torch.zeros(means.shape[0], dtype=torch.bool, device=means.device)
        for R, t, cen, fx, fy, cx, cy, lim in _CAM:
            Xc = means @ R.T + t
            z = Xc[:, 2]; zc = z.clamp_min(1e-6)
            u = fx * Xc[:, 0] / zc + cx; v = fy * Xc[:, 1] / zc + cy
            m |= ((z > 0) & (z < lim) & (u >= 0) & (u < e3.W) & (v >= 0) & (v < e3.H))
            if P15_NP_BALL:
                m |= (means - cen).norm(dim=1) < lim
        return m

    @torch.no_grad()
    def _np_apply(P, opts, S, g0, Lg, K, step):
        n = 0
        for r in range(g0, g0 + Lg, step):
            idx = e7.alive_idx(S, r)
            if len(idx) == 0:
                continue
            hit = idx[_np_mask(e7.curve_at(P, idx, r, g0, Lg, K)['means'])]
            S['death'][hit] = r; n += len(hit)
        if n:
            e7.rebuild(P, opts, S, torch.nonzero(S['birth'] < S['death']).squeeze(1), {}, None)
        return n

    _orig_refine, _orig_train_section = e7.refine, e7.train_section

    def _refine_np(P, opts, S, gsum, gcnt, g0, Lg, K, scene_scale, *extra, **kw):
        # *extra: e7.refine gained the optional P13 `extra` / P17 `dw` arguments after P15; pass them through unchanged
        c, s, p = _orig_refine(P, opts, S, gsum, gcnt, g0, Lg, K, scene_scale, *extra, **kw)
        return c, s, p + _np_apply(P, opts, S, g0, Lg, K, e7.REFRESH)

    def _train_section_np(P, S, gts, g0, g1, K, ext):
        ev = _orig_train_section(P, S, gts, g0, g1, K, ext)
        n = _np_apply(P, {k: torch.optim.Adam([P[k]], lr=0.0) for k in P}, S, g0, g1 - g0, K, 2)
        ev['near_pruned_final'] = n
        print(f'P15 near-prune (f {P15_NP}) final pass of section {g0}: {n} splat deaths, ev {ev}', flush=True)
        return ev

    e7.refine, e7.train_section = _refine_np, _train_section_np
    e7._p15_np_patched = True
    print(f'P15 near-prune in the section training: f {P15_NP}', flush=True)

if __name__ == '__main__':
    e7.main()
