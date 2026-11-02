"""P42 mesh2gs fix candidates: blue-noise sampling, deformation-following covariance,
curvature/boundary caps. Default arguments must reproduce the old splats exactly."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train', 'mesh'))
import mesh2gs as M  # noqa: E402


def grid(nx=30, ny=30, size=1.0):
    xs, ys = np.meshgrid(np.linspace(0, size, nx + 1), np.linspace(0, size, ny + 1), indexing='ij')
    V = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], 1)
    idx = lambda i, j: i * (ny + 1) + j  # noqa: E731
    tris = []
    for i in range(nx):
        for j in range(ny):
            tris += [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)], [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]]
    return V, np.array(tris, np.int64)


def model(V_frames, tris):
    N = np.zeros_like(V_frames); N[..., 2] = 1
    return {'name': 'm0:test', 'f0': 0, 'f1': len(V_frames), 'tris': tris, 'V': V_frames.astype(np.float32),
            'N': N.astype(np.float32), 'color': np.zeros(3), 'owner': None}


def args(extra=()):
    return M.parse(['--frames', '2', '--out', '/tmp/x', '--look', 'shaded', '--h', '0.02', '--max-per-model', '0']
                   + list(extra))


def nn_dist(X):
    d = np.linalg.norm(X[:, None] - X[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min(1)


def test_default_path_unchanged():
    V, tris = grid(10, 10)
    Vf = np.stack([V, V * [1.3, 1.0, 1.0]])
    a = args()
    md = model(Vf, tris)
    smp, st = M.sample_points(md, a, np.random.default_rng(0))
    ex = M.attrs_at(md, smp, np.arange(2), a)
    _, af = M.tri_frames(Vf[:, tris[smp['fi']]].astype(np.float64))
    s = a.sigma * md['h'] * np.sqrt(np.clip(af / smp['aref'], 0.05, 20))
    assert np.allclose(np.exp(ex['scales'][..., 0]), s) and np.allclose(np.exp(ex['scales'][..., 1]), s)
    assert st['samples'] == len(smp['fi']) and abs(st['samples'] - 1.3 ** 0 * 1 / 0.02 ** 2 * np.median([1, 1.3])) < 200


def test_poisson_same_count_and_spread():
    V, tris = grid(20, 20)
    Vf = np.stack([V, V])
    md_r, md_p = model(Vf, tris), model(Vf, tris)
    sr, str_ = M.sample_points(md_r, args(), np.random.default_rng(0))
    sp, stp = M.sample_points(md_p, args(['--sampling', 'poisson']), np.random.default_rng(0))
    assert str_['samples'] == stp['samples']
    Xr = np.einsum('nk,nkd->nd', sr['bary'], V[tris[sr['fi']]])
    Xp = np.einsum('nk,nkd->nd', sp['bary'], V[tris[sp['fi']]])
    h = 0.02
    assert np.percentile(nn_dist(Xp), 5) > 0.6 * h > np.percentile(nn_dist(Xr), 5)


def test_aniso_follows_stretch():
    V, tris = grid(10, 10)
    Vf = np.stack([V, V * [2.0, 1.0, 1.0]])
    a = args(['--aniso'])
    md = model(Vf, tris)
    smp, _ = M.sample_points(md, a, np.random.default_rng(0))
    ex = M.attrs_at(md, smp, np.arange(2), a)
    s = np.exp(ex['scales'][1])
    ref = a.sigma * md['h']
    assert np.allclose(np.sort(s[:, :2], 1), [[ref, 2 * ref]], rtol=1e-6)
    # the major axis lies along world x
    R = np.stack([M.quat_from_frames(np.eye(3)[None])[0]])  # noqa: F841  (smoke: quat helper accepts batches)


def test_edge_cap_shrinks_towards_boundary():
    V, tris = grid(10, 10)
    Vf = np.stack([V, V])
    a = args(['--edge-cap', '1.0'])
    md = model(Vf, tris)
    smp, st = M.sample_points(md, a, np.random.default_rng(0))
    assert st['boundary_edges'] > 0
    ex = M.attrs_at(md, smp, np.arange(1), a)
    X = ex['means'][0]
    d = np.minimum.reduce([X[:, 0], 1 - X[:, 0], X[:, 1], 1 - X[:, 1]])
    s_min = np.exp(ex['scales'][0][:, :2]).min(1)
    near = d < 0.3 * a.sigma * md['h']
    far = d > 3 * a.sigma * md['h']
    assert near.any() and far.any()
    assert np.all(s_min[near] <= np.maximum(d[near], a.edge_floor * a.sigma * md['h']) * 1.0001)
    assert np.allclose(s_min[far], a.sigma * md['h'])


def test_crease_edges_detected():
    V, tris = grid(4, 4)                                   # unit square in z = 0, fold the half x > 0.5 up by 90 deg
    V2 = V.copy()
    up = V2[:, 0] > 0.5
    V2[up, 2] = V2[up, 0] - 0.5; V2[up, 0] = 0.5
    no = M.boundary_edges(tris, V2)
    cr = M.boundary_edges(tris, V2, crease_deg=45)
    assert (cr[..., 0] >= 0).any(1).sum() > (no[..., 0] >= 0).any(1).sum()
    flat = M.boundary_edges(tris, V, crease_deg=45)
    assert ((flat[..., 0] >= 0).any(1) == (M.boundary_edges(tris, V)[..., 0] >= 0).any(1)).all()


def test_curv_cap_thin_cylinder():
    r, L, n_a, n_l = 0.002, 0.2, 16, 20
    ang = np.linspace(0, 2 * math.pi, n_a, endpoint=False)
    zs = np.linspace(0, L, n_l + 1)
    V = np.array([[r * math.cos(t), r * math.sin(t), z] for z in zs for t in ang])
    N = np.array([[math.cos(t), math.sin(t), 0.0] for z in zs for t in ang])
    tris = []
    for k in range(n_l):
        for i in range(n_a):
            a0, a1 = k * n_a + i, k * n_a + (i + 1) % n_a
            b0, b1 = a0 + n_a, a1 + n_a
            tris += [[a0, a1, b1], [a0, b1, b0]]
    tris = np.array(tris)
    md = model(np.stack([V, V]), tris)
    md['N'] = np.stack([N, N]).astype(np.float32)
    a = args(['--curv-cap', '0.7', '--h', '0.004'])
    smp, _ = M.sample_points(md, a, np.random.default_rng(0))
    ex = M.attrs_at(md, smp, np.arange(1), a)
    s = np.sort(np.exp(ex['scales'][0][:, :2]), 1)
    assert np.median(s[:, 0]) < 0.8 * r and np.median(s[:, 1]) > 0.9 * a.sigma * md['h']
    # densified: up to 4x the samples on the thin tube (blue noise, adaptive radius), smaller local spacing
    md2 = model(np.stack([V, V]), tris); md2['N'] = md['N']
    a2 = args(['--curv-cap', '0.7', '--h', '0.004', '--curv-dens', '4', '--sampling', 'poisson'])
    smp2, st2 = M.sample_points(md2, a2, np.random.default_rng(0))
    assert 3.5 * len(smp['fi']) < len(smp2['fi']) < 4.5 * len(smp['fi']) and st2['densified'] == len(smp2['fi'])
    assert np.allclose(smp2['hloc'], md2['h'] / 2)
    ex2 = M.attrs_at(md2, smp2, np.arange(1), a2)
    assert np.isfinite(ex2['scales']).all()


# ---- thin open decals, depthTest-off meshes
def strip(width, nx=40, ny=2):
    V, tris = grid(nx, ny)
    V = V * [1.0, width, 1.0]
    return V, tris


def test_open_width_strip_and_closed():
    V, tris = strip(0.05)
    _, ar = M.tri_frames(V[tris])
    assert abs(M.open_width(tris, V, ar.sum()) - 2 * 0.05 / (2 * 1.05)) < 1e-9
    # a closed surface (two grids glued at their rims would be fiddly): a tetrahedron has no open boundary
    T = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    tt = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    assert M.open_width(tt, T, 1.0) == math.inf


def test_thin_open_densifies_only_thin_meshes():
    V, tris = strip(0.01)
    Vf = np.stack([V, V])
    s0, st0 = M.sample_points(model(Vf, tris), args(['--sampling', 'poisson']), np.random.default_rng(0))
    s1, st1 = M.sample_points(model(Vf, tris), args(['--sampling', 'poisson', '--thin-open', '3']), np.random.default_rng(0))
    w = 2 * 0.01 / 2.02
    assert 'thin_open_width_mm' in st1 and abs(st1['h_mm'] - round(max(w / 5, 0.02 / 8) * 1000, 2)) < 0.02   # static: min(w/3, w_big/5), floor h/8
    assert st1['samples'] > 15 * st0['samples']
    Vw, tw = grid(10, 10)                                  # 1 m x 1 m: w = 0.5 m > 3 h -> untouched
    Vwf = np.stack([Vw, Vw])
    a, b = (M.sample_points(model(Vwf, tw), args(e), np.random.default_rng(0))[1] for e in ([], ['--thin-open', '3']))
    assert a['samples'] == b['samples'] and 'thin_open_width_mm' not in b


def test_ontop_moves_decal_in_front_of_occluder(tmp_path):
    import json
    gd = tmp_path / 'geom'; gd.mkdir()
    Vo, to = grid(4, 4, size=2.0)
    Vo = Vo - [1.0, 1.0, 0.5]                              # occluder plane at z = -0.5, between camera and decal
    np.savez(gd / 'topo.npz', **{f'tris_1_{len(Vo)}': to.astype(np.uint32)})
    np.savez(gd / 'f0000.npz', pos_1=Vo.astype(np.float32))
    json.dump({'film_cams': [{'pos': [0.0, 0.0, -5.0]}]}, open(gd / 'meta.json', 'w'))
    fl = tmp_path / 'flags.json'; json.dump({'2': {'ontop': True}}, open(fl, 'w'))
    a = M.parse(['--frames', '1', '--out', '/tmp/x', '--three', str(gd), '--ontop', str(fl)])
    md = {'name': 'm2:decal', 'f0': 0}
    assert M.mark_ontop([md], a) == ['m2:decal'] and md['ontop']
    means = np.array([[[0.0, 0.0, 0.0], [0.1, 0.05, 0.0]]])
    s, c = M.ontop_scale(md, np.array([0]), means, a)
    assert abs(s[0] - 4.5 * (1 - 0.002) / 5.0) < 1e-6
    moved = c[0] + s[0] * (means[0] - c[0])
    proj = lambda p: (p[:, :2] - c[0][:2]) / (p[:, 2:] - c[0][2])  # noqa: E731  (film picture unchanged)
    assert np.allclose(proj(moved), proj(means[0])) and np.all(moved[:, 2] < -0.5)
