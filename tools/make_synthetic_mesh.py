#!/usr/bin/env python3
"""Synthetic animated-mesh scene for the standard recipe's mesh source (no third-party assets).

  python tools/make_synthetic_mesh.py <out dir> [--frames 36] [--res 32]

Writes the per-frame mesh export that train/mesh/mesh2gs.py --three reads (format: docs/recipe.md, "Mesh geometry"):
  <out>/geom/meta.json      {"meshes": {"<id>": {"path": <name>, "color": [r, g, b]}}, "film_cams": [{"pos", "fov"}, ...]}
  <out>/geom/topo.npz       tris_<id>_<nv>: int32 [n_tris, 3] per mesh id and vertex count
  <out>/geom/f<frame:04d>.npz  pos_<id>, nrm_<id>: float32 [nv, 3] world positions / vertex normals of the visible meshes
  <out>/owners.json         {"<id>": "<owner>"}  (groups meshes; the recipe splits figures from a static far backdrop)
  <out>/cams.json           {"H": <image height>, "cams": [{"pos": [x, y, z], "fov": <deg>}, ...]}
  <out>/scene.json          scene file for tools/faltwerk_recipe.py --source mesh
Content: a pulsating, bouncing sphere ("figure"), a waving ribbon that appears after a third of the clip ("ribbon",
an open mesh) and a static wall far behind them ("backdrop").
"""
import argparse, json, math, os

import numpy as np


def uv_sphere(n):
    th = np.linspace(0, math.pi, n + 1)
    ph = np.linspace(0, 2 * math.pi, 2 * n, endpoint=False)
    V = [[0, 0, 1]] + [[math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)] for t in th[1:-1] for p in ph] \
        + [[0, 0, -1]]
    V = np.array(V, np.float64)
    m = len(ph); F = []
    for j in range(m):                                          # north cap
        F.append([0, 1 + j, 1 + (j + 1) % m])
    for i in range(n - 2):
        for j in range(m):
            a, b = 1 + i * m + j, 1 + i * m + (j + 1) % m
            c, d = a + m, b + m
            F += [[a, c, b], [b, c, d]]
    s = len(V) - 1; base = 1 + (n - 2) * m
    for j in range(m):                                          # south cap
        F.append([base + j, s, base + (j + 1) % m])
    return V, np.array(F, np.int32)


def grid(nx, ny):
    xs, ys = np.meshgrid(np.linspace(0, 1, nx + 1), np.linspace(0, 1, ny + 1), indexing='ij')
    V = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], 1)
    idx = lambda i, j: i * (ny + 1) + j  # noqa: E731
    F = []
    for i in range(nx):
        for j in range(ny):
            F += [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)], [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]]
    return V, np.array(F, np.int32)


def normals(V, F):
    n = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--frames', type=int, default=36); ap.add_argument('--res', type=int, default=16)
    a = ap.parse_args()
    T = a.frames
    gd = os.path.join(a.out, 'geom'); os.makedirs(gd, exist_ok=True)
    Vs, Fs = uv_sphere(a.res)
    Vr, Fr = grid(2 * a.res, 4)
    Vw, Fw = grid(4, 4)
    meshes = {'0': ('figure_sphere', Vs, Fs, [0.92, 0.88, 0.82]), '1': ('ribbon', Vr, Fr, [0.2, 0.4, 0.9]),
              '2': ('backdrop_wall', Vw, Fw, [0.5, 0.5, 0.5])}
    np.savez(os.path.join(gd, 'topo.npz'), **{f'tris_{i}_{len(V)}': F for i, (_, V, F, _) in meshes.items()})
    cams = [{'pos': [4.0 * math.cos(k * math.pi / 2), 0.4, 4.0 * math.sin(k * math.pi / 2)], 'fov': 40.0} for k in range(4)]
    film = [{'pos': [4.0 * math.cos(0.3 + 0.02 * t), 0.5, 4.0 * math.sin(0.3 + 0.02 * t)], 'fov': 40.0} for t in range(T)]
    json.dump({'meshes': {i: {'path': m[0], 'color': m[3]} for i, m in meshes.items()}, 'film_cams': film},
              open(os.path.join(gd, 'meta.json'), 'w'), indent=1)
    json.dump({'0': 'figure', '1': 'ribbon', '2': 'backdrop'}, open(os.path.join(a.out, 'owners.json'), 'w'))
    json.dump({'H': 180, 'cams': cams}, open(os.path.join(a.out, 'cams.json'), 'w'), indent=1)
    for t in range(T):
        ph = 2 * math.pi * t / T
        z = {}
        r = 0.45 * (1 + 0.1 * math.sin(2 * ph))                 # figure: pulsating sphere on a bounce path
        P = Vs * np.array([r, r * (1 - 0.15 * math.sin(ph) ** 2), r]) + np.array([0.3 * math.sin(ph), 0.2 * abs(math.sin(ph)), 0])
        z['pos_0'] = P.astype(np.float32); z['nrm_0'] = normals(P, Fs).astype(np.float32)
        if t >= T // 3:                                         # ribbon: born later (own lifetime), waving open strip
            u, v = Vr[:, 0], Vr[:, 1]
            P = np.stack([-0.8 + 1.6 * u, 0.6 + 0.12 * v, 0.6 + 0.1 * np.sin(4 * math.pi * u + 3 * ph)], 1)
            z['pos_1'] = P.astype(np.float32); z['nrm_1'] = normals(P, Fr).astype(np.float32)
        P = np.stack([-10 + 20 * Vw[:, 0], -5 + 10 * Vw[:, 1], np.full(len(Vw), -80.0)], 1)    # static far wall
        z['pos_2'] = P.astype(np.float32); z['nrm_2'] = normals(P, Fw).astype(np.float32)
        np.savez(os.path.join(gd, f'f{t:04d}.npz'), **z)
    sc = {'name': 'synthetic_mesh', 'looks': ['shaded'],
          'geom': [{'dir': os.path.abspath(gd), 'f0': 0, 'frames': T, 'cams': os.path.abspath(os.path.join(a.out, 'cams.json'))}]}
    json.dump(sc, open(os.path.join(a.out, 'scene.json'), 'w'), indent=1)
    print(os.path.join(a.out, 'scene.json'))


if __name__ == '__main__':
    main()
