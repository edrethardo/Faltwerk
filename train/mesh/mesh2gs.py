#!/usr/bin/env python3
"""Gaussians directly from animated meshes (no training from scratch).

Sources (one of):
  --three <geom dir>   per-frame world geometry (format: docs/recipe.md "Mesh geometry";
                       tools/make_synthetic_mesh.py writes an example)
                       (topo.npz tris_<id>_<nv>, f<frame>.npz pos_<id>/nrm_<id>, meta.json colour per mesh id). Every
                       contiguous run of frames in which a mesh is visible becomes one model with its own lifetime.
  --mesh-dir + --cfg   the Blender/MMD export of bl_export_mesh.py (P41 scene or a detailed model; textures + UVs).
Output: a run dir in the p8_e7 layout (codec.measure / e8-style evaluation work unchanged):
  <out>/k<K>/curves.npz   per 32-frame section: control points (hat basis, knots on frames g0..g0+L-1, soft_knot0 =
                          every section fitted on its own) of means/quats/scales/opacities/sh0
  <out>/k<K>/tracks.npz   per-frame means (NaN when not alive), birth, death, shN (zeros: the looks are view-independent)
  <out>/k<K>/mesh2gs.json sampling + fit statistics (splats per model, curve fit error vs the exact mesh values)

Method (per model):
  1. surface samples: every triangle gets A_i / h^2 samples on average (systematic sampling over the cumulative area,
     area = median over the model's frames), uniform barycentric positions; a sample keeps its (triangle, barycentric)
     for its whole lifetime, so it FOLLOWS the animated surface exactly.
  2. colour (--look): sil = the mesh's flat colour (palette 0 of the film: ink = black; Blender: white like P41 sw);
     shaded = view-independent clay shading albedo * (amb + (1 - amb) |n.L|) from the interpolated vertex normal at
     every frame (same formula as the GT shader in ba3_page.js); color = Blender textures (x MMD diffuse).
  3. flat splats: rotation = the triangle's frame (edge, in-plane normal, normal) per frame (sign-continuous quats),
     in-plane sigma = --sigma x h x sqrt(area_t / area_ref), normal sigma = --flat x in-plane sigma.
  4. curves: weighted least-squares fit of the exact per-frame values onto K hat knots per section (frames outside a
     model's lifetime are edge-padded with weight 1e-3, so knots stay near the surface). No images involved.
  python mesh2gs.py --three <geom> --frames 444 --look sil --out <run> [--h 0.006 --K 9]
"""
import argparse, glob, json, math, os, time

import numpy as np

C0 = 0.28209479177387814
LIGHT = np.array([0.45, 0.8, 0.4]); LIGHT = LIGHT / np.linalg.norm(LIGHT)
ALBEDO = np.array([0.92, 0.88, 0.82]); AMB = 0.3          # = ba3_page.js shadeMat


def quat_from_frames(m):
    """rotation matrices [..., 3, 3] (columns = local axes in world) -> unit quaternions wxyz [..., 4]."""
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = np.zeros(m.shape[:-2] + (4,), np.float64)
    c0 = tr > 0
    c1 = ~c0 & (m[..., 0, 0] >= m[..., 1, 1]) & (m[..., 0, 0] >= m[..., 2, 2])
    c2 = ~c0 & ~c1 & (m[..., 1, 1] >= m[..., 2, 2])
    c3 = ~c0 & ~c1 & ~c2
    s = np.sqrt(np.maximum(tr + 1, 1e-12)) * 2
    q[c0] = np.stack([0.25 * s, (m[..., 2, 1] - m[..., 1, 2]) / s, (m[..., 0, 2] - m[..., 2, 0]) / s,
                      (m[..., 1, 0] - m[..., 0, 1]) / s], -1)[c0]
    s = np.sqrt(np.maximum(1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], 1e-12)) * 2
    q[c1] = np.stack([(m[..., 2, 1] - m[..., 1, 2]) / s, 0.25 * s, (m[..., 0, 1] + m[..., 1, 0]) / s,
                      (m[..., 0, 2] + m[..., 2, 0]) / s], -1)[c1]
    s = np.sqrt(np.maximum(1 + m[..., 1, 1] - m[..., 0, 0] - m[..., 2, 2], 1e-12)) * 2
    q[c2] = np.stack([(m[..., 0, 2] - m[..., 2, 0]) / s, (m[..., 0, 1] + m[..., 1, 0]) / s, 0.25 * s,
                      (m[..., 1, 2] + m[..., 2, 1]) / s], -1)[c2]
    s = np.sqrt(np.maximum(1 + m[..., 2, 2] - m[..., 0, 0] - m[..., 1, 1], 1e-12)) * 2
    q[c3] = np.stack([(m[..., 1, 0] - m[..., 0, 1]) / s, (m[..., 0, 2] + m[..., 2, 0]) / s,
                      (m[..., 1, 2] + m[..., 2, 1]) / s, 0.25 * s], -1)[c3]
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def tri_frames(P):
    """P [..., 3 verts, 3] -> (R [..., 3, 3] columns t, b, n; area [...])."""
    e1 = P[..., 1, :] - P[..., 0, :]; e2 = P[..., 2, :] - P[..., 0, :]
    cr = np.cross(e1, e2)
    a2 = np.linalg.norm(cr, axis=-1)
    n = cr / np.maximum(a2, 1e-20)[..., None]
    t = e1 / np.maximum(np.linalg.norm(e1, axis=-1), 1e-20)[..., None]
    b = np.cross(n, t)
    return np.stack([t, b, n], -1), 0.5 * a2


def vertex_normals(V, tris):
    """area-weighted smooth vertex normals per frame: V [nT, nV, 3] -> [nT, nV, 3]."""
    P = V[:, tris]
    cr = np.cross(P[:, :, 1] - P[:, :, 0], P[:, :, 2] - P[:, :, 0])
    N = np.zeros_like(V)
    for k in range(3):
        for t in range(V.shape[0]):
            np.add.at(N[t], tris[:, k], cr[t])
    return N / np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-20)


def shade(n):
    return ALBEDO * (AMB + (1 - AMB) * np.abs(n @ LIGHT))[..., None]


# ---------------------------------------------------------------- sources
def models_three(gd, T, only=None, f0=0):
    """frames f0 .. f0+T-1 of the export; local frame indices 0 .. T-1 (--f0: a later segment of the film)."""
    meta = json.load(open(os.path.join(gd, 'meta.json')))
    topo = np.load(os.path.join(gd, 'topo.npz'))
    owners_p = os.path.join(gd, '..', 'owners.json')
    owners = json.load(open(owners_p)) if os.path.exists(owners_p) else {}
    runs = {}                                     # (id, nv) -> list of (frame, pos, nrm)
    for f in range(T):
        z = np.load(os.path.join(gd, f'f{f0 + f:04d}.npz'))
        for k in z.files:
            if not k.startswith('pos_'):
                continue
            i = k[4:]
            if only and owners.get(i) not in only:
                continue
            P = z[k]; N = z['nrm_' + i]
            runs.setdefault((i, P.shape[0]), []).append((f, P, N))
    models = []
    for (i, nv), lst in runs.items():
        mm = meta['meshes'][i]
        tris = topo[f'tris_{i}_{nv}'].astype(np.int64)
        # contiguous frame runs = separate lifetimes
        start = 0
        for j in range(1, len(lst) + 1):
            if j == len(lst) or lst[j][0] != lst[j - 1][0] + 1:
                seg = lst[start:j]
                models.append({'name': f"m{i}:{mm.get('path', '')[-40:]}", 'f0': seg[0][0], 'f1': seg[-1][0] + 1,
                               'owner': owners.get(i),
                               'tris': tris, 'V': np.stack([s[1] for s in seg]),
                               'N': np.stack([s[2] for s in seg]),
                               'color': np.array(mm.get('color') or [0, 0, 0], np.float64)})
                start = j
    return models


def load_tex(mesh_dir, mats):
    from PIL import Image
    return [np.asarray(Image.open(os.path.join(mesh_dir, d['tex'])).convert('RGBA')).astype(np.float32) / 255.0
            if d.get('tex') else None for d in mats]


def models_blender(mesh_dir, cfg):
    models = []
    for m in cfg['models']:
        z = np.load(os.path.join(mesh_dir, f"{m['name']}.npz"))
        meta = json.load(open(os.path.join(mesh_dir, f"{m['name']}.json")))
        models.append({'name': m['name'], 'f0': int(m['f0']), 'f1': int(m['f1']), 'tris': z['tris'].astype(np.int64),
                       'V': z['verts'].astype(np.float32), 'N': None, 'color': np.ones(3),
                       'uvt': z['uv'], 'mat': z['mat'], 'mats': meta['materials'], 'tex': load_tex(mesh_dir, meta['materials'])})
    return models


def tex_lookup(img, uv):
    """bilinear, repeat wrap; uv [n,2] Blender convention (v up) -> [n,4]."""
    H, W = img.shape[:2]
    x = (uv[:, 0] % 1.0) * W - 0.5
    y = (1.0 - (uv[:, 1] % 1.0)) * H - 0.5
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    x0 %= W; y0 %= H; x1 = (x0 + 1) % W; y1 = (y0 + 1) % H
    return ((img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy) + (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy)


def texture_colour(md, fi, bary, P0, a, rng):
    """Blender textured colour + alpha of the samples, averaged over the splat footprint (affine UV map per triangle)."""
    uvt, mat, mats, tex = md['uvt'], md['mat'], md['mats'], md['tex']
    sig = a.sigma * a.h
    uv0 = np.einsum('nk,nkd->nd', bary, uvt[fi])
    R0, _ = tri_frames(P0)
    e1 = P0[:, 1] - P0[:, 0]; e2 = P0[:, 2] - P0[:, 0]
    L2 = np.stack([np.einsum('nd,nd->n', e1, R0[..., 0]), np.einsum('nd,nd->n', e1, R0[..., 1]),
                   np.einsum('nd,nd->n', e2, R0[..., 0]), np.einsum('nd,nd->n', e2, R0[..., 1])], 1).reshape(-1, 2, 2)
    du = np.stack([uvt[fi, 1] - uvt[fi, 0], uvt[fi, 2] - uvt[fi, 0]], 1)
    det = L2[:, 0, 0] * L2[:, 1, 1] - L2[:, 0, 1] * L2[:, 1, 0]
    good = np.abs(det) > 1e-14
    Linv = np.zeros_like(L2)
    Linv[good, 0, 0] = L2[good, 1, 1] / det[good]; Linv[good, 1, 1] = L2[good, 0, 0] / det[good]
    Linv[good, 0, 1] = -L2[good, 0, 1] / det[good]; Linv[good, 1, 0] = -L2[good, 1, 0] / det[good]
    JT = np.einsum('nij,njk->nik', Linv, du)
    rgba = np.zeros((len(fi), 4))
    m_of = mat[fi]
    for s in range(a.tex_ss):
        off = rng.standard_normal((len(fi), 2)) * sig * (1.0 if s else 0.0)
        uv = uv0 + np.einsum('ni,nik->nk', off, JT)
        col = np.ones((len(fi), 4))
        for mi in np.unique(m_of):
            sel = m_of == mi
            if mi < len(tex) and tex[mi] is not None:
                col[sel] = tex_lookup(tex[mi], uv[sel])
        rgba += col / a.tex_ss
    diff = np.array([mats[mi].get('diffuse', [1, 1, 1]) for mi in range(len(mats))] or [[1, 1, 1]])
    malpha = np.array([mats[mi].get('alpha', 1.0) for mi in range(len(mats))] or [1.0])
    return np.clip(rgba[:, :3] * diff[m_of] * a.gain, 0, 1), rgba[:, 3] * malpha[m_of]


def sample_points(md, a, rng):
    """pass 1: fixed (triangle, barycentric) samples + static colour/alpha; area = median over <= 24 frames."""
    tris, V = md['tris'], md['V']
    nT = V.shape[0]
    sub = np.unique(np.linspace(0, nT - 1, min(nT, 24)).astype(int))
    _, area = tri_frames(V[sub][:, tris].astype(np.float64))
    aref = np.median(area, 0)
    if np.max(area.sum(1)) > a.grow_ratio * np.median(area.sum(1)):     # a mesh that scales up a lot (the apple core
        aref = np.max(area, 0)                                            # swells x14.7): sample for its largest size
        md['grown'] = True
    ok = aref > 1e-12
    h = h0 = a.h
    if getattr(a, 'thin_open', 0) > 0 and ok.any():
        # thin OPEN meshes (decals, ribbons, flat cut-outs): width w = 2 A / P_open (A area,
        # P_open open-boundary length; a strip of width w gives w). With h > w/2 the few samples cannot draw the shape
        # (smile: 15 splats for a 4 px x 51 px crescent -> broken, jagged line). Spacing <= w / thin_open, floored at
        # --h-floor (no finer than ~0.5 px in the closest view, set per mesh by the recipe), and sampled at the LARGEST
        # size (a decal that widens 3x would otherwise open gaps; such meshes are small, the extra samples are cheap).
        w = open_width(tris, V[sub[len(sub) // 2]].astype(np.float64), float(area[len(sub) // 2].sum()))
        md['open_width'] = w
        if w < a.thin_open * h:
            # the samples sit at the LARGEST frame and --thin-inset drops a 1-sigma band at its edge: keep >= ~3 rows
            # there too (w_big / (k + 2)); else a decal that is wide all the time (the grin in the push, D2) gets 1-2
            # patchy rows while the same crescent in D1 (thin at its median frame) was fine.
            kb = int(np.argmax(area.sum(1)))
            wb = open_width(tris, V[sub[kb]].astype(np.float64), float(area[kb].sum()))
            h = h0 = max(min(w / a.thin_open, wb / (a.thin_open + 2)), getattr(a, 'h_floor', 0.0) or 0.0, a.h / 8)
            aref = np.max(area, 0); ok = aref > 1e-12
            md['thin'] = md['grown'] = True
    cap = a.cap_owner.get(md.get('owner'), a.max_per_model)               # --cap-owner <owner>=5000: far scenery coarser
    if cap and aref[ok].sum() / h ** 2 > cap:   # cap huge-area meshes (broom bristles, hat)
        h = math.sqrt(aref[ok].sum() / cap)
    md['h'] = h
    tot = area.sum(1)
    if a.sampling == "poisson" or a.aniso or a.curv_dens > 1 or _aniso(md, a):   # reference frame: median (grown: largest) total area
        md['kref'] = int(sub[np.argmax(tot) if md.get('grown') else np.argmin(np.abs(tot - np.median(tot)))])
    m_t = np.ones(len(tris))
    if a.curv_dens > 1 and a.curv_cap > 0 and ok.any():
        # thin/curved parts: the curvature cap shrinks their splats to c/|kappa|; sample them up to curv_dens x denser
        # (spacing h/sqrt(m)) so the smaller splats still close the surface (apple stem, strands, rims). Capped meshes
        # keep their sample budget: the extra density there is taken from the flat parts (coarser h).
        kr = md['kref']
        Nk = md['N'][kr] if md.get('N') is not None else vertex_normals(V[[kr]].astype(np.float64), tris)[0]
        kv, _ = shape_op(V[kr][tris].astype(np.float64), np.asarray(Nk, np.float64)[tris])
        kmax = np.abs(kv).max(-1)
        for _ in range(3):
            rho = np.clip(a.curv_cap / np.maximum(kmax, 1e-9) / (a.sigma * h), 1e-6, 1.0)
            m_t = np.minimum(1.0 / rho ** 2, a.curv_dens)
            aeff = float((aref[ok] * m_t[ok]).sum())
            h = math.sqrt(aeff / cap) if cap and aeff / h0 ** 2 > cap else h0
        md['h'] = h
    w = np.where(ok, aref * m_t / h ** 2, 0.0)
    c = np.cumsum(w) + rng.random()
    n_i = np.diff(np.floor(np.concatenate([[c[0] - w[0]], c]))).astype(np.int64)
    if a.sampling == 'poisson' and n_i.sum() > 1:              # blue noise: k x candidates, weighted elimination
        n = int(n_i.sum())
        wk = w * a.poisson_k
        ck = np.cumsum(wk) + rng.random()
        fi = np.repeat(np.arange(len(tris)), np.diff(np.floor(np.concatenate([[ck[0] - wk[0]], ck]))).astype(np.int64))
        r1, r2 = rng.random(len(fi)), rng.random(len(fi))
        s1 = np.sqrt(r1)
        bary = np.stack([1 - s1, s1 * (1 - r2), s1 * r2], 1)
        Pref = V[md['kref']][tris[fi]].astype(np.float64)
        aK = area[list(sub).index(md['kref'])]
        keep = poisson_eliminate(np.einsum('nk,nkd->nd', bary, Pref), n, float((aK * m_t).sum()),
                                 scale=None if a.curv_dens <= 1 else 1.0 / np.sqrt(m_t[fi]))
        keep = np.sort(keep)
        fi, bary = fi[keep], bary[keep]
    else:
        fi = np.repeat(np.arange(len(tris)), n_i)
        r1, r2 = rng.random(len(fi)), rng.random(len(fi))
        s1 = np.sqrt(r1)
        bary = np.stack([1 - s1, s1 * (1 - r2), s1 * r2], 1)
    k_in = getattr(a, 'thin_inset', 0) if md.get('thin') else getattr(a, 'open_inset', 0)
    if k_in > 0 and len(fi):
        # --thin-inset k: the 50 % coverage line of a dense Gaussian layer lies ~0.72 sigma OUTSIDE its last
        # sample row (decal 1.5x too wide: a 4 px crescent drawn 6 px). Samples closer than k sigma to the open boundary
        # (reference frame) are dropped, so the drawn edge lands on the mesh edge. --open-inset k: the same for every
        # other mesh with an open boundary (hems, brims; A/B for the contour spill).
        kr = md.get('kref', sub[len(sub) // 2])
        Vk = V[kr].astype(np.float64)
        sa, sb = open_segments(tris, Vk)
        if len(sa):
            X = np.einsum('nk,nkd->nd', bary, Vk[tris[fi]])
            dmin = np.full(len(X), np.inf)
            for i0 in range(0, len(X), 4096):
                Xi = X[i0:i0 + 4096]
                for j in range(0, len(sa), 512):
                    A_, B_ = sa[j:j + 512], sb[j:j + 512]
                    ab = B_ - A_
                    t_ = np.clip(np.einsum('nsd,sd->ns', Xi[:, None] - A_[None], ab) / np.maximum((ab * ab).sum(1), 1e-20), 0, 1)
                    dmin[i0:i0 + 4096] = np.minimum(dmin[i0:i0 + 4096],
                                                    np.linalg.norm(Xi[:, None] - (A_[None] + t_[..., None] * ab[None]), axis=-1).min(1))
            hl = h / np.sqrt(m_t[fi]) if a.curv_dens > 1 else h
            delta = k_in * _sig(md, a) * hl
            keep = dmin >= delta
            if keep.sum() >= max(8, 0.25 * len(fi)):
                fi, bary = fi[keep], bary[keep]
                md['inset_dropped'] = int((~keep).sum())
    st = {'name': md['name'], 'f0': int(md['f0']), 'f1': int(md['f1']), 'triangles': int(len(tris)),
          'area_m2': float(aref.sum()), 'samples': int(len(fi)), 'h_mm': round(h * 1000, 2)}
    if md.get('thin'):
        st['thin_open_width_mm'] = round(md['open_width'] * 1000, 2)
    alpha = np.ones(len(fi)); rgb = None
    if a.look == 'color':
        if 'uvt' not in md:
            raise SystemExit('--look color needs a textured (Blender) source')
        rgb, alpha = texture_colour(md, fi, bary, V[0][tris[fi]].astype(np.float64), a, rng)
        keep = alpha >= a.alpha_min
        fi, bary, rgb, alpha = fi[keep], bary[keep], rgb[keep], alpha[keep]
        st['alpha_dropped'] = int((~keep).sum())
    elif a.look == 'sil':
        rgb = np.broadcast_to(md['color'], (len(fi), 3)).copy()
    st['kept'] = int(len(fi))
    if a.look == 'shaded' and md.get('N') is None:
        md['N'] = vertex_normals(V.astype(np.float64), tris).astype(np.float32)
    smp = {'fi': fi, 'bary': bary, 'rgb': rgb, 'alpha': alpha, 'aref': aref[fi]}
    if a.curv_dens > 1 and len(fi):                            # local sample spacing (densified thin parts)
        smp['hloc'] = h / np.sqrt(m_t[fi])
        st['densified'] = int((m_t[fi] > 1.01).sum())
    if (a.aniso or _aniso(md, a)) and len(fi):                                    # in-plane edge matrix of the reference frame (inverse)
        smp['Linv_ref'] = np.linalg.pinv(edge_mat(V[md['kref']][tris[fi]].astype(np.float64)))
    if a.edge_cap > 0 and len(fi):
        md['bedges'] = boundary_edges(tris, V[sub[len(sub) // 2]], crease_deg=a.crease_deg)
        st['boundary_edges'] = int((md['bedges'][..., 0] >= 0).any(1).sum())
    return smp, st


def edge_mat(P):
    """[..., 3 verts, 3] -> 2x2 matrices (columns = edges e1, e2 in the triangle's own (t, b) frame)."""
    R, _ = tri_frames(P)
    e1 = P[..., 1, :] - P[..., 0, :]; e2 = P[..., 2, :] - P[..., 0, :]
    t, b = R[..., 0], R[..., 1]
    return np.stack([np.stack([np.einsum('...d,...d->...', e1, t), np.einsum('...d,...d->...', e2, t)], -1),
                     np.stack([np.einsum('...d,...d->...', e1, b), np.einsum('...d,...d->...', e2, b)], -1)], -2)


def open_segments(tris, V0):
    """open-boundary edges of a mesh (welded by position) as segment end points (A [m, 3], B [m, 3])."""
    q = np.round(np.asarray(V0, np.float64) / 1e-6).astype(np.int64)
    _, weld = np.unique(q, axis=0, return_inverse=True)
    T = weld.ravel()[tris]
    E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    Eo = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    _, inv, cnt = np.unique(E, axis=0, return_inverse=True, return_counts=True)
    b = (cnt[inv.ravel()] == 1) & (E[:, 0] != E[:, 1])
    V0 = np.asarray(V0, np.float64)
    return V0[Eo[b, 0]], V0[Eo[b, 1]]


def open_width(tris, V0, area):
    """width of an open mesh = 2 area / open-boundary length (welded by position like boundary_edges; closed: inf)."""
    A, B = open_segments(tris, V0)
    if not len(A):
        return math.inf
    return 2 * area / max(float(np.linalg.norm(A - B, axis=1).sum()), 1e-12)


def boundary_edges(tris, V0, maxe=4, crease_deg=0.0):
    """open-boundary edges near every triangle: edges used by one triangle only, after welding vertices by position
    (UV seams split vertices without being an edge of the surface); crease_deg > 0 adds sharp creases (two faces whose
    normals differ by more than crease_deg, e.g. a hat brim's rim with split normals): splats must not reach across
    them either. Per triangle the edges that touch one of its vertices -> [nT, maxe, 2] vertex ids (-1 = none)."""
    q = np.round(np.asarray(V0, np.float64) / 1e-6).astype(np.int64)
    _, weld = np.unique(q, axis=0, return_inverse=True)
    weld = weld.ravel()
    T = weld[tris]
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    Eo = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])   # original ids (positions per frame)
    key = np.sort(E, 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    bnd = (cnt[inv] == 1) & (key[:, 0] != key[:, 1])
    if crease_deg > 0:
        P = np.asarray(V0, np.float64)[tris]
        fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-20)
        two = np.nonzero(cnt[inv] == 2)[0]
        o = two[np.argsort(inv[two], kind='stable')]
        a_, b_ = o[0::2], o[1::2]
        nT = len(tris)
        cosang = np.einsum('nd,nd->n', fn[a_ % nT], fn[b_ % nT])
        sharp = cosang < math.cos(math.radians(crease_deg))
        bnd[a_[sharp]] = True
    out = np.full((len(tris), maxe, 2), -1, np.int64)
    if not bnd.any():
        return out
    be_w, be_o = E[bnd], Eo[bnd]
    nv = weld.max() + 1
    inc = [[] for _ in range(nv)]                                       # welded vertex -> boundary edges
    for k, (u, v) in enumerate(be_w):
        inc[u].append(k); inc[v].append(k)
    for ti in np.nonzero(np.isin(T, np.unique(be_w)).any(1))[0]:
        ks = []
        for u in T[ti]:
            for k in inc[u]:
                if k not in ks:
                    ks.append(k)
        for j, k in enumerate(ks[:maxe]):
            out[ti, j] = be_o[k]
    return out


def poisson_eliminate(X, n, area, scale=None):
    """weighted sample elimination (Yuksel 2015, 'Sample Elimination for Generating Poisson Disk Sample Sets'):
    candidates X [M, 3] on a surface of total area `area` -> indices of n blue-noise samples. Neighbours from a uniform
    grid of cell 2 r_max; weight (1 - d/2r_max)^8 with d >= r_min; greedy removal of the most crowded candidate.
    scale [M] (<= 1): local radius factor (adaptive density: r_i = r_max * scale_i, pair radius = mean of both)."""
    import heapq
    M = len(X)
    if n >= M:
        return np.arange(M)
    rmax = math.sqrt(area / (2 * math.sqrt(3) * n)); R = 2 * rmax
    rmin = rmax * (1 - (n / M) ** 1.5) * 0.65
    rs = np.ones(M) if scale is None else np.asarray(scale, np.float64)
    c = np.floor(X / R).astype(np.int64); c -= c.min(0) - 1
    dims = c.max(0) + 2
    code = (c[:, 0] * dims[1] + c[:, 1]) * dims[2] + c[:, 2]
    order = np.argsort(code, kind='stable'); sc = code[order]
    uc, start = np.unique(sc, return_index=True); end = np.r_[start[1:], len(sc)]
    I, J, D = [], [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                nc = code + (dx * dims[1] + dy) * dims[2] + dz
                pos = np.minimum(np.searchsorted(uc, nc), len(uc) - 1)
                i = np.nonzero(uc[pos] == nc)[0]
                if not len(i):
                    continue
                s, e = start[pos[i]], end[pos[i]]
                cnt = e - s
                ii = np.repeat(i, cnt)
                jj = order[np.repeat(s, cnt) + np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)]
                d = np.linalg.norm(X[ii] - X[jj], axis=1)
                k = (ii != jj) & (d < R * 0.5 * (rs[ii] + rs[jj]))
                I.append(ii[k]); J.append(jj[k]); D.append(d[k])
    I, J, D = np.concatenate(I), np.concatenate(J), np.concatenate(D)
    sp = 0.5 * (rs[I] + rs[J])
    wgt = (1 - np.maximum(D, rmin * sp) / (R * sp)) ** 8
    W = np.bincount(I, wgt, minlength=M)
    o = np.argsort(I, kind='stable'); I, J, wgt = I[o], J[o], wgt[o]
    ptr = np.r_[0, np.cumsum(np.bincount(I, minlength=M))]
    alive = np.ones(M, bool)
    heap = [(-W[i], i) for i in range(M)]
    heapq.heapify(heap)
    Wl, Jl, wl, pl = W.tolist(), J.tolist(), wgt.tolist(), ptr.tolist()
    removed = 0
    while removed < M - n:
        w_, i = heapq.heappop(heap)
        if not alive[i] or -w_ != Wl[i]:
            continue
        alive[i] = False; removed += 1
        for p in range(pl[i], pl[i + 1]):
            j = Jl[p]
            if alive[j]:
                Wl[j] -= wl[p]
                heapq.heappush(heap, (-Wl[j], j))
    return np.nonzero(alive)[0]


_GEO = {}


def _geo(gd):
    if gd not in _GEO:
        _GEO[gd] = {'topo': np.load(os.path.join(gd, 'topo.npz')), 'tris': {},
                    'film': json.load(open(os.path.join(gd, 'meta.json')))['film_cams']}
    return _GEO[gd]


def filter_ids(models, a):
    """--mesh-ids / --skip-mesh-ids (export mesh ids, comma lists): a decal stream with its own position step."""
    inc = {x for x in (getattr(a, 'mesh_ids', None) or '').split(',') if x}
    exc = {x for x in (getattr(a, 'skip_mesh_ids', None) or '').split(',') if x}
    mid = lambda md: md['name'].split(':')[0][1:]              # noqa: E731
    return [md for md in models if (not inc or mid(md) in inc) and mid(md) not in exc]


def mark_ontop(models, a):
    """--ontop flags.json (ba3_export.py flags): models of meshes drawn with depthTest off get md['ontop']."""
    if not getattr(a, 'ontop', None):
        return []
    fl = json.load(open(a.ontop))
    hit = []
    for md in models:
        i = md['name'].split(':')[0][1:]
        if fl.get(i, {}).get('ontop'):
            md['ontop'] = True; md['mesh_id'] = i; hit.append(md['name'])
    return hit


def _occl(c, m, z, ontop_ids, g, slack=1e-4):
    """nearest hit distance along the rays from c through the points m [n, 3] with any non-decal triangle of frame npz z
    that lies in front of the point (inf: free)."""
    d = m - c; tp = np.linalg.norm(d, axis=1); u = d / tp[:, None]
    uc = u.mean(0); uc /= np.linalg.norm(uc)
    cosr = float(np.min(u @ uc)) - 1e-4                       # angular radius of the decal seen from the camera
    best = np.full(len(m), np.inf)
    for k in z.files:
        if not k.startswith('pos_') or k[4:] in ontop_ids:
            continue
        V = z[k].astype(np.float64)
        key = f'tris_{k[4:]}_{V.shape[0]}'
        if key not in g['tris']:
            g['tris'][key] = g['topo'][key].astype(np.int64)
        tr = g['tris'][key]
        dv = V - c; dn = np.linalg.norm(dv, axis=1)
        inside = (dv @ uc) / np.maximum(dn, 1e-12) > cosr - 0.02
        if not inside.any():
            continue
        tr = tr[inside[tr].any(1)]
        ch = max(256, int(2e6 // max(len(m), 1)))
        for j in range(0, len(tr), ch):                       # Moeller-Trumbore, rays x triangles
            P0, P1, P2 = V[tr[j:j + ch, 0]], V[tr[j:j + ch, 1]], V[tr[j:j + ch, 2]]
            e1, e2 = P1 - P0, P2 - P0
            pv = np.cross(u[:, None], e2[None])
            det = np.einsum('td,ntd->nt', e1, pv)
            ok = np.abs(det) > 1e-14
            inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
            sv = c - P0
            uu = np.einsum('td,ntd->nt', sv, pv) * inv
            qv = np.cross(sv, e1)
            vv = np.einsum('nd,td->nt', u, qv) * inv
            th = np.einsum('td,td->t', e2, qv)[None] * inv
            hitm = ok & (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (th > 0) & (th < tp[:, None] * (1 - slack))
            if hitm.any():
                best = np.minimum(best, np.where(hitm, th, np.inf).min(1))
    return best, tp


def _qfwd(q):
    """three.js camera quaternion [x, y, z, w] -> forward vector (local -Z in world)."""
    x, y, z, w = q
    return -np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def _extra_cams(a, F):
    """--ontop-cams cams.json: positions / forward / fov of the non-film cameras at global frame F."""
    if not getattr(a, 'ontop_cams', None):
        return []
    if a.ontop_cams not in _GEO:
        _GEO[a.ontop_cams] = json.load(open(a.ontop_cams)).get('cams', [])
    out = []
    for cm in _GEO[a.ontop_cams]:
        if cm.get('film'):
            continue
        c = cm['track'][F] if isinstance(cm.get('track'), list) else cm
        if not isinstance(c, dict) or 'pos' not in c:
            continue
        out.append((np.asarray(c['pos'], np.float64), _qfwd(c['quat']) if 'quat' in c else None, float(c.get('fov', 30.0))))
    return out


def ontop_scale(md, idx, means, a, margin=0.002):
    """per frame s <= 1: the decal (all its splats, one rigid scale about the film camera centre) sits in front of the
    nearest other surface hit by the camera rays through its splats (margin = relative gap). With --ontop-cams the scale
    is lowered further (down to --ontop-smin) until the decal is also unoccluded in every other camera that faces it and
    has it in view (other camera passes: a decal drawn over
    everything must show there too). -> s [nt], centres [nt, 3]."""
    g = _geo(a.three)
    if a.ontop not in _GEO:                                    # ids drawn on top (kept off `a`: vars(a) goes to JSON)
        _GEO[a.ontop] = {k for k, v in json.load(open(a.ontop)).items() if v.get('ontop')}
    ontop_ids = _GEO[a.ontop]
    smin = getattr(a, 'ontop_smin', 0.85)
    S, C = [], []
    for t, li in enumerate(np.asarray(idx)):
        F = a.f0 + md['f0'] + int(li)
        c = np.asarray(g['film'][F]['pos'], np.float64)
        m = means[t]
        if len(m) > 512:                                       # one rigid scale per frame: a subset of rays suffices
            m = m[np.linspace(0, len(m) - 1, 512).astype(np.int64)]
        z = np.load(os.path.join(a.three, f'f{F:04d}.npz'))
        best, tp = _occl(c, m, z, ontop_ids, g)
        s = max(float(np.min(np.minimum(best * (1 - margin) / tp, 1.0))), 0.05)
        cams = []
        pc = m.mean(0); nf = (c - pc) / np.linalg.norm(c - pc)
        for ck, fw, fov in _extra_cams(a, F):
            v = pc - ck; dist = np.linalg.norm(v)
            if np.dot(nf, -v / dist) <= 0.2:                   # sees the decal from behind / grazing: not its view
                continue
            if fw is not None and np.dot(fw, v / dist) < math.cos(math.radians(0.75 * fov)):
                continue                                       # decal outside that camera's view
            cams.append(ck)
        while cams and s > smin:
            mm = c + s * (m - c)
            if all(np.isinf(_occl(ck, mm, z, ontop_ids, g)[0]).all() for ck in cams):
                break
            s -= 0.004
        S.append(s); C.append(c)
    return np.array(S), np.array(C)


def _sig(md, a):
    """in-plane sigma / h of a mesh."""
    return a.thin_sigma if md.get('thin') and getattr(a, 'thin_sigma', 0) > 0 else a.sigma


def _aniso(md, a):
    """--thin-aniso: thin decals follow their deformation gradient (the smile opens 3.2x in height, 1.7x in
    width; an isotropic sqrt(area) size leaves gaps along the crescent -> a row of dots)."""
    return bool(md.get('thin') and getattr(a, 'thin_aniso', False))


def attrs_at(md, smp, idx, a, qprev=None):
    """pass 2: exact per-frame splat values at local frame indices idx (sign-continuous quats from qprev)."""
    tris, fi, bary = md['tris'], smp['fi'], smp['bary']
    Pf = md['V'][idx][:, tris[fi]].astype(np.float64)                        # [nt, n, 3, 3]
    Rf, af = tri_frames(Pf)
    means = np.einsum('nk,tnkd->tnd', bary, Pf)
    nt, n = len(idx), len(fi)
    if a.look == 'shaded':
        nn = np.einsum('nk,tnkd->tnd', bary, md['N'][idx][:, tris[fi]].astype(np.float64))
        nl = np.linalg.norm(nn, axis=-1, keepdims=True)
        nn = np.where(nl > 1e-8, nn / np.maximum(nl, 1e-20), Rf[..., 2])
        rgb_t = shade(nn)
    else:
        rgb_t = np.broadcast_to(smp['rgb'], (nt, n, 3))
    q = quat_from_frames(Rf)
    if a.aniso or _aniso(md, a) or a.curv_cap > 0 or a.edge_cap > 0:
        q, scales = aniso_cov(md, smp, idx, a, Pf, Rf, af, means)
    else:
        ratio = np.sqrt(np.clip(af / np.maximum(smp['aref'], 1e-12), 0.002 if md.get('grown') else 0.05, 20.0))
        s_in = _sig(md, a) * smp.get('hloc', md['h']) * ratio
        scales = np.log(np.stack([s_in, s_in, np.maximum(a.flat * s_in, a.min_flat)], -1))
    for t in range(nt):
        ref = q[t - 1] if t else qprev
        if ref is not None:
            q[t, np.einsum('nd,nd->n', q[t], ref) < 0] *= -1
    if md.get('ontop') and n:
        # drawn over everything in their film (depthTest off: the smile shows over the fringe when she bows).
        # Splats are depth-sorted, so the decal moves along the film camera's rays to just in front of the nearest
        # occluder: a scale s <= 1 about the camera centre (positions AND sizes) leaves the film picture unchanged.
        s, c = ontop_scale(md, idx, means, a)
        means = c[:, None] + s[:, None, None] * (means - c[:, None])
        scales = scales + np.log(s)[:, None, None]
        md.setdefault('ontop_s', []).extend(s.tolist())
    op = np.clip(smp['alpha'] * a.opacity, 1e-3, 0.995)
    logit = np.log(op / (1 - op))
    return {'means': means, 'quats': q, 'scales': scales, 'opacities': np.broadcast_to(logit, (nt, n))[..., None],
            'sh0': (rgb_t - 0.5) / C0}


def shape_op(P, N):
    """principal curvatures per triangle from its vertex normals (least-squares shape operator, Rusinkiewicz 2004):
    P, N [..., 3 verts, 3] -> kappa [..., 2] (ascending), directions [..., 2, 2] (columns, in the (t, b) frame of
    tri_frames(P))."""
    R, _ = tri_frames(P)
    t_, b_ = R[..., 0], R[..., 1]
    A_ = np.zeros(P.shape[:-2] + (6, 3)); rhs = np.zeros(P.shape[:-2] + (6,))
    for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
        de = P[..., j, :] - P[..., i, :]; dn = N[..., j, :] - N[..., i, :]
        eu, ev_ = np.einsum('...d,...d->...', de, t_), np.einsum('...d,...d->...', de, b_)
        A_[..., 2 * k, 0], A_[..., 2 * k, 1] = eu, ev_
        A_[..., 2 * k + 1, 1], A_[..., 2 * k + 1, 2] = eu, ev_
        rhs[..., 2 * k] = np.einsum('...d,...d->...', dn, t_); rhs[..., 2 * k + 1] = np.einsum('...d,...d->...', dn, b_)
    AtA = np.einsum('...ki,...kj->...ij', A_, A_) + 1e-12 * np.eye(3)
    x = np.linalg.solve(AtA, np.einsum('...ki,...k->...i', A_, rhs)[..., None])[..., 0]
    II = np.stack([np.stack([x[..., 0], x[..., 1]], -1), np.stack([x[..., 1], x[..., 2]], -1)], -2)
    return np.linalg.eigh(II)


def _clamp_dir(S, u, lim):
    """2x2 covariances S [..., 2, 2]: limit the std along unit direction u [..., 2] to lim [...] (A S A, A = I - (1-l) u u^T)."""
    var = np.einsum('...i,...ij,...j->...', u, S, u)
    lam = np.minimum(1.0, lim / np.sqrt(np.maximum(var, 1e-30)))
    A = np.eye(2) - (1 - lam)[..., None, None] * np.einsum('...i,...j->...ij', u, u)
    return A @ S @ np.swapaxes(A, -1, -2)


def aniso_cov(md, smp, idx, a, Pf, Rf, af, means):
    """P42 fix candidates (owner 2026-10-31): per sample a full in-plane 2x2 covariance in the current triangle frame
       --aniso       follows the triangle's deformation gradient F = L_t L_ref^-1 (sigma0^2 F F^T) instead of the
                     isotropic sqrt(area ratio): anisotropic stretch (flapping skirt) no longer opens gaps
       --curv-cap c  std along each principal curvature direction <= c / |kappa| (shape operator from the vertex
                     normals): thin strands/bristles/rims keep their width instead of sigma (needles, not blobs)
       --edge-cap c  std towards the nearest open boundary edge <= c * distance (+ floor): no overhang at hems/brims
    -> quats (triangle frame rotated by the major-axis angle, continuous over frames) and log scales."""
    nt, n = Pf.shape[:2]
    s0 = (_sig(md, a) * np.broadcast_to(np.asarray(smp.get('hloc', md['h']), np.float64), (n,)))   # [n] (densified parts)
    s0q = (s0 ** 2)[:, None, None]
    rmin, rmax = (0.002 if md.get('grown') else 0.05), 20.0
    if a.aniso or _aniso(md, a):
        F = edge_mat(Pf) @ smp['Linv_ref'][None]                       # [nt, n, 2, 2]
        S = s0q * F @ np.swapaxes(F, -1, -2)
    else:
        ratio2 = np.clip(af / np.maximum(smp['aref'], 1e-12), rmin, rmax)
        S = (s0 ** 2 * ratio2)[..., None, None] * np.eye(2)
    # clip the principal stds like the isotropic path clips the area ratio
    ev, evec = np.linalg.eigh(S)
    ev = np.clip(ev, (s0 ** 2 * rmin)[:, None], (s0 ** 2 * rmax)[:, None])
    S = evec @ (ev[..., None] * np.swapaxes(evec, -1, -2))
    t_, b_ = Rf[..., 0], Rf[..., 1]
    fi = smp['fi']
    if a.curv_cap > 0 and md.get('N') is not None:
        Nf = md['N'][idx][:, md['tris'][fi]].astype(np.float64)        # [nt, n, 3, 3] vertex normals
        kv, kd = shape_op(Pf, Nf)
        for k in (0, 1):
            lim = a.curv_cap / np.maximum(np.abs(kv[..., k]), 1e-9)
            S = _clamp_dir(S, kd[..., :, k], np.maximum(lim, a.min_flat))
    if a.edge_cap > 0 and 'bedges' in md:
        be = md['bedges'][fi]                                            # [n, maxe, 2]
        has = (be[..., 0] >= 0).any(1)
        if has.any():
            hi = np.nonzero(has)[0]
            V = md['V'][idx].astype(np.float64)                          # [nt, nV, 3]
            best = np.full((nt, len(hi)), np.inf); bdir = np.zeros((nt, len(hi), 3))
            for e in range(be.shape[1]):
                ok = be[hi, e, 0] >= 0
                if not ok.any():
                    continue
                pa = V[:, be[hi, e, 0].clip(0)]; pb = V[:, be[hi, e, 1].clip(0)]
                m = means[:, hi]
                ab = pb - pa
                tt = np.clip(np.einsum('tnd,tnd->tn', m - pa, ab) / np.maximum(np.einsum('tnd,tnd->tn', ab, ab), 1e-20), 0, 1)
                cp = pa + tt[..., None] * ab
                dv = m - cp
                d = np.linalg.norm(dv, axis=-1)
                d = np.where(ok[None], d, np.inf)
                upd = d < best
                best = np.where(upd, d, best); bdir = np.where(upd[..., None], dv, bdir)
            u2 = np.stack([np.einsum('tnd,tnd->tn', bdir, t_[:, hi]), np.einsum('tnd,tnd->tn', bdir, b_[:, hi])], -1)
            un = np.linalg.norm(u2, axis=-1, keepdims=True)
            u2 = np.where(un > 1e-12, u2 / np.maximum(un, 1e-20), np.array([1.0, 0.0]))
            ec = a.thin_edge_cap if md.get("thin") and getattr(a, "thin_edge_cap", 0) > 0 else a.edge_cap   # 
            lim = np.maximum(ec * best, a.edge_floor * s0[hi])
            S[:, hi] = _clamp_dir(S[:, hi], u2, lim)
    # major/minor axes; angle continuous over frames (ellipse angle is defined mod 90 deg with swapped axes)
    th = 0.5 * np.arctan2(2 * S[..., 0, 1], S[..., 0, 0] - S[..., 1, 1])
    tr, df = 0.5 * (S[..., 0, 0] + S[..., 1, 1]), np.sqrt(0.25 * (S[..., 0, 0] - S[..., 1, 1]) ** 2 + S[..., 0, 1] ** 2)
    l1, l2 = tr + df, np.maximum(tr - df, 1e-30)
    prev = smp.get('th_prev')
    s1, s2 = np.sqrt(l1), np.sqrt(l2)
    for t in range(nt):
        if prev is not None:
            k = np.round((prev - th[t]) / (0.5 * math.pi))
            th[t] = th[t] + k * 0.5 * math.pi
            odd = (k.astype(np.int64) % 2) == 1
            s1[t], s2[t] = np.where(odd, s2[t], s1[t]), np.where(odd, s1[t], s2[t])
        prev = th[t]
    smp['th_prev'] = prev
    c, s = np.cos(th), np.sin(th)
    Rz = np.zeros(th.shape + (3, 3)); Rz[..., 0, 0], Rz[..., 0, 1], Rz[..., 1, 0], Rz[..., 1, 1], Rz[..., 2, 2] = c, -s, s, c, 1
    q = quat_from_frames(Rf @ Rz)
    sn = np.maximum(a.flat * np.maximum(s1, s2), a.min_flat)
    return q, np.log(np.stack([s1, s2, sn], -1))


def hat(Lg, K):
    knots = np.linspace(0, max(Lg - 1, 1), K)
    t = np.arange(Lg)[:, None]
    return np.clip(1 - np.abs(t - knots[None]) / (knots[1] - knots[0]), 0, None)


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--three'); ap.add_argument('--mesh-dir'); ap.add_argument('--cfg')
    ap.add_argument('--frames', type=int, required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--f0', type=int, default=0, help='three: first exported frame of this segment (stream frame 0)')
    ap.add_argument('--h', type=float, default=0.006, help='mean sample spacing on the surface (m)')
    ap.add_argument('--sigma', type=float, default=0.7, help='in-plane sigma / h')
    ap.add_argument('--flat', type=float, default=0.1, help='normal sigma / in-plane sigma')
    ap.add_argument('--min-flat', type=float, default=2e-4)
    ap.add_argument('--opacity', type=float, default=0.98)
    ap.add_argument('--alpha-min', type=float, default=0.15)
    ap.add_argument('--gain', type=float, default=1.0)
    ap.add_argument('--tex-ss', type=int, default=8)
    ap.add_argument('--look', choices=('sil', 'shaded', 'color'), default='sil')
    ap.add_argument('--K', type=int, default=9); ap.add_argument('--L', type=int, default=32)
    ap.add_argument('--sh-deg', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-per-model', type=int, default=25000, help='cap samples per mesh (coarser h for huge meshes); 0 = off')
    ap.add_argument('--only', help='three: comma list of owners (<geom>/../owners.json: mesh id -> owner)')
    ap.add_argument('--full-tracks', action='store_true', help='tracks.npz with every attribute per frame (float-path evaluation)')
    ap.add_argument('--cap-owner', default='', help='three: per-owner sample cap per mesh, e.g. backdrop=5000 (default: '
                                                    '--max-per-model for all)')
    ap.add_argument('--grow-ratio', type=float, default=1e9, help='meshes whose total area grows beyond this factor '
                    'over their median get samples for their LARGEST size (clip: 4; default off = old behaviour)')
    # splat fix (speckles on surfaces, fringed edges); defaults = old behaviour
    ap.add_argument('--sampling', choices=('random', 'poisson'), default='random',
                    help='poisson: blue-noise sample positions (weighted sample elimination from k x candidates)')
    ap.add_argument('--poisson-k', type=float, default=4.0)
    ap.add_argument('--aniso', action='store_true', help='in-plane covariance follows the triangle deformation gradient')
    ap.add_argument('--curv-cap', type=float, default=0.0, help='std along a principal curvature dir <= c/|kappa| (0 off)')
    ap.add_argument('--edge-cap', type=float, default=0.0, help='std towards an open boundary edge <= c*distance (0 off)')
    ap.add_argument('--edge-floor', type=float, default=0.15, help='--edge-cap floor in units of sigma*h')
    ap.add_argument('--crease-deg', type=float, default=0.0, help='--edge-cap also at creases sharper than this (0 off)')
    ap.add_argument('--curv-dens', type=float, default=1.0, help='with --curv-cap: sample curved/thin parts up to this '
                    'factor denser (local spacing h/sqrt(m), m = min((sigma h kappa / c)^2, curv_dens)); 1 = off')
    # thin decals/cut-outs and depthTest-off meshes
    ap.add_argument('--thin-open', type=float, default=0.0, help='open meshes of width w = 2A/P_open < k h: spacing w/k, '
                    'sampled at their largest size (0 = off)')
    ap.add_argument('--h-floor', type=float, default=0.0, help='--thin-open: spacing never below this (m; recipe: per mesh)')
    ap.add_argument('--thin-edge-cap', type=float, default=0.0, help='--thin-open meshes: --edge-cap value for them (0 = same)')
    ap.add_argument('--thin-sigma', type=float, default=0.0, help='--thin-open meshes: sigma / spacing for them (0 = --sigma)')
    ap.add_argument('--thin-aniso', action='store_true', help='--thin-open meshes: covariance follows the deformation gradient')
    ap.add_argument('--thin-inset', type=float, default=0.0, help='--thin-open meshes: drop samples closer than k sigma '
                    'to the open boundary (Gaussian rim lies ~0.72 sigma outside; 0 = off)')
    ap.add_argument('--open-inset', type=float, default=0.0, help='the same for every other mesh with an open boundary '
                    '(A/B only)')
    ap.add_argument('--ontop', help='three: flags.json (mesh id -> {"ontop": true}): depthTest-off meshes are moved along the '
                    'film camera rays in front of the nearest occluder (film picture unchanged)')
    ap.add_argument('--mesh-ids', help='three: only these export mesh ids (comma list; decal stream)')
    ap.add_argument('--ontop-cams', help='--ontop: cams.json of the other camera passes; the decal is also moved in front '
                    'of what hides it from them (as far as --ontop-smin)')
    ap.add_argument('--ontop-smin', type=float, default=0.85)
    ap.add_argument('--skip-mesh-ids', help='three: without these export mesh ids (comma list)')
    return ap


def parse(argv=None):
    a = parser().parse_args(argv)
    a.cap_owner = {k: int(v) for k, v in (x.split('=') for x in a.cap_owner.split(',') if x)}
    return a


def main():
    a = parse()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    T, L, K = a.frames, a.L, a.K
    if a.three:
        models = models_three(a.three, T, a.only.split(',') if a.only else None, a.f0)
    else:
        models = models_blender(a.mesh_dir, json.load(open(a.cfg)))
    models.sort(key=lambda m: (m['f0'], m['name']))           # ids in birth order (continuing splats precede newborns)
    models = filter_ids(models, a)
    ontop = mark_ontop(models, a)
    print(len(models), 'models', f'{time.time() - t0:.0f}s', 'ontop', ontop, flush=True)
    kd = os.path.join(a.out, f'k{K}'); os.makedirs(kd, exist_ok=True)
    TR = ('means', 'quats', 'scales', 'opacities', 'sh0')
    stats, smps = [], []
    off = 0
    for md in models:                                          # pass 1: samples (ids in birth order)
        smp, st = sample_points(md, a, rng)
        st['id0'] = off; off += st['kept']
        smps.append(smp); stats.append(st)
    Ng = off
    # range guard: codec/prep quantises every attribute on its [min, max] range; a range of exactly 0 (all splats the
    # same opacity / the same black) divides noise by 1e-12 and costs ~4 MB of random knots. One splat 1 % off makes the
    # range non-zero, every other splat then sits exactly on a grid level (K=1, ~0 bytes).
    first = next(i for i, st in enumerate(stats) if st['kept'])
    smps[first]['alpha'] = smps[first]['alpha'].copy(); smps[first]['alpha'][0] *= 0.99
    if smps[first]['rgb'] is not None:
        smps[first]['rgb'][0] = np.clip(smps[first]['rgb'][0] + 2 / 255, 0, 1)
    birth = np.concatenate([np.full(st['kept'], md['f0'], np.int32) for md, st in zip(models, stats)])
    death = np.concatenate([np.full(st['kept'], min(md['f1'], T), np.int32) for md, st in zip(models, stats)])
    print('splats', Ng, f'{time.time() - t0:.0f}s', flush=True)
    mm_path = os.path.join(kd, 'means_tmp.npy')
    means_all = np.lib.format.open_memmap(mm_path, mode='w+', dtype=np.float32, shape=(T, Ng, 3))
    for t in range(T):
        means_all[t] = np.nan
    extra_tr = {}                                              # --full-tracks: per-frame values of every attribute
    if a.full_tracks:                                          # (float-path evaluations need them)
        for k, shp, dt in (('quats', (4,), np.float32), ('scales', (3,), np.float16), ('opacities', (), np.float16),
                           ('sh0', (1, 3), np.float16)):
            extra_tr[k] = np.lib.format.open_memmap(os.path.join(kd, f'{k}_tmp.npy'), mode='w+', dtype=dt, shape=(T, Ng) + shp)
            for t in range(T):
                extra_tr[k][t] = np.nan
    fit = {k: [] for k in TR}
    merr = {mi: [] for mi in range(len(models))}
    qprev = {}
    import zipfile
    cpath = os.path.join(kd, 'curves.npz')
    zf = zipfile.ZipFile(cpath, 'w', zipfile.ZIP_STORED, allowZip64=True)

    def put(name, arr):                                        # one npz member, written now (memory: one section)
        with zf.open(name + '.npy', 'w', force_zip64=True) as fh:
            np.lib.format.write_array(fh, np.asarray(arr), allow_pickle=False)
    for k, v in (('L', L), ('K', K), ('knots', 'gop'), ('soft_knot0', True), ('basis', 'hat')):
        put(k, np.array(v))
    for gi, g0 in enumerate(range(0, T, L)):                  # pass 2: per section and model, exact values -> knots
        g1 = min(T, g0 + L)
        Lg = g1 - g0
        B = hat(Lg, K)
        D2 = np.diff(np.eye(K), 2, axis=0) if K > 2 else np.zeros((0, K))   # smoothness on knots no frame pins down
        parts = []
        for mi, (md, smp, st) in enumerate(zip(models, smps, stats)):
            f0, f1, n, o = md['f0'], min(md['f1'], T), st['kept'], st['id0']
            lo, hi = max(g0, f0), min(g1, f1)
            if n == 0 or lo >= hi:
                continue
            w = np.full(Lg, 1e-3); w[lo - g0:hi - g0] = 1.0
            M = np.linalg.solve(B.T @ (w[:, None] * B) + 1e-6 * D2.T @ D2 + 1e-10 * np.eye(K), (B * w[:, None]).T)   # [K, Lg]
            src = np.clip(np.arange(g0, g1), lo, hi - 1) - f0                    # edge padding outside the lifetime
            ex = attrs_at(md, smp, np.arange(lo, hi) - f0, a, qprev.get(mi))
            qprev[mi] = ex['quats'][-1]
            sec = {}
            for k in TR:
                Y = ex[k][src - (lo - f0)].reshape(Lg, n, -1)
                X = np.einsum('kl,lnd->nkd', M, Y)
                Rr = np.einsum('lk,nkd->lnd', B, X)
                e = np.abs(Rr - Y)[lo - g0:hi - g0].max(0).max(-1).astype(np.float32)
                fit[k].append(e)
                if k == 'means':
                    merr[mi].append(e)
                    means_all[lo:hi, o:o + n] = Rr[lo - g0:hi - g0].astype(np.float32)
                elif k in extra_tr:
                    dst = extra_tr[k]
                    dst[lo:hi, o:o + n] = Rr[lo - g0:hi - g0].reshape((hi - lo, n) + dst.shape[2:]).astype(dst.dtype)
                sec[k] = X.astype(np.float32)
            parts.append((o, n, sec))
            del ex
            if hi == f1:                                       # model ends in this section
                md.pop('V'); md.pop('N', None)
        if not parts:                                          # a section without any model (decal stream gap)
            put(f'g{gi}_ids', np.zeros(0, np.int32)); put(f'g{gi}_birth', np.zeros(0, np.int32)); put(f'g{gi}_death', np.zeros(0, np.int32))
            for k, dim in (('means', 3), ('quats', 4), ('scales', 3), ('opacities', 0), ('sh0', 3)):
                put(f'g{gi}_{k}', np.zeros((0, K) + ((dim,) if dim else ()), np.float32))
            print('section', gi, 0, 'splats', flush=True)
            continue
        ids = np.concatenate([np.arange(o, o + n) for o, n, _ in parts]).astype(np.int32)
        put(f'g{gi}_ids', ids); put(f'g{gi}_birth', birth[ids]); put(f'g{gi}_death', death[ids])
        for k in TR:
            X = np.concatenate([s[k] for _, _, s in parts])
            put(f'g{gi}_{k}', X[..., 0] if k == 'opacities' else X)
        del parts
        print('section', gi, len(ids), 'splats', f'{time.time() - t0:.0f}s', flush=True)
    zf.close()
    for mi, st in enumerate(stats):
        if models[mi].get('ontop_s'):
            s_ = np.array(models[mi]['ontop_s'])
            st['ontop_scale_min'] = round(float(s_.min()), 4); st['ontop_frames_moved'] = int((s_ < 0.9999).sum())
        if merr[mi]:
            e = np.concatenate(merr[mi])
            st['fit_means_p99_mm'] = round(float(np.percentile(e, 99)) * 1000, 2); st['fit_means_max_mm'] = round(float(e.max()) * 1000, 2)
    nsh = (a.sh_deg + 1) ** 2 - 1
    np.savez(os.path.join(kd, 'tracks.npz'), birth=birth, death=death, shN=np.zeros((Ng, nsh, 3), np.float16), means=means_all,
             **extra_tr)
    del means_all; os.remove(mm_path)
    for k in list(extra_tr):
        del extra_tr[k]; os.remove(os.path.join(kd, f'{k}_tmp.npy'))
    fe = {}
    for k in TR:
        e = np.concatenate(fit[k])
        fe[k] = {'mean': float(e.mean()), 'p99': float(np.percentile(e, 99)), 'max': float(e.max())}
    alive = [int(((birth <= t) & (death > t)).sum()) for t in range(T)]
    res = {'args': vars(a), 'N_total': int(Ng), 'alive_mean': float(np.mean(alive)), 'alive_max': int(max(alive)),
           'models': stats, 'curve_fit_max_err_per_splat': fe, 'secs': round(time.time() - t0, 1),
           'source': 'per-frame mesh export (--three)' if a.three else 'Blender export',
           'note': 'fit error = max over the alive frames of a section of |curve - exact mesh value| (means in m)'}
    json.dump(res, open(os.path.join(kd, 'mesh2gs.json'), 'w'), indent=1)
    print('MESH2GS OK', kd, json.dumps({k: res[k] for k in ('N_total', 'alive_mean', 'alive_max', 'curve_fit_max_err_per_splat', 'secs')}),
          flush=True)


if __name__ == '__main__':
    main()
