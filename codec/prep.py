"""numpy front-ends: P1 data -> integer sections for the .csgs container (NOT part of the portable decoder core).

  sections_from_tracks(tracks.npz, qpos, tol, ks)   e6 symbols: tracks quantised exactly like e4.quantise(step, dz=0),
                                                   per section and attribute group the smallest K (e6.fit_group) whose
                                                   curve stays within `tol` quantisation steps on all alive frames.
                                                   Also returns e6's own zstd-19 byte count (same blob layout as
                                                   e6.encode_curves) as a cross-check of the 54.2 Mbit/s.
  sections_from_curves(curves.npz, tracks.npz, qpos, ks)   P1 knots directly (trained curves): knots quantised on the
                                                   same grids, shared boundary knot implicit, K reduced only where
                                                   the smaller curve reproduces all quantised knots exactly.
Both return (params, [Section]) with Python ints, ready for codec.csgs.Encoder.
"""
import numpy as np

from .csgs import Params, Section, apply_follow, needs_rescale, rescale_value

GROUPS = (('P', 3), ('R', 4), ('S', 3), ('O', 1), ('D', 3))


def q8(a, mn, mx, levels=255):
    return np.round(np.clip((np.nan_to_num(a) - mn) / (mx - mn + 1e-12), 0, 1) * levels).astype(np.int16)


def zero_sn_bands(SN, sn_deg=3):
    """SN [n, 3*((deg+1)^2-1)] (coefficient-major, 3 channels each): bands above sn_deg set to 0 (copy)."""
    SN = SN.copy()
    SN[:, ((sn_deg + 1) ** 2 - 1) * 3:] = 0
    return SN


def quant_sn(SN, sn_deg=3, sn_bits=8):
    """SN [n, 45] (coefficient-major, 3 channels each) -> (q in 0..2^sn_bits-1, ranges). Bands above sn_deg are sent
    as a constant 0 (offset 0, scale 0): the format and decoders stay at 45 scalars, the rANS cost of those is ~0.
    (sections_from_curves does the same via zero_sn_bands + the sn_levels of its plan.)"""
    SN = zero_sn_bands(SN, sn_deg)
    mn, mx = SN.min(0), SN.max(0)
    return q8(SN, mn, mx, (1 << sn_bits) - 1), (mn, mx)


def _quant_floats(lo, step, ranges, n_sn, attr_levels=255, rot_levels=127, sn_levels=255):
    """affine dequantisation value = offset + q * scale per scalar: P(3) R(4) S(3) O(1) D(3) SN(n_sn).
    attr_levels / rot_levels / sn_levels (S7): number of grid steps over the S/O/D range, the rotation unit and the SN range
    (defaults = the fixed 8-bit / 1/127 / 255 tables of all earlier runs, bit-identical floats)."""
    off, sc = [], []
    off += [float(x) for x in lo]; sc += [float(step)] * 3
    off += [0.0] * 4; sc += [1.0 / rot_levels] * 4
    for k, d in (('S', 3), ('O', 1), ('D', 3), ('SN', n_sn)):
        mn, mx = (np.broadcast_to(np.asarray(v, np.float64), (d,)) for v in ranges[k])
        lv = sn_levels if k == 'SN' else attr_levels
        off += mn.tolist(); sc += ((mx - mn) / float(lv)).tolist()
    return off, sc


def hat_basis(L, K):
    if K == 1:
        return np.ones((L, 1))
    knots = np.linspace(0, L - 1, K)
    t = np.arange(L)[:, None]
    h = knots[1] - knots[0]
    return np.clip(1 - np.abs(t - knots[None]) / h, 0, None)


def fit_group(X, A, tol, KS):
    """copy of e6.fit_group: X [L,n,d] ints, A [L,n] alive -> best K [n], [(K, idx, C)], decoded [L,n,d]."""
    L, n, d = X.shape
    Xf = X.astype(np.float64)
    best_k = np.full(n, -1); dec = np.zeros_like(X); groups = []
    Af = A.astype(np.float64)
    ks = [k for k in KS if k <= L]
    for K in ks:
        B = hat_basis(L, K)
        BtWB = np.einsum('lk,ln,lj->nkj', B, Af, B) + 1e-6 * np.eye(K)[None]
        BtWX = np.einsum('lk,ln,lnd->nkd', B, Af, Xf)
        C = np.round(np.linalg.solve(BtWB, BtWX)).astype(np.int64)
        Xh = np.round(np.einsum('lk,nkd->lnd', B, C.astype(np.float64))).astype(np.int64)
        err = np.where(A[..., None], np.abs(Xh - X), 0).max(axis=(0, 2))
        take = (best_k < 0) & ((err <= tol) | (K == ks[-1]))
        idx = np.nonzero(take)[0]
        best_k[idx] = K
        dec[:, idx] = Xh[:, idx]
        groups.append((K, idx, C[idx]))
    return best_k, groups, dec


def load_tracks(path, qpos):
    """e4.quantise(step, 0) on a tracks.npz; returns dict with qi, alive, birth, death, params pieces."""
    tr = np.load(path)
    birth, death = tr['birth'], tr['death']
    T, NG = tr['means'].shape[:2]
    alive = (np.arange(T)[:, None] >= birth[None]) & (np.arange(T)[:, None] < death[None])
    M = tr['means']
    Q = tr['quats'].copy(); Q /= np.linalg.norm(Q, axis=-1, keepdims=True); Q *= np.where(Q[..., :1] < 0, -1, 1)
    S = tr['scales'].astype(np.float32); O = tr['opacities'].astype(np.float32)
    D = tr['sh0'].astype(np.float32).reshape(T, NG, 3)
    SN = tr['shN'].astype(np.float32).reshape(NG, -1)
    lo = np.nanmin(M.reshape(-1, 3), 0); ext = float(np.linalg.norm(np.nanmax(M.reshape(-1, 3), 0) - lo))
    ranges = {'S': (np.nanmin(S.reshape(-1, 3), 0), np.nanmax(S.reshape(-1, 3), 0)),
              'O': (float(np.nanmin(O)), float(np.nanmax(O))),
              'D': (np.nanmin(D.reshape(-1, 3), 0), np.nanmax(D.reshape(-1, 3), 0)),
              'SN': (SN.min(0), SN.max(0))}
    step = qpos * ext
    qi = {'P': np.round((np.nan_to_num(M) - lo) / step).astype(np.int32),
          'R': np.round(np.nan_to_num(Q) * 127).astype(np.int16),
          'S': q8(S, *ranges['S']), 'O': q8(O[..., None], ranges['O'][0], ranges['O'][1])[..., 0], 'D': q8(D, *ranges['D'])}
    qi['SN'] = q8(SN, *ranges['SN'])
    for k in ('P', 'R', 'S', 'O', 'D'):
        qi[k][~alive] = 0
    return {'qi': qi, 'alive': alive, 'birth': birth, 'death': death, 'T': T, 'NG': NG, 'lo': lo, 'ext': ext,
            'step': step, 'ranges': ranges, 'n_sn': SN.shape[1]}


def _death_code(death, g0, g1):
    return np.where(death > g1, g1 - g0 + 1, death - g0)


def sections_from_tracks(path, qpos=0.001, tol=1.0, ks=(1, 2, 3, 5, 9), L=32, key_interval=4, zstd_level=19,
                         max_sections=None):
    """returns params, sections, e6 dict {'zstd_bytes_per_gop': [...], 'ref_mismatch': n}."""
    import zstandard
    Z = zstandard.ZstdCompressor(level=zstd_level)
    d = load_tracks(path, qpos)
    qi, alive, birth, death, T = d['qi'], d['alive'], d['birth'], d['death'], d['T']
    off, sc = _quant_floats(d['lo'], d['step'], d['ranges'], d['n_sn'])
    params = Params(L=L, ks=list(ks), groups=list(GROUPS), n_sn=d['n_sn'], offsets=off, scales=sc,
                    shared_knot=False, key_interval=key_interval)
    dq = {k: np.zeros_like(qi[k]) for k in ('P', 'R', 'S', 'O', 'D')}
    sections, e6_bytes, ref_mismatch = [], [], 0
    for gi, g0 in enumerate(range(0, T, L)):
        if max_sections is not None and gi >= max_sections:
            break
        g1 = min(T, g0 + L)
        ids = np.nonzero(alive[g0:g1].any(0))[0]
        A = alive[g0:g1, ids]
        cont = (birth[ids] < g0) if g0 > 0 else np.zeros(len(ids), bool)
        n_cont = int(cont.sum())
        assert cont[:n_cont].all() and not cont[n_cont:].any(), 'continuing splats must precede newborns (id order)'
        # e6 blob (exact layout of e6.encode_curves, curve mode with SN)
        new = np.nonzero((birth >= g0) & (birth < g1))[0]
        dead = np.nonzero((death >= g0) & (death < g1))[0]
        parts = [new.astype(np.int32).tobytes(), (birth[new] - g0).astype(np.uint8).tobytes(),
                 dead.astype(np.int32).tobytes(), (death[dead] - g0).astype(np.uint8).tobytes(),
                 qi['SN'][new].astype(np.uint8).tobytes()]
        kk_all, knots_all = [], []
        for k, dim in GROUPS:
            X = qi[k][g0:g1, ids]
            sh = X.shape
            X3 = X.reshape(sh[0], sh[1], -1).astype(np.int64)
            kk, groups, dec = fit_group(X3, A, tol, list(ks))
            prev = dq[k][g0 - 1, ids].reshape(len(ids), -1).astype(np.int64) if g0 > 0 else np.zeros((len(ids), dim), np.int64)
            parts.append(kk.astype(np.uint8).tobytes())
            knots = [None] * len(ids)
            for K, idx, C in groups:
                first = C[:, 0] - prev[idx] * cont[idx, None]
                parts += [first.astype(np.int32).tobytes(), np.diff(C, axis=1).astype(np.int32).tobytes()]
                for i, row in zip(idx.tolist(), C.reshape(len(idx), K * dim).tolist()):
                    knots[i] = row
            # the container predicts knot 0 of continuing splats from the previous section's LAST KNOT; e6 from the
            # decoded value at frame g0-1. Count where they differ (should be 0: hat basis is 1 at the last knot).
            if g0 > 0:
                pk = sections[-1].knots[[g for g, _ in GROUPS].index(k)]
                prev_ids = sections[-1].ids
                pos = {sid: j for j, sid in enumerate(prev_ids)}
                for i in np.nonzero(cont)[0].tolist():
                    j = pos[int(ids[i])]
                    lastk = pk[j][-dim:]
                    if lastk != prev[i].tolist():
                        ref_mismatch += 1
            kk_all.append(kk.tolist()); knots_all.append(knots)
            dec = np.where(A[..., None], dec, 0)
            dq[k][g0:g1, ids] = dec.reshape(sh).astype(dq[k].dtype)
        e6_bytes.append(len(Z.compress(b''.join(parts))))
        sec = Section(index=gi, g0=g0, nframes=g1 - g0, ids=ids.tolist(), n_cont=n_cont,
                      birth=(birth[ids[n_cont:]] - g0).tolist(), death=_death_code(death[ids], g0, g1).tolist(),
                      k=kk_all, knots=knots_all, sn=qi['SN'][ids].tolist())
        sections.append(sec)
    return params, sections, {'zstd_bytes_per_gop': e6_bytes, 'ref_mismatch': ref_mismatch, 'T': int(T),
                              'alive_mean': float(alive.sum(1).mean())}


def _norm_quats(q):
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def _reduce_k(C, ks, tol=0, k1_exact=True):
    """C [n,K,d] ints -> (K' per splat, list of flat knot rows); smallest K' in ks with (K-1) % (K'-1) == 0 whose
    hat curve reproduces every quantised knot within `tol` quantisation steps (tol=0: lossless w.r.t. the quantised
    K-knot curve). Knot 0 and the last knot are always kept, so shared boundary knots stay exact."""
    n, K, d = C.shape
    best = np.full(n, K)
    for Kp in sorted(ks, reverse=True):
        if Kp > K or (Kp > 1 and (K - 1) % (Kp - 1)) or Kp == K:
            continue
        B = hat_basis(K, Kp)                                   # evaluate the K'-curve at the K knot positions
        stride = (K - 1) // (Kp - 1) if Kp > 1 else K
        sub = C[:, ::stride][:, :Kp] if Kp > 1 else C[:, :1]
        rec = np.round(np.einsum('kj,njd->nkd', B, sub.astype(np.float64))).astype(np.int64)
        ok = (np.abs(rec - C) <= tol).all(axis=(1, 2))
        if Kp == 1 and k1_exact:                               # K=1 holds knot 0: the last knot must match exactly
            ok &= (C[:, -1] == C[:, 0]).all(-1)
        best[ok] = Kp
    rows = []
    for i in range(n):
        Kp = int(best[i])
        stride = (K - 1) // (Kp - 1) if Kp > 1 else K
        rows.append(C[i, ::stride][:Kp].reshape(-1).tolist())
    return best.tolist(), rows


def _reduce_k_bspline(C, ks, tol, nframes, k1_exact=True):
    """B-spline counterpart of _reduce_k (P16, flag BSPLINE): C [n,K,d] ints are control points of a clamped uniform
    B-spline (csgs.bspline_weights). For every K' in ks below K the control points are refitted by least squares to the
    quantised K-curve at the integer frames 0..nframes-1 (first and last control point fixed = boundary values, so shared /
    soft knots stay exact), rounded, and K' is taken when the decoded K'-curve stays within `tol` quantisation steps of the
    K-curve at every frame. K'=1 needs a constant curve (and last == first knot, like the hat reduction, unless
    k1_exact=False: P22 per-group tolerance with soft knot 0, where the end may differ within `tol`). Smallest K' wins."""
    from .csgs import bspline_weights
    n, K, d = C.shape
    span = max(nframes - 1, 1)

    def basis(Kx):
        return np.array([bspline_weights(Kx, f / span) for f in range(nframes)], np.float64)      # [F, Kx]
    best = np.full(n, K)
    Cf = C.astype(np.float64)
    target = np.rint(np.einsum('fk,nkd->nfd', basis(K), Cf)) if K > 1 else np.repeat(Cf[:, :1], nframes, 1)
    sol = {K: C}
    for Kp in sorted(ks, reverse=True):
        if Kp >= K or nframes < Kp:
            continue
        if Kp == 1:
            ok = (np.abs(C[:, :1] - target) <= tol).all(axis=(1, 2))
            if k1_exact:
                ok &= (C[:, -1] == C[:, 0]).all(-1)
            sol[1] = C[:, :1]
        else:
            B = basis(Kp)
            c0, cl = Cf[:, 0], Cf[:, -1]
            rec = np.zeros((n, Kp, d))
            rec[:, 0], rec[:, -1] = c0, cl
            if Kp > 2:
                rhs = target - B[None, :, 0, None] * c0[:, None] - B[None, :, -1, None] * cl[:, None]     # [n,F,d]
                inner = np.einsum('kf,nfd->nkd', np.linalg.pinv(B[:, 1:-1]), rhs)
                rec[:, 1:-1] = np.rint(inner)
            rec = rec.astype(np.int64)
            got = np.rint(np.einsum('fk,nkd->nfd', B, rec.astype(np.float64)))
            ok = (np.abs(got - target) <= tol).all(axis=(1, 2))
            sol[Kp] = rec
        best[ok] = Kp
    rows = [sol[int(best[i])][i].reshape(-1).tolist() for i in range(n)]
    return best.tolist(), rows


def _reduce_adaptive(C, kn, ks, tol, bspline, nframes, k1_exact=True):
    """K reduction for splats trained with their own K (P16 adaptive K): kn [n] = trained K per splat (<= C.shape[1]),
    the columns beyond kn are padding. Splats with the same kn are reduced together; uniform kn = the old call."""
    n = C.shape[0]
    kk, rows = [0] * n, [None] * n
    for kv in sorted(set(int(x) for x in kn)):
        sel = np.nonzero(kn == kv)[0]
        sub = C[sel, :kv]
        k_, r_ = _reduce_k_bspline(sub, ks, tol, nframes, k1_exact) if bspline else _reduce_k(sub, ks, tol, k1_exact)
        for j, i in enumerate(sel.tolist()):
            kk[i], rows[i] = k_[j], r_[j]
    return kk, rows


def _raw_quat_scale(cv, nsec):
    """per global id one scalar 1/max|q| over all its knots in all sections (raw, unnormalised trained quats).
    Scaling every knot of a splat by the same scalar leaves nlerp(lerp(q)) unchanged, so the decoded rotation equals
    the trained one (gsplat normalises after the lerp too) and the knots stay piecewise linear for the K reduction."""
    mx = {}
    for gi in range(nsec):
        ids = cv[f'g{gi}_ids']
        nr = np.linalg.norm(cv[f'g{gi}_quats'].astype(np.float64), axis=-1).max(-1)
        for i, v in zip(ids.tolist(), nr.tolist()):
            if v > mx.get(i, 0.0):
                mx[i] = v
    return mx


def _select(cv, tr_birth, tr_death, gi, L, T, keep=None):
    """splats of section gi in id order with lifetimes from tracks.npz (final): rows into cv['g{gi}_*'], ids, birth,
    death. A later section may still kill a splat at its first frame, which the per-section death of curves.npz does
    not know yet. keep (matte_cull, ): optional bool per trainer id; dropped splats are not selected."""
    g0 = gi * L; g1 = min(T, g0 + L)
    ids = cv[f'g{gi}_ids']; b = tr_birth[ids]; dth = tr_death[ids]
    sel = (b < g1) & (dth > g0)
    if keep is not None:
        sel &= keep[ids]
    order = np.argsort(ids[sel], kind='stable')
    rows = np.nonzero(sel)[0][order]
    return rows, ids[rows], b[rows], dth[rows]


def _fit_sn_vq(cv, tr_birth, tr_death, SN, nsec, L, T, key_interval, opts, off_sn, sc_sn, stats, keep=None):
    """one SN-VQ codebook set per key interval (sections [k, k+key_interval)): every splat that appears in the interval
    is assigned once (degree by rule, indices by nearest quantised entry). Returns codec.snvq.Assignment."""
    from . import snvq
    A = opts.get('from')
    if A is None:
        A = snvq.Assignment(opts['levels'], off_sn, sc_sn, snvq.zero_row(off_sn, sc_sn))
    degs = [0] + A.degs()
    stats['sn_vq'] = {'levels': A.levels, 'weight': opts.get('weight', 'none'), 'q': opts.get('q'), 'thr': opts.get('thr'),
                      'from': opts.get('from_path'), 'groups': {}}
    for k0 in range(0, nsec, key_interval):
        if k0 in A.groups:                                  # reused assignment (retraining with fixed SN)
            g = A.groups[k0]
            hist = {int(d): int((g['deg'] == d).sum()) for d in degs}
            stats['sn_vq']['groups'][k0] = {'n': len(g['ids']), 'deg_hist': hist, 'reused': True}
            continue
        first = {}
        for gi in range(k0, min(nsec, k0 + key_interval)):
            rows, ids, _, _ = _select(cv, tr_birth, tr_death, gi, L, T, keep)
            w = snvq.importance(cv[f'g{gi}_opacities'][rows], cv[f'g{gi}_scales'][rows]) \
                if opts.get('weight', 'none') == 'importance' else np.ones(len(rows))
            for i, wi in zip(ids.tolist(), w.tolist()):
                if i not in first:
                    first[i] = wi
        ids = np.array(sorted(first), np.int64)
        w = np.array([first[int(i)] for i in ids])
        X = SN[ids].astype(np.float64)
        if opts.get('thr') is not None:
            deg = snvq.degree_rule_abs(X, w, degs, opts['thr'])
        else:
            deg = snvq.degree_rule(X, w, degs, opts.get('q') or [0.0] * (len(degs) - 1))
        g = A.fit_group(k0, ids, X, w, deg, iters=int(opts.get('iters', 25)), seed=int(opts.get('seed', 0)))
        rec = A.float_rows(k0, range(len(ids)))
        err = ((rec - X) ** 2).sum(-1)
        hist = {int(d): int((deg == d).sum()) for d in degs}
        stats['sn_vq']['groups'][k0] = {'n': len(ids), 'deg_hist': hist, 'reused': False,
                                        'sn_mse': float(err.mean()), 'sn_mse_weighted': float((err * w).sum() / max(w.sum(), 1e-12)),
                                        'sn_energy_mean': float((X ** 2).sum(-1).mean())}
    return A


CULL_LOGIT = float(np.log(1 / 255 / (1 - 1 / 255)))     # gsplat skips alpha < 1/255; opacity knots below are invisible


def sections_from_curves(curves_path, tracks_path, qpos=0.001, ks=None, key_interval=4, max_sections=None,
                         quat='norm', tol=0, shared_knot=None, sn_vq=None, ext=None, cull_floor=False,
                         newborn_order='id', pred=None, knn_k=0, follow=True, plan=None, basis=None,
                         sn_deg=3, sn_bits=8, keep=None, ktol_groups=None):
    """ktol_groups (P22, ): optional {group: tol} overriding the K reduction tolerance per attribute group
    (P/R/S/O/D) in every section, e.g. {'D': 2} lets nearly constant colour curves collapse to K=1; None = unchanged.
    quat: 'norm' = every knot normalised + sign-aligned (P1/P4 path, reproduces 40.10 Mbit/s), 'raw' = trained
    unnormalised knots times one scalar per splat (_raw_quat_scale). tol: K reduction tolerance in quantisation steps.
    shared_knot: None = from curves.npz (False when the trainer let knot 0 of continuing splats move, soft_knot0).
    sn_vq: None (v1 SN rows) or dict for the v2 SN-VQ mode: levels [(deg, [sizes])], q (fractions per
    degree below the highest) or thr (absolute weighted-energy thresholds), weight 'none'|'importance', iters, seed,
    from (codec.snvq.Assignment to reuse incl. its SN grid), save (path for the assignment .npz).
    ext (P10): fixed scene extent for the position step (default None = norm of the min/max box of all tracked means;
    one runaway splat then coarsens the step for everyone). cull_floor (P10): opacity knots below the 1/255 cull are
    rounded DOWN to the grid, so invisible (zombie) knots never become visible after decoding (default off = round).
    newborn_order (S2, ): 'id' = trainer ids; 'morton' = the newborns of a section are renumbered along a
    Morton curve of their quantised knot-0 position (ids stay consecutive, the stream stays v1/v2; the DPCM of the
    first knots then runs between spatial neighbours). pred/knn_k: .csgs v3 motion-prediction modes (codec.csgs.Params),
    combinable with sn_vq (version 3 carries both).
    follow: when curves.npz carries dormant knots (g<i>_dk, p8_e7 E7_DORMANT, ), position knots 1.. of dormant
    splats are not sent but predicted from their neighbours (csgs follow mode, flag 16 / version 4); those splats keep
    the trained K for P. No effect without g<i>_dk in curves.npz (all existing runs).
    plan (S7, ): None (one quantisation table for the whole stream, as before) or a list with one dict per section
    {'qpos': .., 'attr_levels': 255, 'rot_levels': 127, 'ktol': ..} (missing keys = the call's defaults); every section
    then has its own table (Params), returned in stats['section_params']. A table change inside a key interval is carried
    by csgs flag RESCALE (shared/soft knot 0 is rescaled from the previous section's last knot); SN tables stay fixed.
    basis (P16, ): None = from curves.npz ('basis' field of the trainer, default hat), 'hat' or 'bspline'; bspline = the
    knots are control points of a clamped cubic B-spline (csgs flag 64 / version 6), the K reduction then refits B-spline
    control points (_reduce_k_bspline). Adaptive K: curves.npz may carry g<i>_kn (trained K per splat, columns beyond are
    padding); the K reduction then runs per distinct kn (_reduce_adaptive).
    sn_deg / sn_bits (the contributor, ): highest SH degree sent (higher bands are zeroed before the ranges, so they cost
    ~0 bytes; format and decoders stay at 45 scalars) and the SN grid bits (default sn_levels of the plan =
    2^sn_bits - 1; a plan entry 'sn_levels' still overrides it per key interval). sn_bits != 8 is not combinable with sn_vq.
    keep (matte_cull, ): optional bool per trainer id; dropped splats vanish from every section, the rest are
    renumbered densely in id order (newborns of a section stay consecutive). Quantiser ranges and the position grid
    ignore keep."""
    if sn_vq is not None and sn_bits != 8:
        raise ValueError('sn_bits != 8 is not combinable with sn_vq (the SN-VQ grid is 8 bit)')
    cv = np.load(curves_path)
    tr = np.load(tracks_path)
    L, K = int(cv['L']), int(cv['K'])
    if basis is None:
        basis = str(cv['basis']) if 'basis' in cv.files else 'hat'
    if basis not in ('hat', 'bspline'):
        raise ValueError('basis hat|bspline')
    bspline = basis == 'bspline'
    if shared_knot is None:
        shared_knot = not ('soft_knot0' in cv and bool(cv['soft_knot0']))
    if ks is None:
        ks = tuple(k for k in (1, 2, 3, 5, 9) if k <= K)
    T = tr['means'].shape[0]
    tr_birth, tr_death = tr['birth'], tr['death']
    Mt = tr['means']
    lo = np.nanmin(Mt.reshape(-1, 3), 0); ext_auto = float(np.linalg.norm(np.nanmax(Mt.reshape(-1, 3), 0) - lo))
    del Mt
    ext = ext_auto if ext is None else float(ext)
    step = qpos * ext
    SN = tr['shN'].astype(np.float32).reshape(tr['shN'].shape[0], -1)
    nsec = (T + L - 1) // L
    if max_sections is not None:
        nsec = min(nsec, max_sections)
    keys = {'S': 'scales', 'O': 'opacities', 'D': 'sh0'}
    ranges = {}
    for g, name in keys.items():
        allv = np.concatenate([cv[f'g{gi}_{name}'].reshape(-1, 1 if g == 'O' else 3) for gi in range((T + L - 1) // L)])
        ranges[g] = (allv.min(0) if g != 'O' else float(allv.min()), allv.max(0) if g != 'O' else float(allv.max()))
    if sn_deg < 3:                                          # (the contributor): bands above sn_deg sent as constant 0
        SN = zero_sn_bands(SN, sn_deg)
    ranges['SN'] = (SN.min(0), SN.max(0))
    if sn_vq is not None and sn_vq.get('from') is not None:   # reused assignment: its SN grid (retrained SN sits on it)
        A0 = sn_vq['from']
        ranges['SN'] = (A0.off.astype(np.float32), (A0.off + 255 * A0.scale).astype(np.float32))
    qSN = q8(SN, *ranges['SN'])
    plan = plan or None
    default_pl = {'qpos': qpos, 'attr_levels': 255, 'rot_levels': 127, 'ktol': tol, 'sn_levels': (1 << sn_bits) - 1}

    def pl_of(gi):
        return {**default_pl, **(plan[gi] if plan is not None and gi < len(plan) else {})}

    def sn_lv(gi):                                          # the SN grid is fixed per key interval: the key section's entry counts
        return int(pl_of((gi // key_interval) * key_interval)['sn_levels'])

    _qsn = {255: qSN}

    def qSN_of(gi):
        lv = sn_lv(gi)
        if lv not in _qsn:
            _qsn[lv] = q8(SN, *ranges['SN'], lv)
        return _qsn[lv]

    def table_of(gi):
        pl = pl_of(gi)
        return _quant_floats(lo, pl['qpos'] * ext, ranges, SN.shape[1], int(pl['attr_levels']), int(pl['rot_levels']), sn_lv(gi))
    off, sc = table_of(0)
    step = pl_of(0)['qpos'] * ext
    newid = None
    if keep is not None:                                   # (the contributor): dense stream ids of the kept splats
        keep = np.asarray(keep, bool)
        newid = np.cumsum(keep) - 1
    stats = {'knot0_float_mismatch': 0, 'k_hist': {g: {} for g, _ in GROUPS}, 'quat': quat, 'tol': tol,
             'shared_knot': shared_knot, 'K_trained': K, 'knots_sent': 0, 'knot_scalars_sent': 0,
             'knots_at_K_trained': 0, 'ext': ext, 'ext_auto': ext_auto, 'pos_step': step, 'cull_floor': bool(cull_floor),
             'newborn_order': newborn_order, 'pred': pred, 'knn_k': knn_k}
    if newborn_order not in ('id', 'morton'):
        raise ValueError('newborn_order id|morton')
    A = None
    if sn_vq is not None:
        n_sn = SN.shape[1]
        A = _fit_sn_vq(cv, tr_birth, tr_death, SN, nsec, L, T, key_interval, sn_vq, off[-n_sn:], sc[-n_sn:], stats, keep)
        if sn_vq.get('save'):
            A.save(sn_vq['save'])
        def mk_params(off_, sc_):
            return Params(L=L, ks=list(ks), groups=list(GROUPS), n_sn=n_sn, offsets=off_, scales=sc_,
                          shared_knot=shared_knot, key_interval=key_interval, sn_vq=A.levels, sn_zero=A.zero,
                          pred=pred, knn_k=knn_k, bspline=bspline)
    else:
        def mk_params(off_, sc_):
            return Params(L=L, ks=list(ks), groups=list(GROUPS), n_sn=SN.shape[1], offsets=off_, scales=sc_,
                          shared_knot=shared_knot, key_interval=key_interval, pred=pred, knn_k=knn_k, bspline=bspline)
    params = mk_params(off, sc)
    sec_params = [params]
    for gi in range(1, nsec):                                  # S7: one table per section (all identical without a plan)
        sec_params.append(params if plan is None else mk_params(*table_of(gi)))
    qscale = _raw_quat_scale(cv, (T + L - 1) // L) if quat == 'raw' else None
    sections, last = [], {}
    idmap = {}                                             # trainer id -> stream id (identity for 'id')
    for gi in range(nsec):
        params = sec_params[gi]
        pl = pl_of(gi)
        step = pl['qpos'] * ext
        lv_a, lv_r, tol_i = int(pl['attr_levels']), int(pl['rot_levels']), pl['ktol']
        if gi and needs_rescale(sec_params[gi - 1], params):
            rs = []                                        # (group, per-dim (off_o, sc_o, off_n, sc_n)) for the previous last knots
            q0_ = 0
            for _, d_ in GROUPS:
                rs.append([(sec_params[gi - 1].offsets[q0_ + c], sec_params[gi - 1].scales[q0_ + c],
                            params.offsets[q0_ + c], params.scales[q0_ + c]) for c in range(d_)])
                q0_ += d_
            for gk, (g_, d_) in enumerate(GROUPS):
                last[g_] = {i: [rescale_value(v[c], *rs[gk][c]) for c in range(d_)] for i, v in last[g_].items()}
        g0 = gi * L; g1 = min(T, g0 + L)
        rows, ids, b, dth = _select(cv, tr_birth, tr_death, gi, L, T, keep)
        cont = b < g0
        n_cont = int(cont.sum())
        assert cont[:n_cont].all() and not cont[n_cont:].any()
        tr_ids = ids                                       # trainer ids (index shN / quat scale); ids = stream ids
        if newid is not None:                              # keep: dense ids; newborns stay consecutive
            ids = newid[ids]
        if newborn_order == 'morton':
            # continuing splats: stream ids from the map, re-sorted (strictly increasing); newborns: consecutive
            # (dense) ids id_base.. are permuted along the Morton curve of their quantised knot-0 position
            from .knn import morton_order
            sid = np.array([idmap[int(i)] for i in ids[:n_cont]], np.int64)
            oc = np.argsort(sid, kind='stable')
            nn = len(ids) - n_cont
            if nn:
                nb_ids = ids[n_cont:]
                assert (nb_ids == nb_ids[0] + np.arange(nn)).all(), 'newborn trainer ids must be consecutive'
                p0 = np.round((cv[f'g{gi}_means'][rows[n_cont:], 0] - lo) / step).astype(np.int64)
                om = np.array(morton_order(p0.tolist()), np.int64)
                for r_, t_ in zip(om.tolist(), range(nn)):
                    idmap[int(nb_ids[r_])] = int(nb_ids[0]) + t_
            else:
                om = np.zeros(0, np.int64)
            perm = np.concatenate([oc, n_cont + om])
            rows, b, dth, tr_ids = rows[perm], b[perm], dth[perm], tr_ids[perm]
            ids = np.array([idmap[int(i)] for i in ids[perm]], np.int64)
            cont = b < g0
        fmask = None
        if follow and f'g{gi}_dk' in cv.files and K >= 2:
            dk = cv[f'g{gi}_dk'][rows].astype(bool)
            dk[:, 0] = False
            fmask = (dk * (1 << np.arange(K))).sum(1).astype(np.int64)
            stats.setdefault('follow_splats', 0); stats.setdefault('follow_knots', 0)
            stats['follow_splats'] += int((fmask > 0).sum()); stats['follow_knots'] += int(dk.sum())
        q = {'P': np.round((cv[f'g{gi}_means'][rows] - lo) / step).astype(np.int64),
             'S': q8(cv[f'g{gi}_scales'][rows], *ranges['S'], lv_a).astype(np.int64),
             'O': q8(cv[f'g{gi}_opacities'][rows][..., None], *ranges['O'], lv_a).astype(np.int64),
             'D': q8(cv[f'g{gi}_sh0'][rows], *ranges['D'], lv_a).astype(np.int64)}
        if cull_floor:
            O = cv[f'g{gi}_opacities'][rows][..., None].astype(np.float64)
            o_mn, o_mx = ranges['O']
            below = O < CULL_LOGIT
            q['O'] = np.where(below, np.floor(np.clip((O - o_mn) / (o_mx - o_mn + 1e-12), 0, 1) * lv_a).astype(np.int64), q['O'])
            stats['cull_floor_knots'] = stats.get('cull_floor_knots', 0) + int(below.sum())

        quats_raw = cv[f'g{gi}_quats'][rows].astype(np.float64)
        if quat == 'raw':
            sc_i = np.array([1.0 / max(qscale[int(i)], 1e-12) for i in tr_ids])
            q['R'] = np.round(quats_raw * sc_i[:, None, None] * lv_r).astype(np.int64)
        else:
            qf = _norm_quats(quats_raw)
            sign = np.ones(qf.shape[:2])
            sign[:, 0] = np.where(qf[:, 0, 0] < 0, -1, 1)
            for j in range(1, K):
                dots = (qf[:, j] * qf[:, j - 1] * sign[:, j - 1, None]).sum(-1)
                sign[:, j] = np.where(dots < 0, -1, 1)
            q['R'] = np.round(qf * sign[..., None] * lv_r).astype(np.int64)
        kn_i = cv[f'g{gi}_kn'][rows].astype(np.int64) if f'g{gi}_kn' in cv.files else np.full(len(rows), K, np.int64)
        kk_all, knots_all = [], []
        for g, dim in GROUPS:
            C = q[g]
            tol_g = (ktol_groups or {}).get(g, tol_i)
            k1x = shared_knot or g not in (ktol_groups or {})       # P22: soft knot 0 + own tolerance -> K=1 within tol
            if g == 'R' and n_cont and quat == 'norm':
                # shared/predicted knot: keep the whole curve's sign consistent with the previous last knot
                prevR = np.array([last['R'][int(i)] for i in ids[:n_cont]])
                flip = (C[:n_cont] * prevR[:, None]).sum(-1)[:, :1] < 0
                C[:n_cont] = np.where(flip[..., None], -C[:n_cont], C[:n_cont])
            if n_cont and shared_knot:
                # shared knot: knot 0 of a continuing splat IS the quantised last knot of the previous section
                prevk = np.array([last[g][int(i)] for i in ids[:n_cont]])
                stats['knot0_float_mismatch'] += int((np.abs(C[:n_cont, 0] - prevk).max(-1) > 0).sum())
                C[:n_cont, 0] = prevk
            kk, rows_flat = _reduce_adaptive(C, kn_i, ks, tol_g, bspline, g1 - g0, k1x) if (bspline or (len(kn_i) and kn_i.min() < K)) \
                else _reduce_k(C, ks, tol_g, k1x)
            if g == 'P' and fmask is not None:                 # followers keep the trained K (masks are per knot)
                for i in np.nonzero(fmask)[0].tolist():
                    kk[i] = K; rows_flat[i] = C[i].reshape(-1).tolist()
            for kv in kk:
                stats['k_hist'][g][kv] = stats['k_hist'][g].get(kv, 0) + 1
            stats['knots_sent'] += int(sum(kk)); stats['knot_scalars_sent'] += int(sum(kk)) * dim
            stats['knots_at_K_trained'] += int(kn_i.sum())
            kk_all.append(kk); knots_all.append(rows_flat)
        dcode = np.where(dth > g1, g1 - g0 + 1, dth - g0)
        sn_kw = {}
        if A is not None:
            k0 = (gi // key_interval) * key_interval
            pos = A.lookup(k0)
            grp = A.groups[k0]
            which = [pos[int(i)] for i in tr_ids.tolist()]       # the SN assignment is keyed by trainer ids
            sn_rows = A.rows(k0, which)
            sn_kw = {'sn_deg': [int(grp['deg'][j]) for j in which], 'sn_idx': [list(grp['idx'][j]) for j in which],
                     'sn_cb': grp['cb'] if gi == k0 else None}
        else:
            sn_rows = qSN_of(gi)[tr_ids].tolist()
        sec = Section(index=gi, g0=g0, nframes=g1 - g0, ids=ids.tolist(), n_cont=n_cont,
                      birth=(b[n_cont:] - g0).tolist(), death=dcode.tolist(), k=kk_all, knots=knots_all,
                      sn=sn_rows, follow=fmask.tolist() if fmask is not None else None, **sn_kw)
        sec = apply_follow(params, sec)
        knots_all = sec.knots
        for g, dim in GROUPS:
            gi_ = [x for x, _ in GROUPS].index(g)
            last.setdefault(g, {})
            for i, row in zip(ids.tolist(), knots_all[gi_]):
                last[g][i] = row[-dim:]
        sections.append(sec)
    if keep is not None:
        stats['keep_kept'] = int(keep.sum()); stats['keep_total'] = int(len(keep))
    if plan is not None:
        stats['section_params'] = sec_params               # NOT json-serialisable: the caller (codec.measure, s7) pops it
    return sec_params[0], sections, stats
