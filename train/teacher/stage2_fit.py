#!/usr/bin/env python3
"""stage 2a: least-squares fit of our curve model to the stage-1 per-frame tracks (CPU, no images).
Reads a dense e3/stage-1 tracks.npz:
sections of L frames, per splat and section K control points on a hat basis with knots on frames g0 .. g0+Lg-1 ('gop'),
soft knot 0 (every section fits its own knot 0, the codec sends it), lifetimes [birth, death) from stage 1's identity.
Per splat and section: weighted LS, weight 1 on alive frames, eps on the other frames (target = nearest alive value, keeps
knots outside the lifetime sane for the quantiser) + a small second-difference penalty. Quaternions normalised and
sign-aligned along the track first. SH: stage 1 is SH0 only -> shN = 0 (15 bands, encode with --sn-deg 0).
Rows are processed in chunks (fits in ~15 GB RAM).
Writes <out>/curves.npz (e7 keys), <out>/tracks.npz (dense per-frame samples of the fitted curves, like e7's tracks.npz:
for float-path evaluations) and <out>/fit.json (fit error on alive frames).
  python stage2_fit.py <stage1 dir> <out k dir> [--L 32] [--K 9]
"""
import argparse
import json
import os
import time

import numpy as np

ATTRS = ('means', 'quats', 'scales', 'opacities', 'sh0')


def hat_basis(Lg, K):
    span = max(Lg - 1, 1)
    A = np.zeros((Lg, K))
    for t in range(Lg):
        tau = t * (K - 1) / span
        j = min(int(np.floor(tau)), K - 2)
        w = tau - j
        A[t, j] = 1 - w
        A[t, j + 1] = w
    return A


def ffill_bfill(mask):
    Lg, n = mask.shape
    t = np.arange(Lg)[:, None]
    prev = np.maximum.accumulate(np.where(mask, t, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(mask, t, Lg)[::-1], axis=0)[::-1]
    return np.where(prev >= 0, prev, nxt).clip(0, Lg - 1)


def sig(x):
    return 1 / (1 + np.exp(-x))


KS = (1, 2, 3, 5, 9, 17)
GROUP = {'means': 'P', 'quats': 'R', 'scales': 'S', 'opacities': 'O', 'sh0': 'D'}


def per_splat_err(k, P, Y, M):
    """[n] RMS error over alive frames, in the group's tolerance unit (m, deg, log-scale, alpha, SH0 coefficient)."""
    if k == 'means':
        e = np.linalg.norm(P - Y, axis=-1)
    elif k == 'quats':
        qp = P / np.maximum(np.linalg.norm(P, axis=-1, keepdims=True), 1e-12)
        e = np.degrees(2 * np.arccos(np.clip(np.abs((qp * Y).sum(-1)), 0, 1)))
    elif k == 'opacities':
        e = np.abs(sig(P[..., 0]) - sig(Y[..., 0]))
    else:
        e = np.abs(P - Y).max(-1)
    e = np.where(M, e, 0.0)
    return np.sqrt((e ** 2).sum(0) / np.maximum(M.sum(0), 1)), np.where(M, e, 0).max(0)


def lsq(A, W, Y, smooth):
    Kp = A.shape[1]
    G = np.einsum('tn,tk,tl->nkl', W, A, A)
    if Kp >= 3:
        D2 = np.diff(np.eye(Kp), 2, axis=0)
        G = G + smooth * (D2.T @ D2)[None]
    G = G + 1e-9 * np.eye(Kp)[None]
    return np.linalg.solve(G, np.einsum('tk,tn,tnd->nkd', A, W, Y))


def fit_chunk(z, ids, g0, g1, birth, death, A, a):
    Lg, K = A.shape
    n = len(ids)
    M = np.zeros((Lg, n), bool)
    for t in range(g0, g1):
        M[t - g0] = (birth[ids] <= t) & (death[ids] > t)
    fi = ffill_bfill(M)
    cols = np.arange(n)
    Y = {}
    for k in ATTRS:
        y = z[k][g0:g1][:, ids].astype(np.float64)
        y = y.reshape(y.shape[0], y.shape[1], -1)
        Y[k] = y[fi, cols[None, :]]
    q = Y['quats']
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    ref = q[0].copy()
    for t in range(Lg):
        s = np.sign((q[t] * ref).sum(-1))
        s[s == 0] = 1
        q[t] *= s[:, None]
        ref = q[t]
    W = np.where(M, 1.0, a.eps)
    X, Kp = {}, {}
    tol = a.tols
    for k in ATTRS:
        if tol is None:                                   # plain K-knot fit (no adaptive K)
            X[k] = lsq(A, W, Y[k], a.smooth)
            Kp[k] = np.full(n, K, np.int8)
            continue
        best = None
        kp = np.full(n, K, np.int8)
        done = np.zeros(n, bool)
        for k2 in [x for x in KS if x <= K and (x == 1 or (K - 1) % (x - 1) == 0)]:
            if k2 == 1:
                xs = (np.einsum('tn,tnd->nd', W, Y[k]) / W.sum(0)[:, None])[:, None]
                full = np.repeat(xs, K, axis=1)
            else:
                xs = lsq(hat_basis(Lg, k2), W, Y[k], a.smooth)
                full = np.einsum('jk,nkd->njd', hat_basis(K, k2), xs) if k2 < K else xs
            P = np.einsum('tk,nkd->tnd', A, full)
            rms, mx = per_splat_err(k, P, Y[k], M)
            tl = tol[GROUP[k]] * (a.impm[ids] if a.impm is not None else 1.0)
            ok = (rms <= tl) & (mx <= 3 * tl) & ~done
            if best is None:
                best = full.copy()
            best[ok] = full[ok]
            kp[ok] = k2
            done |= ok
            if k2 == K:
                best[~done] = full[~done]
        X[k] = best
        Kp[k] = kp
    P = {k: np.einsum('tk,nkd->tnd', A, X[k]) for k in ATTRS}
    err = {'dm': np.linalg.norm(P['means'] - Y['means'], axis=-1)[M]}
    qp = P['quats'] / np.maximum(np.linalg.norm(P['quats'], axis=-1, keepdims=True), 1e-12)
    err['ang'] = np.degrees(2 * np.arccos(np.clip(np.abs((qp * Y['quats']).sum(-1)), 0, 1)))[M]
    err['oa'] = np.abs(sig(P['opacities'][..., 0]) - sig(Y['opacities'][..., 0]))[M]
    err['sc2'] = ((P['scales'] - Y['scales'])[M] ** 2).ravel()
    err['dc2'] = ((P['sh0'] - Y['sh0'])[M] ** 2).ravel()
    return X, P, M, err, Kp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('s1')
    ap.add_argument('out')
    ap.add_argument('--L', type=int, default=32)
    ap.add_argument('--K', type=int, default=9)
    ap.add_argument('--eps', type=float, default=1e-4)
    ap.add_argument('--smooth', type=float, default=1e-3)
    ap.add_argument('--chunk', type=int, default=30000)
    ap.add_argument('--tols', default='P=0.0015,R=3,S=0.05,O=0.02,D=0.03',
                    help='adaptive K: per group the smallest nested K (1,2,3,5,9) whose fit has RMS <= tol and max <= 3 tol '
                         '(P m, R deg, S log-scale, O alpha, D SH0 coeff.); "none" = plain K-knot fit')
    ap.add_argument('--imp', default='', help='contrib.npz (train/recipe/contrib.py, same id space): splats below the '
                    'median contribution get a looser tolerance (x (median/score)^imp-pow, at most imp-max)')
    ap.add_argument('--imp-pow', type=float, default=0.5)
    ap.add_argument('--imp-max', type=float, default=8.0)
    a = ap.parse_args()
    a.tols = None if a.tols == 'none' else {kv.split('=')[0]: float(kv.split('=')[1]) for kv in a.tols.split(',')}
    a.impm = None
    if a.imp:                    # importance-weighted tolerance: tol x clip((median score / score)^pow, 1, max)
        sc = np.load(a.imp)['score'].astype(np.float64)
        med = float(np.median(sc[sc > 0]))
        a.impm = np.clip((med / np.maximum(sc, 1e-6)) ** a.imp_pow, 1.0, a.imp_max)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    z = np.load(os.path.join(a.s1, 'tracks.npz'))
    z = {k: z[k] for k in ATTRS + ('birth', 'death')}
    birth, death = z['birth'].astype(np.int32), z['death'].astype(np.int32)
    T, Ng = z['means'].shape[:2]
    L, K = a.L, a.K
    curves, rep = {}, {'frames': T, 'L': L, 'K': K, 'N_ids': int(Ng), 'tols': a.tols, 'imp': a.imp or None,
                       'imp_pow': a.imp_pow, 'imp_max': a.imp_max, 'sections': []}
    dense = {k: np.full(z[k].shape, np.nan, np.float32 if k in ('means', 'quats') else np.float16) for k in ATTRS}
    for gi, g0 in enumerate(range(0, T, L)):
        g1 = min(T, g0 + L)
        A = hat_basis(g1 - g0, K)
        ids = np.nonzero((birth < g1) & (death > g0))[0]
        Xs = {k: [] for k in ATTRS}
        Ks = {k: [] for k in ATTRS}
        errs = {}
        for c0 in range(0, len(ids), a.chunk):
            ic = ids[c0:c0 + a.chunk]
            X, P, M, err, Kp = fit_chunk(z, ic, g0, g1, birth, death, A, a)
            for k in ATTRS:
                Xs[k].append(X[k].astype(np.float32))
                Ks[k].append(Kp[k])
                for t in range(g1 - g0):
                    al = M[t]
                    v = P[k][t, al]
                    slot = dense[k][g0 + t, ic[al]]
                    dense[k][g0 + t, ic[al]] = v.reshape(slot.shape)
            for k, v in err.items():
                errs.setdefault(k, []).append(v)
            del X, P, M
        e = {k: np.concatenate(v) for k, v in errs.items()}
        sec = {'g0': g0, 'g1': g1, 'rows': len(ids), 'born_in_section': int((birth[ids] > g0).sum()),
               'died_in_section': int(((death[ids] > g0) & (death[ids] < g1)).sum()),
               'means_l2_mean': float(e['dm'].mean()), 'means_l2_p99': float(np.percentile(e['dm'], 99)),
               'quat_deg_mean': float(e['ang'].mean()), 'opa_mae': float(e['oa'].mean()),
               'scales_rmse': float(np.sqrt(e['sc2'].mean())), 'sh0_rmse': float(np.sqrt(e['dc2'].mean()))}
        rep['sections'].append(sec)
        print('section', sec, flush=True)
        curves[f'g{gi}_ids'] = ids.astype(np.int32)
        curves[f'g{gi}_birth'] = birth[ids]
        curves[f'g{gi}_death'] = np.minimum(death[ids], T).astype(np.int32)
        for k in ATTRS:
            x = np.concatenate(Xs[k])
            curves[f'g{gi}_{k}'] = x[..., 0] if k == 'opacities' else x
            kp = np.concatenate(Ks[k])
            curves[f'g{gi}_p43k{GROUP[k]}'] = kp
            sec.setdefault('k_hist', {})[GROUP[k]] = {int(v): int((kp == v).sum()) for v in np.unique(kp)}
        print('  k_hist', sec['k_hist'], flush=True)
    np.savez(os.path.join(a.out, 'curves.npz'), L=L, K=K, knots='gop', soft_knot0=True, basis='hat', **curves)
    np.savez(os.path.join(a.out, 'tracks.npz'), birth=birth, death=np.minimum(death, T),
             shN=np.zeros((Ng, 15, 3), np.float16), **dense)
    rep['secs'] = round(time.time() - t0, 1)
    json.dump(rep, open(os.path.join(a.out, 'fit.json'), 'w'), indent=1)
    print('S2FIT OK', a.out, rep['secs'], flush=True)


if __name__ == '__main__':
    main()
