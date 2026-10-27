"""SN-VQ encoder side: SH degree per splat and
vector-quantised higher SH bands per key interval.

  degree_rule(SN, w, degs, q)      per splat the smallest allowed degree whose dropped bands carry little weighted
                                   energy: score_d = w * sum(SN[bands > d]**2); the q[j] lowest-score fraction of the
                                   remaining splats takes the j-th degree (q = quantile fractions per allowed degree
                                   except the highest, which takes the rest). w = 1 or opacity x area (importance).
  train_codebooks(X, w, sizes, ...)  weighted k-means in FLOAT space (what the renderer sees), stage 0 on X, every
                                   further stage on the residual; entries quantised to the SN grid (stage 0: 0..255,
                                   residual stages: signed steps), assignment re-done against the QUANTISED entries so
                                   the decoder's integer reconstruction is what was optimised.
  Assignment: per key interval (key section + its delta sections) one codebook set; every splat first coded in that
  interval (newborns, and all splats of the key section) is assigned once; `sn` rows = codec.csgs.vq_row(...).
  save/load: the assignment (degrees, indices, codebooks, SN quantisation table) as .npz so a retraining with fixed SN
  and the following encode reproduce exactly the same bitstream SN part (prep.sections_from_curves(sn_vq_from=...)).
"""
import numpy as np

from .csgs import sn_dim, vq_row

DEG_DIMS = {d: sn_dim(d) for d in range(4)}


def parse_levels(spec):
    """'1:1024,3:4096+1024' -> [(1, [1024]), (3, [4096, 1024])]"""
    out = []
    for part in spec.split(','):
        deg, sizes = part.split(':')
        out.append((int(deg), [int(s) for s in sizes.split('+')]))
    return out


def importance(opac_logit, scales_log):
    """opacity (sigmoid of the max logit over the knots) x area of the two largest axes (exp of the max log-scale
    over the knots): a rough weight for how much a splat's SH bands can change the picture."""
    op = 1.0 / (1.0 + np.exp(-np.asarray(opac_logit, np.float64).max(-1)))
    s = np.exp(np.asarray(scales_log, np.float64).max(1))                # [n,3] max over knots
    s.sort(-1)
    return op * s[:, -1] * s[:, -2]


def band_energy(SN, lo_deg):
    """sum of squares of the SN scalars of bands above degree lo_deg (coefficient-major, 3 channels)."""
    return (np.asarray(SN, np.float64)[:, DEG_DIMS[lo_deg]:] ** 2).sum(-1)


def degree_rule(SN, w, degs, q):
    """degs: allowed degrees incl. 0, ascending (e.g. [0, 1, 3]); q: fraction per degree except the last.
    Returns int array of degrees. Splats are ranked by w * energy(bands above the degree); the lowest q[j] fraction
    of the splats not yet assigned takes degs[j]."""
    n = len(SN)
    deg = np.full(n, degs[-1], np.int64)
    left = np.ones(n, bool)
    for j, d in enumerate(degs[:-1]):
        frac = float(q[j])
        if frac <= 0:
            continue
        score = w * band_energy(SN, d)
        idx = np.nonzero(left)[0]
        k = int(round(frac * len(idx)))
        if k <= 0:
            continue
        take = idx[np.argsort(score[idx], kind='stable')[:k]]
        deg[take] = d
        left[take] = False
    return deg


def degree_rule_abs(SN, w, degs, thresholds):
    """same, with absolute thresholds on w * energy instead of fractions: deg = smallest d with score_d <= thr[j]."""
    n = len(SN)
    deg = np.full(n, degs[-1], np.int64)
    for j, d in list(enumerate(degs[:-1]))[::-1]:          # highest first so the smallest passing degree wins
        score = w * band_energy(SN, d)
        deg[score <= float(thresholds[j])] = d
    return deg


def _sqdist_argmin(X, C, w=None, chunk=8192):
    """nearest codebook entry per row of X (float64), chunked; returns (argmin, min squared distance)."""
    Cn = (C ** 2).sum(-1)
    a = np.empty(len(X), np.int64); dmin = np.empty(len(X))
    for i in range(0, len(X), chunk):
        xb = X[i:i + chunk]
        d = Cn[None] - 2 * xb @ C.T
        j = d.argmin(-1)
        a[i:i + chunk] = j
        dmin[i:i + chunk] = d[np.arange(len(xb)), j] + (xb ** 2).sum(-1)
    return a, np.maximum(dmin, 0)


def kmeans(X, w, k, iters=25, seed=0):
    """weighted k-means (Lloyd), k-means++-like seeding on a weighted subsample; X [n,d] float64, w [n] >= 0."""
    rng = np.random.default_rng(seed)
    n = len(X)
    if n == 0:
        return np.zeros((k, X.shape[1]))
    w = np.asarray(w, np.float64)
    p = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
    k_eff = min(k, n)
    # seeding: first centre by weight, then D^2 sampling on a subsample (cheap for k up to a few thousand)
    sub = rng.choice(n, min(n, 20000), replace=False, p=None)
    Xs, ps = X[sub], p[sub] / p[sub].sum()
    C = np.empty((k_eff, X.shape[1]))
    C[0] = Xs[rng.choice(len(Xs), p=ps)]
    d2 = ((Xs - C[0]) ** 2).sum(-1)
    for j in range(1, k_eff):
        pr = ps * d2
        if pr.sum() <= 0:
            pr = ps
        C[j] = Xs[rng.choice(len(Xs), p=pr / pr.sum())]
        d2 = np.minimum(d2, ((Xs - C[j]) ** 2).sum(-1))
    for _ in range(iters):
        a, _ = _sqdist_argmin(X, C)
        cnt = np.bincount(a, weights=w, minlength=k_eff)
        S = np.zeros_like(C)
        np.add.at(S, a, X * w[:, None])
        nz = cnt > 0
        C[nz] = S[nz] / cnt[nz, None]
        empty = np.nonzero(~nz)[0]
        if len(empty):                                      # re-seed empty clusters at the worst-fitted points
            _, dm = _sqdist_argmin(X, C)
            far = np.argsort(dm * w)[::-1][:len(empty)]
            C[empty] = X[far]
    if k_eff < k:                                           # pad (fewer points than entries): repeat the first entry
        C = np.concatenate([C, np.repeat(C[:1], k - k_eff, 0)])
    return C


def quantise_cb(C, off, scale, stage):
    """stage 0: grid indices 0..255 of the entries; residual stages: signed step counts (clipped to +-255)."""
    q = np.round(C / scale) if stage else np.round((C - off) / scale)
    lo, hi = (-255, 255) if stage else (0, 255)
    return np.clip(q, lo, hi).astype(np.int64)


def train_level(X, w, sizes, off, scale, iters=25, seed=0):
    """X [n, dim] float SN of the splats of one degree level; returns (quantised codebooks per stage [size, dim] int,
    indices [n, stages] int, float reconstruction [n, dim])."""
    n, dim = X.shape
    cbs, idx = [], np.zeros((n, len(sizes)), np.int64)
    rec = np.zeros_like(X)
    target = X
    for s, size in enumerate(sizes):
        C = kmeans(target, w, size, iters=iters, seed=seed + s)
        Cq = quantise_cb(C, off, scale, s)
        Cf = Cq * scale if s else off + Cq * scale
        if n:
            a, _ = _sqdist_argmin(target, Cf)
            idx[:, s] = a
            rec = rec + Cf[a]
        cbs.append(Cq)
        target = X - rec
    return cbs, idx, rec


class Assignment:
    """per key interval: ids, deg, idx (list per splat), codebooks (per level per stage, python ints) + the SN grid."""

    def __init__(self, levels, off, scale, zero):
        self.levels = levels                                 # [(deg, [sizes])]
        self.off, self.scale, self.zero = np.asarray(off, np.float64), np.asarray(scale, np.float64), [int(v) for v in zero]
        self.groups = {}                                     # key index -> dict(ids, deg, idx, cb)

    def degs(self):
        return [d for d, _ in self.levels]

    def level_of(self, deg):
        return 0 if deg == 0 else self.degs().index(deg) + 1

    def fit_group(self, kidx, ids, SN, w, deg, iters=25, seed=0):
        """SN [n, n_sn] float (dequantised grid or trained floats), deg [n] per splat; trains one codebook set."""
        n = len(ids)
        idx = [[] for _ in range(n)]
        cb = []
        for li, (d, sizes) in enumerate(self.levels):
            dim = DEG_DIMS[d]
            sel = np.nonzero(deg == d)[0]
            cbs, ix, _ = train_level(SN[sel][:, :dim], w[sel], sizes, self.off[:dim], self.scale[:dim], iters, seed + 97 * li)
            cb.append([c.tolist() for c in cbs])
            for r, row in zip(sel.tolist(), ix.tolist()):
                idx[r] = row
        self.groups[kidx] = {'ids': np.asarray(ids, np.int64), 'deg': np.asarray(deg, np.int64), 'idx': idx, 'cb': cb}
        return self.groups[kidx]

    def rows(self, kidx, which):
        """integer sn rows (lists) for the splats `which` (positions in the group) of key interval kidx."""
        g = self.groups[kidx]
        degs = self.degs()
        return [vq_row(self.level_of(int(g['deg'][i])), g['idx'][i], g['cb'], self.zero, degs) for i in which]

    def float_rows(self, kidx, which):
        return self.off[None] + np.asarray(self.rows(kidx, which), np.float64) * self.scale[None]

    def lookup(self, kidx):
        """id -> position in the group."""
        return {int(i): j for j, i in enumerate(self.groups[kidx]['ids'].tolist())}

    def save(self, path):
        out = {'levels_deg': np.array([d for d, _ in self.levels]), 'levels_sizes': np.array([len(s) for _, s in self.levels]),
               'sizes_flat': np.array([x for _, s in self.levels for x in s]), 'off': self.off, 'scale': self.scale,
               'zero': np.array(self.zero), 'keys': np.array(sorted(self.groups))}
        for kidx, g in self.groups.items():
            ns = max(len(s) for _, s in self.levels)
            idx = np.full((len(g['ids']), ns), -1, np.int64)
            for i, row in enumerate(g['idx']):
                idx[i, :len(row)] = row
            out[f'k{kidx}_ids'] = g['ids']; out[f'k{kidx}_deg'] = g['deg']; out[f'k{kidx}_idx'] = idx
            for li, lvl in enumerate(g['cb']):
                for si, c in enumerate(lvl):
                    out[f'k{kidx}_cb_l{li}_s{si}'] = np.asarray(c, np.int64)
        np.savez(path, **out)

    @staticmethod
    def load(path):
        z = np.load(path)
        sizes_flat = z['sizes_flat'].tolist(); levels = []; o = 0
        for d, ns in zip(z['levels_deg'].tolist(), z['levels_sizes'].tolist()):
            levels.append((d, sizes_flat[o:o + ns])); o += ns
        A = Assignment(levels, z['off'], z['scale'], z['zero'].tolist())
        for kidx in z['keys'].tolist():
            idx = z[f'k{kidx}_idx']
            deg = z[f'k{kidx}_deg']
            rows = []
            for i in range(len(deg)):
                lv = A.level_of(int(deg[i]))
                rows.append(idx[i, :len(levels[lv - 1][1])].tolist() if lv else [])
            cb = [[z[f'k{kidx}_cb_l{li}_s{si}'].tolist() for si in range(len(sizes))] for li, (_, sizes) in enumerate(levels)]
            A.groups[kidx] = {'ids': z[f'k{kidx}_ids'], 'deg': deg, 'idx': rows, 'cb': cb}
        return A


def zero_row(off, scale):
    return np.clip(np.round(-np.asarray(off, np.float64) / np.asarray(scale, np.float64)), 0, 255).astype(np.int64).tolist()
