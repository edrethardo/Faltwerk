#!/usr/bin/env python3
"""synthetic .csgs fixtures for the TypeScript decoder tests (random content, no scene data).

usage: python web/test/make_fixture.py web/test/fixtures
Writes three streams with the Python reference encoder (codec/csgs.py):
  synth_shared_s000{0,1,2}.csgs   shared_knot=True,  key_interval=2 (key, delta, key), last section shorter
  synth_res_s000{0,1}.csgs        shared_knot=False (first knot sent as residual), key_interval=4
  synth_vq_s000{0,1,2}.csgs       v2 SN_VQ: levels deg 1 [4] and deg 3 [8, 4] (residual VQ), key_interval=2;
                                  fresh codebooks + indices in every key section, newborns of the delta section indexed
                                  against the carried codebooks
and dumps the decoded reference next to them with web/bench/dump_ref.py.
"""
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from codec.csgs import Encoder, Params, Section, apply_follow, sn_dim, vq_row  # noqa: E402

GROUPS = [('P', 3), ('R', 4), ('S', 3), ('O', 1), ('D', 3)]
N_SN = 45
VQ_LEVELS = [(1, [4]), (3, [8, 4])]
KS = [1, 2, 3, 5]
KS_FOLLOW = [1, 2, 3, 5, 9]          # follow mode: every K-1 divides max(ks)-1


def make_codebooks(rng):
    cb = []
    for deg, sizes in VQ_LEVELS:
        d = sn_dim(deg)
        stages = []
        for si, size in enumerate(sizes):
            if si == 0:
                stages.append([[max(0, min(255, 120 + rng.randint(-40, 40))) if rng.random() < .9 else rng.randint(0, 255)
                                for _ in range(d)] for _ in range(size)])
            else:
                stages.append([[rng.randint(-25, 25) if rng.random() < .95 else rng.choice([-255, 255]) for _ in range(d)]
                               for _ in range(size)])
        cb.append(stages)
    return cb


def make_stream(rng, shared, nsec, L, n0, key_interval, last_short, vq=False, pred_modes=None, knn_k=0, follow=False, rescale=False,
                bspline=False):
    """pred_modes: None (v1/v2 knot coding) or a list with one mode list (per group) per section (cycled): v3.
    follow: random follow masks on group P, canonicalised with apply_follow (v4). pred/follow streams get clustered,
    duplicated and far-away P positions (k-NN ties, sparse-grid brute force) and no 2^30 knot jumps (position span
    stays below 2^25, the exact-double limit of the TS k-NN). bspline: v6, K up to 9."""
    ks = [1, 2, 3, 5, 9] if bspline else (KS_FOLLOW if follow else KS)
    nq = sum(d for _, d in GROUPS) + N_SN
    offs = [rng.uniform(-1, 1) for _ in range(nq)]
    scs = [rng.uniform(0.001, 0.05) for _ in range(nq)]
    for c in range(3, 7):          # R: offset 0, scale 1/127 like prep.py
        offs[c], scs[c] = 0.0, 1 / 127
    sn_zero = [rng.randint(100, 156) for _ in range(N_SN)] if vq else None
    degs = [d for d, _ in VQ_LEVELS]
    spatial = pred_modes is not None or follow

    def params_of(si):
        modes = None if pred_modes is None else pred_modes[si % len(pred_modes)]
        kk = knn_k if modes is not None and any(m & 2 for m in modes) else 0
        o2, s2 = offs, scs
        if rescale and si > 0:         # /S7: knot table (P R S O D) differs from section to section, SN table fixed
            nk_ = sum(d for _, d in GROUPS)
            o2 = [o + 0.013 * si * (1 + i % 3) if i < nk_ and i not in range(3, 7) else o for i, o in enumerate(offs)]
            s2 = [s * (1 + 0.07 * si + 0.02 * (i % 4)) if i < nk_ and i not in range(3, 7) else s for i, s in enumerate(scs)]
        return Params(L=L, ks=ks, groups=GROUPS, n_sn=N_SN, offsets=o2, scales=s2, shared_knot=shared,
                      key_interval=key_interval, sn_vq=VQ_LEVELS if vq else None, sn_zero=sn_zero, pred=modes, knn_k=kk,
                      bspline=bspline)
    p = params_of(0)
    plist = []
    secs, cont, next_id = [], [], 0          # cont: list of (id, last knots per group, sn, sn_deg, sn_idx)
    cb = None

    def pick(cb):
        lv = rng.choice([0, 1, 2, 2])
        idxs = [rng.randrange(s) for s in VQ_LEVELS[lv - 1][1]] if lv else []
        return (degs[lv - 1] if lv else 0), idxs, vq_row(lv, idxs, cb, sn_zero, degs)

    for si in range(nsec):
        nf = L if not (last_short and si == nsec - 1) else L // 2 + 1
        g0 = si * L
        nn = n0 if si == 0 else rng.randint(n0 // 4, n0 // 2)
        ids = [c[0] for c in cont] + list(range(next_id, next_id + nn))
        next_id += nn
        nc = len(cont)
        n = nc + nn
        key = si % key_interval == 0
        birth = [rng.randrange(nf) for _ in range(nn)]
        death = []
        for i in range(n):
            b = birth[i - nc] if i >= nc else -1
            r = rng.random()
            death.append(nf + 1 if r < 0.6 else rng.randint(max(b + 1, 1), nf))
        k, knots = [], []
        for g, (tag, d) in enumerate(GROUPS):
            kg, rows = [], []
            for i in range(n):
                K = rng.choice(ks)
                if i < nc:
                    first = list(cont[i][1][g]) if shared else [v + rng.randint(-3, 3) for v in cont[i][1][g]]
                else:
                    first = [rng.randint(-2000, 2000) if tag == 'P' else rng.randint(-127, 127) for _ in range(d)]
                    if spatial and tag == 'P':
                        u = rng.random()
                        if u < .45:                   # tight cluster with duplicates: k-NN ties
                            first = [rng.randint(-3, 3) for _ in range(3)]
                        elif u < .5:                  # far outlier: sparse grid -> brute-force fallback
                            first = [rng.randint(-900000, 900000) for _ in range(3)]
                row = list(first)
                for j in range(1, K):
                    row += [row[(j - 1) * d + c] + rng.choice([0, 0, 1, -1, rng.randint(-40, 40), rng.randint(-70000, 70000)])
                            for c in range(d)]
                kg.append(K); rows.append(row)
            k.append(kg); knots.append(rows)
        if si == 0 and not spatial:      # extreme knot jumps: zig-zag values near 2**32 exercise the >15-bit raw paths of the decoder
            lo, hi = -(1 << 30), (1 << 30) - 1
            k[0][0], knots[0][0] = 2, [lo, 5, hi, hi, -3, lo]
            k[0][1], knots[0][1] = 3, [hi, lo, 0, lo, hi, 1, 0, 0, 0]
        sn_deg = sn_idx = sn_cb = None
        if vq:
            if key:          # fresh codebooks: every splat (also continuing ones) re-indexed
                cb = sn_cb = make_codebooks(rng)
                picks = [pick(cb) for _ in range(n)]
            else:
                picks = [(cont[i][3], cont[i][4], cont[i][2]) for i in range(nc)] + [pick(cb) for _ in range(nn)]
            sn_deg = [x[0] for x in picks]; sn_idx = [x[1] for x in picks]; sn = [x[2] for x in picks]
        else:
            sn = [list(cont[i][2]) if i < nc else [rng.randint(0, 255) if rng.random() < .2 else 120 + rng.randint(-6, 6)
                                                     for _ in range(N_SN)] for i in range(n)]
        secs.append(Section(index=si, g0=g0, nframes=nf, ids=ids, n_cont=nc, birth=birth, death=death, k=k,
                            knots=knots, sn=sn, sn_deg=sn_deg, sn_idx=sn_idx, sn_cb=sn_cb))
        if follow:       # random masks on group P (never knot 0, only K >= 2), then the prediction replaces the masked knots
            masks = []
            for i in range(n):
                m = 0
                if k[0][i] >= 2 and rng.random() < .35:
                    for j in range(1, k[0][i]):
                        if rng.random() < .6:
                            m |= 1 << j
                masks.append(m)
            secs[-1].follow = masks if any(masks) else None
            secs[-1] = apply_follow(params_of(si), secs[-1])
            knots = secs[-1].knots
        plist.append(params_of(si))
        cont = [(ids[i], [knots[g][i][-GROUPS[g][1]:] for g in range(len(GROUPS))], sn[i],
                 sn_deg[i] if vq else None, sn_idx[i] if vq else None)
                for i in range(n) if death[i] == nf + 1]
    return p, secs, plist


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    rng = random.Random(1267)
    dump = os.path.join(os.path.dirname(__file__), '..', 'bench', 'dump_ref.py')
    # (name, shared, nsec, key_interval, vq, pred modes per section, knn_k, follow)
    # pred bits: 1 temporal, 2 spatial (k-NN), 4 vel0; order of the groups: P R S O D
    specs = [('synth_shared', True, 3, 2, False, None, 0, False), ('synth_res', False, 2, 4, False, None, 0, False),
             ('synth_vq', False, 3, 2, True, None, 0, False),
             # v3 pred, all modes, modes change per section like `--pred auto`
             ('synth_pred', False, 4, 2, False, [[7, 3, 6, 0, 5], [0, 1, 7, 2, 4], [3, 3, 3, 3, 3], [4, 6, 1, 5, 2]], 4, False),
             ('synth_pred_shared', True, 3, 2, False, [[3, 1, 2, 4, 7], [7, 7, 7, 7, 7], [1, 0, 2, 0, 3]], 6, False),
             ('synth_pred_vq', False, 3, 2, True, [[5, 3, 3, 1, 2], [7, 2, 0, 4, 3], [2, 2, 2, 2, 2]], 1, False),
             # v4 follow alone / with SN_VQ + pred (version 4, flags 16|8|4)
             ('synth_follow', False, 3, 2, False, None, 0, True),
             ('synth_follow_all', False, 4, 2, True, [[7, 3, 6, 0, 5], [3, 3, 7, 1, 2], [1, 2, 4, 0, 7], [2, 1, 3, 0, 4]], 4, True),
             ('synth_follow_shared', True, 3, 2, False, [[3, 1, 2, 4, 7], [1, 1, 1, 0, 2], [2, 3, 0, 1, 1]], 3, True, False),
             # v5 RESCALE (S7): quantisation table of the knots changes at every section (also at the key sections: no flag there)
             ('synth_rescale', False, 4, 3, False, [[7, 3, 6, 0, 5], [5, 4, 7, 4, 4], [3, 3, 3, 3, 3], [4, 6, 5, 5, 4]], 4, False, True),
             ('synth_rescale_all', False, 4, 3, True, [[5, 3, 4, 4, 4], [7, 7, 7, 4, 5], [1, 1, 1, 1, 1], [4, 4, 4, 4, 4]], 3, True, True)]
    streams = []
    for name, shared, nsec, ki, vq, pm, kk, fol, *rs in specs:
        p, secs, plist = make_stream(rng, shared, nsec, L=8, n0=60, key_interval=ki, last_short=(nsec == 3), vq=vq,
                                     pred_modes=pm, knn_k=kk, follow=fol, rescale=bool(rs and rs[0]))
        streams.append((name, p, secs, plist))
    # v6 B-spline stream: its own generator, replaying the three base streams it followed on the
    # branch, so synth_bsp_* stays byte-identical to the fixture committed there and every other fixture unchanged
    rng_b = random.Random(1267)
    for _, shared, nsec, ki, vq, *_ in specs[:3]:
        make_stream(rng_b, shared, nsec, L=8, n0=60, key_interval=ki, last_short=(nsec == 3), vq=vq)
    p, secs, plist = make_stream(rng_b, True, 3, L=8, n0=60, key_interval=2, last_short=True, bspline=True)
    streams.append(('synth_bsp', p, secs, plist))
    # v6 combined with every other flag (SN_VQ, pred, follow, rescale), own generator (other fixtures unchanged)
    rng_c = random.Random(1308)
    p, secs, plist = make_stream(rng_c, False, 4, L=8, n0=60, key_interval=3, last_short=False, vq=True,
                                 pred_modes=[[7, 3, 6, 0, 5], [3, 3, 7, 1, 2], [1, 2, 4, 0, 7], [2, 1, 3, 0, 4]], knn_k=4,
                                 follow=True, rescale=True, bspline=True)
    streams.append(('synth_bsp_all', p, secs, plist))
    for name, p, secs, plist in streams:
        enc = Encoder(p)
        files = []
        for s, ps in zip(secs, plist):
            enc.params = ps
            f = os.path.join(out, f'{name}_s{s.index:04d}.csgs')
            open(f, 'wb').write(enc.encode(s))
            files.append(f)
        subprocess.check_call([sys.executable, dump, out] + files, env=dict(os.environ, REF_SAMPLE='12'))
    # v6 oracle: B-spline weights as the exact IEEE-double bit patterns (the TS port must reproduce them bit for bit)
    import json
    import struct
    from codec.csgs import bspline_weights
    us = [0.0, 1e-9, 1 / 31, 0.1, 0.25, 0.3333333333333333, 0.5, 0.5000000000000001, 0.7, 0.75, 0.9999999999999999, 1.0, 1.0 / 7 * 3]
    rows = [{'K': K, 'u': u, 'bits': [struct.pack('<d', w).hex() for w in bspline_weights(K, u)]}
            for K in (1, 2, 3, 4, 5, 6, 9, 15) for u in us]
    json.dump(rows, open(os.path.join(out, 'synth_bsp_weights.json'), 'w'))


if __name__ == '__main__':
    main()
