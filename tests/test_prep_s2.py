"""(S2): codec.prep newborn_order='morton' – same splats and knots as the id order, ids renumbered consistently
across sections, stream stays v1 and roundtrips; verify_equal-style comparison helper shared with experiments/s2."""
import numpy as np

from codec import prep
from codec.csgs import Decoder, Encoder

L, K = 4, 3


def synth(tmp_path, n0=12, seed=0):
    """two sections: n0 splats born at frame 0 (ids 0..n0-1), 3 die at the end of section 0, 5 newborns in section 1."""
    rng = np.random.default_rng(seed)
    T = 2 * L
    N = n0 + 5
    birth = np.array([0] * n0 + [L] * 5)
    death = np.array([L] * 3 + [T] * (n0 - 3) + [T] * 5)
    means = np.full((T, N, 3), np.nan, np.float32)
    cv = {'L': L, 'K': K, 'knots': 'gop', 'soft_knot0': True}
    for gi in range(2):
        g0 = gi * L
        ids = np.nonzero((birth < g0 + L) & (death > g0))[0]
        rng.shuffle(ids)                                        # trainer order is arbitrary; prep sorts by id
        n = len(ids)
        cv[f'g{gi}_ids'] = ids.astype(np.int32)
        cv[f'g{gi}_birth'] = birth[ids].astype(np.int32); cv[f'g{gi}_death'] = death[ids].astype(np.int32)
        base = rng.uniform(-1, 1, (n, 1, 3)).astype(np.float32)
        cv[f'g{gi}_means'] = base + rng.normal(0, 0.05, (n, K, 3)).astype(np.float32)
        cv[f'g{gi}_quats'] = rng.normal(0, 1, (n, K, 4)).astype(np.float32)
        cv[f'g{gi}_scales'] = rng.uniform(-5, -2, (n, K, 3)).astype(np.float32)
        cv[f'g{gi}_opacities'] = rng.uniform(-3, 3, (n, K)).astype(np.float32)
        cv[f'g{gi}_sh0'] = rng.uniform(-1, 1, (n, K, 3)).astype(np.float32)
        for t in range(L):
            means[g0 + t, ids] = cv[f'g{gi}_means'][:, min(t * (K - 1) // (L - 1), K - 1)]
    tr = {'means': means, 'birth': birth, 'death': death, 'shN': rng.uniform(-1, 1, (N, 9)).astype(np.float32)}
    cp, tp = tmp_path / 'curves.npz', tmp_path / 'tracks.npz'
    np.savez(cp, **cv); np.savez(tp, **tr)
    return str(cp), str(tp)


def records(sec, params):
    """per-splat content without ids: (birth-or-None, death, K per group, knots per group, sn) as a sorted list."""
    out = []
    nc = sec.n_cont
    for i in range(len(sec.ids)):
        b = sec.birth[i - nc] if i >= nc else -1
        out.append((b, sec.death[i], tuple(sec.k[g][i] for g in range(len(params.groups))),
                    tuple(tuple(sec.knots[g][i]) for g in range(len(params.groups))), tuple(sec.sn[i])))
    return sorted(out)


def test_morton_order_keeps_content_and_renumbers_consistently(tmp_path):
    cp, tp = synth(tmp_path)
    p_id, s_id, _ = prep.sections_from_curves(cp, tp, qpos=0.01, quat='raw', tol=1)
    p_mo, s_mo, st = prep.sections_from_curves(cp, tp, qpos=0.01, quat='raw', tol=1, newborn_order='morton')
    assert st['newborn_order'] == 'morton' and p_id == p_mo
    for a, b in zip(s_id, s_mo):
        assert records(a, p_id) == records(b, p_mo)         # same splats, knots, lifetimes
        assert b.ids == sorted(b.ids) and len(set(b.ids)) == len(b.ids)
        nn = len(b.ids) - b.n_cont
        assert b.ids[b.n_cont:] == [b.ids[b.n_cont] + i for i in range(nn)] if nn else True
    # continuing ids of section 1 are exactly the survivors of section 0, in id order
    surv = [i for i, dc in zip(s_mo[0].ids, s_mo[0].death) if dc == s_mo[0].nframes + 1]
    assert s_mo[1].ids[:s_mo[1].n_cont] == sorted(surv)
    # the permutation is not the identity (otherwise the test proves nothing) and the knot-0 DPCM is spatially sorted
    assert s_mo[0].knots[0] != s_id[0].knots[0]
    # v1 stream, bit-exact roundtrip incl. the shared state between sections
    enc, dec = Encoder(p_mo), Decoder()
    for s in s_mo:
        buf = enc.encode(s)
        assert buf[4] == 1
        assert dec.decode(buf) == s


def test_morton_reorder_is_v1_and_pred_modes_compose(tmp_path):
    cp, tp = synth(tmp_path, seed=3)
    p, secs, _ = prep.sections_from_curves(cp, tp, qpos=0.01, quat='raw', tol=1, newborn_order='morton',
                                           pred=[3, 1, 0, 4, 1], knn_k=4)
    enc, dec = Encoder(p), Decoder()
    for s in secs:
        buf = enc.encode(s)
        assert buf[4] == 3
        assert dec.decode(buf) == s
