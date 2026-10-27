"""(S1): SN-VQ encoder side (codec/snvq.py) and its prep integration on synthetic E7-style data."""
import numpy as np
import pytest

from codec import snvq
from codec.csgs import Decoder, Encoder, stream_sizes, vq_row
from codec.prep import sections_from_curves


def synth_run(tmp_path, T=8, L=4, K=3, N=60, seed=0):
    """curves.npz + tracks.npz like e7_curves_train writes them: 2 sections, some splats die / are born in section 1."""
    rng = np.random.default_rng(seed)
    birth = np.zeros(N, np.int32); death = np.full(N, T, np.int32)
    birth[40:] = L + rng.integers(0, L, N - 40)            # 20 newborns in section 1
    death[5:12] = L + rng.integers(1, L + 1, 7)            # 7 die in section 1
    death[12:15] = L                                       # 3 die exactly at the boundary
    shN = rng.normal(size=(N, 15, 3)).astype(np.float32) * np.array([1.0] * 3 + [0.3] * 5 + [0.1] * 7, np.float32)[None, :, None]
    shN[:8] = 0.001 * rng.normal(size=(8, 15, 3))           # 8 splats with (almost) no higher bands
    cv = {'L': L, 'K': K, 'knots': 'gop', 'soft_knot0': True}
    tracks = {k: np.full((T, N) + s, np.nan, np.float32) for k, s in
              (('means', (3,)), ('quats', (4,)), ('scales', (3,)), ('opacities', ()), ('sh0', (1, 3)))}
    for gi in range(2):
        g0 = gi * L
        ids = np.nonzero((birth < g0 + L) & (death > g0))[0] if gi else np.arange(40)
        n = len(ids)
        cv[f'g{gi}_ids'] = ids.astype(np.int32)
        cv[f'g{gi}_birth'] = birth[ids]; cv[f'g{gi}_death'] = death[ids]
        cv[f'g{gi}_means'] = (rng.normal(size=(n, 1, 3)) + 0.01 * rng.normal(size=(n, K, 3))).astype(np.float32)
        cv[f'g{gi}_quats'] = (rng.normal(size=(n, 1, 4)) + 0.01 * rng.normal(size=(n, K, 4))).astype(np.float32)
        cv[f'g{gi}_scales'] = (-3 + rng.normal(size=(n, K, 3)) * 0.1).astype(np.float32)
        cv[f'g{gi}_opacities'] = (2 + rng.normal(size=(n, K)) * 0.1).astype(np.float32)
        cv[f'g{gi}_sh0'] = rng.normal(size=(n, K, 3)).astype(np.float32)
        for t in range(g0, g0 + L):
            alive = ids[(birth[ids] <= t) & (death[ids] > t)]
            tracks['means'][t, alive] = cv[f'g{gi}_means'][np.searchsorted(ids, alive), 0]
            tracks['quats'][t, alive] = 1; tracks['scales'][t, alive] = -3
            tracks['opacities'][t, alive] = 2; tracks['sh0'][t, alive] = 0
    np.savez(tmp_path / 'curves.npz', **cv)
    np.savez(tmp_path / 'tracks.npz', birth=birth, death=death, shN=shN.astype(np.float16), **tracks)
    return str(tmp_path / 'curves.npz'), str(tmp_path / 'tracks.npz'), shN.reshape(N, -1)


def roundtrip(p, secs):
    enc, dec = Encoder(p), Decoder()
    bufs = [enc.encode(s) for s in secs]
    for s, b in zip(secs, bufs):
        out = dec.decode(b)
        assert out == s, out.first_difference(s)
    return bufs


def test_degree_rule_fractions_and_energy_order():
    rng = np.random.default_rng(1)
    SN = rng.normal(size=(100, 45))
    SN[:10] *= 0.01                                     # tiny everywhere -> degree 0
    SN[10:30, 9:] *= 0.01                               # only band 1 -> degree 1
    deg = snvq.degree_rule(SN, np.ones(100), [0, 1, 3], [0.1, 20 / 90])
    assert (deg[:10] == 0).all() and (deg[10:30] == 1).all() and (deg[30:] == 3).all()
    assert snvq.band_energy(SN, 1).shape == (100,) and snvq.band_energy(SN, 3).max() == 0
    # importance weight: an invisible splat drops to degree 0 before a strong one with the same SH energy
    w = np.ones(100); w[50] = 1e-6
    deg = snvq.degree_rule(SN, w, [0, 3], [0.11])
    assert deg[50] == 0


def test_kmeans_quantised_codebook_beats_a_single_entry():
    rng = np.random.default_rng(2)
    X = np.concatenate([rng.normal(loc=c, scale=0.05, size=(200, 9)) for c in (-1.0, 0.0, 1.0)])
    off, scale = np.full(9, -2.0), np.full(9, 4 / 255)
    cbs, idx, rec = snvq.train_level(X, np.ones(len(X)), [3], off, scale)
    assert cbs[0].shape == (3, 9) and cbs[0].min() >= 0 and cbs[0].max() <= 255
    assert ((rec - X) ** 2).mean() < 0.05 ** 2 * 1.5
    # residual stage lowers the error further and is signed
    cbs2, idx2, rec2 = snvq.train_level(X, np.ones(len(X)), [3, 4], off, scale)
    assert ((rec2 - X) ** 2).mean() < ((rec - X) ** 2).mean() and cbs2[1].min() < 0
    # padding when fewer points than entries
    C = snvq.kmeans(X[:2], np.ones(2), 5)
    assert C.shape == (5, 9)


def test_prep_sn_vq_roundtrip_and_v1_unchanged(tmp_path):
    curves, tracks, SN = synth_run(tmp_path)
    p1, s1, st1 = sections_from_curves(curves, tracks, quat='raw', tol=1)
    b1 = roundtrip(p1, s1)
    assert p1.sn_vq is None and all(b[4] == 1 for b in b1)
    opts = {'levels': [(1, [4]), (3, [8, 4])], 'q': [0.1, 0.3], 'weight': 'importance', 'iters': 10, 'seed': 0,
            'save': str(tmp_path / 'assign.npz')}
    p2, s2, st2 = sections_from_curves(curves, tracks, quat='raw', tol=1, sn_vq=opts)
    assert p2.sn_vq == opts['levels'] and len(s2) == 2 and s2[0].sn_cb is not None and s2[1].sn_cb is None
    b2 = roundtrip(p2, s2)
    # knots identical to v1 (only the SN part changed), v2 headers, the raw sn stream empty
    for a, b in zip(s1, s2):
        assert a.knots == b.knots and a.k == b.k and a.ids == b.ids
    assert all(b[4] == 2 for b in b2) and all(stream_sizes(b)['sn'] == 0 for b in b2)
    hist = st2['sn_vq']['groups'][0]['deg_hist']
    assert set(hist) == {0, 1, 3} and hist[0] + hist[1] + hist[3] == 60 and hist[0] == 6 and hist[1] == 16
    # newborns of the delta section are coded against the key section's codebook
    assert stream_sizes(b2[1])['sn_idx'] > 0 and stream_sizes(b2[1])['sn_cb'] == 0
    # reconstruction: degree-0 rows are the zero row, degree-1 rows keep the zero row beyond 9 scalars
    for s in s2:
        for row, dg in zip(s.sn, s.sn_deg):
            if dg == 0:
                assert row == p2.sn_zero
            elif dg == 1:
                assert row[9:] == p2.sn_zero[9:]
    # saved assignment reproduces the identical bitstream (retraining hand-over)
    A = snvq.Assignment.load(str(tmp_path / 'assign.npz'))
    p3, s3, st3 = sections_from_curves(curves, tracks, quat='raw', tol=1, sn_vq={'levels': A.levels, 'from': A})
    assert st3['sn_vq']['groups'][0]['reused']
    assert [Encoder(p3).encode(s) if i == 0 else None for i, s in enumerate(s3[:1])][0] == b2[0]
    assert roundtrip(p3, s3) == b2
    # ... also when the tracks' SN changed to the reconstruction (what the retrained run carries)
    rec = A.float_rows(0, range(len(A.groups[0]['ids'])))
    tr = dict(np.load(tracks))
    shn = tr['shN'].astype(np.float32).reshape(len(tr['birth']), -1)
    shn[A.groups[0]['ids']] = rec
    tr['shN'] = shn.reshape(-1, 15, 3)
    np.savez(tmp_path / 'tracks2.npz', **tr)
    p4, s4, _ = sections_from_curves(curves, str(tmp_path / 'tracks2.npz'), quat='raw', tol=1, sn_vq={'levels': A.levels, 'from': A})
    assert roundtrip(p4, s4) == b2 and p4.offsets[-45:] == p2.offsets[-45:] and p4.scales[-45:] == p2.scales[-45:]


def test_prep_sn_vq_degree_thresholds_and_zero_row(tmp_path):
    curves, tracks, SN = synth_run(tmp_path, seed=3)
    opts = {'levels': [(3, [16])], 'thr': [1e-3], 'iters': 5}
    p, secs, st = sections_from_curves(curves, tracks, quat='raw', sn_vq=opts)
    hist = st['sn_vq']['groups'][0]['deg_hist']
    assert hist[0] == 8 and hist[3] == 52                                   # the 8 near-zero splats
    off, sc = np.array(p.offsets[-45:]), np.array(p.scales[-45:])
    assert np.abs(off + np.array(p.sn_zero) * sc).max() <= sc.max() / 2 + 1e-6   # zero row = grid point nearest 0.0
    roundtrip(p, secs)


def test_parse_levels():
    assert snvq.parse_levels('1:1024,3:4096+1024') == [(1, [1024]), (3, [4096, 1024])]
