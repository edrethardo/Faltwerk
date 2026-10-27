"""(P16): prep.sections_from_curves with basis='bspline' and adaptive K (g<i>_kn): stream roundtrip, flags, K per splat,
decoded curves vs the trained float curves (within the quantisation step)."""
import numpy as np
import pytest

from codec import prep
from codec.csgs import Decoder, Encoder, FLAG_BSPLINE, bspline_weights

L, K, T, N = 32, 9, 64, 60


def make_run(tmp_path, basis, adaptive, seed=0):
    K = 9 if adaptive else 5
    rng = np.random.default_rng(seed)
    kn = np.where(np.arange(N) % 3 == 0, 9, 5) if adaptive else np.full(N, 5)
    cv = {'L': L, 'K': K, 'knots': 'gop', 'soft_knot0': True, 'basis': basis}
    for gi in range(T // L):
        span = L - 1
        base = rng.normal(size=(N, 1, 3)) * 0.3
        # smooth curves: control points on a smooth path (so K reduction has something to do), padded beyond kn
        t = np.linspace(0, 1, K)[None, :, None]
        vel = rng.normal(size=(N, 1, 3)) * 0.2
        P = base + vel * t + 0.05 * np.sin(6 * t + rng.normal(size=(N, 1, 3)))
        for i in range(N):
            P[i, kn[i]:] = P[i, kn[i] - 1]
        cv[f'g{gi}_ids'] = np.arange(N, dtype=np.int32)
        cv[f'g{gi}_birth'] = np.zeros(N, np.int32)
        cv[f'g{gi}_death'] = np.full(N, T, np.int32)
        cv[f'g{gi}_means'] = P.astype(np.float32)
        q = rng.normal(size=(N, K, 4)) * 0.05 + np.array([1.0, 0, 0, 0])
        cv[f'g{gi}_quats'] = q.astype(np.float32)
        cv[f'g{gi}_scales'] = (rng.normal(size=(N, K, 3)) * 0.1 - 5).astype(np.float32)
        cv[f'g{gi}_opacities'] = (rng.normal(size=(N, K)) * 0.5 + 1).astype(np.float32)
        cv[f'g{gi}_sh0'] = (rng.normal(size=(N, K, 3)) * 0.2).astype(np.float32)
        if adaptive:
            cv[f'g{gi}_kn'] = kn.astype(np.int16)
    means = np.full((T, N, 3), np.nan, np.float32)
    means[:] = cv['g0_means'][:, 0][None]
    np.savez(tmp_path / 'curves.npz', **cv)
    np.savez(tmp_path / 'tracks.npz', birth=np.zeros(N, np.int32), death=np.full(N, T, np.int32),
             shN=np.zeros((N, 15, 3), np.float16), means=means)
    return tmp_path / 'curves.npz', tmp_path / 'tracks.npz', kn


@pytest.mark.parametrize('adaptive', [False, True])
def test_bspline_curves_to_csgs_roundtrip(tmp_path, adaptive):
    c, t, kn = make_run(tmp_path, 'bspline', adaptive)
    params, secs, stats = prep.sections_from_curves(str(c), str(t), qpos=0.001, key_interval=4, quat='raw', tol=1, ext=1.5)
    assert params.bspline and params.ks == ([1, 2, 3, 5, 9] if adaptive else [1, 2, 3, 5])
    enc, dec = Encoder(params), Decoder()
    bufs = [enc.encode(s) for s in secs]
    assert all(b[5] & FLAG_BSPLINE and b[4] == 6 for b in bufs)
    for s, b in zip(secs, bufs):
        assert dec.decode(b) == s
    ks = np.array(secs[0].k[0])
    assert ks.max() <= 9
    if adaptive:                                                     # splats trained with K=5 never come back as K=9
        assert (ks[kn == 5] <= 5).all()
        assert (ks[kn == 9] > 5).any()                              # fast (K=9) splats keep more than 5 control points
    # decoded curves vs trained float curves: within a few position steps (K reduction tolerance 1 step + rounding)
    cvf = np.load(c)
    step = 0.001 * 1.5
    p0 = secs[0]
    for i in range(0, N, 7):
        row = np.array(p0.knots[0][i], np.float64).reshape(-1, 3)
        Kd = len(row)
        deq = params.offsets[0:3] + row * np.array(params.scales[0:3])
        for f in (0, 5, 17, 31):
            got = np.array(bspline_weights(Kd, f / (L - 1))) @ deq
            ki = int(kn[i])
            want = np.array(bspline_weights(ki, f / (L - 1))) @ cvf['g0_means'][i, :ki].astype(np.float64)
            assert np.abs(got - want).max() < 4 * step, (i, f, np.abs(got - want).max())


def test_ktol_group_overrides_only_its_group(tmp_path):
    # (P22): per-group K tolerance; None = unchanged, {'D': big} collapses colour curves to K=1 only
    c, t, _ = make_run(tmp_path, 'bspline', False)
    _, s0, _ = prep.sections_from_curves(str(c), str(t), quat='raw', tol=1, ext=1.5)
    _, s1, _ = prep.sections_from_curves(str(c), str(t), quat='raw', tol=1, ext=1.5, ktol_groups=None)
    assert [s.knots for s in s0] == [s.knots for s in s1]
    _, s2, st = prep.sections_from_curves(str(c), str(t), quat='raw', tol=1, ext=1.5, ktol_groups={'D': 1000})
    gi = [g for g, _ in prep.GROUPS].index('D')
    for a, b in zip(s0, s2):
        assert all(k == 1 for k in b.k[gi])
        assert [x for j, x in enumerate(a.k) if j != gi] == [x for j, x in enumerate(b.k) if j != gi]
    enc, dec = Encoder(prep.sections_from_curves(str(c), str(t), quat='raw', tol=1, ext=1.5)[0]), Decoder()
    assert all(dec.decode(enc.encode(s)) == s for s in s2)


def test_hat_default_is_unchanged_by_the_new_arguments(tmp_path):
    c, t, kn = make_run(tmp_path, 'hat', False)
    p_auto, s_auto, _ = prep.sections_from_curves(str(c), str(t), quat='raw', tol=1, ext=1.5)
    p_hat, s_hat, _ = prep.sections_from_curves(str(c), str(t), quat='raw', tol=1, ext=1.5, basis='hat')
    assert not p_auto.bspline and p_auto == p_hat
    assert [s.knots for s in s_auto] == [s.knots for s in s_hat]
    enc = Encoder(p_auto)
    assert all(enc.encode(s)[4] == 1 for s in s_auto[:1])
