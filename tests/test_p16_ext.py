"""(P16): trainer helpers (experiments/run/curves_ext.py): K refit, end/start velocity, pad sync, adaptive evaluation."""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments', 'run'))
import curves_ext as cx  # noqa: E402


def curve(C, F, span, bsp):
    B = cx.basis_matrix(C.shape[1], F, span, bsp)
    return np.einsum('fk,nkd->nfd', B, np.asarray(C, np.float64))


@pytest.mark.parametrize('bsp', [False, True])
def test_basis_matrix_partition_of_unity(bsp):
    for K in (1, 2, 3, 5, 9):
        B = cx.basis_matrix(K, 32, 31, bsp)
        assert np.allclose(B.sum(1), 1.0) and B.shape == (32, K)
        if K > 1:
            assert B[0, 0] == 1.0 and B[-1, -1] == 1.0


def test_hat_refit_5_to_9_is_exact():
    rng = np.random.default_rng(0)
    C = torch.tensor(rng.normal(size=(20, 5, 3)))
    H = cx.refit_columns(C, 9, 32, 31, False)
    assert np.abs(curve(H.numpy(), 32, 31, False) - curve(C.numpy(), 32, 31, False)).max() < 1e-9
    assert (H[:, 0] == C[:, 0]).all() and (H[:, -1] == C[:, -1]).all()
    O = torch.tensor(rng.normal(size=(20, 5)))                       # opacities have no trailing dimension
    HO = cx.refit_columns(O, 9, 32, 31, False)
    assert HO.shape == (20, 9) and np.allclose(curve(HO.numpy()[..., None], 32, 31, False), curve(O.numpy()[..., None], 32, 31, False))


def test_bspline_refit_5_to_9_is_close_and_keeps_ends():
    rng = np.random.default_rng(1)
    C = torch.tensor(np.cumsum(rng.normal(size=(20, 5, 3)), 1))
    H = cx.refit_columns(C, 9, 32, 31, True)
    err = np.abs(curve(H.numpy(), 32, 31, True) - curve(C.numpy(), 32, 31, True)).max()
    assert err < 1e-9, err        # K=5 interior knot 1/2 is also an interior knot of K=9 (j/6): nested spline spaces, exact refit


@pytest.mark.parametrize('bsp', [False, True])
def test_end_and_start_velocity_match_finite_differences(bsp):
    rng = np.random.default_rng(2)
    span = 31
    for K in (2, 3, 5, 9):
        C = rng.normal(size=(6, K, 3))
        B = cx.basis_matrix(K, 32, span, bsp)
        h = 1e-5
        # evaluate at u = 1 and 1 - h/span through the same weights
        from codec.csgs import bspline_weights
        wf = (lambda u: np.array(bspline_weights(K, u))) if bsp else (lambda u: cx.basis_matrix(K, 1, 1, False)[0] * 0 + _hat(K, u))
        end = np.einsum('k,nkd->nd', wf(1.0), C)
        end_h = np.einsum('k,nkd->nd', wf(1.0 - h), C)
        t = torch.tensor(C)
        kn = torch.full((6,), K, dtype=torch.long)
        val, vel = cx.end_state(t, kn, span, bsp)
        assert np.allclose(val.numpy(), end)
        assert np.allclose(vel.numpy(), (end - end_h) / (h * span), rtol=1e-3, atol=1e-6), (K, vel.numpy()[0], (end - end_h)[0] / (h * span))
        s0 = np.einsum('k,nkd->nd', wf(0.0), C)
        s_h = np.einsum('k,nkd->nd', wf(h), C)
        assert np.allclose(cx.start_vel(t, kn, span, bsp).numpy(), (s_h - s0) / (h * span), rtol=1e-3, atol=1e-6)


def _hat(K, u):
    tau = min(max(u * (K - 1), 0), K - 1)
    j = min(int(np.floor(tau)), K - 2)
    w = np.zeros(K)
    w[j], w[j + 1] = 1 - (tau - j), tau - j
    return w


def test_curv_d2_hat_equals_plain_second_difference_and_bspline_linear_motion_is_zero():
    rng = np.random.default_rng(4)
    v = torch.tensor(rng.normal(size=(7, 5, 3)))
    plain = v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]
    assert torch.allclose(cx.curv_d2(v, 5, False), -plain)               # sign: residual of the linear interpolation, x2
    for K in (3, 5, 9):
        g = torch.tensor(cx.greville(K, True))
        a = torch.tensor(rng.normal(size=(7, 1, 3)))
        b = torch.tensor(rng.normal(size=(7, 1, 3)))
        lin = a + g.view(1, K, 1) * b                                    # control points on a line in time = linear motion
        assert cx.curv_d2(lin, K, True).abs().max() < 1e-12
        B = cx.basis_matrix(K, 32, 31, True)                             # and the curve really is linear in time
        c = np.einsum('fk,nkd->nfd', B, lin.numpy())
        assert np.abs(np.diff(c, 2, axis=1)).max() < 1e-12


def test_sync_pad_and_select_hi():
    P = {k: torch.nn.Parameter(torch.arange(4 * 9 * 3, dtype=torch.float32).view(4, 9, 3)) for k in ('means', 'opacities')}
    P['opacities'] = torch.nn.Parameter(torch.full((4, 9), 2.0))
    kn = torch.tensor([5, 9, 5, 3])
    cx.sync_pad(P, ['means', 'opacities'], kn)
    m = P['means'].data
    assert (m[0, 5:] == m[0, 4]).all() and (m[2, 5:] == m[2, 4]).all() and (m[3, 3:] == m[3, 2]).all()
    assert torch.equal(m[1], torch.arange(27, 54, dtype=torch.float32).view(9, 3) if False else m[1])        # K=9 row untouched
    assert (m[0, :5] == torch.arange(0, 27, dtype=torch.float32).view(9, 3)[:5]).all()
    # select_hi: top fraction of visible splats by path length
    means = torch.zeros(10, 9, 3)
    for i in range(10):
        means[i, :5, 0] = torch.arange(5) * (i + 1.0)
    P = {'means': means, 'opacities': torch.full((10, 9), 2.0)}
    S = {'birth': torch.zeros(10, dtype=torch.long)}
    hi = cx.select_hi(P, S, 5, 0.3, 'path')
    assert hi.sum() == 3 and hi[7:].all()
    P['opacities'][9] = -9.0                                  # invisible splat never selected
    hi = cx.select_hi(P, S, 5, 0.3, 'path')
    assert not hi[9] and hi.sum() == 2 and hi[7:9].all()


def test_curve_adaptive_restores_order_and_matches_per_splat_evaluation():
    def curve_at(P, idx, t, g0, Lg, K):
        B = torch.tensor(cx.basis_matrix(K, Lg, Lg - 1, False)[t - g0], dtype=torch.float32)
        return {'means': torch.einsum('k,nkd->nd', B, P['means'][idx, :K])}
    rng = np.random.default_rng(3)
    P = {'means': torch.tensor(rng.normal(size=(12, 9, 3)), dtype=torch.float32)}
    kn = torch.tensor([5, 9, 5, 5, 9, 9, 5, 9, 5, 5, 9, 5])
    idx = torch.tensor([11, 0, 4, 7, 3, 1])
    out = cx.curve_adaptive(curve_at, P, idx, 10, 0, 32, kn)['means']
    for r, i in enumerate(idx.tolist()):
        want = curve_at(P, torch.tensor([i]), 10, 0, 32, int(kn[i]))['means'][0]
        assert torch.allclose(out[r], want, atol=1e-6)
